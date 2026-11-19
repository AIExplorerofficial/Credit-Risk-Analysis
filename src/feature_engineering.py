"""Phase 3: feature engineering + time-based splits.

Run from the creditlens/ folder:  python -m src.feature_engineering

Writes (all keep the ORIGINAL row index of loans_36m.parquet and include the target column):
  train.parquet, early_stop.parquet, calibration.parquet, test_oot.parquet
  models/preprocessor.pkl, and audit decisions merged into models/metadata.json
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config_loader import Cfg, get_banned_columns, load_config

NUM_MEDIAN = ["loan_amnt", "annual_inc", "dti", "fico_avg", "open_acc", "revol_bal",
              "revol_util", "total_acc", "credit_history_years"]
NUM_ZERO = ["emp_length", "pub_rec", "delinq_2yrs", "inq_last_6mths"]   # missing -> 0
RAW_NUMERIC = ["loan_amnt", "annual_inc", "dti", "open_acc", "pub_rec", "revol_bal", "revol_util",
               "total_acc", "delinq_2yrs", "inq_last_6mths", "mort_acc"]
HOME_KEEP = ["RENT", "OWN", "MORTGAGE"]


# ---------------------------------------------------------------- raw feature helpers
def parse_emp_length(series: pd.Series) -> pd.Series:
    """'10+ years' -> 10, '< 1 year' -> 0, '3 years' -> 3, missing -> 0."""
    s = series.astype("object").where(series.notna(), "").astype(str)
    out = s.str.extract(r"(\d+)")[0].astype("float64")
    out[s.str.contains("<", regex=False)] = 0.0
    return out.fillna(0.0)


def compute_fico_avg(df: pd.DataFrame) -> pd.Series:
    return (pd.to_numeric(df["fico_range_low"], errors="coerce")
            + pd.to_numeric(df["fico_range_high"], errors="coerce")) / 2.0


def compute_credit_history_years(df: pd.DataFrame) -> pd.Series:
    yrs = (df["issue_d"] - df["earliest_cr_line"]).dt.days / 365.25
    return yrs.clip(lower=0).astype("float64")        # negative -> 0; NaN stays NaN (imputed later)


def engineer_raw_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw loan rows -> model-ready raw columns (NOT yet scaled or encoded). Same function for training and serving."""
    out = pd.DataFrame(index=df.index)
    for c in RAW_NUMERIC:
        out[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    out["emp_length"] = parse_emp_length(df["emp_length"])
    out["fico_avg"] = compute_fico_avg(df).astype("float64")
    if "credit_history_years" in df.columns:
        out["credit_history_years"] = pd.to_numeric(df["credit_history_years"], errors="coerce").astype("float64")
    else:
        out["credit_history_years"] = compute_credit_history_years(df)
    out["mort_acc_missing"] = df["mort_acc"].isna().astype("float64")

    home = df["home_ownership"].astype("object").where(df["home_ownership"].notna(), "OTHER").astype(str)
    out["home_ownership"] = home.where(home.isin(HOME_KEEP), "OTHER")
    out["purpose"] = df["purpose"].astype("object").where(df["purpose"].notna(), "OTHER").astype(str)
    out["application_type"] = (df["application_type"].astype("object")
                               .where(df["application_type"].notna(), "Individual").astype(str))
    return out


# ---------------------------------------------------------------- audits (run on train rows only)
def audit_mort_acc(df: pd.DataFrame) -> dict:
    """Is mort_acc missingness acting as an issue-year proxy? df must have issue_d and mort_acc."""
    year = df["issue_d"].dt.year
    rate = df["mort_acc"].isna().groupby(year).mean().sort_index()
    spread = float(rate.max() - rate.min())
    d = rate.diff().dropna()
    monotone = len(d) > 0 and bool((d >= 0).all() or (d <= 0).all())
    decision = "drop" if (spread > 0.20 or (monotone and spread > 0.02)) else "keep_with_indicator"
    return {"decision": decision, "missing_rate_by_year": {int(k): round(float(v), 4) for k, v in rate.items()},
            "spread": round(spread, 4), "monotone": monotone}


def audit_application_type(df: pd.DataFrame) -> dict:
    share = df["application_type"].value_counts(normalize=True, dropna=True)
    top = float(share.max())
    return {"decision": "drop" if top > 0.98 else "keep", "top_share": round(top, 4),
            "shares": {str(k): round(float(v), 4) for k, v in share.items()}}


# ---------------------------------------------------------------- preprocessor
def build_preprocessor(cfg: Cfg, mort_decision: str, app_decision: str) -> ColumnTransformer:
    num_zero = list(NUM_ZERO) + (["mort_acc"] if mort_decision == "keep_with_indicator" else [])
    flags = ["mort_acc_missing"] if mort_decision == "keep_with_indicator" else []
    cats = ["home_ownership", "purpose"] + (["application_type"] if app_decision == "keep" else [])

    transformers = [
        ("num_med", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUM_MEDIAN),
        ("num_zero", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value=0.0)),
                               ("sc", StandardScaler())]), num_zero),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                              max_categories=int(cfg.features.purpose_top_n) + 1,
                              sparse_output=False), cats),
    ]
    if flags:
        transformers.append(("flag", "passthrough", flags))
    ct = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=True)
    ct.set_output(transform="pandas")
    return ct


def fit_and_save_preprocessor(train_raw: pd.DataFrame, cfg: Cfg, mort_decision: str,
                              app_decision: str, out_path: str | Path) -> ColumnTransformer:
    pre = build_preprocessor(cfg, mort_decision, app_decision)
    pre.fit(train_raw)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pre, out_path, compress=3)
    return pre


def apply_preprocessor(df_raw: pd.DataFrame, preprocessor: ColumnTransformer) -> pd.DataFrame:
    return preprocessor.transform(df_raw)


# ---------------------------------------------------------------- splits
def build_splits(cfg: Cfg) -> None:
    loans = pd.read_parquet(cfg.paths.loans_parquet)
    assert loans.index.is_unique, "loans index must be unique"
    target = cfg.target.column
    year, month = loans["issue_d"].dt.year, loans["issue_d"].dt.month

    train_years = list(cfg.splits.train_years)
    last_train_year = max(train_years)
    n_es = int(cfg.splits.early_stop_months)

    es_mask = (year == last_train_year) & (month >= 13 - n_es)
    train_mask = year.isin(train_years) & ~es_mask
    cal_mask = year.isin(list(cfg.splits.calibration_years))
    test_mask = year.isin(list(cfg.splits.test_years))
    masks = {"train": train_mask, "early_stop": es_mask, "cal": cal_mask, "test": test_mask}

    names = list(masks)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not (masks[a] & masks[b]).any(), f"splits overlap: {a} and {b}"
        assert masks[a].sum() > 0, f"split {a} is empty"

    train_rows = loans[train_mask]
    mort = audit_mort_acc(train_rows)
    app = audit_application_type(train_rows)
    print("mort_acc audit:", json.dumps(mort))
    print("application_type audit:", json.dumps(app))

    out_paths = {"train": cfg.paths.train_parquet, "early_stop": cfg.paths.early_stop_parquet,
                 "cal": cfg.paths.calibration_parquet, "test": cfg.paths.test_parquet}
    Path(cfg.paths.processed_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.model_dir).mkdir(parents=True, exist_ok=True)

    pre = fit_and_save_preprocessor(engineer_raw_features(train_rows), cfg, mort["decision"], app["decision"],
                                    Path(cfg.paths.model_dir) / "preprocessor.pkl")

    banned = set(get_banned_columns(cfg))
    sizes, rates = {}, {}
    for name, mask in masks.items():
        rows = loans[mask]
        X = apply_preprocessor(engineer_raw_features(rows), pre)
        assert X.index.equals(rows.index), "preprocessor changed the row index"
        stripped = {c.split("__", 1)[-1] for c in X.columns}
        assert not (stripped & banned), f"banned columns in features: {stripped & banned}"
        assert not any(c.startswith("issue_") for c in stripped), "issue_* column in features"
        X[target] = rows[target].astype("int8").to_numpy()
        X.to_parquet(out_paths[name], index=True)
        sizes[name], rates[name] = int(len(X)), round(float(X[target].mean()), 4)
        print(f"{name:11s} shape={X.shape} default_rate={rates[name]}")

    feature_columns = [c for c in X.columns if c != target]
    meta_path = Path(cfg.paths.model_dir) / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update({"audit_mort_acc": mort, "audit_application_type": app, "split_sizes": sizes,
                 "split_default_rates": rates, "feature_columns": feature_columns})
    meta_path.write_text(json.dumps(meta, indent=2))
    print("\nfeature columns:", feature_columns)
    print("saved parquets, preprocessor.pkl, metadata.json")


def main(config_path: str = "config.yaml") -> None:
    build_splits(load_config(config_path))


if __name__ == "__main__":
    main()
