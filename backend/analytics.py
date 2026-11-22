"""Metrics and cohorts computed from the saved 02_training.ipynb data and artifacts."""
from __future__ import annotations

import sys
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as parquet
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from creditview.notebook_service import CreditDecisionService
from creditview.schemas import LoanApplication

_ROOT = Path(__file__).resolve().parents[1]
_RUNTIME = _ROOT / "notebook_runtime"
if str(_RUNTIME) not in sys.path:
    sys.path.insert(0, str(_RUNTIME))

from src.calibrate import expected_calibration_error  # noqa: E402
from src.conformal import check_oos_rules, decide, get_prediction_set  # noqa: E402
from src.feature_engineering import engineer_raw_features  # noqa: E402
from src.explainer import compute_shap_values  # noqa: E402
from src.threshold import compute_profit_per_threshold, select_optimal_threshold  # noqa: E402

_FIELDS = [
    "loan_amnt", "annual_inc", "dti", "fico_avg", "emp_length", "home_ownership", "purpose",
    "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc", "delinq_2yrs",
    "inq_last_6mths", "credit_history_years",
]
_LABELS = {
    "loan_amnt": "Requested loan amount ($)", "annual_inc": "Annual income ($)",
    "dti": "Debt-to-income ratio (%)", "fico_avg": "FICO score",
    "emp_length": "Employment length (years)", "home_ownership": "Home ownership",
    "purpose": "Loan purpose", "open_acc": "Open credit accounts", "pub_rec": "Public records",
    "revol_bal": "Revolving balance ($)", "revol_util": "Revolving utilization (%)",
    "total_acc": "Total credit accounts", "delinq_2yrs": "Delinquencies in past 2 years",
    "inq_last_6mths": "Credit inquiries in past 6 months", "credit_history_years": "Credit history (years)",
}
_LIMITS = {"historical_example_count": 12, "max_historical_examples": 50,
           "max_batch_rows": 1000, "max_upload_bytes": 10_485_760}
_SPLIT_FILES = {"train": "train", "early_stop": "early_stop",
                "cal": "calibration", "test": "test_oot"}
_LOANS_PATH = _RUNTIME / "data" / "processed" / "loans_36m.parquet"


@lru_cache(maxsize=1)
def _loan_column_names() -> tuple[str, ...]:
    return tuple(parquet.ParquetFile(_LOANS_PATH).schema.names)


def _load_split(role: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Read one notebook split for the duration of the current calculation."""
    if role not in _SPLIT_FILES:
        raise ValueError(f"Unknown notebook split: {role}")
    return pd.read_parquet(_RUNTIME / "data" / "processed" / f"{_SPLIT_FILES[role]}.parquet",
                           columns=columns)


def _split_index_column(parquet_file) -> str:
    index_columns = parquet_file.schema_arrow.pandas_metadata.get("index_columns", [])
    names = parquet_file.schema.names
    for index_column in index_columns:
        if isinstance(index_column, str) and index_column in names:
            return index_column
    raise RuntimeError("Notebook split is missing its original row index")


def _iter_split_frames(role: str, columns: list[str], selected_indices=None,
                       batch_size: int = 16_384):
    """Yield split frames in stored row order, optionally selecting original row ids."""
    if role not in _SPLIT_FILES:
        raise ValueError(f"Unknown notebook split: {role}")
    path = _RUNTIME / "data" / "processed" / f"{_SPLIT_FILES[role]}.parquet"
    parquet_file = parquet.ParquetFile(path)
    index_column = _split_index_column(parquet_file)
    batch_columns = [index_column] + [name for name in columns if name != index_column]
    wanted = None
    if selected_indices is not None:
        selected = np.asarray(selected_indices, dtype=np.int64)
        loan_count = int(parquet.ParquetFile(_LOANS_PATH).metadata.num_rows)
        if selected.size and (selected.min() < 0 or selected.max() >= loan_count):
            raise KeyError("Requested row index is outside the notebook loans dataset")
        wanted = np.zeros(loan_count, dtype=bool)
        wanted[selected] = True
    for batch in parquet_file.iter_batches(columns=batch_columns, batch_size=batch_size):
        frame = batch.to_pandas()
        if index_column in frame.columns:
            row_indices = frame.pop(index_column).to_numpy(dtype=np.int64, copy=False)
            frame.index = pd.Index(row_indices)
        else:
            row_indices = frame.index.to_numpy(dtype=np.int64, copy=False)
        if wanted is not None:
            mask = wanted[row_indices]
            if not mask.any():
                continue
            frame = frame.iloc[np.flatnonzero(mask)]
        yield frame


def _read_split_rows(role: str, indices, columns: list[str]) -> pd.DataFrame:
    requested = pd.Index(indices)
    pieces = list(_iter_split_frames(role, columns, selected_indices=requested))
    result = pd.concat(pieces, axis=0) if pieces else pd.DataFrame(columns=columns)
    if len(result) != len(requested):
        raise KeyError("One or more requested notebook split row indices were not found")
    return result.reindex(requested)


def _score_split(role: str, *, selected_indices=None, include_base: bool = False,
                 include_lr: bool = False, include_sets: bool = False) -> dict[str, np.ndarray]:
    """Score notebook rows in bounded batches, retaining only result vectors."""
    svc = CreditDecisionService.get_instance()
    target = str(svc.cfg.target.column)
    columns = [*svc.feature_columns, target]
    row_indices, labels, calibrated, base, logistic, prediction_sets = [], [], [], [], [], []
    for frame in _iter_split_frames(role, columns, selected_indices=selected_indices):
        X = frame[svc.feature_columns]
        row_indices.append(frame.index.to_numpy(dtype=np.int64, copy=True))
        labels.append(frame[target].to_numpy(dtype=int, copy=True))
        with svc.inference_lock:
            calibrated.append(svc.lgbm_calibrated.predict_proba(X)[:, 1])
            if include_base:
                base.append(svc.lgbm_base.predict_proba(X)[:, 1])
            if include_lr:
                logistic.append(svc.lr_baseline.predict_proba(X)[:, 1])
            if include_sets:
                prediction_sets.append(get_prediction_set(svc.conformal_model, X))
    output = {
        "indices": np.concatenate(row_indices) if row_indices else np.asarray([], dtype=np.int64),
        "y": np.concatenate(labels) if labels else np.asarray([], dtype=int),
        "p_calibrated": np.concatenate(calibrated) if calibrated else np.asarray([], dtype=float),
    }
    if include_base:
        output["p_base"] = np.concatenate(base) if base else np.asarray([], dtype=float)
    if include_lr:
        output["p_lr"] = np.concatenate(logistic) if logistic else np.asarray([], dtype=float)
    if include_sets:
        output["prediction_sets"] = np.concatenate(prediction_sets, axis=0) if prediction_sets else np.empty((0, 2), dtype=bool)
    if selected_indices is not None:
        requested = pd.Index(selected_indices)
        order = pd.Index(output["indices"]).get_indexer(requested)
        if np.any(order < 0) or len(order) != len(requested):
            raise KeyError("One or more requested notebook split rows were not found")
        for key, values in output.items():
            output[key] = values[order]
    return output


@lru_cache(maxsize=1)
def _cal_b_indices() -> tuple[int, ...]:
    """Return the exact seeded cal_B row order using only the target column."""
    svc = CreditDecisionService.get_instance()
    target = str(svc.cfg.target.column)
    calibration = _load_split("cal", columns=[target])
    _, cal_b = train_test_split(
        calibration.index, test_size=0.5, stratify=calibration[target],
        random_state=int(svc.cfg.modeling.random_seed),
    )
    return tuple(int(index) for index in cal_b)


def _calibration_halves(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    svc = CreditDecisionService.get_instance()
    target = str(svc.cfg.target.column)
    return train_test_split(
        frame, test_size=0.5, stratify=frame[target],
        random_state=int(svc.cfg.modeling.random_seed),
    )


def _load_cal_b() -> pd.DataFrame:
    calibration = _load_split("cal")
    cal_a, cal_b = _calibration_halves(calibration)
    del calibration, cal_a
    return cal_b


def _read_loans_rows(indices, columns: list[str]) -> pd.DataFrame:
    """Select original row-indexed loan observations in bounded Parquet batches."""
    requested = pd.Index(indices)
    if requested.empty:
        return pd.DataFrame(columns=columns, index=requested)
    row_ids = requested.to_numpy(dtype=np.int64, copy=False)
    parquet_file = parquet.ParquetFile(_LOANS_PATH)
    row_count = int(parquet_file.metadata.num_rows)
    if row_ids.min() < 0 or row_ids.max() >= row_count:
        raise KeyError("Requested notebook row index is outside the loans dataset")
    wanted = np.zeros(row_count, dtype=bool)
    wanted[row_ids] = True
    pieces: list[pd.DataFrame] = []
    offset = 0
    for batch in parquet_file.iter_batches(columns=columns, batch_size=65_536):
        stop = offset + batch.num_rows
        mask = wanted[offset:stop]
        if mask.any():
            frame = batch.to_pandas()
            frame.index = pd.RangeIndex(offset, stop)
            pieces.append(frame.iloc[np.flatnonzero(mask)])
        offset = stop
    if offset != row_count:
        raise RuntimeError("Notebook loans Parquet row count changed during batched read")
    result = pd.concat(pieces, axis=0) if pieces else pd.DataFrame(columns=columns)
    if len(result) != len(requested):
        raise KeyError("One or more requested notebook row indices were not found")
    return result.reindex(requested)


def _score_role(role: str) -> pd.DataFrame:
    svc = CreditDecisionService.get_instance()
    if role == "cal_b":
        split = _load_cal_b()
    else:
        split = _load_split(role)
    X = split[svc.feature_columns]
    with svc.inference_lock:
        p = svc.lgbm_calibrated.predict_proba(X)[:, 1]
        sets = get_prediction_set(svc.conformal_model, X)
    raw = _read_loans_rows(split.index, list(_loan_column_names()))
    oos = check_oos_rules(raw, svc.oos_rules)
    decisions, reasons = decide(p, sets, oos, svc.threshold, svc.band_halfwidth)
    result = raw.copy()
    result["p_default"] = p
    result["decision"] = decisions
    result["refer_reason"] = reasons
    result["prediction_set"] = list(sets)
    return result


def _scored(role: str) -> pd.DataFrame:
    if role not in {"cal_b", "test"}:
        raise ValueError("Only notebook cal_B selection and read-only 2015 OOT cohorts are available")
    svc = CreditDecisionService.get_instance()
    if role == "cal_b":
        split = _load_cal_b()
    else:
        split = _load_split("test")
    X = split[svc.feature_columns]
    with svc.inference_lock:
        p = svc.lgbm_calibrated.predict_proba(X)[:, 1]
        sets = get_prediction_set(svc.conformal_model, X)
    raw = _read_loans_rows(split.index, list(_loan_column_names()))
    oos = check_oos_rules(raw, svc.oos_rules)
    dec, reason = decide(p, sets, oos, svc.threshold, svc.band_halfwidth)
    raw["p_default"] = p
    raw["prediction_set"] = list(sets)
    raw["decision"] = dec
    raw["refer_reason"] = reason
    return raw


def _cohort(role: str) -> pd.DataFrame:
    # Cohort consumers filter or aggregate without mutating the cached base.
    return _scored(role).copy(deep=False)


def meta_data() -> dict[str, Any]:
    svc = CreditDecisionService.get_instance()
    categories = svc.dataset_categories
    fields = []
    for name in _FIELDS:
        field = LoanApplication.model_fields[name]
        field_meta = field.metadata
        ge = next((getattr(m, "ge", None) for m in field_meta if hasattr(m, "ge")), None)
        gt = next((getattr(m, "gt", None) for m in field_meta if hasattr(m, "gt")), None)
        le = next((getattr(m, "le", None) for m in field_meta if hasattr(m, "le")), None)
        lt = next((getattr(m, "lt", None) for m in field_meta if hasattr(m, "lt")), None)
        minimum = ge if ge is not None else gt
        maximum = le if le is not None else lt
        step = 1 if name in {"open_acc", "pub_rec", "total_acc", "delinq_2yrs", "inq_last_6mths"} else 0.1
        item = {"name": name, "label": _LABELS[name], "type": "number", "min": minimum,
                "max": maximum, "exclusive_min": gt is not None, "exclusive_max": lt is not None, "step": step}
        if name == "home_ownership":
            item.update(type="select", options=categories["home_ownership"])
        elif name == "purpose":
            item.update(type="select", options=categories["purpose"])
        fields.append(item)
    return {
        "fields": fields,
        "policy": {"approve_below": svc.threshold - svc.band_halfwidth,
                   "reject_above": svc.threshold + svc.band_halfwidth,
                   "band_halfwidth": svc.band_halfwidth},
        "policy_sweep": {"start": float(svc.cfg.profit_model.sweep_range[0]),
                         "end": float(svc.cfg.profit_model.sweep_range[1]),
                         "step": float(svc.cfg.profit_model.sweep_step)},
        "scope_labels": {"cal_b": "2014 calibration selection half (seeded stratified split)",
                         "test_oot": "2015 out-of-time evaluation (read-only)"},
        "app_limits": _LIMITS,
        "cohorts": ["cal_b", "test_oot"],
        "categories": {"purpose": categories["purpose"]},
        "disclaimer": "Historical model estimate from the saved notebook artifacts; not an approval or loan offer.",
        "source": "notebooks/02_training.ipynb and its bundled saved artifacts/data",
    }


def _example_application(raw: pd.DataFrame) -> dict[str, Any]:
    derived = engineer_raw_features(raw).iloc[0]
    result = {name: derived[name] for name in _FIELDS}
    for name in _FIELDS:
        value = result[name]
        if pd.isna(value):
            result[name] = None
        elif isinstance(value, np.generic):
            result[name] = value.item()
    return result


def examples_data(limit: int = 12) -> list[dict[str, Any]]:
    svc = CreditDecisionService.get_instance()
    target = str(svc.cfg.target.column)
    candidate_columns = [
        "loan_amnt", "annual_inc", "dti", "fico_range_low", "fico_range_high",
        "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc", "delinq_2yrs",
        "inq_last_6mths", "issue_d", "earliest_cr_line",
    ]
    candidates = _read_loans_rows(_cal_b_indices(), candidate_columns)
    valid_mask = candidates[candidate_columns[:-2]].notna().all(axis=1)
    valid_mask &= candidates["issue_d"].notna() & candidates["earliest_cr_line"].notna()
    valid = candidates.loc[valid_mask].sort_values(["issue_d", "loan_amnt"])
    if valid.empty:
        return []
    positions = np.linspace(0, len(valid) - 1, min(limit, len(valid)), dtype=int)
    selected_indices = valid.index[positions]
    application_columns = list(dict.fromkeys([
        "loan_amnt", "annual_inc", "dti", "emp_length", "home_ownership", "purpose",
        "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc", "delinq_2yrs",
        "inq_last_6mths", "fico_range_low", "fico_range_high", "issue_d", "earliest_cr_line",
        "mort_acc", "application_type", target,
    ]))
    selected_rows = _read_loans_rows(selected_indices, application_columns)
    result = []
    for idx in selected_indices:
        row = selected_rows.loc[idx]
        result.append({"applicant": _example_application(selected_rows.loc[[idx]]),
                       "originated": str(pd.Timestamp(row.issue_d).date()),
                       "historical_outcome": "Charged Off" if int(row[target]) == 1 else "Fully Paid"})
    return result


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, log_loss, roc_curve
    fpr, tpr, _ = roc_curve(y, p)
    brier = float(brier_score_loss(y, p))
    return {"roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
            "brier": brier, "brier_score": brier,
            "ece": expected_calibration_error(y, p), "log_loss": float(log_loss(y, p)),
            "ks": float(np.max(tpr - fpr))}


def dashboard_data() -> dict[str, Any]:
    svc = CreditDecisionService.get_instance()
    test_scores = _score_split("test", include_base=True, include_sets=True)
    test_rows = len(test_scores["y"])
    y, p_base, p, sets = (test_scores["y"], test_scores["p_base"],
                          test_scores["p_calibrated"], test_scores["prediction_sets"])
    raw = _read_loans_rows(test_scores["indices"], ["issue_d", "fico_range_low", "fico_range_high",
                                        "dti", "annual_inc", "loan_amnt"])
    oos = check_oos_rules(raw, svc.oos_rules)
    decisions, reasons = decide(p, sets, oos, svc.threshold, svc.band_halfwidth)
    period = raw.issue_d.dt.to_period("M")
    trend = raw.assign(_period=period, _p=p, _y=y).groupby("_period", observed=True).agg(
        loans=("_y", "size"), default_rate=("_y", "mean"), mean_predicted_risk=("_p", "mean"))
    hist, edges = np.histogram(p, bins=np.linspace(0, 1, 11))
    approved = decisions == "Approve"
    classes = sorted(svc.lgbm_base.classes_.tolist())
    del raw, test_scores
    calb_scores = _score_split("cal", selected_indices=_cal_b_indices())
    calb_p, y_calb = calb_scores["p_calibrated"], calb_scores["y"]
    coverage = float(sets[np.arange(len(y)), y].mean())
    del calb_scores
    return {
        "dataset": "2015 final out-of-time evaluation · read-only; never used for selection",
        "kpis": {"evaluated_loans": test_rows, "observed_default_rate": float(y.mean()),
                 "mean_predicted_risk": float(p.mean()), "approval_rate": float(approved.mean()),
                 "approved_default_rate": float(y[approved].mean()) if approved.any() else None,
                 "conformal_coverage": coverage, "coverage_target": 1 - float(svc.metadata["alpha_operational"]),
                 "mean_predicted_risk_approved": float(p[approved].mean()) if approved.any() else None},
        "trend": [{"period": str(idx), "loans": int(r.loans), "default_rate": float(r.default_rate),
                   "mean_predicted_risk": float(r.mean_predicted_risk)} for idx, r in trend.iterrows()],
        "risk_distribution": [{"range": f"{edges[i]:.0%}–{edges[i+1]:.0%}", "count": int(hist[i])}
                              for i in range(len(hist))],
        "decisions": {name: int((decisions == name).sum()) for name in ("Approve", "Refer", "Reject")},
        "refer_reasons": {str(k): int(v) for k, v in pd.Series(reasons[reasons != ""]).value_counts().items()},
        "policy": {"threshold": svc.threshold, "band_halfwidth": svc.band_halfwidth,
                   "alpha": float(svc.metadata["alpha_operational"]),
                   "approve_below": svc.threshold - svc.band_halfwidth,
                   "reject_above": svc.threshold + svc.band_halfwidth},
        "model": {"name": "LightGBM (Platt calibrated)", "version": svc.model_version,
                  "training_period": "2007–2013", "roc_auc": _metrics(y, p)["roc_auc"],
                  "brier_score": float(brier_score_loss(y, p)),
                  "ece": expected_calibration_error(y, p)},
        "calibration": {"2014_cal_B": {"ece": expected_calibration_error(y_calb, calb_p),
                                        "mean_predicted": float(calb_p.mean()), "observed": float(y_calb.mean()),
                                        "brier": float(brier_score_loss(y_calb, calb_p))},
                        "2015_test": {"ece": expected_calibration_error(y, p),
                                      "mean_predicted": float(p.mean()), "observed": float(y.mean()),
                                      "brier": float(brier_score_loss(y, p))}},
        "model_metrics": {"LightGBM base 2015": _metrics(y, p_base),
                          "LightGBM calibrated 2015": _metrics(y, p)},
        "classes": classes,
        "feature_columns": svc.feature_columns,
    }


def portfolio_data(filters: dict[str, Any] | None = None) -> dict[str, Any]:
    filters = filters or {}
    df = _cohort("cal_b")
    if filters.get("purpose"):
        df = df[df.purpose == filters["purpose"]]
    if filters.get("home_ownership"):
        df = df[df.home_ownership == filters["home_ownership"]]
    if filters.get("year"):
        df = df[df.issue_d.dt.year == int(filters["year"])]
    fico = (df.fico_range_low + df.fico_range_high) / 2
    if filters.get("fico_min") is not None:
        df = df[fico >= float(filters["fico_min"])]
    if filters.get("fico_max") is not None:
        df = df[fico <= float(filters["fico_max"])]
    counts, edges = np.histogram(df.loan_amnt.to_numpy(dtype=float), bins=10) if len(df) else ([], [])
    return {"dataset": "2014 cal_B threshold-selection cohort (actual notebook observations)", "count": len(df),
            "observed_default_rate": float(df.default.mean()) if len(df) else None,
            "mean_predicted_risk": float(df.p_default.mean()) if len(df) else None,
            "mean_loan_amount": float(df.loan_amnt.mean()) if len(df) else None,
            "decisions": {n: int((df.decision == n).sum()) for n in ("Approve", "Refer", "Reject")},
            "by_purpose": [{"name": str(k), "count": int(v)} for k, v in df.purpose.value_counts().items()],
            "by_home": [{"name": str(k), "count": int(v)} for k, v in df.home_ownership.value_counts().items()],
            "loan_amount_distribution": [{"range": f"{edges[i]:,.0f}–{edges[i+1]:,.0f}", "count": int(counts[i])}
                                         for i in range(len(counts))] if len(counts) else [],
            "years": sorted(int(y) for y in df.issue_d.dt.year.unique()),
            "purposes": sorted(df.purpose.dropna().astype(str).unique().tolist()),
            "home_ownerships": sorted(df.home_ownership.dropna().astype(str).unique().tolist()),
            "frozen_policy": {"threshold": CreditDecisionService.get_instance().threshold,
                              "band_halfwidth": CreditDecisionService.get_instance().band_halfwidth,
                              "approve_below": CreditDecisionService.get_instance().threshold - CreditDecisionService.get_instance().band_halfwidth,
                              "reject_above": CreditDecisionService.get_instance().threshold + CreditDecisionService.get_instance().band_halfwidth}}


def policy_boundary_analysis(approve_below: float, reject_above: float) -> dict[str, Any]:
    """Apply the notebook's conformal decision function to actual cal_B records."""
    if not 0 <= approve_below < reject_above <= 1:
        raise ValueError("Threshold bounds must satisfy 0 ≤ approve below < reject above ≤ 1")
    svc = CreditDecisionService.get_instance()
    threshold = (approve_below + reject_above) / 2
    band = (reject_above - approve_below) / 2
    df = _cohort("cal_b")
    # Reuse the same actual cal_B predictions and conformal sets as cohort
    # reporting; only the requested policy boundaries differ.
    p = df["p_default"].to_numpy()
    ps = np.stack(df["prediction_set"].to_numpy())
    oos = check_oos_rules(df, svc.oos_rules)
    decisions, _ = decide(p, ps, oos, threshold, band)
    y = df.default.to_numpy(dtype=int)
    approved = decisions == "Approve"
    return {"dataset": "2014 cal_B threshold-selection cohort · actual observations",
            "count": len(df), "approval_rate": float(approved.mean()),
            "referral_rate": float((decisions == "Refer").mean()),
            "rejection_rate": float((decisions == "Reject").mean()),
            "approved_default_rate": float(y[approved].mean()) if approved.any() else None,
            "confusion_matrix": {"true_default_rejected": int(((y == 1) & (decisions == "Reject")).sum()),
                                 "paid_rejected": int(((y == 0) & (decisions == "Reject")).sum()),
                                 "paid_not_rejected": int(((y == 0) & (decisions != "Reject")).sum()),
                                 "default_not_rejected": int(((y == 1) & (decisions != "Reject")).sum())},
            "thresholds": {"approve_below": approve_below, "reject_above": reject_above,
                           "notebook_threshold": threshold, "band_halfwidth": band}}


def applicant_comparison(applicant: dict[str, Any], p_default: float) -> dict[str, Any]:
    df = _cohort("cal_b")
    risk_percentile = float((df.p_default <= p_default).mean())
    derived_cohort = engineer_raw_features(df)
    raw_derived = engineer_raw_features(pd.DataFrame([{**applicant,
        "fico_range_low": applicant["fico_avg"], "fico_range_high": applicant["fico_avg"],
        "mort_acc": np.nan, "application_type": "Individual"}])).iloc[0]
    fields = ["loan_amnt", "annual_inc", "dti", "fico_avg", "emp_length", "revol_bal",
              "revol_util", "open_acc", "total_acc", "delinq_2yrs", "inq_last_6mths", "credit_history_years"]
    comparisons = []
    for name in fields:
        values = pd.to_numeric(derived_cohort[name], errors="coerce").dropna()
        comparisons.append({"field": name, "applicant": float(raw_derived[name]),
                            "median": float(values.median()), "q25": float(values.quantile(.25)),
                            "q75": float(values.quantile(.75)), "count": int(values.count())})
    return {"dataset": "Actual 2014 cal_B threshold-selection observations", "risk_percentile": risk_percentile,
            "risk_cohort_count": len(df), "field_comparison": comparisons}


def governance_data() -> dict[str, Any]:
    svc = CreditDecisionService.get_instance()
    model_dir = _RUNTIME / "models"
    import hashlib
    manifest_path = _RUNTIME / "artifact_checksums.json"
    expected = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    artifact_hashes = {}
    for relative, expected_hash in expected.items():
        path = _RUNTIME / relative
        digest = hashlib.sha256()
        if path.exists():
            with path.open("rb") as artifact:
                for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
                    digest.update(chunk)
            actual = digest.hexdigest()
        else:
            actual = None
        artifact_hashes[relative] = {"expected": expected_hash, "actual": actual,
                                     "matches": actual == expected_hash}
    target = str(svc.cfg.target.column)
    test_scores = _score_split("test", include_base=True)
    test_rows = len(test_scores["y"])
    y, p, base_p = test_scores["y"], test_scores["p_calibrated"], test_scores["p_base"]
    from sklearn.metrics import precision_recall_curve, roc_curve
    fpr, tpr, _ = roc_curve(y, p)
    precision, recall, _ = precision_recall_curve(y, p)
    from sklearn.calibration import calibration_curve
    observed, predicted = calibration_curve(y, p, n_bins=10, strategy="quantile")
    del test_scores

    cal_b_indices = np.asarray(_cal_b_indices(), dtype=np.int64)
    sample_size = min(2000, len(cal_b_indices))
    random_state = np.random.RandomState(int(svc.cfg.modeling.random_seed))
    sample_positions = random_state.choice(len(cal_b_indices), size=sample_size, replace=False)
    sample_indices = cal_b_indices[sample_positions]
    sample = _read_split_rows("cal", sample_indices, svc.feature_columns)
    shap_values = compute_shap_values(svc.explainer, sample)
    by_source: dict[str, list[np.ndarray]] = {}
    for index, name in enumerate(svc.feature_columns):
        source = name.split("__", 1)[-1]
        if name.startswith("cat__home_ownership_"):
            source = "home_ownership"
        elif name.startswith("cat__purpose_"):
            source = "purpose"
        by_source.setdefault(source, []).append(shap_values[:, index])
    feature_importance = [{"feature": name, "importance": float(np.abs(np.sum(values, axis=0)).mean())}
                          for name, values in by_source.items()]
    feature_importance.sort(key=lambda row: row["importance"])
    cal_scores = _score_split("cal", include_lr=True, include_base=True)
    cal_y, lr_cal_p, lgbm_cal_p = cal_scores["y"], cal_scores["p_lr"], cal_scores["p_base"]
    lr_cal_metrics = _metrics(cal_y, lr_cal_p)
    lgbm_cal_metrics = _metrics(cal_y, lgbm_cal_p)
    del sample, shap_values, cal_scores, cal_y, lr_cal_p, lgbm_cal_p
    base_metrics, cal_metrics = _metrics(y, base_p), _metrics(y, p)
    base_ece = expected_calibration_error(y, base_p)
    del base_p
    split_counts = {}
    split_rates = {}
    for name in ("train", "early_stop", "cal", "test"):
        labels = _load_split(name, columns=[target])
        output_name = {"cal": "calibration", "test": "test_oot"}.get(name, name)
        split_counts[output_name] = len(labels)
        split_rates[output_name] = float(labels[target].mean())
        if name == "train":
            train_indices = labels.index
        del labels
    train_loans = _read_loans_rows(train_indices, ["int_rate"])
    training_apr = float(pd.to_numeric(train_loans["int_rate"], errors="coerce").mean() / 100)
    del train_loans
    selected_threshold = threshold_data()
    model_rows = {
        "logistic_regression_cal_2014": {
            "model_label": "Logistic Regression · calibration 2014",
            **lr_cal_metrics,
        },
        "lightgbm_cal_2014": {
            "model_label": "LightGBM base · calibration 2014",
            **lgbm_cal_metrics,
        },
        "lightgbm_base": {"model_label": "LightGBM base (not calibrated)", "roc_auc": base_metrics["roc_auc"],
                          "pr_auc": base_metrics["pr_auc"], "brier_score": base_metrics["brier"],
                              "brier_skill_score": None, "ece": base_ece,
                          "log_loss": base_metrics["log_loss"], "ks": base_metrics["ks"]},
        "lightgbm_calibrated": {"model_label": "LightGBM (Platt calibrated)", "roc_auc": cal_metrics["roc_auc"],
                                "pr_auc": cal_metrics["pr_auc"], "brier_score": cal_metrics["brier"],
                                "brier_skill_score": None, "ece": expected_calibration_error(y, p),
                                "log_loss": cal_metrics["log_loss"], "ks": cal_metrics["ks"]},
    }
    return {"project": "Notebook-backed CreditView model", "model_version": svc.model_version,
            "evaluation_dataset": "2015 final out-of-time test set (read-only)",
            "calibration": "Sigmoid / Platt; fitted on cal_A from the 2014 calibration split",
            "models": model_rows, "features": svc.feature_columns,
            "threshold_selection": {k: selected_threshold[k] for k in
                                    ("selected_threshold", "notebook_threshold", "matches_saved_notebook", "min_roi")},
            "apr_reference": {"value": training_apr, "metadata_value": float(svc.metadata["assumed_apr"]),
                              "matches_saved_notebook": abs(training_apr - float(svc.metadata["assumed_apr"])) < 1e-10,
                              "definition": "Training-split mean int_rate / 100, as in notebook cell 5"},
            "roc_curve": [{"x": float(x), "y": float(yv)} for x, yv in zip(fpr, tpr)],
            "precision_recall_curve": [{"x": float(x), "y": float(yv)} for x, yv in zip(recall, precision)],
            "calibration_curve": [{"predicted": float(x), "observed": float(yv)} for x, yv in zip(predicted, observed)],
            "feature_importance": feature_importance,
            "split_counts": split_counts, "split_default_rates": split_rates,
            "integrity": artifact_hashes,
            "source_notebook": "notebooks/02_training.ipynb", "source_data": "notebook_runtime/data/processed",
            "train_rows": split_counts["train"],
            "early_stop_rows": split_counts["early_stop"],
            "calibration_rows": split_counts["calibration"],
            "test_oot_rows": test_rows,
            "metadata": svc.metadata, "artifact_sha256": artifact_hashes,
            "notebook_output_comparison": {"threshold_matches": abs(float(svc.metadata["threshold"]) - .23) < 1e-12,
                                             "coverage_target_met": float(svc.metadata["report_2015"]["coverage"]) >= .90}}


def calibration_data() -> dict[str, Any]:
    """Notebook calibration data, with all bins computed from actual cal_B and 2015 rows."""
    svc = CreditDecisionService.get_instance()
    results = {}
    for role, split_name in (("cal_B 2014", "cal"), ("test 2015", "test")):
        selected = _cal_b_indices() if split_name == "cal" else None
        scores = _score_split(split_name, selected_indices=selected, include_base=True)
        y, p_base, p_cal = scores["y"], scores["p_base"], scores["p_calibrated"]
        rows = []
        for i in range(10):
            low, high = i / 10, (i + 1) / 10
            mask = (p_cal >= low) & ((p_cal < high) if i < 9 else (p_cal <= high))
            if mask.any():
                rows.append({"lower": low, "upper": high, "count": int(mask.sum()),
                             "observed": float(y[mask].mean()), "predicted": float(p_cal[mask].mean())})
        results[role] = {"ece_before": expected_calibration_error(y, p_base),
                         "ece_after": expected_calibration_error(y, p_cal),
                         "mean_p_before": float(p_base.mean()), "mean_p_after": float(p_cal.mean()),
                         "observed": float(y.mean()), "brier_after": float(brier_score_loss(y, p_cal)),
                         "bins": rows}
        del scores, y, p_base, p_cal
    return results


def threshold_data() -> dict[str, Any]:
    svc = CreditDecisionService.get_instance()
    scores = _score_split("cal", selected_indices=_cal_b_indices())
    p = scores["p_calibrated"]
    loans = _read_loans_rows(scores["indices"], ["loan_amnt", "total_pymnt"])
    cfg_profit = svc.cfg.profit_model
    sweep = compute_profit_per_threshold(scores["y"], p, loans.loan_amnt,
                                          loans.total_pymnt, cfg_profit.sweep_range,
                                          cfg_profit.sweep_step, min_roi=cfg_profit.min_roi)
    threshold = select_optimal_threshold(sweep)
    return {"selected_threshold": threshold, "notebook_threshold": float(svc.metadata["threshold"]),
            "matches_saved_notebook": abs(threshold - float(svc.metadata["threshold"])) < 1e-10,
            "min_roi": float(cfg_profit.min_roi),
            "curve": [{"threshold": float(r.threshold), "expected_profit": float(r.expected_profit),
                       "raw_profit": float(r.raw_profit), "n_approved": int(r.n_approved),
                       "default_rate_approved": float(r.n_fp / r.n_approved) if r.n_approved else None}
                      for r in sweep.itertuples()]}
