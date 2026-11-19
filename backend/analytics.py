"""Metrics and cohorts computed from the saved 02_training.ipynb data and artifacts."""
from __future__ import annotations

import sys
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
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


@lru_cache(maxsize=1)
def _inputs() -> dict[str, Any]:
    svc = CreditDecisionService.get_instance()
    base = _RUNTIME / "data" / "processed"
    splits = {
        key: pd.read_parquet(base / f"{filename}.parquet")
        for key, filename in (("train", "train"), ("early_stop", "early_stop"),
                              ("cal", "calibration"), ("test", "test_oot"))
    }
    loans = pd.read_parquet(base / "loans_36m.parquet")
    target = str(svc.cfg.target.column)
    cal_a, cal_b = train_test_split(
        splits["cal"], test_size=0.5, stratify=splits["cal"][target],
        random_state=int(svc.cfg.modeling.random_seed),
    )
    if list(splits["train"].drop(columns=[target]).columns) != svc.feature_columns:
        raise RuntimeError("Notebook split feature ordering differs from loaded model metadata")
    return {"svc": svc, "splits": splits, "loans": loans, "target": target,
            "cal_a": cal_a, "cal_b": cal_b}


def _score_role(role: str) -> pd.DataFrame:
    data = _inputs()
    svc, loans = data["svc"], data["loans"]
    split = data["splits"][role]
    X = split[svc.feature_columns]
    with svc.inference_lock:
        p = svc.lgbm_calibrated.predict_proba(X)[:, 1]
        sets = get_prediction_set(svc.conformal_model, X)
    raw = loans.loc[split.index]
    oos = check_oos_rules(raw, svc.oos_rules)
    decisions, reasons = decide(p, sets, oos, svc.threshold, svc.band_halfwidth)
    result = raw.copy()
    result["p_default"] = p
    result["decision"] = decisions
    result["refer_reason"] = reasons
    result["prediction_set"] = list(sets)
    return result


@lru_cache(maxsize=2)
def _scored(role: str) -> pd.DataFrame:
    if role not in {"cal_b", "test"}:
        raise ValueError("Only notebook cal_B selection and read-only 2015 OOT cohorts are available")
    data = _inputs()
    split = data["cal_b"] if role == "cal_b" else data["splits"]["test"]
    svc, loans = data["svc"], data["loans"]
    X = split[svc.feature_columns]
    with svc.inference_lock:
        p = svc.lgbm_calibrated.predict_proba(X)[:, 1]
        sets = get_prediction_set(svc.conformal_model, X)
    raw = loans.loc[split.index].copy()
    oos = check_oos_rules(raw, svc.oos_rules)
    dec, reason = decide(p, sets, oos, svc.threshold, svc.band_halfwidth)
    raw["p_default"] = p
    raw["prediction_set"] = list(sets)
    raw["decision"] = dec
    raw["refer_reason"] = reason
    return raw


def _cohort(role: str) -> pd.DataFrame:
    return _scored(role).copy()


def meta_data() -> dict[str, Any]:
    data = _inputs()
    svc, loans = data["svc"], data["loans"]
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
            item.update(type="select", options=sorted(loans.home_ownership.dropna().astype(str).unique().tolist()))
        elif name == "purpose":
            item.update(type="select", options=sorted(loans.purpose.dropna().astype(str).unique().tolist()))
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
        "categories": {"purpose": sorted(loans.purpose.dropna().astype(str).unique().tolist())},
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
    data = _inputs()
    loans = data["loans"].loc[data["cal_b"].index]
    derived = engineer_raw_features(loans)
    valid = loans.loc[derived.dropna(subset=_FIELDS).index]
    valid = valid.sort_values(["issue_d", "loan_amnt"])
    if valid.empty:
        return []
    positions = np.linspace(0, len(valid) - 1, min(limit, len(valid)), dtype=int)
    result = []
    for pos in positions:
        row = valid.iloc[int(pos)]
        result.append({"applicant": _example_application(valid.loc[[row.name]]),
                       "originated": str(pd.Timestamp(row.issue_d).date()),
                       "historical_outcome": "Charged Off" if int(row[data["target"]]) == 1 else "Fully Paid"})
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
    data = _inputs()
    svc, target = data["svc"], data["target"]
    test = data["splits"]["test"]
    X = test[svc.feature_columns]
    y = test[target].to_numpy(dtype=int)
    with svc.inference_lock:
        p_base = svc.lgbm_base.predict_proba(X)[:, 1]
        p = svc.lgbm_calibrated.predict_proba(X)[:, 1]
        sets = get_prediction_set(svc.conformal_model, X)
    raw = data["loans"].loc[test.index]
    oos = check_oos_rules(raw, svc.oos_rules)
    decisions, reasons = decide(p, sets, oos, svc.threshold, svc.band_halfwidth)
    period = raw.issue_d.dt.to_period("M")
    trend = raw.assign(_period=period, _p=p, _y=y).groupby("_period", observed=True).agg(
        loans=("_y", "size"), default_rate=("_y", "mean"), mean_predicted_risk=("_p", "mean"))
    hist, edges = np.histogram(p, bins=np.linspace(0, 1, 11))
    approved = decisions == "Approve"
    classes = sorted(svc.lgbm_base.classes_.tolist())
    calb = data["cal_b"]
    calb_p = svc.lgbm_calibrated.predict_proba(calb[svc.feature_columns])[:, 1]
    y_calb = calb[target].to_numpy(dtype=int)
    coverage = float(sets[np.arange(len(y)), y].mean())
    return {
        "dataset": "2015 final out-of-time evaluation · read-only; never used for selection",
        "kpis": {"evaluated_loans": len(test), "observed_default_rate": float(y.mean()),
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
    decisions = df.decision.to_numpy()
    # For an explicitly requested boundary pair, apply the notebook function to the same real cal_B scores/sets.
    split = _inputs()["cal_b"]
    with svc.inference_lock:
        p = svc.lgbm_calibrated.predict_proba(split[svc.feature_columns])[:, 1]
        ps = get_prediction_set(svc.conformal_model, split[svc.feature_columns])
    oos = check_oos_rules(_inputs()["loans"].loc[split.index], svc.oos_rules)
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
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        artifact_hashes[relative] = {"expected": expected_hash, "actual": actual,
                                     "matches": actual == expected_hash}
    test = _inputs()["splits"]["test"]
    y = test[_inputs()["target"]].to_numpy(dtype=int)
    p = svc.lgbm_calibrated.predict_proba(test[svc.feature_columns])[:, 1]
    from sklearn.metrics import precision_recall_curve, roc_curve
    fpr, tpr, _ = roc_curve(y, p)
    precision, recall, _ = precision_recall_curve(y, p)
    from sklearn.calibration import calibration_curve
    observed, predicted = calibration_curve(y, p, n_bins=10, strategy="quantile")
    cal_b = _inputs()["cal_b"]
    sample_size = min(2000, len(cal_b))
    sample = cal_b[svc.feature_columns].sample(sample_size, random_state=int(svc.cfg.modeling.random_seed))
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
    splits = _inputs()["splits"]
    split_counts = {"train": len(splits["train"]), "early_stop": len(splits["early_stop"]),
                    "calibration": len(splits["cal"]), "test_oot": len(splits["test"])}
    target = _inputs()["target"]
    split_rates = {name: float(frame[target].mean()) for name, frame in
                   (("train", splits["train"]), ("early_stop", splits["early_stop"]),
                    ("calibration", splits["cal"]), ("test_oot", splits["test"]))}
    base_p = svc.lgbm_base.predict_proba(test[svc.feature_columns])[:, 1]
    base_metrics, cal_metrics = _metrics(y, base_p), _metrics(y, p)
    train_loans = _inputs()["loans"].loc[splits["train"].index]
    training_apr = float(pd.to_numeric(train_loans["int_rate"], errors="coerce").mean() / 100)
    selected_threshold = threshold_data()
    model_rows = {
        "logistic_regression_cal_2014": {
            "model_label": "Logistic Regression · calibration 2014",
            **_metrics(splits["cal"][target].to_numpy(dtype=int),
                      svc.lr_baseline.predict_proba(splits["cal"][svc.feature_columns])[:, 1]),
        },
        "lightgbm_cal_2014": {
            "model_label": "LightGBM base · calibration 2014",
            **_metrics(splits["cal"][target].to_numpy(dtype=int),
                      svc.lgbm_base.predict_proba(splits["cal"][svc.feature_columns])[:, 1]),
        },
        "lightgbm_base": {"model_label": "LightGBM base (not calibrated)", "roc_auc": base_metrics["roc_auc"],
                          "pr_auc": base_metrics["pr_auc"], "brier_score": base_metrics["brier"],
                          "brier_skill_score": None, "ece": expected_calibration_error(y, base_p),
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
            "train_rows": int(_inputs()["splits"]["train"].shape[0]),
            "early_stop_rows": int(_inputs()["splits"]["early_stop"].shape[0]),
            "calibration_rows": int(_inputs()["splits"]["cal"].shape[0]),
            "test_oot_rows": int(_inputs()["splits"]["test"].shape[0]),
            "metadata": svc.metadata, "artifact_sha256": artifact_hashes,
            "notebook_output_comparison": {"threshold_matches": abs(float(svc.metadata["threshold"]) - .23) < 1e-12,
                                             "coverage_target_met": float(svc.metadata["report_2015"]["coverage"]) >= .90}}


def calibration_data() -> dict[str, Any]:
    """Notebook calibration data, with all bins computed from actual cal_B and 2015 rows."""
    data = _inputs()
    svc, target = data["svc"], data["target"]
    results = {}
    for role, split in (("cal_B 2014", data["cal_b"]), ("test 2015", data["splits"]["test"])):
        X, y = split[svc.feature_columns], split[target].to_numpy(dtype=int)
        with svc.inference_lock:
            p_base = svc.lgbm_base.predict_proba(X)[:, 1]
            p_cal = svc.lgbm_calibrated.predict_proba(X)[:, 1]
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
    return results


def threshold_data() -> dict[str, Any]:
    data = _inputs()
    svc, cal_b = data["svc"], data["cal_b"]
    loans = data["loans"].loc[cal_b.index]
    p = svc.lgbm_calibrated.predict_proba(cal_b[svc.feature_columns])[:, 1]
    cfg_profit = svc.cfg.profit_model
    sweep = compute_profit_per_threshold(cal_b[data["target"]], p, loans.loan_amnt,
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
