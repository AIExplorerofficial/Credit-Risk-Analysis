"""Parity checks against real observations and calculations from 02_training.ipynb."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from backend.analytics import _loan_column_names, _read_loans_rows
from creditview.notebook_service import CreditDecisionService
from creditview.schemas import LoanApplication
from src.conformal import get_prediction_set
from src.feature_engineering import engineer_raw_features
from src.threshold import compute_profit_per_threshold, select_optimal_threshold


def _notebook_data(*roles):
    service = CreditDecisionService.get_instance()
    paths = {"cal": service.cfg.paths.calibration_parquet, "test": service.cfg.paths.test_parquet}
    return service, {role: pd.read_parquet(paths[role]) for role in roles}


def _application(raw_row: pd.Series) -> LoanApplication:
    raw = raw_row.to_frame().T
    for column in ("issue_d", "earliest_cr_line"):
        if column in raw.columns:
            raw[column] = pd.to_datetime(raw[column])
    derived = engineer_raw_features(raw).iloc[0]
    fields = ("loan_amnt", "annual_inc", "dti", "emp_length", "home_ownership", "purpose",
              "fico_avg", "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
              "delinq_2yrs", "inq_last_6mths", "credit_history_years")
    values = {}
    for key in fields:
        value = derived[key]
        values[key] = value.item() if isinstance(value, np.generic) else value
    return LoanApplication(**values)


def test_notebook_preprocessing_and_inference_match_real_calibration_rows():
    service, data = _notebook_data("cal")
    target = service.cfg.target.column
    _, cal_b = train_test_split(
        data["cal"], test_size=0.5, stratify=data["cal"][target],
        random_state=int(service.cfg.modeling.random_seed),
    )
    candidate_columns = ["annual_inc", "dti", "revol_util", "home_ownership", "purpose"]
    candidates = _read_loans_rows(cal_b.index, candidate_columns)
    valid_indices = candidates.dropna(subset=candidate_columns).index[:5]
    selected = _read_loans_rows(valid_indices, list(_loan_column_names()))
    assert len(selected) == 5

    raw_transform = service.preprocessor.transform(engineer_raw_features(selected))
    expected = cal_b.loc[selected.index, service.feature_columns]
    assert list(raw_transform.columns) == service.feature_columns
    np.testing.assert_allclose(raw_transform.to_numpy(), expected.to_numpy(), rtol=0, atol=1e-10)

    applications = [_application(row) for _, row in selected.iterrows()]
    scored = service.score_many(applications)
    with service.inference_lock:
        expected_probabilities = service.lgbm_calibrated.predict_proba(expected)[:, 1]
    np.testing.assert_allclose(scored["probabilities"], expected_probabilities, rtol=0, atol=1e-12)


def test_notebook_threshold_is_selected_only_on_cal_b():
    service, data = _notebook_data("cal")
    target = service.cfg.target.column
    _, cal_b = train_test_split(
        data["cal"], test_size=0.5, stratify=data["cal"][target],
        random_state=int(service.cfg.modeling.random_seed),
    )
    raw = _read_loans_rows(cal_b.index, ["loan_amnt", "total_pymnt"])
    p = service.lgbm_calibrated.predict_proba(cal_b[service.feature_columns])[:, 1]
    cfg = service.cfg.profit_model
    sweep = compute_profit_per_threshold(
        cal_b[target], p, raw.loan_amnt, raw.total_pymnt, cfg.sweep_range,
        cfg.sweep_step, min_roi=cfg.min_roi,
    )
    assert select_optimal_threshold(sweep) == float(service.metadata["threshold"]) == 0.23


def test_notebook_2015_conformal_coverage_is_reproduced_as_report_only():
    service, data = _notebook_data("test")
    test = data["test"]
    prediction_sets = get_prediction_set(
        service.conformal_model, test[service.feature_columns]
    )
    y = test[service.cfg.target.column].to_numpy(dtype=int)
    coverage = float(prediction_sets[np.arange(len(y)), y].mean())
    assert abs(coverage - float(service.metadata["report_2015"]["coverage"])) < 1e-4
    assert coverage < float(service.metadata["confidence_level"])
