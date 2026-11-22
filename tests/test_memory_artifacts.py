"""Exact parity checks for compact notebook-derived runtime artifacts."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as parquet
from pandas.testing import assert_frame_equal
from sklearn.model_selection import train_test_split

from creditview.notebook_service import CreditDecisionService, _NOTEBOOK_ROOT
from backend.analytics import (_cal_b_indices, _loan_column_names, _read_loans_rows,
                               _read_split_rows, _score_split, _load_split)
from src.conformal import get_prediction_set
from src.explainer import build_tree_explainer, compute_shap_values
from src.feature_engineering import engineer_raw_features


def _original_support_summaries():
    service = CreditDecisionService.get_instance()
    train = pd.read_parquet(service.cfg.paths.train_parquet)
    loans = _read_loans_rows(train.index, list(_loan_column_names()))
    training_raw = engineer_raw_features(loans.loc[train.index])
    numeric = {}
    for name in training_raw.select_dtypes(include=[np.number]).columns:
        values = pd.to_numeric(training_raw[name], errors="coerce").dropna()
        if not len(values):
            continue
        q1, q3 = values.quantile([0.25, 0.75]).tolist()
        iqr = q3 - q1
        numeric[name] = {
            "min": float(values.min()), "max": float(values.max()),
            "typical_min": float(q1 - 1.5 * iqr), "typical_max": float(q3 + 1.5 * iqr),
        }
    categories = {
        name: set(training_raw[name].dropna().astype(str).unique())
        for name in ("home_ownership", "purpose")
    }
    all_categories = {name: set() for name in ("home_ownership", "purpose")}
    raw_file = parquet.ParquetFile(service.cfg.paths.loans_parquet)
    for batch in raw_file.iter_batches(columns=list(all_categories), batch_size=65_536):
        frame = batch.to_pandas()
        for name in all_categories:
            all_categories[name].update(frame[name].dropna().astype(str).unique().tolist())
    data_categories = {name: sorted(values) for name, values in all_categories.items()}
    return train, loans, numeric, categories, data_categories


def test_precomputed_support_summaries_match_original_calculation_exactly():
    service = CreditDecisionService.get_instance()
    _, _, numeric, categories, data_categories = _original_support_summaries()
    assert service.support_numeric == numeric
    assert service.support_categories == categories
    assert service.dataset_categories == data_categories


def test_precomputed_shap_background_is_exact_configured_seeded_sample():
    service = CreditDecisionService.get_instance()
    train, _, _, _, _ = _original_support_summaries()
    target = str(service.cfg.target.column)
    features = train.drop(columns=[target])
    expected = features.sample(
        min(int(service.cfg.shap.background_sample_n), len(features)),
        random_state=int(service.cfg.modeling.random_seed),
    )
    actual = pd.read_parquet(_NOTEBOOK_ROOT / "memory" / "shap_background.parquet")
    assert len(actual) == int(service.cfg.shap.background_sample_n) == 200
    assert_frame_equal(actual, expected, check_exact=True)


def test_shap_values_match_original_seeded_background_exactly():
    service = CreditDecisionService.get_instance()
    train, _, _, _, _ = _original_support_summaries()
    target = str(service.cfg.target.column)
    features = train.drop(columns=[target])
    background = features.sample(
        min(int(service.cfg.shap.background_sample_n), len(features)),
        random_state=int(service.cfg.modeling.random_seed),
    )
    original_explainer = build_tree_explainer(service.lgbm_base, background)
    observation = features.iloc[[0]]
    with service.inference_lock:
        expected = compute_shap_values(original_explainer, observation)
        actual = compute_shap_values(service.explainer, observation)
    np.testing.assert_array_equal(actual, expected)


def test_governance_shap_sample_rows_match_original_dataframe_sample_exactly():
    service = CreditDecisionService.get_instance()
    cal = _load_split("cal")
    _, cal_b = train_test_split(
        cal, test_size=0.5, stratify=cal[service.cfg.target.column],
        random_state=int(service.cfg.modeling.random_seed),
    )
    expected = cal_b[service.feature_columns].sample(
        min(2000, len(cal_b)), random_state=int(service.cfg.modeling.random_seed)
    )
    cal_b_indices = np.asarray(_cal_b_indices(), dtype=np.int64)
    random_state = np.random.RandomState(int(service.cfg.modeling.random_seed))
    positions = random_state.choice(len(cal_b_indices), size=min(2000, len(cal_b_indices)), replace=False)
    actual = _read_split_rows("cal", cal_b_indices[positions], service.feature_columns)
    assert_frame_equal(actual, expected, check_exact=True)


def test_batched_oot_scoring_matches_full_saved_model_outputs_exactly():
    service = CreditDecisionService.get_instance()
    test = _load_split("test")
    X = test[service.feature_columns]
    with service.inference_lock:
        expected_p = service.lgbm_calibrated.predict_proba(X)[:, 1]
        expected_base = service.lgbm_base.predict_proba(X)[:, 1]
        expected_sets = get_prediction_set(service.conformal_model, X)
    actual = _score_split("test", include_base=True, include_sets=True)
    np.testing.assert_array_equal(actual["indices"], test.index.to_numpy())
    np.testing.assert_array_equal(actual["y"], test[service.cfg.target.column].to_numpy(dtype=int))
    np.testing.assert_array_equal(actual["p_calibrated"], expected_p)
    np.testing.assert_array_equal(actual["p_base"], expected_base)
    np.testing.assert_array_equal(actual["prediction_sets"], expected_sets)
