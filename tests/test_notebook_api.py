"""Route-level checks using real observations and saved notebook artifacts."""
from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from backend.main import app, batch, dashboard, decide, examples, governance, health, thresholds, whatif
from backend.analytics import _inputs
from creditview.notebook_service import CreditDecisionService
from creditview.schemas import LoanApplication, WhatIfRequest
from src.conformal import decide as notebook_decide
from src.conformal import get_prediction_set
from src.feature_engineering import engineer_raw_features


@pytest.fixture(scope="module")
def supported_policy_applications():
    """Find real cal_B observations for each saved in-support policy outcome."""
    service = CreditDecisionService.get_instance()
    data = _inputs()
    cal_b = data["cal_b"]
    with service.inference_lock:
        probabilities = service.lgbm_calibrated.predict_proba(cal_b[service.feature_columns])[:, 1]
        sets = get_prediction_set(service.conformal_model, cal_b[service.feature_columns])
    model_decisions, _ = notebook_decide(
        probabilities, sets, np.zeros(len(cal_b), dtype=bool), service.threshold, service.band_halfwidth
    )
    loans = data["loans"].loc[cal_b.index]
    fields = ("loan_amnt", "annual_inc", "dti", "fico_avg", "emp_length", "home_ownership", "purpose",
              "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc", "delinq_2yrs",
              "inq_last_6mths", "credit_history_years")
    selected = {}
    for expected in ("Approve", "Reject", "Refer"):
        for pos in np.flatnonzero(model_decisions == expected):
            raw = loans.iloc[[int(pos)]].copy()
            for col in ("issue_d", "earliest_cr_line"):
                raw[col] = pd.to_datetime(raw[col])
            derived = engineer_raw_features(raw).iloc[0]
            values = {name: derived[name] for name in fields}
            if any(pd.isna(value) for value in values.values()):
                continue
            values = {name: (value.item() if isinstance(value, np.generic) else value)
                      for name, value in values.items()}
            try:
                applicant = LoanApplication(**values)
            except ValueError:
                continue
            if not service._is_out_of_support(service._support_warnings(applicant)):
                selected[expected] = applicant
                break
    assert set(selected) == {"Approve", "Reject", "Refer"}
    return selected


def test_health_identifies_notebook_artifact_service():
    response = health()
    assert response["status"] == "healthy"
    assert response["version"].startswith("notebook-02-training")


def test_examples_are_actual_2014_cal_b_observations():
    payload = examples(limit=3)
    rows = payload["examples"]
    assert len(rows) == 3
    assert all(row["originated"].startswith("2014-") for row in rows)
    assert all(row["historical_outcome"] in {"Charged Off", "Fully Paid"} for row in rows)
    indices = _inputs()["cal_b"].index
    loans = _inputs()["loans"].loc[indices]
    actual = {(str(row.issue_d.date()), "Charged Off" if int(row.default) == 1 else "Fully Paid")
              for row in loans.itertuples()}
    assert all((row["originated"], row["historical_outcome"]) in actual for row in rows)


def test_single_whatif_and_batch_share_notebook_inference(actual_cal_b_application):
    service = CreditDecisionService.get_instance()
    expected = service.score_many([actual_cal_b_application])
    application = LoanApplication(**actual_cal_b_application)
    single = decide(application)
    assert single["p_default"] == pytest.approx(float(expected["probabilities"][0]), abs=5e-5)
    assert single["decision"] == str(expected["decisions"][0])
    assert single["formal_status"] == single["decision"]
    assert single["indicative_recommendation"] == single["formal_status"]
    assert single["is_provisional"] is False
    assert single["decision_threshold"] == pytest.approx(service.threshold)
    assert single["band_halfwidth"] == pytest.approx(service.band_halfwidth)
    assert single["full_explanation"]
    assert "monthly_payment_illustrative" not in single

    scored = batch([application])
    assert scored["results"][0]["p_default"] == pytest.approx(single["p_default"], abs=5e-5)
    assert scored["results"][0]["formal_status"] == single["formal_status"]
    assert scored["results"][0]["indicative_recommendation"] == single["indicative_recommendation"]
    assert scored["results"][0]["is_provisional"] is False
    assert "support_warnings" in scored["results"][0]

    modified = dict(actual_cal_b_application)
    modified["loan_amnt"] = max(500, float(modified["loan_amnt"]) * 0.9)
    paired = whatif(WhatIfRequest(original_application=application,
                                  modified_application=LoanApplication(**modified)))
    assert paired["original"]["p_default"] == pytest.approx(single["p_default"], abs=5e-5)
    assert paired["original"]["formal_status"] == single["formal_status"]
    assert paired["original"]["indicative_recommendation"] == single["indicative_recommendation"]


def test_dashboard_and_threshold_metrics_use_correct_notebook_roles():
    result = dashboard()
    assert result["kpis"]["evaluated_loans"] == len(_inputs()["splits"]["test"])
    assert result["policy"]["threshold"] == pytest.approx(0.23)
    assert result["kpis"]["conformal_coverage"] == pytest.approx(0.8895, abs=1e-4)
    assert result["kpis"]["conformal_coverage"] < result["kpis"]["coverage_target"]

    selected = thresholds()
    assert selected["selected_threshold"] == pytest.approx(0.23)
    assert selected["matches_saved_notebook"] is True


def test_governance_metrics_are_computed_and_artifacts_match_prototype():
    result = governance()
    assert len(result["features"]) == 28
    assert result["apr_reference"]["matches_saved_notebook"] is True
    assert result["threshold_selection"]["matches_saved_notebook"] is True
    assert result["integrity"]
    assert all(entry["matches"] for entry in result["integrity"].values())


def test_generated_scenario_grid_endpoints_are_not_exposed():
    paths = {route.path for route in app.routes}
    assert "/api/sensitivity" not in paths
    assert "/api/recourse-explore" not in paths


def test_loan_amount_accepts_any_positive_value_and_rejects_nonpositive(actual_cal_b_application):
    base = dict(actual_cal_b_application)
    for amount in (0.01, 75000.0):
        app = LoanApplication(**{**base, "loan_amnt": amount})
        result = CreditDecisionService.get_instance().decide(app)
        assert 0 <= result.p_default <= 1
        assert result.decision == "Manual Review Required — Outside Model Support"
        assert result.is_provisional
    for amount in (0, -1):
        with pytest.raises(ValueError):
            LoanApplication(**{**base, "loan_amnt": amount})


def test_support_warning_does_not_force_referral_and_training_ranges_are_data_derived(actual_cal_b_application):
    service = CreditDecisionService.get_instance()
    unusual = {**actual_cal_b_application, "loan_amnt": 75000, "annual_inc": 1}
    parsed = LoanApplication(**unusual)
    scored = service.score_many([parsed])
    expected, _ = notebook_decide(scored["probabilities"], scored["prediction_sets"],
                                  [False], service.threshold, service.band_halfwidth)
    assert str(expected[0]) in {"Approve", "Refer", "Reject"}
    result = service.decide(parsed)
    assert result.decision == "Manual Review Required — Outside Model Support"
    assert result.is_provisional
    assert result.refer_reason == "out-of-support"
    assert result.support_warnings
    assert "outside the training observations" in " ".join(result.support_warnings)
    assert service.support_numeric["loan_amnt"]["min"] < service.support_numeric["loan_amnt"]["max"]


def test_in_support_application_keeps_saved_approve_refer_reject_policy(actual_cal_b_application):
    service = CreditDecisionService.get_instance()
    scored = service.score_many([actual_cal_b_application])
    result = service.decide(actual_cal_b_application)
    assert not result.is_provisional
    assert result.decision in {"Approve", "Refer", "Reject"}
    assert result.decision == str(scored["decisions"][0])


def test_slightly_out_of_support_input_requires_manual_review(actual_cal_b_application):
    service = CreditDecisionService.get_instance()
    just_outside = service.support_numeric["loan_amnt"]["max"] + 1
    result = service.decide({**actual_cal_b_application, "loan_amnt": just_outside})
    assert result.is_provisional
    assert result.decision == "Manual Review Required — Outside Model Support"
    assert "outside the training observations" in " ".join(result.support_warnings)


def test_extreme_450k_loan_never_receives_ordinary_recommendation(actual_cal_b_application):
    result = CreditDecisionService.get_instance().decide({**actual_cal_b_application, "loan_amnt": 450000})
    assert 0 <= result.p_default <= 1
    assert result.decision == "Manual Review Required — Outside Model Support"
    assert result.is_provisional
    assert "may be unreliable" in result.full_explanation


def test_low_income_and_extreme_loan_requires_manual_review(actual_cal_b_application):
    result = CreditDecisionService.get_instance().decide(
        {**actual_cal_b_application, "loan_amnt": 450000, "annual_inc": 1}
    )
    assert 0 <= result.p_default <= 1
    assert result.decision == "Manual Review Required — Outside Model Support"
    assert result.is_provisional


def test_high_and_low_risk_extrapolations_both_require_manual_review(actual_cal_b_application):
    service = CreditDecisionService.get_instance()
    low_risk = {**actual_cal_b_application, "loan_amnt": 450000, "annual_inc": 200000,
                "fico_avg": 850, "dti": 1, "revol_util": 1}
    high_risk = {**actual_cal_b_application, "loan_amnt": 450000, "annual_inc": 1,
                 "fico_avg": 300, "dti": 100, "revol_util": 150,
                 "delinq_2yrs": 20, "inq_last_6mths": 10}
    low_scored = service.score_many([low_risk])
    high_scored = service.score_many([high_risk])
    assert low_scored["probabilities"][0] < service.threshold
    assert high_scored["probabilities"][0] > service.threshold
    for applicant, scored in ((low_risk, low_scored), (high_risk, high_scored)):
        assert scored["decisions"][0] == "Manual Review Required — Outside Model Support"
        assert scored["is_provisional"][0]
        assert service.decide(applicant).decision == "Manual Review Required — Outside Model Support"


def test_supported_approve_reject_and_refer_remain_unchanged(supported_policy_applications):
    service = CreditDecisionService.get_instance()
    for expected, application in supported_policy_applications.items():
        result = service.decide(application)
        assert result.is_provisional is False
        assert result.formal_status == expected
        assert result.decision == expected
        assert result.indicative_recommendation == expected


def test_extreme_out_of_support_cases_keep_formal_manual_review_and_expose_model_outcome(actual_cal_b_application):
    service = CreditDecisionService.get_instance()
    base = dict(actual_cal_b_application)
    cases = {
        "300k_1200_uncertain": {**base, "loan_amnt": 300000, "annual_inc": 1200},
        "250k_50k_low_risk": {**base, "loan_amnt": 250000, "annual_inc": 50000},
        "250k_strong_profile": {**base, "loan_amnt": 250000, "annual_inc": 200000,
                                 "fico_avg": 850, "dti": 2, "emp_length": 10,
                                 "revol_bal": 1000, "revol_util": 5, "delinq_2yrs": 0,
                                 "inq_last_6mths": 0},
        "300k_high_risk": {**base, "loan_amnt": 300000, "annual_inc": 1200,
                           "fico_avg": 300, "dti": 100, "revol_util": 150,
                           "delinq_2yrs": 20, "inq_last_6mths": 10},
    }
    expected_indicative = {"300k_1200_uncertain": "Refer", "250k_50k_low_risk": "Approve",
                           "250k_strong_profile": "Approve", "300k_high_risk": "Reject"}
    for name, values in cases.items():
        app = LoanApplication(**values)
        scored = service.score_many([app])
        direct, reasons = notebook_decide(
            scored["probabilities"], scored["prediction_sets"], [False],
            service.threshold, service.band_halfwidth,
        )
        result = service.decide(app)
        assert result.formal_status == "Manual Review Required — Outside Model Support"
        assert result.decision == result.formal_status
        assert result.is_provisional
        assert result.indicative_recommendation == str(direct[0]) == expected_indicative[name]
        assert result.indicative_reason == (str(reasons[0]) or None)
        assert result.formal_explanation


def test_whatif_keeps_formal_status_and_reports_indicative_change(actual_cal_b_application):
    service = CreditDecisionService.get_instance()
    base = dict(actual_cal_b_application)
    original = LoanApplication(**{**base, "loan_amnt": 300000, "annual_inc": 1200})
    modified = LoanApplication(**{**base, "loan_amnt": 250000, "annual_inc": 50000})
    response = whatif(WhatIfRequest(original_application=original, modified_application=modified))
    transformed = service.transform_many([original, modified])
    with service.inference_lock:
        direct_probabilities = service.lgbm_calibrated.predict_proba(transformed)[:, 1]
    assert response["original"]["formal_status"] == "Manual Review Required — Outside Model Support"
    assert response["modified"]["formal_status"] == "Manual Review Required — Outside Model Support"
    assert response["original"]["indicative_recommendation"] == "Refer"
    assert response["modified"]["indicative_recommendation"] == "Approve"
    assert response["original"]["p_default"] == pytest.approx(direct_probabilities[0], abs=5e-5)
    assert response["modified"]["p_default"] == pytest.approx(direct_probabilities[1], abs=5e-5)
    assert response["decision_changed"] is False
    assert response["indicative_recommendation_changed"] is True


def test_batch_predictions_match_direct_saved_model_for_supported_and_extreme_inputs(actual_cal_b_application):
    service = CreditDecisionService.get_instance()
    applicants = [dict(actual_cal_b_application),
                  {**actual_cal_b_application, "loan_amnt": 300000, "annual_inc": 1200},
                  {**actual_cal_b_application, "loan_amnt": 250000, "annual_inc": 50000}]
    transformed = service.transform_many(applicants)
    with service.inference_lock:
        baseline = service.lgbm_calibrated.predict_proba(transformed)[:, 1]
    result = batch([LoanApplication(**app) for app in applicants])
    returned = np.asarray([row["p_default"] for row in result["results"]])
    np.testing.assert_allclose(returned, baseline, rtol=0, atol=1e-12)
    assert all({"formal_status", "indicative_recommendation", "is_provisional",
                "support_warnings", "formal_explanation"} <= row.keys() for row in result["results"])
