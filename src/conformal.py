"""Conformal prediction sets, out-of-support rules, three-way decision."""
from __future__ import annotations

import numpy as np
import pandas as pd
from mapie.classification import SplitConformalClassifier


def check_oos_rules(raw: pd.DataFrame, rules) -> np.ndarray:
    """raw needs columns: fico_range_low, fico_range_high, dti, annual_inc, loan_amnt.
    Returns bool array, True = out-of-support."""
    fico = (raw["fico_range_low"] + raw["fico_range_high"]) / 2
    return ((fico < rules.fico_min) | (raw["dti"] > rules.dti_max)
            | (raw["annual_inc"] < rules.annual_inc_min)
            | (raw["loan_amnt"] > rules.loan_amnt_max)).to_numpy()


def build_conformal_model(calibrated, X_cal_B, y_cal_B, confidence_level: float):
    m = SplitConformalClassifier(estimator=calibrated, confidence_level=confidence_level,
                                 conformity_score="lac", prefit=True)
    return m.conformalize(X_cal_B, y_cal_B)


def get_prediction_set(model, X) -> np.ndarray:
    """Bool (n, 2): column k True if class k is in the set."""
    _, ps = model.predict_set(X)
    return np.asarray(ps)[:, :, 0]


def sweep_alpha(calibrated, X_cal_B, y_cal_B, alphas) -> pd.DataFrame:
    """Coverage and both-class rate on cal_B for each alpha (coverage holds by construction)."""
    y = np.asarray(y_cal_B)
    rows = []
    for a in alphas:
        ps = get_prediction_set(build_conformal_model(calibrated, X_cal_B, y_cal_B, 1 - a), X_cal_B)
        rows.append({"alpha": a, "confidence_level": 1 - a,
                     "empirical_coverage": float(ps[np.arange(len(y)), y.astype(int)].mean()),
                     "refer_rate": float(ps.all(axis=1).mean())})
    return pd.DataFrame(rows)


def decide(p_default, pred_set, oos, threshold: float, band_halfwidth: float):
    """Precedence: OOS -> Refer; both-class set AND |p - t| <= band -> Refer; else p < t Approve, else Reject.
    Returns (decision array, reason array)."""
    p = np.asarray(p_default); both = np.asarray(pred_set).all(axis=1); oos = np.asarray(oos)
    ambiguous = both & (np.abs(p - threshold) <= band_halfwidth)
    dec = np.where(oos | ambiguous, "Refer", np.where(p < threshold, "Approve", "Reject"))
    reason = np.where(oos, "out-of-support", np.where(ambiguous, "ambiguous", ""))
    return dec, reason