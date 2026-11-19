"""SHAP explainer for the base LightGBM model (raw margin / log-odds space)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap


def build_tree_explainer(lgbm, background: pd.DataFrame) -> shap.TreeExplainer:
    masker = shap.maskers.Independent(background, max_samples=len(background))
    return shap.TreeExplainer(lgbm, data=masker, feature_perturbation="interventional")


def compute_shap_values(explainer, X: pd.DataFrame) -> np.ndarray:
    """Returns (n, n_features) log-odds contributions for the positive class (default)."""
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        sv = sv[-1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, -1]
    assert sv.shape == (len(X), X.shape[1]), f"unexpected SHAP shape {sv.shape}"
    return sv


def source_feature(c: str) -> str:
    """Map transformed feature name back to its source feature name."""
    if c.startswith("cat__home_ownership"):
        return "home_ownership"
    if c.startswith("cat__purpose"):
        return "purpose"
    return c.split("__", 1)[1] if "__" in c else c


REASON_TEMPLATES: dict[str, dict[str, str]] = {
    "fico_avg": {
        "increases_risk": "Credit score (FICO {val}) is lower than optimal, increasing estimated risk of default.",
        "decreases_risk": "Strong credit score (FICO {val}) provides substantial positive creditworthiness support.",
    },
    "dti": {
        "increases_risk": "Debt-to-income ratio ({val}%) indicates high monthly debt burden relative to income.",
        "decreases_risk": "Low debt-to-income ratio ({val}%) indicates manageable debt obligations.",
    },
    "revol_util": {
        "increases_risk": "Revolving line utilization ({val}%) is elevated, indicating heavy revolving credit reliance.",
        "decreases_risk": "Disciplined revolving utilization ({val}%) indicates credit line buffer.",
    },
    "annual_inc": {
        "increases_risk": "Annual income (${val:,.0f}) is on the lower side for the requested borrowing profile.",
        "decreases_risk": "Annual income (${val:,.0f}) demonstrates strong capacity to service debt.",
    },
    "loan_amnt": {
        "increases_risk": "Requested loan amount (${val:,.0f}) represents substantial principal exposure.",
        "decreases_risk": "Moderate loan amount (${val:,.0f}) reduces principal risk.",
    },
    "delinq_2yrs": {
        "increases_risk": "History of delinquency ({val:.0f} in past 2 years) signals elevated credit risk.",
        "decreases_risk": "Clean 2-year delinquency history supports reliable repayment tendency.",
    },
    "inq_last_6mths": {
        "increases_risk": "Recent credit inquiries ({val:.0f} in past 6 months) indicates active credit seeking.",
        "decreases_risk": "Few recent credit inquiries indicates stable credit demand.",
    },
    "credit_history_years": {
        "increases_risk": "Shorter credit history ({val:.1f} years) limits historical repayment track record.",
        "decreases_risk": "Established credit history ({val:.1f} years) provides solid repayment evidence.",
    },
    "revol_bal": {
        "increases_risk": "Revolving balance (${val:,.0f}) represents substantial outstanding debt.",
        "decreases_risk": "Low revolving balance (${val:,.0f}) reflects limited outstanding debt burden.",
    },
    "open_acc": {
        "increases_risk": "Number of open accounts ({val:.0f}) suggests broader exposure across credit lines.",
        "decreases_risk": "Active open accounts ({val:.0f}) demonstrates active credit management.",
    },
    "total_acc": {
        "increases_risk": "High total account count ({val:.0f}) indicates extensive credit activity history.",
        "decreases_risk": "Established credit account history ({val:.0f} total accounts) confirms credit familiarity.",
    },
    "emp_length": {
        "increases_risk": "Shorter employment tenure ({val:.0f} years) may affect income stability.",
        "decreases_risk": "Stable employment history ({val:.0f} years) reinforces income reliability.",
    },
    "pub_rec": {
        "increases_risk": "Presence of derogatory public records ({val:.0f}) elevates risk assessment.",
        "decreases_risk": "Clean public record history supports overall borrower integrity.",
    },
    "home_ownership": {
        "increases_risk": "Housing ownership status ({val}) carries higher risk profile in historical data.",
        "decreases_risk": "Housing stability ({val}) provides supportive borrower stability.",
    },
    "purpose": {
        "increases_risk": "Stated loan purpose ({val}) is associated with higher default frequency.",
        "decreases_risk": "Stated loan purpose ({val}) is historically associated with lower default frequency.",
    },
}


def get_top_reason_codes(
    shap_row: np.ndarray,
    feature_names: list[str],
    raw_values: dict | None = None,
    top_k: int = 3,
) -> list[dict]:
    """Aggregate transformed features to source features and return top_k plain-language reasons.

    Magnitudes are base-model log-odds contributions for default (class 1).
    shap_row: 1D array of SHAP values for the transformed columns.
    feature_names: list of transformed column names matching shap_row.
    raw_values: optional dict of raw input values for templating (e.g. {'fico_avg': 680, ...}).
    """
    shap_series = pd.Series(shap_row, index=feature_names)
    src_map = pd.Series([source_feature(c) for c in feature_names], index=feature_names)
    agg_shap = shap_series.groupby(src_map).sum()

    # Sort by absolute magnitude descending
    top_sources = agg_shap.abs().sort_values(ascending=False).head(top_k).index.tolist()

    reasons = []
    for feat in top_sources:
        val_shap = float(agg_shap[feat])
        direction = "increases_risk" if val_shap > 0 else "decreases_risk"
        templates = REASON_TEMPLATES.get(feat, {})
        tpl = templates.get(direction, f"{feat} contributes to {direction} (SHAP log-odds: {val_shap:+.3f})")

        val_display = None
        if raw_values and feat in raw_values:
            val_display = raw_values[feat]

        try:
            if val_display is not None:
                text = tpl.format(val=val_display)
            else:
                text = f"{feat} ({direction.replace('_', ' ')}: {val_shap:+.3f} log-odds)"
        except Exception:
            text = f"{feat} ({direction.replace('_', ' ')}: {val_shap:+.3f} log-odds)"

        reasons.append({
            "feature": feat,
            "shap_value": round(val_shap, 4),
            "direction": direction,
            "raw_value": val_display,
            "reason_text": text,
        })

    return reasons
