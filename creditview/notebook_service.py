"""Inference service backed only by the saved artifacts and data from 02_training.ipynb."""
from __future__ import annotations

import json
import hashlib
import sys
import threading
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as parquet

from creditview.schemas import (
    DecisionResponse,
    LoanApplication,
    ReasonFactor,
    WhatIfResponse,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_NOTEBOOK_ROOT = _PROJECT_ROOT / "notebook_runtime"
if str(_NOTEBOOK_ROOT) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOK_ROOT))

from src.config_loader import load_config  # noqa: E402
from src.conformal import decide, get_prediction_set  # noqa: E402
from src.explainer import build_tree_explainer, compute_shap_values, get_top_reason_codes  # noqa: E402
from src.feature_engineering import engineer_raw_features  # noqa: E402

OUTSIDE_SUPPORT_DECISION = "Manual Review Required — Outside Model Support"


class CreditDecisionService:
    """Load the notebook's exact inference artifacts once and share one scoring path."""

    _instance: CreditDecisionService | None = None
    _instance_lock = threading.RLock()

    def __init__(self) -> None:
        self.inference_lock = threading.RLock()
        self.cfg = load_config(_PROJECT_ROOT / "config.yaml")
        model_dir = _NOTEBOOK_ROOT / "models"
        self.metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
        self.preprocessor = joblib.load(model_dir / "preprocessor.pkl")
        self._lr_baseline = None
        self.lgbm_base = joblib.load(model_dir / "lgbm.pkl")
        self.lgbm_calibrated = joblib.load(model_dir / "lgbm_calibrated.pkl")
        self.conformal_model = joblib.load(model_dir / "lgbm_conformal.pkl")

        if list(self.lgbm_base.classes_) != [0, 1] or list(self.lgbm_calibrated.classes_) != [0, 1]:
            raise RuntimeError("Notebook model class order must be [0, 1] with default mapped to class 1")
        if self.metadata.get("calibration_method") != "sigmoid":
            raise RuntimeError("Notebook metadata does not identify the expected sigmoid calibrator")
        if int(self.cfg.target.positive_label) != 1:
            raise RuntimeError("Notebook target mapping changed: expected default class 1")

        self.feature_columns = list(self.metadata["feature_columns"])
        train_schema_names = parquet.read_schema(self.cfg.paths.train_parquet).names
        expected_train_columns = self.feature_columns + [str(self.cfg.target.column)]
        if train_schema_names[:len(expected_train_columns)] != expected_train_columns:
            raise RuntimeError("Notebook training split feature order does not match notebook model metadata")
        transformed_names = list(self.preprocessor.get_feature_names_out())
        if transformed_names != self.feature_columns:
            raise RuntimeError("Notebook preprocessor feature order does not match notebook model metadata")

        summary_path = _NOTEBOOK_ROOT / "memory" / "support_summary.json"
        background_path = _NOTEBOOK_ROOT / "memory" / "shap_background.parquet"
        manifest = json.loads((_NOTEBOOK_ROOT / "artifact_checksums.json").read_text(encoding="utf-8"))
        for relative, path in (("memory/support_summary.json", summary_path),
                               ("memory/shap_background.parquet", background_path)):
            expected = manifest.get(relative)
            digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
            if expected is None or digest != expected:
                raise RuntimeError(f"Notebook-derived runtime artifact integrity check failed: {relative}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("feature_columns") != self.feature_columns:
            raise RuntimeError("Precomputed notebook summary feature order does not match model metadata")
        if summary.get("target") != str(self.cfg.target.column):
            raise RuntimeError("Precomputed notebook summary target does not match active configuration")
        if summary.get("seed") != int(self.cfg.modeling.random_seed):
            raise RuntimeError("Precomputed SHAP background seed does not match active configuration")
        background = pd.read_parquet(background_path)
        if list(background.columns) != self.feature_columns:
            raise RuntimeError("Precomputed SHAP background feature order does not match model metadata")
        if len(background) != int(self.cfg.shap.background_sample_n):
            raise RuntimeError("Precomputed SHAP background sample size does not match active configuration")
        if len(background) != int(summary.get("shap_background_rows", -1)):
            raise RuntimeError("Precomputed SHAP background row count does not match its summary")

        self.support_numeric = summary["support_numeric"]
        self.support_categories = {
            name: set(values) for name, values in summary["support_categories"].items()
        }
        self.dataset_categories = summary["data_categories"]

        # The notebook uses an observed training sample as its SHAP background.
        self.explainer = build_tree_explainer(self.lgbm_base, background)
        del background
        self.threshold = float(self.metadata["threshold"])
        self.band_halfwidth = float(self.metadata["band_halfwidth"])
        self.approve_below = self.threshold - self.band_halfwidth
        self.reject_above = self.threshold + self.band_halfwidth
        self.oos_rules = self.cfg.three_way_decision.oos_rules
        self.model_version = "notebook-02-training-seed-" + str(int(self.cfg.modeling.random_seed))

    @classmethod
    def get_instance(cls, cfg: Any = None) -> "CreditDecisionService":
        del cfg  # The active config is the copy bundled with the notebook artifacts.
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @property
    def lr_baseline(self):
        """Load the governance-only baseline model on first use, not at startup."""
        with self.inference_lock:
            if self._lr_baseline is None:
                self._lr_baseline = joblib.load(_NOTEBOOK_ROOT / "models" / "lr_baseline.pkl")
            return self._lr_baseline

    @staticmethod
    def _as_application(value: dict[str, Any] | LoanApplication) -> LoanApplication:
        return value if isinstance(value, LoanApplication) else LoanApplication(**value)

    @staticmethod
    def _raw_frame(app: LoanApplication) -> pd.DataFrame:
        raw = app.model_dump()
        fico = float(raw["fico_avg"])
        raw["fico_range_low"] = fico
        raw["fico_range_high"] = fico
        raw.setdefault("mort_acc", np.nan)
        raw.setdefault("application_type", "Individual")
        return pd.DataFrame([raw])

    def transform_many(self, applications: list[dict[str, Any] | LoanApplication]) -> pd.DataFrame:
        parsed = [self._as_application(item) for item in applications]
        frames = []
        for app in parsed:
            raw = app.model_dump()
            fico = float(raw["fico_avg"])
            raw["fico_range_low"] = fico
            raw["fico_range_high"] = fico
            raw.setdefault("mort_acc", np.nan)
            raw.setdefault("application_type", "Individual")
            frames.append(raw)
        raw_frame = pd.DataFrame(frames)
        features = engineer_raw_features(raw_frame)
        with self.inference_lock:
            transformed = self.preprocessor.transform(features)
        transformed = transformed.loc[:, self.feature_columns]
        return transformed

    def score_many(self, applications: list[dict[str, Any] | LoanApplication]) -> dict[str, Any]:
        apps = [self._as_application(item) for item in applications]
        transformed = self.transform_many(apps)
        with self.inference_lock:
            probabilities = self.lgbm_calibrated.predict_proba(transformed)[:, 1]
            prediction_sets = get_prediction_set(self.conformal_model, transformed)

        # Risk and the saved threshold/conformal policy are evaluated first. A
        # separate eligibility gate replaces that outcome when training support
        # does not cover the input; historical cohort evaluation remains unchanged.
        support = [self._support_warnings(app) for app in apps]
        live_out_of_support = np.zeros(len(apps), dtype=bool)
        indicative_decisions, indicative_reasons = decide(
            probabilities,
            prediction_sets,
            live_out_of_support,
            self.threshold,
            self.band_halfwidth,
        )
        indicative_decisions = np.asarray(indicative_decisions, dtype=object)
        indicative_reasons = np.asarray(indicative_reasons, dtype=object)
        decisions = indicative_decisions.copy()
        reasons = indicative_reasons.copy()
        provisional = np.asarray([self._is_out_of_support(warnings) for warnings in support])
        decisions[provisional] = OUTSIDE_SUPPORT_DECISION
        reasons[provisional] = "out-of-support"
        formal_explanations = [
            self._formal_explanation(bool(provisional[i]), str(indicative_decisions[i]),
                                     str(indicative_reasons[i]) or None)
            for i in range(len(apps))
        ]
        return {
            "applications": apps,
            "transformed": transformed,
            "probabilities": probabilities.astype(float),
            "prediction_sets": prediction_sets,
            "support_warnings": support,
            "is_provisional": provisional,
            "decisions": decisions,
            "reasons": reasons,
            "indicative_recommendations": indicative_decisions,
            "indicative_reasons": indicative_reasons,
            "formal_explanations": formal_explanations,
        }

    def _formal_explanation(self, provisional: bool, indicative: str,
                            indicative_reason: str | None) -> str:
        if provisional:
            return ("This application is outside the observed training support. Manual review is required; "
                    "the model's risk estimate and indicative recommendation may be unreliable for these inputs.")
        if indicative_reason == "ambiguous":
            return (f"The conformal model considers both outcomes plausible and estimated risk is within "
                    f"{self.band_halfwidth:.1%} of the {self.threshold:.1%} threshold, so the application is referred.")
        if indicative == "Approve":
            return f"Estimated risk is below the saved {self.threshold:.1%} threshold."
        return f"Estimated risk is at or above the saved {self.threshold:.1%} threshold."

    def _response(self, app: LoanApplication, scored: dict[str, Any], i: int) -> DecisionResponse:
        x = scored["transformed"].iloc[[i]]
        with self.inference_lock:
            shap_row = compute_shap_values(self.explainer, x)[0]
        raw = app.model_dump()
        reason_rows = get_top_reason_codes(
            shap_row,
            self.feature_columns,
            raw_values=raw,
            top_k=int(self.cfg.shap.top_k_reasons),
        )
        factors = [
            ReasonFactor(
                feature=row["feature"],
                impact="adverse" if row["direction"] == "increases_risk" else "supportive",
                shap_value=float(row["shap_value"]),
                raw_value=row.get("raw_value"),
                explanation=row["reason_text"],
            )
            for row in reason_rows
        ]
        decision = str(scored["decisions"][i])
        reason = str(scored["reasons"][i]) or None
        indicative = str(scored["indicative_recommendations"][i])
        indicative_reason = str(scored["indicative_reasons"][i]) or None
        warnings = list(scored["support_warnings"][i])
        provisional = bool(scored["is_provisional"][i])
        formal_explanation = str(scored["formal_explanations"][i])
        if provisional:
            warnings.append("This risk estimate is provisional and may be unreliable outside validated training support. A human reviewer must assess the application.")
        return DecisionResponse(
            decision=decision,
            formal_status=decision,
            indicative_recommendation=indicative,
            indicative_reason=indicative_reason,
            formal_explanation=formal_explanation,
            p_default=round(float(scored["probabilities"][i]), 4),
            refer_reason=reason,
            approve_below=self.approve_below,
            reject_above=self.reject_above,
            decision_threshold=self.threshold,
            band_halfwidth=self.band_halfwidth,
            is_out_of_support=provisional,
            is_provisional=provisional,
            support_warnings=warnings,
            top_factors=factors,
            explanation_verified=False,
            full_explanation=(
                f"{formal_explanation} The calibrated default estimate is "
                f"{float(scored['probabilities'][i]):.1%}. The indicative model outcome is {indicative}. "
                "Feature contributions below are "
                "SHAP attributions from the notebook's base LightGBM model."
            ),
            recourse=None,
            model_version=self.model_version,
        )

    def decide(self, application_data: dict[str, Any] | LoanApplication) -> DecisionResponse:
        app = self._as_application(application_data)
        scored = self.score_many([app])
        return self._response(app, scored, 0)

    def what_if(self, original_app: Any, modified_app: Any) -> WhatIfResponse:
        original = self._as_application(original_app)
        modified = self._as_application(modified_app)
        scored = self.score_many([original, modified])
        first = self._response(original, scored, 0)
        second = self._response(modified, scored, 1)
        first_values, second_values = original.model_dump(), modified.model_dump()
        changes = [f"{key}: {first_values[key]} → {second_values[key]}"
                   for key in first_values if first_values.get(key) != second_values.get(key)]
        return WhatIfResponse(
            original=first,
            modified=second,
            delta_p_default=round(second.p_default - first.p_default, 4),
            decision_changed=first.decision != second.decision,
            indicative_recommendation_changed=(first.indicative_recommendation != second.indicative_recommendation),
            key_drivers=changes,
        )

    def get_diagnostics(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "approve_below": self.approve_below,
            "reject_above": self.reject_above,
            "band_halfwidth": self.band_halfwidth,
            "oos_rules": self.metadata["oos_rules"],
            "model_version": self.model_version,
            "notebook_runtime": str(_NOTEBOOK_ROOT),
        }

    @staticmethod
    def _is_out_of_support(warnings: list[str]) -> bool:
        return any("is outside the training observations" in w or "was not observed" in w for w in warnings)

    def _support_warnings(self, app: LoanApplication) -> list[str]:
        """Compare inputs with training-only observed ranges and Tukey fences."""
        raw = app.model_dump()
        engineered = engineer_raw_features(self._raw_frame(app)).iloc[0]
        warnings: list[str] = []
        for name, limits in self.support_numeric.items():
            value = engineered.get(name)
            if value is None or pd.isna(value):
                continue
            value = float(value)
            label = name.replace("_", " ")
            if value < limits["min"] or value > limits["max"]:
                warnings.append(f"{label.capitalize()} {value:g} is outside the training observations ({limits['min']:g}–{limits['max']:g}); this is an extrapolation.")
            elif value < limits["typical_min"] or value > limits["typical_max"]:
                warnings.append(f"{label.capitalize()} {value:g} is unusual compared with the training data (Tukey 1.5×IQR typical range {limits['typical_min']:g}–{limits['typical_max']:g}).")
        for name, values in self.support_categories.items():
            value = str(raw[name])
            if value not in values:
                warnings.append(f"{name.replace('_', ' ').capitalize()} category {value!r} was not observed in the training split.")
        return warnings
