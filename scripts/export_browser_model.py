"""Export the frozen notebook inference artifacts for in-browser execution.

This writes a JSON representation of the saved model and transforms. It does
not fit, retrain, or modify any notebook model or source dataset.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "notebook_runtime"
OUTPUT = ROOT / "frontend" / "runtime" / "model.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_loader import load_config  # noqa: E402


def export() -> None:
    model_dir = RUNTIME / "models"
    config = load_config(ROOT / "config.yaml")
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    preprocessor = joblib.load(model_dir / "preprocessor.pkl")
    calibrated = joblib.load(model_dir / "lgbm_calibrated.pkl")
    conformal = joblib.load(model_dir / "lgbm_conformal.pkl")
    base_model = joblib.load(model_dir / "lgbm.pkl")
    shap_background = pd.read_parquet(RUNTIME / "memory" / "shap_background.parquet")
    expected_background_rows = int(config.shap.background_sample_n)
    if len(shap_background) != expected_background_rows:
        raise RuntimeError("The browser SHAP background size differs from config.yaml")
    if list(shap_background.columns) != list(metadata["feature_columns"]):
        raise RuntimeError("The browser SHAP background feature order differs from the notebook model")

    if len(calibrated.calibrated_classifiers_) != 1:
        raise RuntimeError("Browser export expects the notebook's single frozen sigmoid calibrator")
    fitted_calibrator = calibrated.calibrated_classifiers_[0].calibrators[0]
    if not hasattr(fitted_calibrator, "a_") or not hasattr(fitted_calibrator, "b_"):
        raise RuntimeError("Saved calibration artifact is not a fitted sigmoid calibrator")

    numeric = {}
    for transformer_name, source_fields in (
        ("num_med", preprocessor.transformers_[0][2]),
        ("num_zero", preprocessor.transformers_[1][2]),
    ):
        pipeline = preprocessor.named_transformers_[transformer_name]
        imputer = pipeline.named_steps["imp"]
        scaler = pipeline.named_steps["sc"]
        for index, source in enumerate(source_fields):
            numeric[source] = {
                "impute": float(imputer.statistics_[index]),
                "mean": float(scaler.mean_[index]),
                "scale": float(scaler.scale_[index]),
                "feature": f"{transformer_name}__{source}",
            }

    onehot = preprocessor.named_transformers_["cat"]
    categorical = {}
    for field_index, source in enumerate(preprocessor.transformers_[2][2]):
        categories = [str(value) for value in onehot.categories_[field_index]]
        rare_indices = (onehot._infrequent_indices[field_index]
                        if onehot._infrequent_indices is not None else None)
        rare = [categories[index] for index in rare_indices] if rare_indices is not None else []
        encoded = {}
        for feature in metadata["feature_columns"]:
            prefix = f"cat__{source}_"
            if feature.startswith(prefix):
                encoded[feature[len(prefix):]] = feature
        categorical[source] = {
            "categories": categories,
            "infrequent": rare,
            "encoded_features": encoded,
        }

    conformal_alpha = float(metadata["alpha_operational"])
    conformity_scores = np.asarray(conformal._mapie_classifier.conformity_scores_).reshape(-1)
    rank = int(np.ceil((len(conformity_scores) + 1) * (1 - conformal_alpha)))
    quantile = float(np.quantile(conformity_scores, rank / len(conformity_scores), method="lower"))

    exported = {
        "format_version": 1,
        "source": "notebooks/02_training.ipynb saved artifacts",
        "model_version": f"notebook-02-training-seed-{int(config.modeling.random_seed)}",
        "classes": [int(value) for value in base_model.classes_],
        "feature_columns": list(metadata["feature_columns"]),
        "tree_model": base_model.booster_.dump_model(),
        "preprocessing": {
            "numeric": numeric,
            "categorical": categorical,
        },
        "calibration": {
            "method": "sigmoid",
            "a": float(fitted_calibrator.a_),
            "b": float(fitted_calibrator.b_),
            "formula": "1 / (1 + exp(a * decision_function + b))",
        },
        "conformal": {
            "method": "LAC",
            "alpha": conformal_alpha,
            "quantile": quantile,
            "classes": [int(value) for value in base_model.classes_],
        },
        "decision": {
            "threshold": float(metadata["threshold"]),
            "band_halfwidth": float(metadata["band_halfwidth"]),
        },
        "shap": {
            "algorithm": "SHAP TreeExplainer interventional, model_output=raw",
            "top_k_reasons": int(config.shap.top_k_reasons),
            "background_rows": len(shap_background),
            "background_features": list(shap_background.columns),
            "background": shap_background.to_numpy(dtype=np.float64).tolist(),
        },
        "support": json.loads((RUNTIME / "memory" / "support_summary.json").read_text(encoding="utf-8")),
        "oos_rules": {
            key: float(getattr(config.three_way_decision.oos_rules, key))
            for key in ("fico_min", "dti_max", "annual_inc_min", "loan_amnt_max")
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(exported, handle, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        handle.write("\n")
    print(f"Exported frozen notebook inference artifacts to {OUTPUT.relative_to(ROOT)}")
    print(f"Features: {len(exported['feature_columns'])}; trees: {len(exported['tree_model']['tree_info'])}; "
          f"conformal alpha: {conformal_alpha:g}")


if __name__ == "__main__":
    export()
