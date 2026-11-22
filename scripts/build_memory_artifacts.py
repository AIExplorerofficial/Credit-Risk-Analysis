"""Build small, exact runtime summaries from the notebook's frozen data.

Run from the project root with ``python scripts/build_memory_artifacts.py``.
This utility reads the existing notebook artifacts and processed datasets; it
does not train or alter any model, source dataset, or prediction policy.
"""
from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.feature_engineering import engineer_raw_features


def main() -> None:
    project_root = PROJECT_ROOT
    runtime = project_root / "notebook_runtime"
    processed = runtime / "data" / "processed"
    model_dir = runtime / "models"
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    from src.config_loader import load_config

    cfg = load_config(project_root / "config.yaml")

    target = str(cfg.target.column)
    train = pd.read_parquet(cfg.paths.train_parquet)
    train_features = train.drop(columns=[target])
    if list(train_features.columns) != list(metadata["feature_columns"]):
        raise RuntimeError("Training split feature order differs from saved model metadata")

    # Preserve the serving implementation's exact deterministic pandas sample.
    seed = int(cfg.modeling.random_seed)
    background_size = int(cfg.shap.background_sample_n)
    background = train_features.sample(min(background_size, len(train_features)), random_state=seed)

    # Support is computed from raw observations aligned to the training split,
    # with the same engineering and pandas quantile defaults as the service.
    loans = pd.read_parquet(cfg.paths.loans_parquet)
    training_raw = engineer_raw_features(loans.loc[train.index])
    support_numeric: dict[str, dict[str, float]] = {}
    for name in training_raw.select_dtypes(include=[np.number]).columns:
        values = pd.to_numeric(training_raw[name], errors="coerce").dropna()
        if not len(values):
            continue
        q1, q3 = values.quantile([0.25, 0.75]).tolist()
        iqr = q3 - q1
        support_numeric[name] = {
            "min": float(values.min()),
            "max": float(values.max()),
            "typical_min": float(q1 - 1.5 * iqr),
            "typical_max": float(q3 + 1.5 * iqr),
        }
    support_categories = {
        name: sorted(training_raw[name].dropna().astype(str).unique().tolist())
        for name in ("home_ownership", "purpose")
    }

    # The current /api/meta options use categories from the complete historical
    # loans dataset, so retain those exact values separately from train support.
    data_categories = {
        name: sorted(loans[name].dropna().astype(str).unique().tolist())
        for name in ("home_ownership", "purpose")
    }

    output_dir = runtime / "memory"
    output_dir.mkdir(parents=True, exist_ok=True)
    background_path = output_dir / "shap_background.parquet"
    background.to_parquet(background_path, index=True)
    summary = {
        "source": "notebooks/02_training.ipynb frozen train split and loans_36m.parquet",
        "target": target,
        "feature_columns": list(train_features.columns),
        "training_rows": int(len(train_features)),
        "seed": seed,
        "shap_background_sample_n": background_size,
        "shap_background_rows": int(len(background)),
        "support_numeric": support_numeric,
        "support_categories": support_categories,
        "data_categories": data_categories,
    }
    # Pin newline bytes so the checksum is identical on Windows and Linux.
    with (output_dir / "support_summary.json").open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    manifest_path = runtime / "artifact_checksums.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative in ("memory/support_summary.json", "memory/shap_background.parquet"):
        digest = hashlib.sha256((runtime / relative).read_bytes()).hexdigest()
        manifest[relative] = digest
    with manifest_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote exact support summary ({len(support_numeric)} numeric fields).")
    print(f"Wrote deterministic SHAP background: {len(background)} rows × {background.shape[1]} features.")


if __name__ == "__main__":
    main()
