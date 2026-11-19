"""Phase 2: load config.yaml with attribute access (cfg.paths.model_dir, cfg.splits.train_years, ...)."""
from __future__ import annotations

from pathlib import Path

import yaml

REQUIRED_TOP_KEYS = [
    "project", "paths", "target", "scope", "splits", "features",
    "leakage_banned_columns", "modeling", "calibration", "thresholds",
    "profit_model", "three_way_decision",
]


class Cfg:
    """Read-only wrapper over a dict. Nested dicts become Cfg objects; lists and scalars are returned as is.

    Supports  cfg.a.b  attribute access, and  **cfg.modeling.lightgbm  unpacking into function kwargs.
    """

    def __init__(self, data: dict):
        object.__setattr__(self, "_data", data)

    def __getattr__(self, name: str):
        data = object.__getattribute__(self, "_data")
        if name not in data:
            raise AttributeError(f"config key not found: {name!r}")
        v = data[name]
        return Cfg(v) if isinstance(v, dict) else v

    def __setattr__(self, name, value):
        raise AttributeError("config is read-only")

    def __getitem__(self, key):
        return self._data[key]

    def keys(self):
        return self._data.keys()

    def to_dict(self) -> dict:
        return self._data

    def __repr__(self) -> str:
        return f"Cfg({list(self._data.keys())})"


def load_config(path: str | Path = "config.yaml") -> Cfg:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run from the creditlens/ folder.")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    missing = [k for k in REQUIRED_TOP_KEYS if k not in data]
    if missing:
        raise KeyError(f"config.yaml is missing required keys: {missing}")
    return Cfg(data)


def get_feature_names(cfg: Cfg) -> list[str]:
    """Configured raw feature names (numeric + categorical). Fairness-only columns are excluded."""
    return list(cfg.features.numeric) + list(cfg.features.categorical)


def get_banned_columns(cfg: Cfg) -> list[str]:
    return list(cfg.leakage_banned_columns)


import json


def assert_assumed_apr_not_null(cfg: Cfg, metadata_path: str | Path | None = None) -> None:
    """Reads assumed_apr from metadata.json, not config.yaml. Raises ValueError if null/missing."""
    if metadata_path is None:
        metadata_path = Path(cfg.paths.model_dir) / "metadata.json"
    else:
        metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise ValueError(f"metadata.json not found at {metadata_path}. Run training notebook first.")
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("assumed_apr") is None:
        raise ValueError("assumed_apr is null in metadata.json. Run threshold / training step first.")
