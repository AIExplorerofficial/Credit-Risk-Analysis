"""Package exact notebook rows as static, typed browser data.

This is an offline export step. It reads the frozen notebook splits and never
trains or alters a model. All statistics can then be calculated by the browser.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.analytics import (
    _cal_b_indices,
    _load_split,
    _read_loans_rows,
    _read_split_rows,
    examples_data,
    meta_data,
)
from creditview.notebook_service import CreditDecisionService
from src.feature_engineering import engineer_raw_features


RUNTIME = ROOT / "notebook_runtime"
OUTPUT_DIR = ROOT / "frontend" / "runtime"
TARGET = "default"
RAW_FIELDS = [
    "loan_amnt", "annual_inc", "dti", "open_acc", "pub_rec", "revol_bal",
    "revol_util", "total_acc", "delinq_2yrs", "inq_last_6mths", "mort_acc",
    "emp_length", "fico_range_low", "fico_range_high", "issue_d",
    "earliest_cr_line", "home_ownership", "purpose", "application_type",
    "default", "total_pymnt", "int_rate",
]
COHORT_NUMERIC = [
    "loan_amnt", "annual_inc", "dti", "fico_avg", "emp_length", "revol_bal",
    "revol_util", "open_acc", "total_acc", "delinq_2yrs", "inq_last_6mths",
    "credit_history_years",
]
TEST_RAW_COLUMNS = ["issue_d", "fico_range_low", "fico_range_high", "dti", "annual_inc", "loan_amnt"]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    service = CreditDecisionService.get_instance()
    arrays: list[tuple[str, np.ndarray, str]] = []
    sections: dict[str, dict] = {}
    payload_parts: list[bytes] = []
    offset = 0

    def add(name: str, values, dtype: str, shape: list[int] | None = None) -> None:
        nonlocal offset
        array = np.asarray(values, dtype=np.dtype(dtype), order="C")
        if shape is not None:
            array = array.reshape(shape)
        alignment = max(1, array.dtype.alignment)
        padding = (-offset) % alignment
        if padding:
            payload_parts.append(bytes(padding))
            offset += padding
        raw = array.tobytes(order="C")
        sections[name] = {"dtype": dtype, "shape": list(array.shape), "offset": offset, "nbytes": len(raw)}
        payload_parts.append(raw)
        offset += len(raw)

    def add_split(name: str, frame: pd.DataFrame) -> None:
        X = frame[service.feature_columns].to_numpy(dtype=np.float64, copy=True)
        y = frame[TARGET].to_numpy(dtype=np.uint8, copy=True)
        add(f"{name}.X", X, "<f8")
        add(f"{name}.y", y, "|u1")

    # Keep chronological split row order exactly as saved in the notebook.
    train = _load_split("train", columns=[TARGET])
    early_stop = _load_split("early_stop", columns=[TARGET])
    calibration = _load_split("cal")
    test = _load_split("test")
    add("train.y", train[TARGET].to_numpy(dtype=np.uint8), "|u1")
    add("early_stop.y", early_stop[TARGET].to_numpy(dtype=np.uint8), "|u1")
    add("cal.y", calibration[TARGET].to_numpy(dtype=np.uint8), "|u1")
    add_split("test", test)

    cal_b_index = pd.Index(_cal_b_indices())
    cal_b = _read_split_rows("cal", cal_b_index, [*service.feature_columns, TARGET])
    add_split("cal_b", cal_b)

    train_indices = train.index
    training_apr = _read_loans_rows(train_indices, ["int_rate"])
    apr = float(pd.to_numeric(training_apr["int_rate"], errors="coerce").mean() / 100)

    cal_b_raw = _read_loans_rows(cal_b_index, RAW_FIELDS)
    cal_b_derived = engineer_raw_features(cal_b_raw)
    for field in COHORT_NUMERIC:
        add(f"cal_b.raw.{field}", cal_b_derived[field].to_numpy(dtype=np.float64), "<f8")
    issue_month = (cal_b_raw["issue_d"].dt.year * 100 + cal_b_raw["issue_d"].dt.month).to_numpy(dtype=np.int32)
    add("cal_b.raw.issue_month", issue_month, "<i4")
    purpose_values = sorted(cal_b_derived["purpose"].dropna().astype(str).unique().tolist())
    home_values = sorted(cal_b_derived["home_ownership"].dropna().astype(str).unique().tolist())
    add("cal_b.raw.purpose_code", pd.Categorical(cal_b_derived["purpose"], categories=purpose_values).codes, "<i2")
    add("cal_b.raw.home_code", pd.Categorical(cal_b_derived["home_ownership"], categories=home_values).codes, "<i2")
    add("cal_b.raw.default", cal_b_raw[TARGET].to_numpy(dtype=np.uint8), "|u1")
    add("cal_b.raw.total_pymnt", cal_b_raw["total_pymnt"].to_numpy(dtype=np.float64), "<f8")

    test_raw = _read_loans_rows(test.index, TEST_RAW_COLUMNS)
    test_derived = engineer_raw_features(test_raw.assign(
        mort_acc=np.nan, emp_length=np.nan, home_ownership="OTHER", purpose="OTHER",
        application_type="Individual", revol_bal=np.nan, revol_util=np.nan,
        total_acc=np.nan, open_acc=np.nan, pub_rec=np.nan, delinq_2yrs=np.nan,
        inq_last_6mths=np.nan, earliest_cr_line=pd.NaT,
    ))
    test_month = (test_raw["issue_d"].dt.year * 100 + test_raw["issue_d"].dt.month).to_numpy(dtype=np.int32)
    add("test.raw.issue_month", test_month, "<i4")
    for field in ("loan_amnt", "annual_inc", "dti", "fico_avg"):
        add(f"test.raw.{field}", test_derived[field].to_numpy(dtype=np.float64), "<f8")

    # Exact random sample used by governance's current 2,000-row SHAP chart.
    cal_b_positions = np.arange(len(cal_b_index), dtype=np.int64)
    random_state = np.random.RandomState(int(service.cfg.modeling.random_seed))
    shap_positions = random_state.choice(
        len(cal_b_positions), size=min(2000, len(cal_b_positions)), replace=False
    )
    add("cal_b.shap_sample_positions", shap_positions, "<i8")

    examples = examples_data(int(meta_data()["app_limits"]["historical_example_count"]))
    metadata = {
        "format_version": 1,
        "source": "notebooks/02_training.ipynb frozen artifacts and processed splits",
        "feature_columns": service.feature_columns,
        "target": TARGET,
        "rows": {"train": len(train), "early_stop": len(early_stop), "cal": len(calibration),
                 "cal_b": len(cal_b), "test": len(test)},
        "default_rates": {"train": float(train[TARGET].mean()),
                          "early_stop": float(early_stop[TARGET].mean()),
                          "cal": float(calibration[TARGET].mean()),
                          "cal_b": float(cal_b[TARGET].mean()),
                          "test": float(test[TARGET].mean())},
        "training_apr": apr,
        "purpose_categories": purpose_values,
        "home_categories": home_values,
        "examples": examples,
        "meta": meta_data(),
        "sections": sections,
    }
    header = json.dumps(metadata, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    raw_bundle = b"CVBD0001" + struct.pack("<I", len(header)) + header + b"".join(payload_parts)
    bundle_path = OUTPUT_DIR / "notebook-data.bin.gz"
    with bundle_path.open("wb") as file:
        file.write(gzip.compress(raw_bundle, compresslevel=9, mtime=0))

    files = {}
    for path in (OUTPUT_DIR / "model.json", bundle_path):
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path = OUTPUT_DIR / "manifest.json"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump({"source": metadata["source"], "files": files}, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"Wrote {bundle_path.relative_to(ROOT)} ({bundle_path.stat().st_size / 1048576:.2f} MiB compressed)")
    print(f"Actual notebook rows: cal_B={len(cal_b):,}; OOT={len(test):,}; model features={len(service.feature_columns)}")


if __name__ == "__main__":
    main()
