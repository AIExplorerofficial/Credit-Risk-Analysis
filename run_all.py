"""Read-only verification of the notebook artifacts; this command never trains models."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def verify_notebook_runtime() -> None:
    root = Path(__file__).resolve().parent
    runtime = root / "notebook_runtime"
    manifest = json.loads((runtime / "artifact_checksums.json").read_text(encoding="utf-8"))
    mismatches = []
    for relative, expected in manifest.items():
        path = runtime / relative
        if not path.is_file():
            mismatches.append(f"missing: {relative}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            mismatches.append(f"checksum mismatch: {relative}")
    if mismatches:
        raise SystemExit("Notebook runtime verification failed:\n" + "\n".join(mismatches))
    metadata = json.loads((runtime / "models" / "metadata.json").read_text(encoding="utf-8"))
    print("Notebook runtime verified (read-only).")
    print(f"Threshold: {metadata['threshold']:.2f}; OOT conformal coverage: {metadata['report_2015']['coverage']:.4f}; "
          f"features: {len(metadata['feature_columns'])}.")
    print("No model was trained or rewritten.")


if __name__ == "__main__":
    verify_notebook_runtime()
