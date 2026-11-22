"""Regenerate the static, browser-served notebook runtime before deployment.

Run this locally after replacing or regenerating notebook artifacts. It exports
the frozen model, actual processed observations, and notebook governance output;
Render only serves these files and does no Python/model work.
"""
from __future__ import annotations

import hashlib
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.encoders import jsonable_encoder

from scripts.export_browser_model import export as export_model
from scripts.export_browser_dataset import main as export_dataset


def main() -> None:
    export_model()
    export_dataset()

    from backend.analytics import governance_data

    output = ROOT / "frontend" / "runtime"
    governance = jsonable_encoder(governance_data())
    governance_path = output / "governance.json.gz"
    encoded = json.dumps(governance, ensure_ascii=False, allow_nan=False,
                         separators=(",", ":")).encode("utf-8") + b"\n"
    governance_path.write_bytes(gzip.compress(encoded, compresslevel=9, mtime=0))
    stale_json = output / "governance.json"
    if stale_json.exists():
        stale_json.unlink()
    files = {}
    for path in sorted(p for p in output.iterdir() if p.is_file() and p.name != "manifest.json"):
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"source": "notebooks/02_training.ipynb frozen artifacts and processed splits",
                "files": files}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                            encoding="utf-8")
    print(f"Exported browser runtime and integrity manifest to {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
