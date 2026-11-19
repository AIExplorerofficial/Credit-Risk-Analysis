"""Run the local CreditView FastAPI application and static frontend."""
from __future__ import annotations

import os

import uvicorn

from backend.main import app


def main() -> None:
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
