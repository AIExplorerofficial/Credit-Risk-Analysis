"""Small static host for the browser-computed CreditView application.

This deployment entry point deliberately does not import the model service,
analytics stack, notebook data processor, or SHAP. The browser worker loads the
frozen model and notebook-derived runtime files from /runtime.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

FRONTEND = Path(__file__).resolve().parent / "frontend"
app = FastAPI(title="CreditView static host", docs_url=None, redoc_url=None, openapi_url=None)


@app.get("/health", include_in_schema=False)
def health() -> dict[str, str]:
    return {"status": "healthy", "project": "CreditView", "compute": "browser"}


@app.get("/api/health", include_in_schema=False)
def api_health() -> dict[str, str]:
    return health()


app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
