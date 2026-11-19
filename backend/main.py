"""FastAPI application and local frontend serving for CreditView."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from creditview.schemas import LoanApplication, WhatIfRequest
from creditview.notebook_service import CreditDecisionService
from . import analytics

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    CreditDecisionService.get_instance()
    yield


app = FastAPI(title="CreditView", version="1.0", lifespan=lifespan)
app.mount("/assets", StaticFiles(directory=FRONTEND / "assets"), name="assets")


class BoundaryAnalysisRequest(BaseModel):
    approve_below: float = Field(ge=0, le=1)
    reject_above: float = Field(ge=0, le=1)


def _score_batch(applicants: list[dict[str, Any]]) -> dict[str, Any]:
    service = CreditDecisionService.get_instance()
    return service.score_many(applicants)


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(FRONTEND / "index.html")


@app.get("/api/health")
def health():
    service = CreditDecisionService.get_instance()
    return {"status": "healthy", "project": "CreditView",
            "version": service.model_version, "model_loaded": True}


@app.get("/api/meta")
def meta():
    return analytics.meta_data()


@app.get("/api/dashboard")
def dashboard():
    try:
        return analytics.dashboard_data()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Dashboard data is unavailable. Check the processed data and model artifacts.") from exc


@app.get("/api/examples")
def examples(limit: int | None = None):
    limit = 12 if limit is None else limit
    maximum = 50
    if not 1 <= limit <= maximum:
        raise HTTPException(status_code=422, detail=f"limit must be between 1 and {maximum}")
    return {"dataset": "Actual 2014 calibration threshold-selection observations (cal_B)",
            "examples": analytics.examples_data(limit)}


@app.post("/api/decide")
def decide(applicant: LoanApplication):
    try:
        result = CreditDecisionService.get_instance().decide(applicant)
        payload = result.model_dump()
        payload["cohort_comparison"] = analytics.applicant_comparison(applicant.model_dump(), result.p_default)
        return payload
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Assessment could not be completed. Check the inputs and artifact status.") from exc


@app.post("/api/whatif")
def whatif(request: WhatIfRequest):
    try:
        response = CreditDecisionService.get_instance().what_if(request.original_application, request.modified_application)
        payload = response.model_dump()
        return payload
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Scenario comparison could not be completed.") from exc


@app.post("/api/batch")
def batch(applicants: list[LoanApplication]):
    row_limit = 1000
    if not applicants or len(applicants) > row_limit:
        raise HTTPException(status_code=422, detail=f"Submit between 1 and {row_limit:,} applicant rows.")
    try:
        service = CreditDecisionService.get_instance()
        scored = _score_batch([a.model_dump() for a in applicants])
        return {"count": len(applicants), "results": [
            {
                "row": i + 1,
                "p_default": float(scored["probabilities"][i]),
                "decision": str(scored["decisions"][i]),
                "formal_status": str(scored["decisions"][i]),
                "indicative_recommendation": str(scored["indicative_recommendations"][i]),
                "indicative_reason": str(scored["indicative_reasons"][i]) or None,
                "is_provisional": bool(scored["is_provisional"][i]),
                "support_warnings": list(scored["support_warnings"][i]),
                "formal_explanation": str(scored["formal_explanations"][i]),
            }
            for i in range(len(applicants))]}
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Batch scoring could not be completed. Check row fields and values.") from exc


@app.get("/api/portfolio/summary")
def portfolio_summary(purpose: str | None = None, home_ownership: str | None = None,
                      year: int | None = None, fico_min: float | None = None, fico_max: float | None = None):
    return analytics.portfolio_data({"purpose": purpose, "home_ownership": home_ownership, "year": year,
                                     "fico_min": fico_min, "fico_max": fico_max})


@app.get("/api/portfolio/trend")
def portfolio_trend():
    # Monthly series is based on final OOT evaluation and remains read-only.
    dashboard = analytics.dashboard_data()
    return {"dataset": dashboard["dataset"], "trend": dashboard["trend"]}


@app.get("/api/portfolio/cohorts")
def portfolio_cohorts():
    return analytics.portfolio_data()


@app.post("/api/policy/evaluate")
def policy_evaluate(request: BoundaryAnalysisRequest):
    try:
        return analytics.policy_boundary_analysis(request.approve_below, request.reject_above)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/governance")
def governance():
    return analytics.governance_data()


@app.get("/api/calibration")
def calibration():
    return analytics.calibration_data()


@app.get("/api/thresholds")
def thresholds():
    return analytics.threshold_data()


# Keep the explanation endpoint aligned with the same shared inference service.
@app.post("/api/explain")
def explain(applicant: LoanApplication):
    return CreditDecisionService.get_instance().decide(applicant)


@app.get("/health", include_in_schema=False)
def legacy_health():
    return health()

