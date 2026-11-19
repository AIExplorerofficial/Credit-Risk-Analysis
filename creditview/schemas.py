"""Pydantic v2 request and response schemas for credit decisioning, what-if, and health checks."""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, ConfigDict, Field, model_validator


class LoanApplication(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="ignore")

    loan_amnt: float = Field(..., gt=0.0, description="Requested principal in USD; any positive amount is accepted")
    annual_inc: float = Field(..., ge=0.0, description="Self-reported annual income in USD")
    dti: float = Field(..., ge=0.0, description="Debt-to-income ratio (%)")
    fico_avg: float = Field(700.0, ge=300.0, le=850.0, description="Credit score (FICO average)")
    emp_length: float = Field(0.0, ge=0.0, le=50.0, description="Employment tenure in years")
    home_ownership: str = Field("RENT", description="Housing tenure category")
    purpose: str = Field("debt_consolidation", description="Stated loan purpose category")
    open_acc: int = Field(5, ge=0, description="Open credit lines count")
    pub_rec: int = Field(0, ge=0, description="Derogatory public records count")
    revol_bal: float = Field(5000.0, ge=0.0, description="Total revolving balance in USD")
    revol_util: float = Field(30.0, ge=0.0, description="Revolving line utilization rate (%)")
    total_acc: int = Field(10, ge=0, description="Total historical credit accounts")
    delinq_2yrs: int = Field(0, ge=0, description="30+ days delinquencies in past 2 years")
    inq_last_6mths: int = Field(0, ge=0, description="Inquiries in past 6 months")
    credit_history_years: float = Field(5.0, ge=0.0, description="Years since earliest credit line")

    # Audited & dropped features: optional and ignored during inference
    mort_acc: int | None = Field(default=None, description="Audited: dropped due to monotone year missingness")
    application_type: str | None = Field(default=None, description="Audited: dropped due to zero variance")
    addr_state: str | None = Field(default=None, description="Fairness monitoring only; excluded from model")

    @model_validator(mode="before")
    @classmethod
    def reconcile_fico(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if ("fico_avg" not in data or data["fico_avg"] is None) and "fico_range_low" in data and "fico_range_high" in data:
                data["fico_avg"] = (float(data["fico_range_low"]) + float(data["fico_range_high"])) / 2.0
        return data


class ReasonFactor(BaseModel):
    feature: str
    impact: str  # "adverse" | "supportive"
    shap_value: float
    raw_value: Any = None
    explanation: str


class RecourseDetails(BaseModel):
    feasible: bool
    target_decision: str = "Approve"
    message: str
    changes: dict[str, Any] | None = None
    new_p_default: float | None = None
    disclaimer: str = "Model-based counterfactual scenario. Does not guarantee future loan approval."


class DecisionResponse(BaseModel):
    decision: str = Field(..., description="Compatibility field mirroring formal_status")
    formal_status: str = Field(..., description="Formal application status; manual review applies outside model support")
    indicative_recommendation: str = Field(..., description="Saved threshold/conformal outcome, labelled indicative when support is insufficient")
    indicative_reason: str | None = Field(default=None, description="Conformal rationale for an indicative Refer outcome")
    formal_explanation: str = Field(..., description="Explanation for the formal application status")
    p_default: float = Field(..., description="Saved-model calibrated estimate; provisional when outside validated support")
    refer_reason: str | None = Field(default=None, description="Conformal referral or outside-support manual-review rationale")
    approve_below: float = Field(..., description="Lower edge of the referral band around the saved threshold")
    reject_above: float = Field(..., description="Upper edge of the referral band around the saved threshold")
    decision_threshold: float = Field(..., description="Saved notebook probability threshold used by the decision rule")
    band_halfwidth: float = Field(..., description="Notebook referral band half-width around the saved threshold")
    is_out_of_support: bool = Field(default=False)
    is_provisional: bool = Field(default=False, description="True when inputs extrapolate beyond observed training data")
    support_warnings: list[str] = Field(default_factory=list, description="Training-data support notes, separate from the decision")
    top_factors: list[ReasonFactor] = Field(default_factory=list)
    explanation_verified: bool = Field(default=False)
    full_explanation: str
    recourse: RecourseDetails | None = None
    model_version: str = "1.0.0"


class WhatIfRequest(BaseModel):
    original_application: LoanApplication
    modified_application: LoanApplication


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="Applicant question; treated as untrusted text")
    application: LoanApplication | None = None

    model_config = ConfigDict(json_schema_extra={"examples": [{"message": "Why was I referred?", "application": None}]})


class WhatIfResponse(BaseModel):
    original: DecisionResponse
    modified: DecisionResponse
    delta_p_default: float
    decision_changed: bool
    indicative_recommendation_changed: bool
    key_drivers: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str = "healthy"
    project: str = "CreditView"
    version: str = "1.0.0"
    model_loaded: bool = True
