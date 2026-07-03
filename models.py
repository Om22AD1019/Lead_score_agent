from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict
from datetime import datetime


VALID_INCOME_SOURCES = {
    "salaried_govt",
    "salaried_mnc",
    "salaried_private",
    "self_employed_professional",
    "business_owner",
    "freelance",
    "unemployed",
}

VALID_LOAN_HISTORY = {
    "no_history",
    "all_paid_on_time",
    "minor_delays",
    "one_default",
    "multiple_defaults",
}


class LeadInput(BaseModel):
    lead_id: Optional[str] = Field(None, description="Optional external lead/customer ID")
    name: Optional[str] = Field(None, description="Customer name")
    source: Optional[str] = Field("manual", description="Lead origin e.g. CRM / Website / App")
    cibil_score: int = Field(..., ge=300, le=900, description="CIBIL score (300-900)")
    annual_income: float = Field(..., ge=0, description="Annual income in INR")
    assets_value: float = Field(0, ge=0, description="Total declared assets value in INR")
    income_source: str = Field(..., description=f"One of {sorted(VALID_INCOME_SOURCES)}")
    previous_loan_history: str = Field(..., description=f"One of {sorted(VALID_LOAN_HISTORY)}")
    

    @field_validator("income_source")
    @classmethod
    def validate_income_source(cls, v):
        if v.strip().lower() not in VALID_INCOME_SOURCES:
            raise ValueError(
                f"income_source must be one of {sorted(VALID_INCOME_SOURCES)}, got '{v}'"
            )
        return v.strip().lower()

    @field_validator("previous_loan_history")
    @classmethod
    def validate_loan_history(cls, v):
        if v.strip().lower() not in VALID_LOAN_HISTORY:
            raise ValueError(
                f"previous_loan_history must be one of {sorted(VALID_LOAN_HISTORY)}, got '{v}'"
            )
        return v.strip().lower()


class LeadResult(BaseModel):
    lead_id: Optional[str]
    name: Optional[str]
    source: Optional[str]
    total_score: float
    category: str
    recommendation: str
    breakdown: Dict[str, float]
    reason_codes: List[str]
    timestamp: str