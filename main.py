"""
Lead Score Agent - FastAPI backend.

Run with:
    uvicorn main:app --reload --port 8000

This implements the workflow:
New Lead -> Collect Data -> Validate -> Score Each Parameter ->
Calculate Final Score -> Classify -> Recommend -> Store Result
"""

from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from models import LeadInput, LeadResult
from scoring import calculate_lead_score
import storage

app = FastAPI(
    title="Lead Score Agent",
    description="Scores incoming loan leads (CIBIL, Income, Assets, Income Source, Loan History) "
    "and classifies them into Excellent / Good / Average / Poor leads.",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "agent": "Lead Score Agent",
        "status": "running",
        "endpoints": {
            "POST /score": "Score a new lead",
            "GET /leads": "List all scored leads",
            "GET /leads/{record_id}": "Get a specific scored lead",
        },
    }


@app.post("/score", response_model=LeadResult)
def score_lead(lead: LeadInput):
    """
    Collect Customer Data -> Validate -> Score -> Classify -> Recommend -> Store
    """
    # --- Validate Information step ---
    if lead.annual_income <= 0:
        raise HTTPException(status_code=400, detail="Invalid income: must be greater than 0")

    # --- Score Each Parameter + Calculate Final Lead Score ---
    scoring_output = calculate_lead_score(
        cibil_score=lead.cibil_score,
        annual_income=lead.annual_income,
        assets_value=lead.assets_value,
        income_source=lead.income_source,
        previous_loan_history=lead.previous_loan_history,
    )

    result = {
        "lead_id": lead.lead_id,
        "name": lead.name,
        "source": lead.source,
        "total_score": scoring_output["total_score"],
        "category": scoring_output["category"],
        "recommendation": scoring_output["recommendation"],
        "breakdown": scoring_output["breakdown"],
        "reason_codes": scoring_output["reason_codes"],
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    # --- Store Result step ---
    stored = storage.save_result(result)
    return stored


@app.get("/leads")
def list_leads():
    return storage.get_all()


@app.get("/leads/{record_id}")
def get_lead(record_id: int):
    row = storage.get_by_id(record_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Lead record not found")
    return row