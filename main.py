"""
Lead Score Agent - FastAPI backend.

Run with:
    uvicorn main:app --reload --port 8000

Scoring priority:
    1. Random Forest ML model (model.pkl) — if trained
    2. Rule-based scoring (scoring.py)    — fallback

Train the ML model with:
    python train_model.py
"""

from datetime import datetime
from fastapi import FastAPI, HTTPException

from models import LeadInput, LeadResult
import storage

app = FastAPI(
    title="Lead Score Agent",
    description="Scores loan leads using Random Forest ML or rule-based scoring.",
    version="2.0.0",
)


def get_scorer():
    """Return ML scorer if model exists, else rule-based."""
    try:
        from ml_scoring import predict_lead, model_exists
        if model_exists():
            return predict_lead, "ml"
    except ImportError:
        pass
    from scoring import calculate_lead_score
    return calculate_lead_score, "rules"


@app.get("/")
def root():
    _, method = get_scorer()
    return {
        "agent":          "Lead Score Agent",
        "status":         "running",
        "scoring_method": "Random Forest ML" if method == "ml" else "Rule-based",
        "endpoints": {
            "POST /score":             "Score a new lead",
            "GET /leads":              "List all scored leads",
            "GET /leads/{record_id}":  "Get a specific lead",
            "GET /model/status":       "Check ML model status",
        },
    }


@app.get("/model/status")
def model_status():
    """Check whether the ML model is trained and active."""
    try:
        from ml_scoring import model_exists
        exists = model_exists()
    except ImportError:
        exists = False
    return {
        "ml_model_trained": exists,
        "scoring_method":   "Random Forest ML" if exists else "Rule-based",
        "model_file":       "model.pkl",
        "train_command":    "python train_model.py",
    }


@app.post("/score", response_model=LeadResult)
def score_lead(lead: LeadInput):
    """Score a lead using ML model (or rule-based fallback)."""
    if lead.annual_income <= 0:
        raise HTTPException(status_code=400, detail="Income must be greater than 0")

    scorer, method = get_scorer()
    scoring_output = scorer(
        cibil_score=lead.cibil_score,
        annual_income=lead.annual_income,
        assets_value=lead.assets_value,
        income_source=lead.income_source,
        previous_loan_history=lead.previous_loan_history,
    )

    result = {
        "lead_id":        lead.lead_id,
        "name":           lead.name,
        "source":         lead.source,
        "total_score":    scoring_output["total_score"],
        "category":       scoring_output["category"],
        "recommendation": scoring_output["recommendation"],
        "breakdown":      scoring_output["breakdown"],
        "reason_codes":   scoring_output["reason_codes"],
        "timestamp":      datetime.utcnow().isoformat() + "Z",
        "scoring_method": scoring_output.get("scoring_method",
                          "Random Forest ML" if method == "ml" else "Rule-based"),
    }

    # Add ML-specific fields if available
    if "ml_confidence" in scoring_output:
        result["ml_confidence"]  = scoring_output["ml_confidence"]
        result["category_probs"] = scoring_output["category_probs"]

    return storage.save_result(result)


@app.get("/leads")
def list_leads():
    return storage.get_all()


@app.get("/leads/{record_id}")
def get_lead(record_id: int):
    row = storage.get_by_id(record_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Lead not found")
    return row