"""
Scoring engine for the Lead Score Agent.
Implements the parameter-wise scoring described in the workflow:

    CIBIL Score        -> 35 Marks
    Annual Income       -> 25 Marks
    Assets Value        -> 15 Marks
    Income Source       -> 10 Marks
    Previous Loan Hist. -> 15 Marks
    -----------------------------
    Total               = 100
"""

from typing import Tuple, List


def score_cibil(cibil_score: int) -> Tuple[float, str]:
    """Max 35 marks"""
    if cibil_score >= 750:
        return 35, "CIBIL >= 750 (Excellent)"
    elif cibil_score >= 700:
        return 28, "CIBIL 700-749 (Good)"
    elif cibil_score >= 650:
        return 20, "CIBIL 650-699 (Fair)"
    elif cibil_score >= 600:
        return 10, "CIBIL 600-649 (Weak)"
    else:
        return 0, "CIBIL < 600 (Poor)"


def score_income(annual_income: float) -> Tuple[float, str]:
    """Max 25 marks"""
    if annual_income >= 1_500_000:
        return 25, "Income >= 15L (Excellent)"
    elif annual_income >= 1_000_000:
        return 20, "Income 10L-15L (Good)"
    elif annual_income >= 600_000:
        return 14, "Income 6L-10L (Fair)"
    elif annual_income >= 300_000:
        return 7, "Income 3L-6L (Weak)"
    else:
        return 0, "Income < 3L (Poor)"  
def score_assets(assets_value: float) -> Tuple[float, str]:
    """Max 15 marks"""
    if assets_value >= 5_000_000:
        return 15, "Assets >= 50L (Excellent)"
    elif assets_value >= 2_000_000:
        return 11, "Assets 20L-50L (Good)"
    elif assets_value >= 500_000:
        return 7, "Assets 5L-20L (Fair)"
    elif assets_value > 0:
        return 3, "Assets < 5L (Weak)"
    else:
        return 0, "No declared assets"


def score_income_source(income_source: str) -> Tuple[float, str]:
    """Max 10 marks"""
    src = income_source.strip().lower()
    mapping = {
        "salaried_govt": (10, "Salaried - Government (Most Stable)"),
        "salaried_mnc": (9, "Salaried - MNC/Large Corp"),
        "salaried_private": (7, "Salaried - Private Company"),
        "self_employed_professional": (6, "Self-Employed Professional"),
        "business_owner": (5, "Business Owner"),
        "freelance": (3, "Freelance / Gig Income"),
        "unemployed": (0, "No Stable Income Source"),
    }
    if src in mapping:
        return mapping[src]
    return 4, f"Unrecognized income source '{income_source}' (default mid score)"


def score_previous_loans(previous_loan_history: str) -> Tuple[float, str]:
    """Max 15 marks"""
    hist = previous_loan_history.strip().lower()
    mapping = {
        "no_history": (8, "No prior loan history (neutral)"),
        "all_paid_on_time": (15, "All previous loans paid on time"),
        "minor_delays": (10, "Minor payment delays in history"),
        "one_default": (4, "One past default"),
        "multiple_defaults": (0, "Multiple past defaults"),
    }
    if hist in mapping:
        return mapping[hist]
    return 6, f"Unrecognized loan history '{previous_loan_history}' (default mid score)"


def classify(total_score: float) -> Tuple[str, str]:
    """Returns (category, recommendation)"""
    if total_score >= 80:
        return "Excellent Lead", "Approve for Sales Follow-up"
    elif total_score >= 60:
        return "Good Lead", "Manual Review"
    elif total_score >= 40:
        return "Average Lead", "Low Priority"
    else:
        return "Poor Lead", "Reject"


def calculate_lead_score(
    cibil_score: int,
    annual_income: float,
    assets_value: float,
    income_source: str,
    previous_loan_history: str,
) -> dict:
    cibil_marks, cibil_reason = score_cibil(cibil_score)
    income_marks, income_reason = score_income(annual_income)
    assets_marks, assets_reason = score_assets(assets_value)
    source_marks, source_reason = score_income_source(income_source)
    loan_marks, loan_reason = score_previous_loans(previous_loan_history)

    total = cibil_marks + income_marks + assets_marks + source_marks + loan_marks
    category, recommendation = classify(total)

    reason_codes: List[str] = [
        cibil_reason,
        income_reason,
        assets_reason,
        source_reason,
        loan_reason,
    ]

    breakdown = {
        "cibil_score_marks": cibil_marks,
        "annual_income_marks": income_marks,
        "assets_value_marks": assets_marks,
        "income_source_marks": source_marks,
        "previous_loans_marks": loan_marks,
    }

    return {
        "total_score": total,
        "category": category,
        "recommendation": recommendation,
        "breakdown": breakdown,
        "reason_codes": reason_codes,
    }