"""
ML Scoring Module — Lead Score Agent
=====================================
Loads the trained Random Forest model (model.pkl) and uses it
to predict lead score and category.

Falls back to rule-based scoring (scoring.py) if model.pkl
does not exist yet.

Usage:
    from ml_scoring import predict_lead
    result = predict_lead(cibil_score, annual_income, assets_value,
                          income_source, previous_loan_history)
"""

import os
import joblib
import numpy as np
from typing import Dict

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

# Cache the model in memory so it's only loaded once
_model_cache = None


def _load_model():
    global _model_cache
    if _model_cache is None:
        if not os.path.exists(MODEL_PATH):
            return None
        _model_cache = joblib.load(MODEL_PATH)
    return _model_cache


def _encode_features(model_data, cibil, income, assets, inc_src, loan_hist):
    """Encode raw inputs into the feature vector the model expects."""
    income_source_map = model_data["income_source_map"]
    loan_history_map  = model_data["loan_history_map"]

    inc_enc  = income_source_map.get(inc_src.strip().lower(), 4)
    loan_enc = loan_history_map.get(loan_hist.strip().lower(), 0)

    return np.array([[cibil, income, assets, inc_enc, loan_enc]])


def predict_lead(
    cibil_score: int,
    annual_income: float,
    assets_value: float,
    income_source: str,
    previous_loan_history: str,
) -> Dict:
    """
    Predict lead score and category using Random Forest model.
    Falls back to rule-based scoring if model.pkl is not found.
    """
    model_data = _load_model()

    # ── Fallback to rule-based if model not trained yet ──────
    if model_data is None:
        print("  [ML] model.pkl not found — using rule-based scoring.")
        from scoring import calculate_lead_score
        return calculate_lead_score(
            cibil_score, annual_income, assets_value,
            income_source, previous_loan_history
        )

    clf              = model_data["classifier"]
    reg              = model_data["regressor"]
    category_reverse = model_data["category_reverse"]
    rec_map          = model_data["recommendation_map"]

    # ── Prepare features ──────────────────────────────────────
    X = _encode_features(
        model_data, cibil_score, annual_income, assets_value,
        income_source, previous_loan_history
    )

    # ── Predict ───────────────────────────────────────────────
    cat_enc    = clf.predict(X)[0]
    cat_proba  = clf.predict_proba(X)[0]
    ml_score   = float(np.clip(reg.predict(X)[0], 0, 100))

    category       = category_reverse[cat_enc]
    recommendation = rec_map[category]
    confidence     = round(float(cat_proba[cat_enc]) * 100, 1)

    # ── Confidence scores for all categories ─────────────────
    all_categories = ["Poor Lead", "Average Lead", "Good Lead", "Excellent Lead"]
    category_probs = {
        cat: round(float(prob) * 100, 1)
        for cat, prob in zip(all_categories, cat_proba)
    }

    # ── Build breakdown (ML version) ─────────────────────────
    # Approximate marks using feature importances × max marks
    importances = clf.feature_importances_  # [cibil, income, assets, src, hist]
    max_marks   = [35, 25, 15, 10, 15]
    raw_vals    = [cibil_score, annual_income, assets_value,
                   model_data["income_source_map"].get(income_source, 4),
                   model_data["loan_history_map"].get(previous_loan_history, 0)]

    # Scale feature value to 0-1 range for contribution estimate
    feat_ranges = [(300, 900), (0, 3_000_000), (0, 10_000_000), (0, 6), (0, 4)]
    contributions = []
    for i, (lo, hi) in enumerate(feat_ranges):
        norm = (raw_vals[i] - lo) / (hi - lo) if hi > lo else 0
        norm = float(np.clip(norm, 0, 1))
        contributions.append(round(norm * max_marks[i], 1))

    breakdown = {
        "cibil_score_marks":    contributions[0],
        "annual_income_marks":  contributions[1],
        "assets_value_marks":   contributions[2],
        "income_source_marks":  contributions[3],
        "previous_loans_marks": contributions[4],
    }

    # ── Reason codes ─────────────────────────────────────────
    src_labels = {
        "salaried_govt":              "Salaried - Government (Most Stable)",
        "salaried_mnc":               "Salaried - MNC/Large Corp",
        "salaried_private":           "Salaried - Private Company",
        "self_employed_professional": "Self-Employed Professional",
        "business_owner":             "Business Owner",
        "freelance":                  "Freelance / Gig Income",
        "unemployed":                 "No Stable Income Source",
    }
    loan_labels = {
        "all_paid_on_time":  "All previous loans paid on time",
        "no_history":        "No prior loan history (neutral)",
        "minor_delays":      "Minor payment delays in history",
        "one_default":       "One past default",
        "multiple_defaults": "Multiple past defaults",
    }

    if cibil_score >= 750:   cibil_label = "CIBIL >= 750 (Excellent)"
    elif cibil_score >= 700: cibil_label = "CIBIL 700-749 (Good)"
    elif cibil_score >= 650: cibil_label = "CIBIL 650-699 (Fair)"
    elif cibil_score >= 600: cibil_label = "CIBIL 600-649 (Weak)"
    else:                    cibil_label = "CIBIL < 600 (Poor)"

    if annual_income >= 1_500_000:   inc_label = "Income >= 15L (Excellent)"
    elif annual_income >= 1_000_000: inc_label = "Income 10L-15L (Good)"
    elif annual_income >= 600_000:   inc_label = "Income 6L-10L (Fair)"
    elif annual_income >= 300_000:   inc_label = "Income 3L-6L (Weak)"
    else:                            inc_label = "Income < 3L (Poor)"

    if assets_value >= 5_000_000:   ast_label = "Assets >= 50L (Excellent)"
    elif assets_value >= 2_000_000: ast_label = "Assets 20L-50L (Good)"
    elif assets_value >= 500_000:   ast_label = "Assets 5L-20L (Fair)"
    elif assets_value > 0:          ast_label = "Assets < 5L (Weak)"
    else:                           ast_label = "No declared assets"

    reason_codes = [
        cibil_label,
        inc_label,
        ast_label,
        src_labels.get(income_source, income_source),
        loan_labels.get(previous_loan_history, previous_loan_history),
    ]

    return {
        "total_score":      round(ml_score, 1),
        "category":         category,
        "recommendation":   recommendation,
        "breakdown":        breakdown,
        "reason_codes":     reason_codes,
        "ml_confidence":    confidence,
        "category_probs":   category_probs,
        "scoring_method":   "Random Forest ML Model",
    }


def model_exists() -> bool:
    return os.path.exists(MODEL_PATH)
