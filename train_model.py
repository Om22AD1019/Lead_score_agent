"""
Random Forest Model Trainer for Lead Score Agent
=================================================
Run this once to train and save the model:
    python train_model.py

What it does:
1. Generates 2000 synthetic leads using the scoring rules
2. Also loads real leads from leads_store.json (if available)
3. Trains a Random Forest Classifier to predict lead category
4. Trains a Random Forest Regressor to predict score (0-100)
5. Saves both models to model.pkl
6. Prints accuracy and feature importance report
"""

import json
import os
import random
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_absolute_error

# ── Encoding Maps ──────────────────────────────────────────────
# These must match exactly what ml_scoring.py uses

INCOME_SOURCE_MAP = {
    "salaried_govt":             0,
    "salaried_mnc":              1,
    "salaried_private":          2,
    "self_employed_professional":3,
    "business_owner":            4,
    "freelance":                 5,
    "unemployed":                6,
}

LOAN_HISTORY_MAP = {
    "no_history":        0,
    "all_paid_on_time":  1,
    "minor_delays":      2,
    "one_default":       3,
    "multiple_defaults": 4,
}

CATEGORY_MAP = {
    "Poor Lead":      0,
    "Average Lead":   1,
    "Good Lead":      2,
    "Excellent Lead": 3,
}

CATEGORY_REVERSE = {v: k for k, v in CATEGORY_MAP.items()}

RECOMMENDATION_MAP = {
    "Poor Lead":      "Reject",
    "Average Lead":   "Low Priority",
    "Good Lead":      "Manual Review",
    "Excellent Lead": "Approve for Sales Follow-up",
}


# ══════════════════════════════════════════════════════════════
#  SCORING RULES (same as scoring.py — used to generate labels)
# ══════════════════════════════════════════════════════════════

def rule_based_score(cibil, income, assets, inc_src, loan_hist):
    """Apply existing rules to generate training labels."""

    # CIBIL (35 marks)
    if cibil >= 750:   cm = 35
    elif cibil >= 700: cm = 28
    elif cibil >= 650: cm = 20
    elif cibil >= 600: cm = 10
    else:              cm = 0

    # Income (25 marks)
    if income >= 1_500_000:   im = 25
    elif income >= 1_000_000: im = 20
    elif income >= 600_000:   im = 14
    elif income >= 300_000:   im = 7
    else:                     im = 0

    # Assets (15 marks)
    if assets >= 5_000_000:   am = 15
    elif assets >= 2_000_000: am = 11
    elif assets >= 500_000:   am = 7
    elif assets > 0:          am = 3
    else:                     am = 0

    # Income source (10 marks)
    src_marks = {
        "salaried_govt": 10, "salaried_mnc": 9, "salaried_private": 7,
        "self_employed_professional": 6, "business_owner": 5,
        "freelance": 3, "unemployed": 0,
    }
    sm = src_marks.get(inc_src, 4)

    # Loan history (15 marks)
    loan_marks = {
        "all_paid_on_time": 15, "minor_delays": 10,
        "no_history": 8, "one_default": 4, "multiple_defaults": 0,
    }
    lm = loan_marks.get(loan_hist, 6)

    total = cm + im + am + sm + lm

    if total >= 80:   cat = "Excellent Lead"
    elif total >= 60: cat = "Good Lead"
    elif total >= 40: cat = "Average Lead"
    else:             cat = "Poor Lead"

    return total, cat


# ══════════════════════════════════════════════════════════════
#  SYNTHETIC DATA GENERATOR
# ══════════════════════════════════════════════════════════════

def generate_synthetic_data(n=2000):
    """
    Generate n synthetic leads by randomly sampling parameter ranges.
    Labels are computed using the rule-based scoring so the model
    learns the same patterns — but it will generalise beyond the rules.
    """
    random.seed(42)
    np.random.seed(42)

    income_sources  = list(INCOME_SOURCE_MAP.keys())
    loan_histories  = list(LOAN_HISTORY_MAP.keys())

    rows = []
    for _ in range(n):
        cibil  = random.randint(300, 900)
        income = random.choice([
            random.uniform(100_000,  300_000),
            random.uniform(300_000,  600_000),
            random.uniform(600_000,  1_000_000),
            random.uniform(1_000_000, 1_500_000),
            random.uniform(1_500_000, 3_000_000),
        ])
        assets = random.choice([
            0,
            random.uniform(50_000,   500_000),
            random.uniform(500_000,  2_000_000),
            random.uniform(2_000_000,5_000_000),
            random.uniform(5_000_000,15_000_000),
        ])
        inc_src   = random.choice(income_sources)
        loan_hist = random.choice(loan_histories)

        score, cat = rule_based_score(cibil, income, assets, inc_src, loan_hist)

        rows.append({
            "cibil_score":           cibil,
            "annual_income":         income,
            "assets_value":          assets,
            "income_source_enc":     INCOME_SOURCE_MAP[inc_src],
            "loan_history_enc":      LOAN_HISTORY_MAP[loan_hist],
            "total_score":           score,
            "category_enc":          CATEGORY_MAP[cat],
        })

    return rows


# ══════════════════════════════════════════════════════════════
#  LOAD REAL LEADS FROM leads_store.json
# ══════════════════════════════════════════════════════════════

def load_real_leads():
    path = os.path.join(os.path.dirname(__file__), "leads_store.json")
    if not os.path.exists(path):
        print("  leads_store.json not found — using synthetic data only.")
        return []

    with open(path) as f:
        data = json.load(f)

    rows = []
    for l in data:
        inc_src   = l.get("income_source", "")
        loan_hist = l.get("previous_loan_history", "")
        cat       = l.get("category", "")

        if inc_src not in INCOME_SOURCE_MAP: continue
        if loan_hist not in LOAN_HISTORY_MAP: continue
        if cat not in CATEGORY_MAP: continue

        rows.append({
            "cibil_score":       l.get("cibil_score", 600),
            "annual_income":     l.get("annual_income", 500000),
            "assets_value":      l.get("assets_value", 0),
            "income_source_enc": INCOME_SOURCE_MAP[inc_src],
            "loan_history_enc":  LOAN_HISTORY_MAP[loan_hist],
            "total_score":       l.get("total_score", 50),
            "category_enc":      CATEGORY_MAP[cat],
        })

    print(f"  Loaded {len(rows)} real leads from leads_store.json")
    return rows


# ══════════════════════════════════════════════════════════════
#  TRAIN
# ══════════════════════════════════════════════════════════════

def train():
    print("\n" + "=" * 55)
    print("  RANDOM FOREST TRAINER — Lead Score Agent")
    print("=" * 55)

    # 1. Build dataset
    print("\n  [1/5] Generating synthetic training data (2000 records)...")
    synthetic = generate_synthetic_data(2000)

    print("  [2/5] Loading real leads...")
    real = load_real_leads()

    all_rows = synthetic + real
    print(f"  Total training records: {len(all_rows)}")

    # 2. Prepare features & labels
    features = ["cibil_score", "annual_income", "assets_value",
                "income_source_enc", "loan_history_enc"]

    X = np.array([[r[f] for f in features] for r in all_rows])
    y_cat   = np.array([r["category_enc"]  for r in all_rows])
    y_score = np.array([r["total_score"]   for r in all_rows])

    # 3. Train/test split
    X_train, X_test, yc_train, yc_test, ys_train, ys_test = train_test_split(
        X, y_cat, y_score, test_size=0.2, random_state=42, stratify=y_cat
    )

    # 4. Train Random Forest Classifier (predicts category)
    print("\n  [3/5] Training Random Forest Classifier...")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, yc_train)
    yc_pred = clf.predict(X_test)

    acc = (yc_pred == yc_test).mean() * 100
    print(f"  Classifier Accuracy: {acc:.1f}%")
    print("\n  Classification Report:")
    target_names = ["Poor Lead", "Average Lead", "Good Lead", "Excellent Lead"]
    print(classification_report(yc_test, yc_pred, target_names=target_names))

    # 5. Train Random Forest Regressor (predicts score 0-100)
    print("  [4/5] Training Random Forest Regressor (score predictor)...")
    reg = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    reg.fit(X_train, ys_train)
    ys_pred = reg.predict(X_test)
    mae = mean_absolute_error(ys_test, ys_pred)
    print(f"  Score MAE: {mae:.2f} points (avg error on 0-100 scale)")

    # 6. Feature importance
    print("\n  Feature Importance (Classifier):")
    for fname, imp in sorted(
        zip(features, clf.feature_importances_), key=lambda x: -x[1]
    ):
        bar = "█" * int(imp * 40)
        print(f"    {fname:<25} {imp:.3f}  {bar}")

    # 7. Save model
    print("\n  [5/5] Saving model to model.pkl...")
    model_data = {
        "classifier":        clf,
        "regressor":         reg,
        "features":          features,
        "income_source_map": INCOME_SOURCE_MAP,
        "loan_history_map":  LOAN_HISTORY_MAP,
        "category_map":      CATEGORY_MAP,
        "category_reverse":  CATEGORY_REVERSE,
        "recommendation_map":RECOMMENDATION_MAP,
    }
    joblib.dump(model_data, os.path.join(os.path.dirname(__file__), "model.pkl"))

    print("\n  ✔ model.pkl saved successfully!")
    print("  The agent will now use ML scoring automatically.")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    train()
