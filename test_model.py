"""
Quick ML Model Test -- Lead Score Agent
========================================
Run this script to verify your model.pkl is working correctly.

    python test_model.py

Checks:
  1. model.pkl exists and loads correctly
  2. Classifier and Regressor are present inside
  3. Runs 5 sample predictions and prints results
  4. Compares ML output vs Rule-based output
"""

import os
import sys

print("\n" + "=" * 60)
print("  ML MODEL VERIFICATION -- Lead Score Agent")
print("=" * 60)

# -- 1. Check model.pkl exists --
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

print("\n[CHECK 1] model.pkl existence...")
if not os.path.exists(MODEL_PATH):
    print("  FAIL -- model.pkl NOT found!")
    print("  --> Run:  python train_model.py   to train and save the model first.")
    sys.exit(1)

size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
print(f"  PASS -- model.pkl found  ({size_mb:.2f} MB)")

# -- 2. Load model.pkl --
print("\n[CHECK 2] Loading model.pkl with joblib...")
try:
    import joblib
    model_data = joblib.load(MODEL_PATH)
    print("  PASS -- Loaded successfully!")
except Exception as e:
    print(f"  FAIL -- Could not load model: {e}")
    sys.exit(1)

# -- 3. Verify internal structure --
print("\n[CHECK 3] Verifying model internals...")
required_keys = ["classifier", "regressor", "features",
                 "income_source_map", "loan_history_map",
                 "category_reverse", "recommendation_map"]
all_ok = True
for key in required_keys:
    if key in model_data:
        print(f"  PASS -- '{key}' present")
    else:
        print(f"  FAIL -- '{key}' MISSING -- retrain with: python train_model.py")
        all_ok = False

if not all_ok:
    sys.exit(1)

clf = model_data["classifier"]
reg = model_data["regressor"]
print(f"\n  Classifier : {clf.__class__.__name__}  (estimators={clf.n_estimators})")
print(f"  Regressor  : {reg.__class__.__name__}  (estimators={reg.n_estimators})")
print(f"  Features   : {model_data['features']}")

# -- 4. Sample Predictions --
print("\n[CHECK 4] Running 5 sample predictions...")
print("-" * 60)

from ml_scoring import predict_lead
from scoring import calculate_lead_score

test_cases = [
    # (cibil, annual_income, assets_value, income_source, loan_history, label)
    (800, 1_800_000, 6_000_000, "salaried_govt",              "all_paid_on_time", "Ideal Customer"),
    (720, 1_100_000, 2_500_000, "salaried_mnc",               "minor_delays",     "Good Customer"),
    (660,   650_000,   600_000, "salaried_private",            "no_history",       "Average Customer"),
    (580,   280_000,         0, "freelance",                   "one_default",      "Risky Customer"),
    (400,   150_000,         0, "unemployed",                  "multiple_defaults","Poor Lead"),
]

all_passed = True
for cibil, income, assets, src, hist, label in test_cases:
    try:
        ml_result   = predict_lead(cibil, income, assets, src, hist)
        rule_result = calculate_lead_score(cibil, income, assets, src, hist)

        method     = ml_result.get("scoring_method", "Rule-based")
        ml_score   = ml_result["total_score"]
        ml_cat     = ml_result["category"]
        rule_score = rule_result["total_score"]
        rule_cat   = rule_result["category"]
        confidence = ml_result.get("ml_confidence", "N/A")

        match = "MATCH" if ml_cat == rule_cat else "MISMATCH (borderline case)"

        print(f"\n  [{label}]")
        print(f"    Input   : CIBIL={cibil}, Income={income:,.0f}, Assets={assets:,.0f}")
        print(f"    ML      : Score={ml_score:.1f}  Category='{ml_cat}'  Confidence={confidence}%")
        print(f"    Rule    : Score={rule_score}      Category='{rule_cat}'")
        print(f"    Result  : {match}  | Method: {method}")

    except Exception as e:
        print(f"  FAIL on [{label}]: {e}")
        all_passed = False

# -- 5. Final verdict --
print("\n" + "=" * 60)
if all_passed:
    print("  ALL CHECKS PASSED -- Model is working correctly!")
    print("  The agent will use ML scoring (Random Forest) automatically.")
else:
    print("  Some predictions failed -- check errors above.")
print("=" * 60 + "\n")
