"""
Model Inspector -- Open & Read model.pkl
==========================================
Run:  python inspect_model.py
"""
import joblib, os
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

print("\n" + "=" * 55)
print("  model.pkl INSPECTOR")
print("=" * 55)

# ── Load ──────────────────────────────────────────────────────
m = joblib.load(MODEL_PATH)
size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
print(f"\n  File size : {size_mb:.2f} MB")
print(f"  Keys      : {list(m.keys())}")

# ── Classifier Info ───────────────────────────────────────────
clf = m["classifier"]
print("\n--- CLASSIFIER (predicts category) ---")
print(f"  Type            : {clf.__class__.__name__}")
print(f"  Number of trees : {clf.n_estimators}")
print(f"  Max tree depth  : {clf.max_depth}")
print(f"  Classes         : {clf.classes_}  --> {[m['category_reverse'][c] for c in clf.classes_]}")
print(f"  Trained on      : {clf.n_features_in_} features")

# ── Regressor Info ────────────────────────────────────────────
reg = m["regressor"]
print("\n--- REGRESSOR (predicts score 0-100) ---")
print(f"  Type            : {reg.__class__.__name__}")
print(f"  Number of trees : {reg.n_estimators}")
print(f"  Max tree depth  : {reg.max_depth}")

# ── Feature Importance ────────────────────────────────────────
print("\n--- FEATURE IMPORTANCE (which inputs matter most) ---")
features = m["features"]
importances = clf.feature_importances_
for feat, imp in sorted(zip(features, importances), key=lambda x: -x[1]):
    bar = "#" * int(imp * 40)
    print(f"  {feat:<25}  {imp:.3f}  {bar}")

# ── Encoding Maps ─────────────────────────────────────────────
print("\n--- INCOME SOURCE MAP ---")
for k, v in m["income_source_map"].items():
    print(f"  {v} = {k}")

print("\n--- LOAN HISTORY MAP ---")
for k, v in m["loan_history_map"].items():
    print(f"  {v} = {k}")

print("\n--- CATEGORY MAP ---")
for k, v in m["category_map"].items():
    print(f"  {v} --> {k}  ({m['recommendation_map'][k]})")

print("\n" + "=" * 55 + "\n")
