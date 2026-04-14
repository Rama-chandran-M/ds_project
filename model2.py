"""
Churn Prediction Pipeline — sample_04.csv
Handles: duplicates, data leakage, missing values, class imbalance,
         encoding, train/test split, model training & evaluation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────
df = pd.read_csv("sample_04.csv")
print(f"Loaded: {df.shape[0]} rows x {df.shape[1]} columns")

# ─────────────────────────────────────────────
# 2. FIX: EXACT DUPLICATES  (critical — 94.1%)
# ─────────────────────────────────────────────
before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"After dedup: {len(df)} rows  (removed {before - len(df)} duplicates)")

# ─────────────────────────────────────────────
# 3. TARGET
# ─────────────────────────────────────────────
df["target"] = (df["Prospect_Outcome"] == "Churned").astype(int)
df = df.drop(columns=["Prospect_Outcome"])

# ─────────────────────────────────────────────
# 4. LEAKAGE: drop Status_Scores
#    (Score=0 → 69.5% churn — almost certainly a post-outcome metric)
#    Comment out the line below if you are CERTAIN it is available
#    at prediction time and truly reflects a pre-outcome signal.
# ─────────────────────────────────────────────
LEAKAGE_COLS = ["Status_Scores"]
df = df.drop(columns=LEAKAGE_COLS)
print(f"Dropped leakage columns: {LEAKAGE_COLS}")

# ─────────────────────────────────────────────
# 5. DROP NEAR-EMPTY COLUMNS (>80% missing)
# ─────────────────────────────────────────────
threshold = 0.80
missing_frac = df.isnull().mean()
drop_cols = missing_frac[missing_frac > threshold].index.tolist()
df = df.drop(columns=drop_cols)
print(f"Dropped near-empty columns (>{threshold*100:.0f}% missing): {drop_cols}")

# ─────────────────────────────────────────────
# 6. NORMALISE SENTINEL VALUES
#    Unify 'unknown' / 'Unknown' / 'Not Discussed' → 'Unknown'
# ─────────────────────────────────────────────
str_cols = df.select_dtypes(include="object").columns.tolist()
for col in str_cols:
    df[col] = df[col].str.strip()
    df[col] = df[col].replace(
        {"unknown": "Unknown", "Not Discussed": "Unknown", "not discussed": "Unknown"}
    )

# ─────────────────────────────────────────────
# 7. CAST NUMERIC-LOOKING STRINGS
# ─────────────────────────────────────────────
for col in str_cols:
    converted = pd.to_numeric(df[col], errors="coerce")
    if converted.notna().sum() > 0.5 * df[col].notna().sum():
        df[col] = converted
        print(f"  Cast to numeric: {col}")

# ─────────────────────────────────────────────
# 8. SPLIT FEATURES / TARGET
# ─────────────────────────────────────────────
X = df.drop(columns=["target"])
y = df["target"]
print(f"\nClass distribution:\n{y.value_counts()}")
print(f"Churn rate: {y.mean():.1%}")

# ─────────────────────────────────────────────
# 9. ENCODE CATEGORICALS
#    Use ordinal encoding — XGBoost handles it natively.
#    Missing values in categoricals → kept as NaN (XGBoost handles).
# ─────────────────────────────────────────────
cat_cols = X.select_dtypes(include="object").columns.tolist()
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    # fit on non-null, transform including NaN → -1
    non_null_mask = X[col].notna()
    le.fit(X.loc[non_null_mask, col])
    X[col] = X[col].map(lambda v: le.transform([v])[0] if pd.notna(v) else np.nan)
    encoders[col] = le

print(f"\nEncoded categorical columns: {cat_cols}")

# ─────────────────────────────────────────────
# 10. TRAIN / TEST SPLIT (stratified)
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\nTrain: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")
print(f"Train churn rate: {y_train.mean():.1%} | Test churn rate: {y_test.mean():.1%}")

# ─────────────────────────────────────────────
# 11. HANDLE IMBALANCE
#
#  Option A (default): scale_pos_weight inside XGBoost — no oversampling needed.
#  Option B (SMOTE):   oversampling — works well with RF. Uncomment to use.
#
# ─────────────────────────────────────────────

# --- Option A (XGBoost built-in) ---
neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
scale_pos_weight = neg / pos
print(f"\nscale_pos_weight = {scale_pos_weight:.2f}  (for XGBoost)")

# --- Option B (SMOTE — for RF or if you prefer oversampling) ---
# from imblearn.over_sampling import SMOTE
# smote = SMOTE(random_state=42, k_neighbors=3)
# X_train_res, y_train_res = smote.fit_resample(X_train.fillna(-999), y_train)
# print(f"After SMOTE: {pd.Series(y_train_res).value_counts().to_dict()}")

# ─────────────────────────────────────────────
# 12. TRAIN XGBOOST (primary model)
# ─────────────────────────────────────────────
xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,   # handles imbalance
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
    enable_categorical=False,
)

# XGBoost handles NaN natively — no imputation needed for training
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False,
)
print("\n=== XGBoost trained ===")

# ─────────────────────────────────────────────
# 13. TRAIN RANDOM FOREST (baseline comparison)
# ─────────────────────────────────────────────
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

rf_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("rf", RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight="balanced",           # handles imbalance
        random_state=42,
        n_jobs=-1,
    ))
])
rf_pipeline.fit(X_train, y_train)
print("=== Random Forest trained ===")

# ─────────────────────────────────────────────
# 14. EVALUATE
# ─────────────────────────────────────────────
def evaluate(name, model, X_t, y_t, needs_impute=False):
    if needs_impute:
        X_t = SimpleImputer(strategy="most_frequent").fit_transform(X_t)
    preds = model.predict(X_t)
    proba = model.predict_proba(X_t)[:, 1]
    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    print(classification_report(y_t, preds, target_names=["Won", "Churned"]))
    print(f"  ROC-AUC: {roc_auc_score(y_t, proba):.4f}")

evaluate("XGBoost", xgb_model, X_test, y_test)
evaluate("Random Forest", rf_pipeline, X_test, y_test)

# ─────────────────────────────────────────────
# 15. CROSS-VALIDATION (XGBoost, 5-fold)
# ─────────────────────────────────────────────
# XGBoost requires no NaN for cross_val_score wrapper;
# use imputed copy for CV
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy="most_frequent")
X_imp = pd.DataFrame(imp.fit_transform(X), columns=X.columns)

cv_scores = cross_val_score(
    xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False, eval_metric="logloss",
        random_state=42, enable_categorical=False,
    ),
    X_imp, y,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="roc_auc",
    n_jobs=-1,
)
print(f"\nXGBoost 5-fold CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ─────────────────────────────────────────────
# 16. FEATURE IMPORTANCE PLOT
# ─────────────────────────────────────────────
importances = pd.Series(xgb_model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=True)

plt.figure(figsize=(8, 5))
importances.plot(kind="barh", color="#378ADD")
plt.title("XGBoost Feature Importance", fontsize=13)
plt.xlabel("Importance score")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()
print("\nSaved: feature_importance.png")

# ─────────────────────────────────────────────
# 17. CONFUSION MATRIX
# ─────────────────────────────────────────────
cm = confusion_matrix(y_test, xgb_model.predict(X_test))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Won", "Churned"])
fig, ax = plt.subplots(figsize=(5, 4))
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("XGBoost — Confusion Matrix (Test set)", fontsize=12)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()
print("Saved: confusion_matrix.png")

print("\n=== Pipeline complete ===")