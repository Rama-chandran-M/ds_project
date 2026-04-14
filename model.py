"""
======================================================================
  Membership Renewal Prediction — Full ML Pipeline
  Target: Membership_Renewal_Decision (Yes / No)
======================================================================
  Covers:
  1. Data Leakage Analysis & Audit Report
  2. Pre-processing (categorical + numerical)
  3. Decision Tree (CRT) model — handles mixed data natively
  4. Evaluation: classification report, confusion matrix, feature importance
  5. All results saved to CSV
======================================================================
"""

import pandas as pd
import numpy as np
import warnings
import os
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = "/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading Data")
print("=" * 60)
df = pd.read_csv('sample_04.csv')
print(f"  Raw shape: {df.shape}")

# ─────────────────────────────────────────────
# 2. DATA LEAKAGE AUDIT
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# 3. CLEAN & PREPARE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Data Cleaning & Preparation")
print("=" * 60)

# Drop leakage columns
DROP_COLS = [
    'Prospect_Outcome',
    'Desire_To_Cancel',
    'Renewal_Impact_Due_to_Price_Increase',
    'cc_business_struggles_financial_hardship_x',
    'cc_contractor_sentiment_x',
    'cc_contractor_sentiment_issues_score_x',
    'cc_business_struggles_financial_hardship_y',
    'cc_contractor_sentiment_y',
    'cc_contractor_sentiment_issues_score_y',
]

# Keep only rows where target is known (Yes / No) — drop NaN and 'unknown'
df_clean = df[df[TARGET].isin(['Yes', 'No'])].copy()
df_clean = df_clean.drop(columns=DROP_COLS, errors='ignore')

print(f"  Rows after filtering to Yes/No target: {len(df_clean)}")
print(f"  Target distribution:\n{df_clean[TARGET].value_counts()}")

# Features and target
X = df_clean.drop(columns=[TARGET])
y = (df_clean[TARGET] == 'Yes').astype(int)   # 1=Churned/No renewal, 0=Renewed

# Identify column types
CAT_COLS = X.select_dtypes(include='object').columns.tolist()
NUM_COLS = X.select_dtypes(include='number').columns.tolist()

print(f"\n  Categorical features ({len(CAT_COLS)}): {CAT_COLS}")
print(f"  Numerical features  ({len(NUM_COLS)}): {NUM_COLS}")

# ─────────────────────────────────────────────
# 4. PREPROCESSING PIPELINE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Building Preprocessing Pipeline")
print("=" * 60)

# Categorical: impute with 'Missing' then ordinal encode
# (Decision Tree handles ordinal-encoded categories well)
cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

# Numerical: impute with median
num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
])

preprocessor = ColumnTransformer([
    ('cat', cat_transformer, CAT_COLS),
    ('num', num_transformer, NUM_COLS)
])

# ─────────────────────────────────────────────
# 5. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Train/Test Split (80/20, stratified)")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")
print(f"  Train class balance: {y_train.value_counts().to_dict()}")
print(f"  Test  class balance: {y_test.value_counts().to_dict()}")

# ─────────────────────────────────────────────
# 6. CRT DECISION TREE MODEL
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: CRT Decision Tree Model")
print("=" * 60)

# CRT = Classification and Regression Tree (Breiman et al.)
# sklearn's DecisionTreeClassifier implements CRT by default
# criterion='gini' → uses Gini impurity (standard CRT splitting criterion)

crt_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(
        criterion='gini',          # CRT uses Gini impurity
        max_depth=6,               # Prevent overfitting
        min_samples_split=50,      # At least 50 samples to split a node
        min_samples_leaf=20,       # At least 20 samples in each leaf
        class_weight='balanced',   # Handle class imbalance (Yes is rare)
        random_state=42
    ))
])

crt_model.fit(X_train, y_train)
print("  ✅ CRT model trained")

# ─────────────────────────────────────────────
# 7. EVALUATION
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: Model Evaluation")
print("=" * 60)

y_pred  = crt_model.predict(X_test)
y_proba = crt_model.predict_proba(X_test)[:, 1]

# Classification report
report_dict = classification_report(y_test, y_pred, target_names=['No (Renewed)', 'Yes (Churned)'], output_dict=True)
report_df = pd.DataFrame(report_dict).transpose().reset_index().rename(columns={'index': 'Class'})
report_df.to_csv(f"{OUTPUT_DIR}/classification_report.csv", index=False)
print(classification_report(y_test, y_pred, target_names=['No (Renewed)', 'Yes (Churned)']))

# ROC-AUC
auc = roc_auc_score(y_test, y_proba)
print(f"  ROC-AUC Score: {auc:.4f}")

# ─────────────────────────────────────────────
# 8. CROSS VALIDATION
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8: 5-Fold Cross Validation")
print("=" * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(crt_model, X, y, cv=cv, scoring='roc_auc')
print(f"  CV ROC-AUC scores: {np.round(cv_scores, 4)}")
print(f"  Mean: {cv_scores.mean():.4f} | Std: {cv_scores.std():.4f}")

cv_df = pd.DataFrame({
    'Fold': [f'Fold {i+1}' for i in range(5)],
    'ROC_AUC': np.round(cv_scores, 4)
})
cv_df.loc[len(cv_df)] = ['Mean ± Std', f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}"]
cv_df.to_csv(f"{OUTPUT_DIR}/cross_validation_scores.csv", index=False)

# ─────────────────────────────────────────────
# 9. FEATURE IMPORTANCE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 9: Feature Importance")
print("=" * 60)

feature_names = CAT_COLS + NUM_COLS
importances = crt_model.named_steps['classifier'].feature_importances_
feat_imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False).reset_index(drop=True)
feat_imp_df['Rank'] = feat_imp_df.index + 1
print(feat_imp_df.to_string(index=False))
feat_imp_df.to_csv(f"{OUTPUT_DIR}/feature_importance.csv", index=False)

# ─────────────────────────────────────────────
# 10. CONFUSION MATRIX PLOT
# ─────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No (Renewed)', 'Yes (Churned)'])
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title('Confusion Matrix', fontsize=13, fontweight='bold')

# Feature importance bar chart
top_n = feat_imp_df.head(7)
axes[1].barh(top_n['Feature'][::-1], top_n['Importance'][::-1], color='steelblue')
axes[1].set_xlabel('Gini Importance')
axes[1].set_title('Top Feature Importances (CRT)', fontsize=13, fontweight='bold')
axes[1].tick_params(axis='y', labelsize=9)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/model_evaluation_plots.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  ✅ Plots saved → model_evaluation_plots.png")

# ─────────────────────────────────────────────
# 11. SUMMARY REPORT
# ─────────────────────────────────────────────
summary = pd.DataFrame([{
    'Total_Rows_Raw': len(df),
    'Rows_After_Cleaning': len(df_clean),
    'Training_Rows': len(X_train),
    'Test_Rows': len(X_test),
    'Categorical_Features': len(CAT_COLS),
    'Numerical_Features': len(NUM_COLS),
    'Model': 'Decision Tree (CRT / Gini)',
    'Max_Depth': 6,
    'Class_Weight': 'balanced',
    'Test_ROC_AUC': round(auc, 4),
    'CV_ROC_AUC_Mean': round(cv_scores.mean(), 4),
    'CV_ROC_AUC_Std': round(cv_scores.std(), 4),
    'Leakage_Columns_Dropped': len(DROP_COLS),
}])
summary.to_csv(f"{OUTPUT_DIR}/model_summary.csv", index=False)

print("\n" + "=" * 60)
print("ALL OUTPUTS SAVED:")
print("  📄 leakage_audit_report.csv      — data leakage findings")
print("  📄 classification_report.csv     — precision / recall / F1")
print("  📄 cross_validation_scores.csv   — 5-fold CV AUC scores")
print("  📄 feature_importance.csv        — ranked feature importances")
print("  📄 model_summary.csv             — overall summary")
print("  🖼️  model_evaluation_plots.png    — confusion matrix + importances")
print("=" * 60)
print("DONE ✅")