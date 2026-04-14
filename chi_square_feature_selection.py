"""
Chi-Square Feature Selection for Prospect Outcome Classification
================================================================
Features tested (your original selection):
    Membership_Renewal_Decision, Serious_Complaint, Other_Complaint,
    Discussion_on_Price_Increase, Renewal_Impact_Due_to_Price_Increase,
    Discount_or_Waiver_Requested, Call_Reschedule_Request,
    Explicit_Competitor_Mention, Explicit_Switching_Intent,
    Mentioned_Competitors, Desire_To_Cancel, Discount_Offered

Target: Prospect_Outcome (Won / Churned)

Hypotheses
----------
For every feature below, the test is structured as:

  H0 (Null Hypothesis):
      The feature is INDEPENDENT of Prospect_Outcome.
      Knowing the feature value tells us nothing about whether a
      prospect will churn or renew.

  H1 (Alternative Hypothesis):
      The feature is DEPENDENT on Prospect_Outcome.
      There is a statistically significant association between the
      feature and the outcome — it carries predictive signal.

Feature-specific hypotheses
-----------------------------
1. Membership_Renewal_Decision
   H0: Whether a member decided to renew has no association with Prospect_Outcome.
   H1: Members who decided to renew are significantly more likely to be Won.

2. Serious_Complaint
   H0: Having a serious complaint is unrelated to Prospect_Outcome.
   H1: Customers with serious complaints are significantly more likely to Churn.

3. Other_Complaint
   H0: Having other (non-serious) complaints is unrelated to Prospect_Outcome.
   H1: Customers with other complaints are significantly more likely to Churn.

4. Discussion_on_Price_Increase
   H0: Whether a price increase was discussed has no bearing on Prospect_Outcome.
   H1: Discussing a price increase is significantly associated with Churn.

5. Renewal_Impact_Due_to_Price_Increase
   H0: Price increase impact on renewal is unrelated to Prospect_Outcome.
   H1: Customers flagged as impacted by price increase are significantly more likely to Churn.

6. Discount_or_Waiver_Requested
   H0: Requesting a discount or waiver is unrelated to Prospect_Outcome.
   H1: Customers who requested a discount/waiver differ significantly in outcome.

7. Call_Reschedule_Request
   H0: Requesting to reschedule a call is unrelated to Prospect_Outcome.
   H1: Call reschedule requests are significantly associated with a particular outcome.

8. Explicit_Competitor_Mention
   H0: Explicitly mentioning a competitor is unrelated to Prospect_Outcome.
   H1: Customers who mention competitors are significantly more likely to Churn.

9. Explicit_Switching_Intent
   H0: Expressing explicit switching intent is unrelated to Prospect_Outcome.
   H1: Customers who express switching intent are significantly more likely to Churn.

10. Mentioned_Competitors
    H0: Mentioning competitors (broadly) is unrelated to Prospect_Outcome.
    H1: Customers who mention competitors show a significantly different outcome distribution.

11. Desire_To_Cancel
    H0: Expressing a desire to cancel is unrelated to Prospect_Outcome.
    H1: Customers who expressed a desire to cancel are significantly more likely to Churn.

12. Discount_Offered
    H0: Whether a discount was offered is unrelated to Prospect_Outcome.
    H1: Offering a discount is significantly associated with a particular outcome.

Decision rules (how to interpret results)
------------------------------------------
After running the test, use the following criteria for each feature:

  p-value < 0.05  AND  Cramér's V >= 0.10  →  INCLUDE (strong signal)
  p-value < 0.05  AND  Cramér's V  < 0.10  →  CONSIDER (weak but real signal; use domain judgment)
  p-value >= 0.05                           →  DROP (no significant association)

Cramér's V strength guide:
  V < 0.10   →  Negligible association
  0.10–0.29  →  Weak association
  0.30–0.49  →  Moderate association
  V >= 0.50  →  Strong association
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings("ignore")

# ── 1. Load data ────────────────────────────────────────────────────────────
df = pd.read_csv("rc_merged_01.csv")

# Work only on rows that have call data (rc_flag = 1)
df_calls = df[df["rc_flag"] == 1].copy()
print(f"Rows used for testing (rc_flag=1): {len(df_calls):,}")
print(f"Prospect_Outcome distribution:\n{df_calls['Prospect_Outcome'].value_counts()}\n")

# ── 2. Define features to test ───────────────────────────────────────────────
features = [
    "Membership_Renewal_Decision",
    "Serious_Complaint",
    "Other_Complaint",
    "Discussion_on_Price_Increase",
    "Renewal_Impact_Due_to_Price_Increase",
    "Discount_or_Waiver_Requested",
    "Call_Reschedule_Request",
    "Explicit_Competitor_Mention",
    "Explicit_Switching_Intent",
    "Mentioned_Competitors",
    "Desire_To_Cancel",
    "Discount_Offered",
]

target = "Prospect_Outcome"

# ── 3. Chi-square test for each feature ─────────────────────────────────────
ALPHA = 0.05  # significance threshold

results = []

for feature in features:
    # Drop rows where either column is NaN
    subset = df_calls[[feature, target]].dropna()
    n = len(subset)

    # Build contingency table
    contingency_table = pd.crosstab(subset[feature], subset[target])

    # Run chi-square test
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)

    # Cramér's V (effect size, scale-independent)
    cramers_v = np.sqrt(chi2 / (chi2 + n))

    # Decision logic
    if p_value >= ALPHA:
        decision = "DROP"
        reason   = "Not significant (p >= 0.05)"
    elif cramers_v >= 0.10:
        decision = "INCLUDE"
        reason   = "Significant + meaningful effect size"
    else:
        decision = "CONSIDER"
        reason   = "Significant but weak effect (use domain judgment)"

    results.append({
        "Feature"   : feature,
        "N"         : n,
        "Chi2"      : round(chi2, 2),
        "p_value"   : round(p_value, 6),
        "DoF"       : dof,
        "Cramers_V" : round(cramers_v, 4),
        "Decision"  : decision,
        "Reason"    : reason,
    })

results_df = pd.DataFrame(results).sort_values("Chi2", ascending=False)

# ── 4. Print results ─────────────────────────────────────────────────────────
print("=" * 100)
print(f"{'Feature':<40} {'Chi2':>8} {'p-value':>10} {'DoF':>4} {'Cramér V':>9}  {'Decision':<10}  Reason")
print("=" * 100)

for _, row in results_df.iterrows():
    print(
        f"{row['Feature']:<40} "
        f"{row['Chi2']:>8.1f} "
        f"{row['p_value']:>10.6f} "
        f"{row['DoF']:>4} "
        f"{row['Cramers_V']:>9.4f}  "
        f"{row['Decision']:<10}  "
        f"{row['Reason']}"
    )

print("=" * 100)

# ── 5. Summary counts ────────────────────────────────────────────────────────
print("\nDecision summary:")
print(results_df["Decision"].value_counts().to_string())

# ── 6. Contingency tables (for manual inspection) ───────────────────────────
print("\n\n── Contingency tables (row = feature value, col = outcome) ──────────────")
for feature in features:
    subset = df_calls[[feature, target]].dropna()
    ct = pd.crosstab(subset[feature], subset[target], margins=True, margins_name="Total")
    # Add row percentages
    ct_pct = pd.crosstab(subset[feature], subset[target], normalize="index").round(3) * 100
    ct_pct.columns = [f"{c}_%" for c in ct_pct.columns]
    combined = pd.concat([ct, ct_pct], axis=1)
    print(f"\n{feature}")
    print(combined.to_string())
    print()

# ── 7. Export results to CSV ─────────────────────────────────────────────────
results_df.to_csv("chi_square_results.csv", index=False)
print("\nResults saved to: chi_square_results.csv")
