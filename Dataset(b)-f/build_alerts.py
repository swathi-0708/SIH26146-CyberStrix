"""
Combine XGBoost (supervised, primary) and Isolation Forest (unsupervised,
blind, secondary) flags into one alert list.

Rule (per spec): tag by detector, never blend scores.
  - both        -> highest priority (two independent methods agree)
  - xgboost     -> high priority (trained detector, confident on known patterns)
  - isolation_forest -> lower priority, "worth reviewing" tier (only a
                    structural outlier by the blind model -- this is exactly
                    the tier meant to catch pattern types XGBoost was never
                    trained on)

XGBoost's y_proba (a probability) and IF's decision_function (an unbounded,
differently-scaled anomaly score) are NEVER averaged or combined into one
number. Each keeps its own column; only the boolean flags are combined for
tiering.
"""
import pandas as pd

xgb = pd.read_csv('output/xgboost_test_scores.csv')          # test-set only (has labels held out)
iso = pd.read_csv('output/isolation_forest_scores.csv')      # all rows (train+test)

# XGBoost only scored the test split (that's the honest, held-out evaluation set);
# restrict the merge to that same set so "both"/"xgboost only" comparisons are apples-to-apples.
merged = xgb.merge(
    iso[['txid', 'if_flag', 'if_score']],
    on='txid', how='left'
)

merged['xgb_flag'] = merged['xgb_pred']

def detector_tag(row):
    if row['xgb_flag'] == 1 and row['if_flag'] == 1:
        return 'both'
    if row['xgb_flag'] == 1:
        return 'xgboost'
    if row['if_flag'] == 1:
        return 'isolation_forest'
    return None

merged['detector'] = merged.apply(detector_tag, axis=1)

def priority_tier(row):
    if row['detector'] == 'both':
        return 'high'
    if row['detector'] == 'xgboost':
        return 'medium-high'
    if row['detector'] == 'isolation_forest':
        return 'worth reviewing'
    return None

merged['priority_tier'] = merged.apply(priority_tier, axis=1)

alerts = merged[merged['detector'].notna()].copy()
# sort: 'both' first, then by xgb_proba desc within tier (never mixed with if_score)
tier_order = {'high': 0, 'medium-high': 1, 'worth reviewing': 2}
alerts['tier_order'] = alerts['priority_tier'].map(tier_order)
alerts = alerts.sort_values(['tier_order', 'xgb_proba'], ascending=[True, False])

# ---- Demo-facing alerts: NO ground truth columns. This is what an analyst
# would see on real seized data, where is_anomaly/anomaly_type don't exist.
# Don't show the answer key next to the predictions.
demo_cols = ['txid', 'canonical_wallet_id', 'detector', 'priority_tier', 'xgb_proba', 'if_score']
alerts[demo_cols].to_csv('output/alerts.csv', index=False)

# ---- Dev-only file WITH ground truth, for your own precision checking / write-up.
# Not for the demo.
eval_cols = demo_cols + ['anomaly_type', 'is_anomaly']
alerts[eval_cols].to_csv('output/alerts_eval.csv', index=False)

print(f"Total alerts: {len(alerts)} (of {len(merged)} test-set rows)")
print(alerts['detector'].value_counts().to_string())
print()
print("=== Precision by tier (against ground truth, for reporting only) ===")
for tier in ['high', 'medium-high', 'worth reviewing']:
    sub = alerts[alerts['priority_tier'] == tier]
    if len(sub) == 0:
        continue
    precision = sub['is_anomaly'].mean()
    print(f"  {tier}: {len(sub)} alerts, {precision:.1%} true positive")

print("\nDemo alerts (no ground truth) -> output/alerts.csv")
print("Dev/eval alerts (with ground truth, NOT for demo) -> output/alerts_eval.csv")
