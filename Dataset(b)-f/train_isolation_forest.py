"""
Secondary, BLIND detector for Dataset B.

Purpose (per spec): catch structurally anomalous patterns outside XGBoost's
trained categories -- real value for unseen/future anomaly types on real
seized data, where you won't have ground-truth labels to train against.

This is conceptually the same idea as Jayadev/dsBtrained.py (Isolation Forest,
unsupervised) but rebuilt to:
  - use the SAME frozen FEATURE_COLS as train_compare.py's XGBoost, computed
    once by split_dataset.py, instead of recomputing a separate, non-causal
    feature set (dsBtrained.py's time_since_last_tx used a plain groupby-diff
    with no 24h cap, and its tx_freq_last_24h was an O(n^2) python loop)
  - NEVER read is_anomaly / anomaly_type as model input -- labels are loaded
    only afterward, for evaluation/reporting, never for fitting
  - run over ALL rows (train+test), since an unsupervised model has no
    train/test leakage concept the way a supervised one does
"""
import pandas as pd
from sklearn.ensemble import IsolationForest

train_df = pd.read_csv('output/train.csv')
test_df = pd.read_csv('output/test.csv')
all_df = pd.concat([train_df, test_df], ignore_index=True)

# Same frozen feature list as train_compare.py -- kept in sync manually since
# these are two separate detectors, not to be merged.
FEATURE_COLS = [
    'n_inputs', 'n_outputs', 'n_unique_input_addresses',
    'n_unique_output_addresses', 'total_input_btc',
    'input_output_ratio', 'fan_in_5plus',
    'output_min_max_ratio', 'fee', 'fee_ratio',
    'input_addr_is_recent_output', 'minutes_since_addr_last_output',
    'sender_tx_count_1h', 'sender_tx_count_24h', 'sender_time_since_last_tx_min',
    'sender_distinct_asn_last10', 'sender_distinct_ip_last10', 'sender_amount_zscore',
]

X = all_df[FEATURE_COLS]

# contamination left at the actual anomaly rate in the generated dataset as a
# starting point (~4.8%) -- on real seized data this would need re-tuning,
# there's no ground truth to calibrate against there.
contamination = round(all_df['is_anomaly'].mean(), 4)
print(f"Fitting IsolationForest BLIND on {len(X)} rows, {len(FEATURE_COLS)} features, "
      f"contamination={contamination} (no labels used in fit)\n")

iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
iso.fit(X)  # labels never touch this call

all_df['if_pred'] = iso.predict(X)          # -1 = anomaly, 1 = normal
all_df['if_flag'] = (all_df['if_pred'] == -1).astype(int)
all_df['if_score'] = iso.decision_function(X)  # lower = more anomalous

all_df[['txid', 'canonical_wallet_id', 'if_flag', 'if_score']].to_csv(
    'output/isolation_forest_scores.csv', index=False
)

# ---- Post-hoc evaluation against labels (labels used ONLY for reporting here) ----
flagged = all_df[all_df['if_flag'] == 1]
print(f"IF flagged {len(flagged)}/{len(all_df)} rows ({len(flagged)/len(all_df):.1%})\n")

print("=== IF recall by anomaly_type (post-hoc, not used in training) ===")
for atype in all_df[all_df['is_anomaly'] == 1]['anomaly_type'].unique():
    subset = all_df[all_df['anomaly_type'] == atype]
    caught = (subset['if_flag'] == 1).sum()
    total = len(subset)
    print(f"  {atype}: {caught}/{total} caught ({caught/total:.1%})")

n_normal = (all_df['is_anomaly'] == 0).sum()
fp = ((all_df['is_anomaly'] == 0) & (all_df['if_flag'] == 1)).sum()
print(f"\nFalse positive rate on normal rows: {fp}/{n_normal} ({fp/n_normal:.2%})")
print("Scores -> output/isolation_forest_scores.csv")