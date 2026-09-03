import pandas as pd
from sklearn.metrics import roc_auc_score

from network_corr import NETWORK_CORR_COLS

train_df = pd.read_csv("output/train.csv")
feature_cols = [
    "n_inputs",
    "n_outputs",
    "n_unique_input_addresses",
    "n_unique_output_addresses",
    "total_input_btc",
    "input_output_ratio",
    "fan_in_5plus",
    "output_min_max_ratio",
    "fee",
    "fee_ratio",
    "sender_tx_count_1h",
    "sender_tx_count_24h",
    "sender_time_since_last_tx_min",
    "sender_distinct_asn_last10",
    "sender_distinct_ip_last10",
    "sender_amount_zscore",
] + NETWORK_CORR_COLS

anomaly_types = train_df["anomaly_type"].dropna().unique()

print("Single-feature AUC per anomaly_type (train set)")
print(
    "AUC near 1.0 or 0.0 for a feature = that feature alone near-perfectly separates that type = leak candidate\n"
)

any_suspect = False
for atype in anomaly_types:
    y_type = (train_df["anomaly_type"] == atype).astype(int)
    # only meaningful if there are both anomaly-of-this-type rows and normal rows to compare against
    mask = (train_df["is_anomaly"] == 0) | (train_df["anomaly_type"] == atype)
    print(f"--- {atype} (n={y_type.sum()}) ---")
    for col in feature_cols:
        try:
            auc = roc_auc_score(y_type[mask], train_df.loc[mask, col])
            suspect = auc > 0.9 or auc < 0.1
            flag = "  <-- SUSPECT" if suspect else ""
            any_suspect = any_suspect or suspect
            print(f"  {col:30s} AUC={auc:.3f}{flag}")
        except Exception as e:
            print(f"  {col:30s} skipped ({e})")
    print()

print("=== RESULT ===")
print(
    "SUSPECT features found (AUC>0.9 or <0.1) - investigate before training"
    if any_suspect
    else "No single-feature AUC >0.9 or <0.1 - no leak candidates found"
)
