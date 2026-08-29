import pandas as pd
import random
import math
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve, classification_report

random.seed(42)

train_df = pd.read_csv('output/train.csv')
test_df = pd.read_csv('output/test.csv')

feature_cols = [
    'n_inputs', 'n_outputs', 'n_unique_input_addresses',
    'n_unique_output_addresses', 'total_input_btc', 'total_output_btc',
    'input_output_ratio', 'fan_in_5plus', 'fee',
    'sender_tx_count_1h', 'sender_tx_count_24h', 'sender_time_since_last_tx_min',
    'sender_distinct_asn_last10', 'sender_distinct_ip_last10', 'sender_amount_zscore',
]

# ---- Stratified-by-type fit/val split ----
# Plain GroupShuffleSplit only knows about canonical_wallet_id groups, not anomaly_type,
# so with rare types down to 4 wallets in train it can (and did) draw a random 20% slice
# that misses that type entirely -- meaning the tuned threshold was never actually checked
# against it. Mirrors the same "force minimum wallet representation" approach split_dataset.py
# already uses for train/test: for every anomaly_type, force at least 1 of its wallets into val.
VAL_FRACTION = 0.2
anomaly_df = train_df[train_df['is_anomaly'] == 1]

val_groups = set()
for atype in anomaly_df['anomaly_type'].unique():
    wallets = sorted(anomaly_df[anomaly_df['anomaly_type'] == atype]['canonical_wallet_id'].unique().tolist())
    n_val = max(1, math.ceil(len(wallets) * VAL_FRACTION))
    random.shuffle(wallets)
    val_groups.update(wallets[:n_val])

all_groups = set(train_df['canonical_wallet_id'].unique())
remaining_groups = sorted(all_groups - val_groups)
random.shuffle(remaining_groups)
n_remaining_val = int(len(remaining_groups) * VAL_FRACTION)
val_groups.update(remaining_groups[:n_remaining_val])
fit_groups = all_groups - val_groups

fit_df = train_df[train_df['canonical_wallet_id'].isin(fit_groups)].reset_index(drop=True)
val_df = train_df[train_df['canonical_wallet_id'].isin(val_groups)].reset_index(drop=True)

overlap = set(fit_df['canonical_wallet_id']) & set(val_df['canonical_wallet_id'])
assert len(overlap) == 0, f"LEAKAGE in fit/val split: {len(overlap)} groups shared"

print("=== fit/val anomaly_type coverage check ===")
for atype in train_df[train_df['is_anomaly'] == 1]['anomaly_type'].unique():
    n_fit = (fit_df['anomaly_type'] == atype).sum()
    n_val = (val_df['anomaly_type'] == atype).sum()
    flag = "  <-- WARNING: zero in val, threshold for this type not checkable" if n_val == 0 else ""
    print(f"  {atype}: fit={n_fit}, val={n_val}{flag}")

X_fit, y_fit = fit_df[feature_cols], fit_df['is_anomaly']
X_val, y_val = val_df[feature_cols], val_df['is_anomaly']
X_test, y_test = test_df[feature_cols], test_df['is_anomaly']

neg, pos = (y_fit == 0).sum(), (y_fit == 1).sum()
scale_pos_weight = neg / pos

models = {
    # scaled: LR's lbfgs solver converges slowly / warns when features span very
    # different ranges (tx counts 0-30 vs zscore -10..10 vs time-since-last in the
    # thousands of minutes). Trees (RF/XGBoost) are scale-invariant, so leave them as-is.
    'Logistic Regression': make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    ),
    'Random Forest': RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42),
    'XGBoost': XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        scale_pos_weight=scale_pos_weight, eval_metric='logloss', random_state=42
    ),
}

for name, model in models.items():
    model.fit(X_fit, y_fit)

    # pick threshold on VAL only
    val_probs = model.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = f1s[:-1].argmax()  # last point has no threshold
    best_thresh = thresholds[best_idx]

    print(f"\n=== {name} ===")
    print(f"Val-selected threshold: {best_thresh:.3f} (val F1={f1s[best_idx]:.3f}, "
          f"val precision={precisions[best_idx]:.3f}, val recall={recalls[best_idx]:.3f})")

    # apply chosen threshold ONCE to untouched test set
    test_probs = model.predict_proba(X_test)[:, 1]
    test_pred = (test_probs >= best_thresh).astype(int)

    print(f"--- {name}: TEST at tuned threshold (default 0.5 for comparison too) ---")
    print("Tuned threshold:")
    print(classification_report(y_test, test_pred, target_names=['Normal', 'Anomalous']))
    default_pred = (test_probs >= 0.5).astype(int)
    print("Default 0.5 threshold (baseline for comparison):")
    print(classification_report(y_test, default_pred, target_names=['Normal', 'Anomalous']))

    test_df[f'{name}_pred_tuned'] = test_pred
    print(f"--- {name}: Recall by anomaly_type (tuned threshold) ---")
    for atype in test_df[test_df['is_anomaly'] == 1]['anomaly_type'].unique():
        subset = test_df[test_df['anomaly_type'] == atype]
        caught = (subset[f'{name}_pred_tuned'] == 1).sum()
        total = len(subset)
        print(f"  {atype}: {caught}/{total} caught ({caught/total:.1%})")