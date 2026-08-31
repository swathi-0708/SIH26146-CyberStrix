import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix
)
import joblib

# ---- Load split data ----
train_df = pd.read_csv('output/train.csv')
test_df = pd.read_csv('output/test.csv')

# FROZEN — do not edit without re-running check.py (leakage check) on the new set.
# Ported from swathi/train_compare.py (Dataset A) with fee_ratio added, since the
# fee leak that check.py found lived in fee_ratio and the model needs to see a
# *clean* fee_ratio to prove it's no longer a shortcut.
FEATURE_COLS = [
    'n_inputs', 'n_outputs', 'n_unique_input_addresses',
    'n_unique_output_addresses', 'total_input_btc',
    'input_output_ratio', 'fan_in_5plus',
    'output_min_max_ratio', 'fee', 'fee_ratio',
    # behavioral / velocity features (causal, per-sender history) --
    # these are what give the model any chance on structuring, ip_hopping,
    # wallet_reuse_burst and profile_deviation, which have near-zero signal
    # in the purely structural features above (see check.py output)
    'sender_tx_count_1h', 'sender_tx_count_24h', 'sender_time_since_last_tx_min',
    'sender_distinct_asn_last10', 'sender_distinct_ip_last10', 'sender_amount_zscore',
]

X_train = train_df[FEATURE_COLS]
y_train = train_df['is_anomaly']
X_test = test_df[FEATURE_COLS]
y_test = test_df['is_anomaly']

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos
print(f"Train imbalance: {neg} normal, {pos} anomaly, scale_pos_weight={scale_pos_weight:.2f}\n")

models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42),
    'XGBoost': XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        scale_pos_weight=scale_pos_weight, eval_metric='logloss', random_state=42
    ),
}

results = []
trained_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    results.append({
        'Model': name,
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_proba),
        'PR-AUC': average_precision_score(y_test, y_proba),
    })
    trained_models[name] = (model, y_pred, y_proba)

# ---- Summary table ----
results_df = pd.DataFrame(results)
print("=== Model Comparison (test set) ===")
print(results_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# ---- Confusion matrix + per-anomaly-type recall for each model ----
for name, (model, y_pred, y_proba) in trained_models.items():
    print(f"\n=== {name}: Confusion Matrix ===")
    cm = confusion_matrix(y_test, y_pred)
    print("        Pred Normal  Pred Anomaly")
    print(f"True Normal   {cm[0][0]:>6}       {cm[0][1]:>6}")
    print(f"True Anomaly  {cm[1][0]:>6}       {cm[1][1]:>6}")

    test_df['pred'] = y_pred
    print(f"=== {name}: Recall by anomaly_type ===")
    for atype in test_df[test_df['is_anomaly'] == 1]['anomaly_type'].unique():
        subset = test_df[test_df['anomaly_type'] == atype]
        caught = (subset['pred'] == 1).sum()
        total = len(subset)
        print(f"  {atype}: {caught}/{total} caught ({caught/total:.1%})")

# ---- Save best model (by PR-AUC, most meaningful for imbalanced data) ----
best_name = results_df.loc[results_df['PR-AUC'].idxmax(), 'Model']
best_model, best_pred, best_proba = trained_models[best_name]
joblib.dump(best_model, 'output/baseline_model.pkl')
print(f"\nBest model by PR-AUC: {best_name} -> saved to output/baseline_model.pkl")

# ---- Save XGBoost test-set predictions specifically for the alert-tagging step ----
# (XGBoost is the primary detector per spec, regardless of which model wins PR-AUC)
xgb_model, xgb_pred, xgb_proba = trained_models['XGBoost']
xgb_scores = test_df[['txid', 'canonical_wallet_id', 'is_anomaly', 'anomaly_type']].copy()
xgb_scores['xgb_pred'] = xgb_pred
xgb_scores['xgb_proba'] = xgb_proba
xgb_scores.to_csv('output/xgboost_test_scores.csv', index=False)
print("XGBoost test-set scores -> output/xgboost_test_scores.csv")
