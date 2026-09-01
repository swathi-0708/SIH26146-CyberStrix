"""
Explainability layer for SIH PS 26146.

PS requirement: "Generate a ranked, explainable alert list (why a wallet/
transaction was flagged, with a confidence score)." alerts.csv already has
the confidence score (xgb_proba) and the rank (priority_tier). It does NOT
have the "why" -- a judge reading it sees THAT wallet #499 is high-priority,
not WHICH features drove that call. This script closes that gap with SHAP
on the XGBoost model (the primary, supervised detector per spec).

Why SHAP and not just "feature importance": global feature importance
(e.g. XGBoost's built-in gain) tells you what matters on average across
all transactions. It can't tell you why THIS SPECIFIC transaction was
flagged -- one alert might be driven by a fan-in spike, another by a
velocity anomaly, even if both hit "high" tier. SHAP gives a per-row
attribution, which is what "explainable" actually means here.

Approach:
  1. Re-fit XGBoost with the exact same config/features/random_state as
     train_compare.py (that script only persists LogisticRegression as
     output/baseline_model.pkl when it wins PR-AUC -- XGBoost itself
     isn't saved anywhere, so it's refit here rather than assuming a
     pickle exists).
  2. Run shap.TreeExplainer on the test-set rows that ended up in
     alerts.csv (only alerted rows need explaining -- SHAP-ing all 2589
     test rows for a 322-row alert list would be wasted computation).
  3. For each alerted txid, take the top-3 |SHAP value| features and
     render them as a plain-English reason string, e.g.
     "high fan_in_5plus (+0.41), low sender_time_since_last_tx_min (+0.22),
      elevated sender_amount_zscore (+0.15)".
  4. Write output/alerts_explained.csv = alerts.csv + a `top_reasons`
     column + the 3 raw (feature, shap_value) pairs for anyone who wants
     the numbers instead of the sentence.

Usage:
    python3 explain_alerts.py
"""
import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier

FEATURE_COLS = [
    'n_inputs', 'n_outputs', 'n_unique_input_addresses',
    'n_unique_output_addresses', 'total_input_btc',
    'input_output_ratio', 'fan_in_5plus',
    'output_min_max_ratio', 'fee', 'fee_ratio',
    'input_addr_is_recent_output', 'minutes_since_addr_last_output',
    'sender_tx_count_1h', 'sender_tx_count_24h', 'sender_time_since_last_tx_min',
    'sender_distinct_asn_last10', 'sender_distinct_ip_last10', 'sender_amount_zscore',
]

# human-readable phrasing per feature: (high_phrase, low_phrase)
FEATURE_PHRASING = {
    'n_inputs': ("unusually many inputs", "unusually few inputs"),
    'n_outputs': ("unusually many outputs", "unusually few outputs"),
    'n_unique_input_addresses': ("many distinct input addresses", "few distinct input addresses"),
    'n_unique_output_addresses': ("many distinct output addresses", "few distinct output addresses"),
    'total_input_btc': ("large total BTC moved", "small total BTC moved"),
    'input_output_ratio': ("input/output amount mismatch", "near-perfect input/output balance"),
    'fan_in_5plus': ("5+ address fan-in pattern", "no fan-in pattern"),
    'output_min_max_ratio': ("very uneven output split", "even output split"),
    'fee': ("abnormal fee amount", "abnormally low fee"),
    'fee_ratio': ("fee out of line with tx size", "unusually low fee ratio"),
    'input_addr_is_recent_output': ("input addr reused right after receiving", "input addr not recently active"),
    'minutes_since_addr_last_output': ("very fast address reuse", "long dormancy before reuse"),
    'sender_tx_count_1h': ("burst of tx in the last hour", "quiet in the last hour"),
    'sender_tx_count_24h': ("high tx volume in 24h", "low tx volume in 24h"),
    'sender_time_since_last_tx_min': ("rapid repeat transacting", "long gap since last tx"),
    'sender_distinct_asn_last10': ("hopping across many ASNs recently", "consistent ASN recently"),
    'sender_distinct_ip_last10': ("hopping across many IPs recently", "consistent IP recently"),
    'sender_amount_zscore': ("amount far from sender's usual pattern", "amount close to sender's usual pattern"),
}


def reason_phrase(feature, shap_value, feature_value):
    high_phrase, low_phrase = FEATURE_PHRASING.get(feature, (feature, feature))
    direction = high_phrase if shap_value > 0 else low_phrase
    return f"{direction} ({feature}={feature_value:.4g}, impact={shap_value:+.3f})"


def main():
    train_df = pd.read_csv('output/train.csv')
    test_df = pd.read_csv('output/test.csv')
    alerts_df = pd.read_csv('output/alerts.csv')

    X_train = train_df[FEATURE_COLS]
    y_train = train_df['is_anomaly']
    X_test = test_df[FEATURE_COLS]

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    model = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        scale_pos_weight=neg / pos, eval_metric='logloss', random_state=42
    )
    model.fit(X_train, y_train)

    # Only explain the rows that made it into alerts.csv -- that's the
    # actual deliverable, not every test-set transaction.
    alerted_txids = set(alerts_df['txid'])
    test_df = test_df.set_index('txid')
    alert_rows = test_df.loc[test_df.index.intersection(alerted_txids)]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(alert_rows[FEATURE_COLS])

    reasons = {}
    for i, txid in enumerate(alert_rows.index):
        row_shap = shap_values[i]
        row_vals = alert_rows.iloc[i]
        top_idx = np.argsort(-np.abs(row_shap))[:3]
        phrases = [
            reason_phrase(FEATURE_COLS[j], row_shap[j], row_vals[FEATURE_COLS[j]])
            for j in top_idx
        ]
        reasons[txid] = {
            "top_reasons": "; ".join(phrases),
            "top_feature_1": FEATURE_COLS[top_idx[0]], "shap_1": round(float(row_shap[top_idx[0]]), 4),
            "top_feature_2": FEATURE_COLS[top_idx[1]], "shap_2": round(float(row_shap[top_idx[1]]), 4),
            "top_feature_3": FEATURE_COLS[top_idx[2]], "shap_3": round(float(row_shap[top_idx[2]]), 4),
        }

    reasons_df = pd.DataFrame.from_dict(reasons, orient='index').reset_index().rename(columns={'index': 'txid'})

    # isolation_forest-only alerts have no XGBoost explanation (XGBoost
    # didn't flag them) -- mark plainly rather than silently dropping them.
    explained = alerts_df.merge(reasons_df, on='txid', how='left')
    no_xgb = explained['top_reasons'].isna()
    explained.loc[no_xgb, 'top_reasons'] = (
        "not flagged by XGBoost -- structural outlier per Isolation Forest only "
        "(no SHAP attribution available for this detector)"
    )

    explained.to_csv('output/alerts_explained.csv', index=False)

    print(f"Explained {(~no_xgb).sum()} of {len(explained)} alerts via SHAP "
          f"({no_xgb.sum()} were isolation_forest-only, no XGBoost explanation to give)")
    print("\nSample explained alerts:\n")
    sample_cols = ['txid', 'canonical_wallet_id', 'priority_tier', 'xgb_proba', 'top_reasons']
    print(explained[sample_cols].head(5).to_string(index=False))
    print("\nWrote output/alerts_explained.csv")


if __name__ == "__main__":
    main()
