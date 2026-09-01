import re
import pandas as pd
import joblib

from sklearn.ensemble import IsolationForest


FEATURES_FILE = "output/ml_features_with_tx_anomalies.csv"
GROUND_TRUTH_FILE = "output/ground_truth_labels.csv"
WALLETS_FILE = "output/wallets_reference.csv"
XGBOOST_MODEL = "output/xgboost_wallet_model.pkl"
PEELING_FILE = "output/peeling_chain_results.csv"
PROFILE_FILE = "output/profile_deviation_results.csv"

FEATURE_COLUMNS = [
    "transaction_count",
    "incoming_count",
    "outgoing_count",
    "total_incoming_btc",
    "total_outgoing_btc",
    "avg_transaction_amount",
    "avg_time_gap_minutes",
    "unique_counterparties",
    "fan_in",
    "fan_out",
    "unique_ips",
    "unique_asns",
    "rapid_transfer_ratio",
    "split_ratio",
    "merge_ratio",
    "max_amount_ratio",
    "outgoing_small_amount_ratio",
    "tx_anomaly_count",
    "tx_total_count",
    "tx_anomaly_ratio",
    "tx_anomaly_score_mean",
    "tx_anomaly_score_min",
    "tx_anomaly_score_max",
]


def extract_wallet_id(wallet_label):

    if pd.isna(wallet_label):
        return None

    match = re.search(r"_(\d+)", str(wallet_label))

    if match:
        return int(match.group(1))

    return None


def main():

    # ---------------------------------------------------------
    # 1. Load files
    # ---------------------------------------------------------

    df = pd.read_csv(FEATURES_FILE)
    ground_truth_df = pd.read_csv(GROUND_TRUTH_FILE)
    wallets_df = pd.read_csv(WALLETS_FILE)

    peeling_df = pd.read_csv(PEELING_FILE)

    profile_df = pd.read_csv(PROFILE_FILE)

    profile_wallets = set(
        profile_df["wallet"]
    )

    df["profile_deviation_detected"] = (
        df["wallet"]
        .isin(profile_wallets)
        .astype(int)
    )

    peeling_wallets = set(
        peeling_df["input_wallet"]
    )

    df["peeling_chain_detected"] = (
        df["wallet"]
        .isin(peeling_wallets)
        .astype(int)
    )
    # ---------------------------------------------------------
    # 2. Convert anomaly labels → wallet IDs
    # ---------------------------------------------------------

    anomalous_labels = ground_truth_df[
        ground_truth_df["is_anomaly"] == 1
    ].copy()

    anomalous_labels["wallet_id"] = (
        anomalous_labels["wallet_ids"]
        .apply(extract_wallet_id)
    )

    anomalous_wallet_ids = set(
        anomalous_labels["wallet_id"]
        .dropna()
        .astype(int)
    )

    # ---------------------------------------------------------
    # 3. Convert wallet IDs → wallet addresses
    # ---------------------------------------------------------

    id_to_address = dict(
        zip(
            wallets_df["wallet_id"],
            wallets_df["address"]
        )
    )

    anomalous_addresses = {
        id_to_address[wallet_id]
        for wallet_id in anomalous_wallet_ids
        if wallet_id in id_to_address
    }

    # ---------------------------------------------------------
    # 4. Create actual anomaly labels
    # ---------------------------------------------------------

    df["actual_anomaly"] = (
        df["wallet"]
        .isin(anomalous_addresses)
        .astype(int)
    )

    print(f"Wallets loaded: {len(df)}")
    print(
        f"Anomalous wallets: "
        f"{df['actual_anomaly'].sum()}"
    )

    # ---------------------------------------------------------
    # 5. Prepare features
    # ---------------------------------------------------------

    X = df[FEATURE_COLUMNS].fillna(0)

    # ---------------------------------------------------------
    # 6. Isolation Forest
    # ---------------------------------------------------------

    isolation_forest = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42
    )

    isolation_forest.fit(X)

    df["if_anomaly_score"] = (
        -isolation_forest.decision_function(X)
    )

    # Normalize IF score
    if_min = df["if_anomaly_score"].min()
    if_max = df["if_anomaly_score"].max()

    if if_max > if_min:
        df["if_score_normalized"] = (
            (df["if_anomaly_score"] - if_min)
            / (if_max - if_min)
        )
    else:
        df["if_score_normalized"] = 0.0

    # ---------------------------------------------------------
    # 7. XGBoost
    # ---------------------------------------------------------

    xgb_model = joblib.load(XGBOOST_MODEL)

    df["xgb_probability"] = (
        xgb_model.predict_proba(X)[:, 1]
    )

    # ---------------------------------------------------------
    # 8. Combine scores
    # ---------------------------------------------------------

    df["risk_score"] = (
    0.30 * df["if_score_normalized"]
    + 0.45 * df["xgb_probability"]
    + 0.15 * df["peeling_chain_detected"]
    + 0.10 * df["profile_deviation_detected"]
)
# Specialist detectors override weak ML confidence
    specialist_detected = (
        (df["peeling_chain_detected"] == 1) |
        (df["profile_deviation_detected"] == 1)
    )

    df.loc[specialist_detected, "risk_score"] = (
        df.loc[specialist_detected, "risk_score"].clip(lower=0.50)
    )

    # ---------------------------------------------------------
    # 9. Risk categories
    # ---------------------------------------------------------

    def risk_category(score):

        if score >= 0.80:
            return "CRITICAL"

        elif score >= 0.60:
            return "HIGH"

        elif score >= 0.40:
            return "MEDIUM"

        else:
            return "LOW"

    df["risk_level"] = df["risk_score"].apply(
        risk_category
    )

    # ---------------------------------------------------------
    # 10. Display top wallets
    # ---------------------------------------------------------

    top_wallets = df.sort_values(
        "risk_score",
        ascending=False
    ).head(20)

    print("\nTop 20 highest-risk wallets:")

    print(
        top_wallets[
            [
                "wallet",
                "risk_score",
                "risk_level",
                "actual_anomaly",
                "if_score_normalized",
                "xgb_probability",
                "peeling_chain_detected",
                "profile_deviation_detected",
                "transaction_count",
                "fan_in",
                "fan_out",
                "unique_counterparties",
                "unique_ips",
                "rapid_transfer_ratio",
                "max_amount_ratio",
                "tx_anomaly_count",
                "tx_total_count",
                "tx_anomaly_ratio",
                "tx_anomaly_score_mean",
                "tx_anomaly_score_min",
                "tx_anomaly_score_max", 
            ]
        ].to_string(index=False)
    )

    # ---------------------------------------------------------
    # 11. Risk distribution
    # ---------------------------------------------------------

    print("\nRisk-level distribution:")

    print(
        df["risk_level"].value_counts()
    )

    # ---------------------------------------------------------
    # 12. Save results
    # ---------------------------------------------------------

    output_file = "output/hybrid_risk_results.csv"

    df.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nResults written to: {output_file}"
    )


if __name__ == "__main__":
    main()
