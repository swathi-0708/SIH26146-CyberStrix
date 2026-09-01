import pandas as pd


TRANSACTION_FILE = "output/transaction_anomaly_scores.csv"
WALLET_FEATURES_FILE = "output/ml_features.csv"
OUTPUT_FILE = "output/ml_features_with_tx_anomalies.csv"


def main():

    # ---------------------------------------------------------
    # 1. Load files
    # ---------------------------------------------------------

    tx_df = pd.read_csv(TRANSACTION_FILE)
    wallet_df = pd.read_csv(WALLET_FEATURES_FILE)

    print(f"Transactions loaded: {len(tx_df)}")
    print(f"Wallets loaded: {len(wallet_df)}")

    # ---------------------------------------------------------
    # 2. Convert Isolation Forest output into anomaly flag
    # ---------------------------------------------------------

    # Her detector uses:
    # -1 = anomalous
    #  1 = normal

    tx_df["tx_is_anomaly"] = (
        tx_df["anomaly"] == -1
    ).astype(int)

    # ---------------------------------------------------------
    # 3. Aggregate transaction anomalies per wallet
    # ---------------------------------------------------------

    wallet_tx = (
        tx_df
        .groupby("sender_wallet")
        .agg(
            tx_anomaly_count=(
                "tx_is_anomaly",
                "sum"
            ),

            tx_total_count=(
                "txid",
                "count"
            ),

            tx_anomaly_score_mean=(
                "anomaly_score",
                "mean"
            ),

            tx_anomaly_score_min=(
                "anomaly_score",
                "min"
            ),

            tx_anomaly_score_max=(
                "anomaly_score",
                "max"
            )
        )
        .reset_index()
    )

    # ---------------------------------------------------------
    # 4. Calculate anomaly ratio
    # ---------------------------------------------------------

    wallet_tx["tx_anomaly_ratio"] = (
        wallet_tx["tx_anomaly_count"]
        / wallet_tx["tx_total_count"]
    )

    # ---------------------------------------------------------
    # 5. Rename wallet column for merging
    # ---------------------------------------------------------

    wallet_tx = wallet_tx.rename(
        columns={
            "sender_wallet": "wallet"
        }
    )

    # ---------------------------------------------------------
    # 6. Merge with your wallet-level features
    # ---------------------------------------------------------

    merged_df = wallet_df.merge(
        wallet_tx,
        on="wallet",
        how="left"
    )

    # ---------------------------------------------------------
    # 7. Fill wallets with no detected transactions
    # ---------------------------------------------------------

    count_columns = [
        "tx_anomaly_count",
        "tx_total_count"
    ]

    score_columns = [
        "tx_anomaly_score_mean",
        "tx_anomaly_score_min",
        "tx_anomaly_score_max",
        "tx_anomaly_ratio"
    ]

    for col in count_columns:
        merged_df[col] = merged_df[col].fillna(0)

    for col in score_columns:
        merged_df[col] = merged_df[col].fillna(0)

    # ---------------------------------------------------------
    # 8. Save combined wallet features
    # ---------------------------------------------------------

    merged_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ---------------------------------------------------------
    # 9. Display results
    # ---------------------------------------------------------

    print(
        f"\nWallets with transaction anomalies: "
        f"{(merged_df['tx_anomaly_count'] > 0).sum()}"
    )

    print(
        f"Total transaction anomalies: "
        f"{int(merged_df['tx_anomaly_count'].sum())}"
    )

    print("\nTop wallets by transaction anomaly count:")

    print(
        merged_df[
            [
                "wallet",
                "tx_anomaly_count",
                "tx_total_count",
                "tx_anomaly_ratio",
                "tx_anomaly_score_mean",
                "tx_anomaly_score_min",
                "tx_anomaly_score_max"
            ]
        ]
        .sort_values(
            "tx_anomaly_count",
            ascending=False
        )
        .head(20)
        .to_string(index=False)
    )

    print(
        f"\nCombined features written to: "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
