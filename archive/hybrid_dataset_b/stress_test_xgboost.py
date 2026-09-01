import re
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score

from xgboost import XGBClassifier


FEATURES_FILE = "output/ml_features_with_tx_anomalies.csv"
GROUND_TRUTH_FILE = "output/ground_truth_labels.csv"
WALLETS_FILE = "output/wallets_reference.csv"


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
    # 1. Load data
    # ---------------------------------------------------------

    features_df = pd.read_csv(FEATURES_FILE)
    ground_truth_df = pd.read_csv(GROUND_TRUTH_FILE)
    wallets_df = pd.read_csv(WALLETS_FILE)

    # ---------------------------------------------------------
    # 2. Find anomalous wallet IDs
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
    # 3. Map wallet IDs -> addresses
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
    # 4. Create wallet labels
    # ---------------------------------------------------------

    features_df["actual_anomaly"] = (
        features_df["wallet"]
        .isin(anomalous_addresses)
        .astype(int)
    )

    # ---------------------------------------------------------
    # 5. Build wallet -> anomaly types mapping
    # ---------------------------------------------------------

    anomalous_labels["wallet"] = (
        anomalous_labels["wallet_id"]
        .map(id_to_address)
    )

    wallet_types = (
        anomalous_labels
        .dropna(subset=["wallet"])
        .groupby("wallet")["anomaly_type"]
        .apply(lambda x: set(x))
        .to_dict()
    )

    # ---------------------------------------------------------
    # 6. Prepare ML data
    # ---------------------------------------------------------

    X = features_df[FEATURE_COLUMNS].fillna(0)
    y = features_df["actual_anomaly"]

    print("Wallets:", len(X))
    print("Normal:", (y == 0).sum())
    print("Anomalous:", (y == 1).sum())

    # ---------------------------------------------------------
    # 7. Cross-validation
    # ---------------------------------------------------------

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    normal_count = (y == 0).sum()
    anomaly_count = (y == 1).sum()

    scale_pos_weight = normal_count / anomaly_count

    print(
        f"\nscale_pos_weight: "
        f"{scale_pos_weight:.2f}"
    )

    fold_precision = []
    fold_recall = []
    fold_f1 = []

    # Track anomaly-type performance
    anomaly_types = [
        "peeling_chain",
        "structuring",
        "wallet_reuse_burst",
        "ip_hopping",
        "fan_out_mixer",
        "profile_deviation"
    ]

    type_total = {
        anomaly_type: 0
        for anomaly_type in anomaly_types
    }

    type_caught = {
        anomaly_type: 0
        for anomaly_type in anomaly_types
    }

    print("\nRunning 5-fold cross-validation...\n")

    # ---------------------------------------------------------
    # 8. Train each fold
    # ---------------------------------------------------------

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y),
        start=1
    ):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42
    )

        model.fit(
            X_train,
            y_train
        )

        probabilities = model.predict_proba(
            X_test
        )[:, 1]

        predictions = (
            probabilities >= 0.5
        ).astype(int)

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        fold_precision.append(precision)
        fold_recall.append(recall)
        fold_f1.append(f1)

        print(
            f"Fold {fold}: "
            f"Precision={precision:.3f}, "
            f"Recall={recall:.3f}, "
            f"F1={f1:.3f}"
        )

        # -----------------------------------------------------
        # Anomaly-type recall for this fold
        # -----------------------------------------------------

        test_rows = features_df.iloc[test_idx]

        for row_index, prediction in zip(
            test_rows.index,
            predictions
        ):

            wallet = features_df.loc[
                row_index,
                "wallet"
            ]

            types = wallet_types.get(
                wallet,
                set()
            )

            if not types:
                continue

            for anomaly_type in types:

                if anomaly_type not in anomaly_types:
                    continue

                type_total[anomaly_type] += 1

                if prediction == 1:
                    type_caught[anomaly_type] += 1

    # ---------------------------------------------------------
    # 9. Overall CV summary
    # ---------------------------------------------------------

    print("\n=== Cross-validation summary ===")

    print(
        f"Precision: "
        f"{np.mean(fold_precision):.3f} "
        f"+/- "
        f"{np.std(fold_precision):.3f}"
    )

    print(
        f"Recall: "
        f"{np.mean(fold_recall):.3f} "
        f"+/- "
        f"{np.std(fold_recall):.3f}"
    )

    print(
        f"F1: "
        f"{np.mean(fold_f1):.3f} "
        f"+/- "
        f"{np.std(fold_f1):.3f}"
    )

    # ---------------------------------------------------------
    # 10. Overall anomaly-type recall
    # ---------------------------------------------------------

    print("\n=== Recall by anomaly type across all folds ===")

    for anomaly_type in anomaly_types:

        total = type_total[anomaly_type]
        caught = type_caught[anomaly_type]

        if total == 0:
            print(
                f"{anomaly_type}: no wallets found"
            )
            continue

        recall = caught / total

        print(
            f"{anomaly_type}: "
            f"{caught}/{total} caught "
            f"({recall * 100:.1f}%)"
        )


if __name__ == "__main__":
    main()
