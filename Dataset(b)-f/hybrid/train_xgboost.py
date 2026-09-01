import re
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score
)

from xgboost import XGBClassifier


FEATURES_FILE = "output/ml_features_with_tx_anomalies.csv"
GROUND_TRUTH_FILE = "output/ground_truth_labels.csv"
WALLETS_FILE = "output/wallets_reference.csv"


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
    # 3. Map wallet IDs → addresses
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
    # 4. Create target labels
    # ---------------------------------------------------------

    features_df["actual_anomaly"] = (
        features_df["wallet"]
        .isin(anomalous_addresses)
        .astype(int)
    )

    print(
        f"Total wallets: {len(features_df)}"
    )

    print(
        f"Normal wallets: "
        f"{(features_df['actual_anomaly'] == 0).sum()}"
    )

    print(
        f"Anomalous wallets: "
        f"{features_df['actual_anomaly'].sum()}"
    )

    # ---------------------------------------------------------
    # 5. Features
    # ---------------------------------------------------------

    feature_columns = [
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

    X = features_df[feature_columns].fillna(0)
    y = features_df["actual_anomaly"]

    # ---------------------------------------------------------
    # 6. Train/test split
    # ---------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=42
    )
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        stratify=y_train,
        random_state=42
    )

    print(
        f"\nTraining wallets: {len(X_train)}"
    )

    print(
        f"Testing wallets: {len(X_test)}"
    )

    print(
        f"Training anomalies: {y_train.sum()}"
    )

    print(
        f"Testing anomalies: {y_test.sum()}"
    )
    print(
    f"Fitting wallets: {len(X_fit)}"
    )

    print(
        f"Validation wallets: {len(X_val)}"
    )

    print(
        f"Fitting anomalies: {y_fit.sum()}"
    )

    print(
        f"Validation anomalies: {y_val.sum()}"
    )


    # ---------------------------------------------------------
    # 7. Handle class imbalance
    # ---------------------------------------------------------

    normal_count = (y_fit == 0).sum()
    anomaly_count = (y_fit == 1).sum()

    scale_pos_weight = (
        normal_count / anomaly_count
    )

    print(
        f"\nscale_pos_weight: "
        f"{scale_pos_weight:.2f}"
    )

    # ---------------------------------------------------------
    # 8. Train XGBoost
    # ---------------------------------------------------------

    model = XGBClassifier(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.75,
        colsample_bytree=0.75,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=2.0,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        early_stopping_rounds=30,
        random_state=42
    )

    model.fit(
        X_fit,
        y_fit,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # ---------------------------------------------------------
# Training vs Test performance
# ---------------------------------------------------------

    train_probabilities = model.predict_proba(X_fit)[:, 1]
    test_probabilities = model.predict_proba(X_test)[:, 1]

    train_predictions = (
        train_probabilities >= 0.5
    ).astype(int)

    test_predictions = (
        test_probabilities >= 0.5
    ).astype(int)

    print("\n=== TRAINING SET ===")

    print(
        classification_report(
            y_fit,
            train_predictions,
            target_names=["Normal", "Anomalous"],
            zero_division=0
        )
    )

    print(
        f"Train ROC-AUC: "
        f"{roc_auc_score(y_fit, train_probabilities):.3f}"
    )

    print(
        f"Train PR-AUC: "
        f"{average_precision_score(y_fit, train_probabilities):.3f}"
    )

    print("\n=== TEST SET ===")

    print(
        classification_report(
            y_test,
            test_predictions,
            target_names=["Normal", "Anomalous"],
            zero_division=0
        )
    )

    print(
        f"Test ROC-AUC: "
        f"{roc_auc_score(y_test, test_probabilities):.3f}"
    )

    print(
        f"Test PR-AUC: "
        f"{average_precision_score(y_test, test_probabilities):.3f}"
    )

    # ---------------------------------------------------------
    # 9. Predictions
    # ---------------------------------------------------------

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    # ---------------------------------------------------------
    # 10. Evaluation
    # ---------------------------------------------------------

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Normal",
                "Anomalous"
            ],
            zero_division=0
        )
    )

    print(
        f"ROC-AUC: "
        f"{roc_auc_score(y_test, probabilities):.3f}"
    )

    print(
        f"PR-AUC: "
        f"{average_precision_score(y_test, probabilities):.3f}"
    )

    # ---------------------------------------------------------
    # 11. Feature importance
    # ---------------------------------------------------------

    importance = pd.DataFrame({
        "feature": feature_columns,
        "importance": model.feature_importances_
    }).sort_values(
        "importance",
        ascending=False
    )

    print("\nFeature importance:")

    print(
        importance.to_string(index=False)
    )

    # ---------------------------------------------------------
    # 12. Save model
    # ---------------------------------------------------------

    import joblib

    output_file = "output/xgboost_wallet_model.pkl"

    joblib.dump(
        model,
        output_file
    )

    print(
        f"\nModel saved to: {output_file}"
    )


if __name__ == "__main__":
    main()
