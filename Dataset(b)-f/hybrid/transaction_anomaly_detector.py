import pandas as pd
import json
from sklearn.ensemble import IsolationForest


# -------------------------
# 1. Load and prepare data
# -------------------------

df = pd.read_csv("output/transactions.csv")

for col in [
    "input_addresses",
    "output_addresses",
    "input_amounts",
    "output_amounts"
]:
    df[col] = df[col].apply(json.loads)

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    format="ISO8601"
)


# -------------------------
# 2. Feature engineering
# -------------------------

df["amount"] = df["output_amounts"].apply(sum)

df["fee_ratio"] = df["fee"] / df["amount"]

df["num_inputs"] = df["input_addresses"].apply(len)

df["num_outputs"] = df["output_addresses"].apply(len)

df["sender_wallet"] = df["input_addresses"].apply(
    lambda x: x[0]
)

df = df.sort_values(
    ["sender_wallet", "timestamp"]
)

df["time_since_last_tx_min"] = (
    df.groupby("sender_wallet")["timestamp"]
      .diff()
      .dt.total_seconds()
      .div(60)
)


# Transaction frequency in previous 24 heures

df["tx_freq_last_24h"] = 0

for wallet, group in df.groupby("sender_wallet"):

    times = group["timestamp"].tolist()

    for i, current_time in zip(group.index, times):

        count = sum(
            current_time - pd.Timedelta(hours=24)
            <= previous_time
            < current_time
            for previous_time in times
        )

        df.loc[i, "tx_freq_last_24h"] = count


# -------------------------
# 3. Isolation Forest
# -------------------------

features = [
    "amount",
    "fee_ratio",
    "num_inputs",
    "num_outputs",
    "time_since_last_tx_min",
    "tx_freq_last_24h"
]

X = df[features].copy()

X["time_since_last_tx_min"] = (
    X["time_since_last_tx_min"]
    .fillna(X["time_since_last_tx_min"].median())
)

model = IsolationForest(
    n_estimators=200,
    contamination=0.02,
    random_state=42
)

model.fit(X)

df["anomaly"] = model.predict(X)

df["anomaly_score"] = model.decision_function(X)


# -------------------------
# 4. Explain anomalies
# -------------------------

def explain_transaction(row, df):

    reasons = []

    if row["amount"] >= df["amount"].quantile(0.95):
        reasons.append(
            "Unusually high transaction amount"
        )

    if row["fee_ratio"] >= df["fee_ratio"].quantile(0.95):
        reasons.append(
            "Unusually high fee ratio"
        )

    if row["num_inputs"] >= df["num_inputs"].quantile(0.95):
        reasons.append(
            "Unusually many inputs"
        )

    if row["num_outputs"] >= df["num_outputs"].quantile(0.95):
        reasons.append(
            "Unusually many outputs"
        )

    if row["time_since_last_tx_min"] >= df[
        "time_since_last_tx_min"
    ].quantile(0.95):
        reasons.append(
            "Unusually long inactivity period"
        )

    if row["tx_freq_last_24h"] >= df[
        "tx_freq_last_24h"
    ].quantile(0.95):
        reasons.append(
            "Unusually high transaction frequency"
        )

    return reasons


df["reasons"] = df.apply(
    lambda row: explain_transaction(row, df),
    axis=1
)


# -------------------------
# 5. Generate alerts
# -------------------------

alerts = df[df["anomaly"] == -1].copy()

alerts = alerts.sort_values(
    "anomaly_score"
)

alerts = alerts[
    [
        "txid",
        "sender_wallet",
        "amount",
        "fee_ratio",
        "num_inputs",
        "num_outputs",
        "time_since_last_tx_min",
        "tx_freq_last_24h",
        "anomaly_score",
        "reasons"
    ]
]

alerts.head(20).to_string(index=False)

alerts.to_csv(
    "output/investigative_alerts.csv",
    index=False
)

print(alerts.head(20).to_string(index=False))

alerts.to_csv(
    "investigative_alerts.csv",
    index=False
)
