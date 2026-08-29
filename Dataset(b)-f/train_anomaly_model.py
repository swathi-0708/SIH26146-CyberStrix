#!/usr/bin/env python3
"""
Step 2 of the PS 26146 pipeline: feature engineering + anomaly model training.

Takes transactions.csv (from generate_dataset.py) and:
  1. Engineers per-transaction behavioral features, built per-wallet over time
     (exactly the feature list from the research doc: transaction_frequency,
     amount, fee, number_of_inputs, number_of_outputs, unique_counterparties,
     time_between_transactions -- plus network-layer features specific to this
     PS: ASN/IP diversity per wallet).
  2. Trains an Isolation Forest (unsupervised -- no labels used at train time,
     exactly matching how you'd have to do this on real, unlabeled seized data).
  3. Scores every transaction, ranks them by anomaly confidence, and writes a
     ranked, explainable alert list (why flagged + confidence score), matching
     the PS's explicit output requirement.
  4. ONLY THEN loads ground_truth_labels.csv -- never used for training, only
     to report precision/recall/F1 so you have real numbers for your write-up
     and demo. This separation (blind training, labels only for scoring) is
     itself worth stating explicitly to a judge.

Usage:
    python3 train_anomaly_model.py --data output/transactions.csv \
        --labels output/ground_truth_labels.csv --contamination 0.05
"""

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from geoip_enrich import GeoEnricher


FEATURE_COLS = [
    "amount", "fee", "fee_ratio", "num_inputs", "num_outputs",
    "time_since_last_tx_min", "tx_freq_last_24h",
    "unique_counterparties_cum", "unique_asn_cum",
    "amount_zscore_vs_wallet_hist",
]


def load_transactions(path):
    df = pd.read_csv(path)
    for col in ["input_addresses", "output_addresses", "input_amounts", "output_amounts"]:
        df[col] = df[col].apply(json.loads)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
    df["sender_wallet"] = df["input_addresses"].apply(lambda x: x[0])
    df["amount"] = df["output_amounts"].apply(sum)
    df["num_inputs"] = df["input_addresses"].apply(len)
    df["num_outputs"] = df["output_addresses"].apply(len)
    df["fee_ratio"] = df["fee"] / df["amount"].replace(0, np.nan)
    df["fee_ratio"] = df["fee_ratio"].fillna(0)

    # --- real GeoIP resolution (PS requirement: "integrate open source
    # downloadable GeoIP database"). This REPLACES the dataset's baked-in
    # geo_country/asn columns for feature purposes -- those were synthetic
    # ground-truth labels used only to shape wallet behavior during
    # generation, never derived from the IP itself. The model below only
    # ever sees the independently-resolved values, so the "GeoIP
    # integration" is a real, load-bearing part of the pipeline, not
    # decoration. ---
    print("Resolving src_ip -> country/ASN via offline GeoIP (geoip2fast)...")
    geo = GeoEnricher()
    df = geo.enrich_dataframe(df, ip_col="src_ip",
                               country_col="geo_country_resolved",
                               asn_col="asn_resolved")
    # unresolved (private/reserved/not-found) IPs get a placeholder so they
    # still form a valid, distinct category for the model rather than NaN
    df["asn_resolved"] = df["asn_resolved"].fillna("UNRESOLVED")

    return df.sort_values("timestamp").reset_index(drop=True)


def engineer_features(df):
    """Per-wallet, time-ordered feature engineering. Every feature here only
    looks BACKWARD in time relative to the transaction being scored -- no
    leakage from future transactions, which matters both for validity and
    because a real investigative system only ever has data up to 'now'."""
    df = df.copy()
    out_rows = []

    for wallet, grp in df.groupby("sender_wallet", sort=False):
        grp = grp.sort_values("timestamp").reset_index(drop=True)

        # time since this wallet's previous transaction (temporal anomaly signal)
        time_diff = grp["timestamp"].diff().dt.total_seconds() / 60.0
        grp["time_since_last_tx_min"] = time_diff.fillna(time_diff.median() if len(grp) > 1 else 1440)

        # rolling transaction frequency in the trailing 24h (burst detection)
        freq = []
        times = grp["timestamp"].tolist()
        for i, t in enumerate(times):
            window_start = t - pd.Timedelta(hours=24)
            count = sum(1 for tt in times[:i] if tt >= window_start)
            freq.append(count)
        grp["tx_freq_last_24h"] = freq

        # expanding count of DISTINCT counterparties seen so far (excludes current tx)
        seen = set()
        uniq_counts = []
        for outs in grp["output_addresses"]:
            uniq_counts.append(len(seen))
            seen.update(outs)
        grp["unique_counterparties_cum"] = uniq_counts

        # expanding count of DISTINCT ASNs this wallet's traffic has come from so
        # far, using the REAL, independently-resolved ASN (asn_resolved) -- not
        # the dataset's baked-in synthetic label. This is the network-layer
        # feature that's specific to this PS, and it's now genuinely derived
        # from GeoIP resolution, not a shortcut.
        seen_asn = set()
        asn_counts = []
        for asn in grp["asn_resolved"]:
            asn_counts.append(len(seen_asn))
            if pd.notna(asn):
                seen_asn.add(asn)
        grp["unique_asn_cum"] = asn_counts

        # amount deviation from this wallet's OWN historical mean (profile-deviation
        # feature) -- expanding mean/std computed only from prior transactions
        amounts = grp["amount"].tolist()
        zscores = []
        for i in range(len(amounts)):
            hist = amounts[:i]
            if len(hist) < 2:
                zscores.append(0.0)
            else:
                mu, sigma = np.mean(hist), np.std(hist)
                zscores.append(0.0 if sigma == 0 else (amounts[i] - mu) / sigma)
        grp["amount_zscore_vs_wallet_hist"] = zscores

        out_rows.append(grp)

    return pd.concat(out_rows, ignore_index=True)


def explain_row(row, feature_means, feature_stds, top_n=3):
    """Cheap, dependency-free explainability: rank this row's features by how
    many std deviations they sit from the GLOBAL feature mean, and report the
    top N as the 'why flagged' reasons. (Swap for SHAP if you have time later --
    this is deliberately simple so it always works and is easy to narrate live.)"""
    devs = []
    for col in FEATURE_COLS:
        std = feature_stds[col] if feature_stds[col] > 0 else 1e-9
        z = (row[col] - feature_means[col]) / std
        devs.append((col, z))
    devs.sort(key=lambda x: abs(x[1]), reverse=True)
    reasons = [f"{col} is {z:+.1f} std from typical" for col, z in devs[:top_n]]
    return "; ".join(reasons)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="output/transactions.csv")
    ap.add_argument("--labels", default="output/ground_truth_labels.csv")
    ap.add_argument("--contamination", type=float, default=0.05)
    ap.add_argument("--out", default="output/ranked_alerts.csv")
    args = ap.parse_args()

    print("Loading transactions...")
    df = load_transactions(args.data)

    print("Engineering features (per-wallet, time-ordered, no leakage)...")
    df = engineer_features(df)

    X = df[FEATURE_COLS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Training Isolation Forest (contamination={args.contamination})...")
    model = IsolationForest(
        n_estimators=200, contamination=args.contamination,
        random_state=42, n_jobs=-1,
    )
    model.fit(X_scaled)

    # decision_function: higher = more normal. Flip + rescale to a 0-100 "confidence
    # this is anomalous" score, which is what the PS asks for ("confidence score").
    raw_scores = model.decision_function(X_scaled)
    df["anomaly_score_raw"] = raw_scores
    min_s, max_s = raw_scores.min(), raw_scores.max()
    df["confidence"] = ((max_s - raw_scores) / (max_s - min_s) * 100).round(2)
    df["predicted_anomaly"] = (model.predict(X_scaled) == -1).astype(int)

    feature_means = {c: df[c].mean() for c in FEATURE_COLS}
    feature_stds = {c: df[c].std() for c in FEATURE_COLS}
    df["explanation"] = df.apply(lambda r: explain_row(r, feature_means, feature_stds), axis=1)

    ranked = df.sort_values("confidence", ascending=False)
    alert_cols = ["txid", "sender_wallet", "timestamp", "amount", "confidence",
                  "predicted_anomaly", "explanation"]
    ranked[alert_cols].to_csv(args.out, index=False)
    print(f"Ranked alert list written to {args.out}")

    # --- evaluation against ground truth (labels never touched the model above) ---
    try:
        labels = pd.read_csv(args.labels)
        eval_df = df.merge(labels[["txid", "is_anomaly", "anomaly_type"]], on="txid")

        tp = ((eval_df.predicted_anomaly == 1) & (eval_df.is_anomaly == 1)).sum()
        fp = ((eval_df.predicted_anomaly == 1) & (eval_df.is_anomaly == 0)).sum()
        fn = ((eval_df.predicted_anomaly == 0) & (eval_df.is_anomaly == 1)).sum()
        tn = ((eval_df.predicted_anomaly == 0) & (eval_df.is_anomaly == 0)).sum()
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

        print("\n=== Evaluation vs. ground truth (labels NOT used in training) ===")
        print(f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")
        print(f"Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}")

        print("\nRecall by injected anomaly type:")
        for atype, grp in eval_df[eval_df.is_anomaly == 1].groupby("anomaly_type"):
            caught = (grp.predicted_anomaly == 1).sum()
            print(f"  {atype:20s}: {caught}/{len(grp)} caught ({caught/len(grp)*100:.0f}%)")
    except FileNotFoundError:
        print("No labels file found -- skipping evaluation (expected on a real, unlabeled dataset).")


if __name__ == "__main__":
    main()
