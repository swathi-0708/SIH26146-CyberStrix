"""
Final Investigative Confidence
================================

Combines every evidence type this pipeline separately computes into ONE
number an investigator can act on, instead of leaving them scattered as
disconnected numbers across output/correlation_scores.csv,
output/xgboost_test_scores.csv, output/isolation_forest_scores.csv, and
output/train.csv / output/test.csv:

    investigative_confidence = ml_evidence        (ML anomaly score)
                              + network_evidence   (network correlation)
                              + temporal_evidence  (temporal correlation)
                              + wallet_evidence    (wallet evidence)
                              + amount_evidence    (amount evidence)
                              + graph_evidence      (graph evidence)

Same additive, transparent, rule-based philosophy as correlation_score.py:
each term is in [0, 1] (so investigative_confidence is in [0, 6];
investigative_confidence_pct rescales to [0, 1]), nothing here is fit,
trained, or thresholded against labels, and every term is a documented
ratio/count/value already sitting in this pipeline's own state.

Four of the six terms are NOT recomputed here -- read straight from
correlation_score.py's own output, unchanged:

  network_evidence, temporal_evidence, wallet_evidence  <- correlation_scores.csv
  graph_evidence  <- correlation_scores.csv's blockchain_evidence, renamed to
                      match the PS's six-box diagram. Still the same
                      common-input / likely-change-address on-chain
                      structural edge count, still deliberately excluding
                      shared_src_ip edges (see correlation_score.py's own
                      docstring for why: counting them here too would let
                      graph_evidence just echo network_evidence back at
                      itself).

The two NEW terms this script adds:

  ml_evidence -- combines XGBoost's y_proba (supervised, test-split only --
                 see train_compare.py) and Isolation Forest's
                 decision_function (unsupervised, blind, every row -- see
                 train_isolation_forest.py) into one [0, 1] number.
                 Deliberately NOT an average of the two raw scores --
                 build_alerts.py already established why not (xgb_proba is a
                 calibrated probability, if_score is an unbounded,
                 differently-scaled decision_function; "NEVER averaged or
                 combined into one number" per that script's own docstring).
                 Instead: max(xgb_proba, if_evidence), where if_evidence is
                 if_score min-max normalized across this dataset and
                 inverted (lower if_score = more anomalous, so smaller raw
                 values map to evidence near 1). This mirrors
                 build_alerts.py's OWN existing agreement logic
                 (detector_tag: "both" beats either detector alone, i.e. an
                 OR across detectors, not a blend of their raw numbers) --
                 just expressed as a continuous number instead of a
                 discrete tier. XGBoost only ever scores its held-out test
                 split, by design, forever (scoring a model on the rows it
                 was trained on would be meaningless) -- so most rows will
                 only have if_evidence. `ml_source` records which
                 detector(s) actually contributed, so a low ml_evidence
                 from "no test-split coverage" is never mistaken for
                 "ML actively confirmed this is clean".

  amount_evidence -- does the MONEY ITSELF look anomalous, independent of
                 network/timing/wallet/graph data entirely?
                 = avg(zscore_evidence, skew_evidence, ratio_evidence)

                 zscore_evidence = min(1, |sender_amount_zscore| / 3)
                   how many std-devs this tx's amount sits from this
                   sender's OWN causal running history (already computed,
                   causal-by-construction, by split_dataset.py -- see that
                   script's docstring on why it's safe to reuse as-is here).

                 skew_evidence = 1 - output_min_max_ratio
                   output_min_max_ratio = min(outputs)/max(outputs) is in
                   (0, 1]; near 0 means one dominant output plus a tiny
                   remainder -- the classic peeling-chain / change-address
                   shape -- so 1 minus that ratio is high exactly when that
                   shape is present.

                 ratio_evidence = min(1, |input_output_ratio - 1|)
                   input_output_ratio (total_input_btc / total_output_btc)
                   should sit close to 1.0 for an ordinary transaction
                   (inputs ~= outputs + fee); distance from 1.0 is itself
                   the evidence.

                 fee_ratio is deliberately NOT used here: check.py's own
                 history found it was the exact feature an earlier version
                 of this dataset leaked labels through (see split_dataset.py's
                 comment on fee_ratio). Leaving it out of a rule-based score
                 an investigator reads as ground truth is the conservative
                 choice.

Depends on (run first if missing):
  - output/correlation_scores.csv         (correlation_score.py)
  - output/train.csv, output/test.csv     (split_dataset.py) -- amount_evidence, tx_amount_btc
  - output/isolation_forest_scores.csv    (train_isolation_forest.py) -- ml_evidence, all rows

Optional (graceful, not required):
  - output/xgboost_test_scores.csv        (train_compare.py) -- ml_evidence
    refinement, test-split rows only. If missing, ml_evidence falls back to
    if_evidence alone for every row (never a hard failure -- XGBoost
    coverage being partial is expected, not an error state).

Output: output/investigative_confidence.csv, one row per transaction,
carrying the chain (src_ip, txid, input_wallet, output_wallets, entity_id)
over from correlation_scores.csv, all 6 component scores, tx_amount_btc,
investigative_confidence (sum, [0, 6]), investigative_confidence_pct
([0, 1]), confidence_level (Low / Moderate / High / Critical, thresholded on
the pct), and an `explanation` string. This is also the file
multi_hop_investigation.py now reads (CORR_SCORES_PATH there points here,
not at correlation_scores.csv, which was never meant to carry
investigative_confidence / confidence_level / tx_amount_btc -- those belong
to this broader, six-term artifact, not the narrower four-term IP<->wallet
score).

Usage:
    python3 entity_clustering.py
    python3 correlation_score.py
    python3 ../split_dataset.py            # amount_evidence + tx_amount_btc
    python3 ../train_isolation_forest.py   # ml_evidence (all rows)
    python3 ../train_compare.py            # ml_evidence refinement (test rows, optional)
    python3 investigative_confidence.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

CORR_SCORES_PATH = "output/correlation_scores.csv"
TRAIN_PATH = "output/train.csv"
TEST_PATH = "output/test.csv"
IF_SCORES_PATH = "output/isolation_forest_scores.csv"
XGB_SCORES_PATH = "output/xgboost_test_scores.csv"  # optional
OUT_PATH = "output/investigative_confidence.csv"

AMOUNT_COLS = ["txid", "total_input_btc", "output_min_max_ratio", "input_output_ratio", "sender_amount_zscore"]

LEVEL_THRESHOLDS = [  # (min_pct, label) -- checked high to low
    (0.85, "Critical"),
    (0.65, "High"),
    (0.40, "Moderate"),
    (0.0, "Low"),
]


def _require_inputs():
    required = [CORR_SCORES_PATH, TRAIN_PATH, TEST_PATH, IF_SCORES_PATH]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "investigative_confidence.py needs correlation_score.py's, "
            "split_dataset.py's, and train_isolation_forest.py's output first "
            f"(missing: {missing}). Run those and retry -- amount_evidence and "
            "ml_evidence have no honest value without them."
        )


def _confidence_level(pct: float) -> str:
    for threshold, label in LEVEL_THRESHOLDS:
        if pct >= threshold:
            return label
    return "Low"  # unreachable given the 0.0 floor above, kept for safety


def _load_amount_evidence() -> pd.DataFrame:
    """One row per txid, ALL rows (train+test concatenated) -- amount
    features are computed over every transaction by split_dataset.py
    regardless of which split it later lands in."""
    train_df = pd.read_csv(TRAIN_PATH, usecols=AMOUNT_COLS)
    test_df = pd.read_csv(TEST_PATH, usecols=AMOUNT_COLS)
    amounts = pd.concat([train_df, test_df], ignore_index=True)

    zscore_evidence = (amounts["sender_amount_zscore"].abs() / 3).clip(upper=1.0)
    skew_evidence = (1.0 - amounts["output_min_max_ratio"]).clip(lower=0.0, upper=1.0)
    ratio_evidence = (amounts["input_output_ratio"] - 1.0).abs().clip(upper=1.0)

    amounts["amount_evidence"] = (
        (zscore_evidence + skew_evidence + ratio_evidence) / 3
    ).round(4)
    amounts["tx_amount_btc"] = amounts["total_input_btc"]
    return amounts[["txid", "amount_evidence", "tx_amount_btc"]]


def _load_ml_evidence() -> pd.DataFrame:
    """One row per txid, ALL rows -- IF covers everything; XGBoost (if
    present) only refines the test-split subset. ml_source records which
    detector(s) actually informed each row's ml_evidence."""
    iso = pd.read_csv(IF_SCORES_PATH, usecols=["txid", "if_score"])

    lo, hi = iso["if_score"].min(), iso["if_score"].max()
    span = hi - lo
    iso["if_evidence"] = (
        ((hi - iso["if_score"]) / span).clip(lower=0.0, upper=1.0)
        if span > 1e-12
        else 0.5  # every row equally (un)anomalous by this detector -- no signal either way
    )

    if os.path.exists(XGB_SCORES_PATH):
        xgb = pd.read_csv(XGB_SCORES_PATH, usecols=["txid", "xgb_proba"])
        merged = iso.merge(xgb, on="txid", how="left")
        has_xgb = merged["xgb_proba"].notna()
        merged["ml_evidence"] = np.where(
            has_xgb,
            np.maximum(merged["xgb_proba"], merged["if_evidence"]),
            merged["if_evidence"],
        )
        merged["ml_source"] = np.where(has_xgb, "xgboost+isolation_forest", "isolation_forest_only")
    else:
        merged = iso.copy()
        merged["ml_evidence"] = merged["if_evidence"]
        merged["ml_source"] = "isolation_forest_only"

    merged["ml_evidence"] = merged["ml_evidence"].round(4)
    return merged[["txid", "ml_evidence", "ml_source"]]


def _explain(row) -> str:
    def level(x):
        return "Strong" if x >= 0.66 else ("Moderate" if x >= 0.33 else "Weak")

    parts = [row["explanation"]]  # the IP -> TXID -> wallet -> wallet chain + its 4 evidence terms, from correlation_score.py
    ml_note = "no XGBoost coverage (outside test split)" if row["ml_source"] == "isolation_forest_only" else "XGBoost + Isolation Forest agree"
    parts.append(f"{level(row['ml_evidence'])} ML anomaly evidence ({ml_note}, ml_evidence={row['ml_evidence']:.2f})")
    parts.append(f"{level(row['amount_evidence'])} amount-shape evidence (tx_amount_btc={row['tx_amount_btc']:.8f}, amount_evidence={row['amount_evidence']:.2f})")
    parts.append(f"FINAL INVESTIGATIVE CONFIDENCE: {row['investigative_confidence']:.2f}/6.00 ({row['confidence_level']})")
    return " | ".join(parts)


def compute_investigative_confidence() -> pd.DataFrame:
    _require_inputs()

    scores = pd.read_csv(CORR_SCORES_PATH)
    scores = scores.rename(columns={"blockchain_evidence": "graph_evidence"})

    amount = _load_amount_evidence()
    ml = _load_ml_evidence()

    df = scores.merge(amount, on="txid", how="left").merge(ml, on="txid", how="left")

    missing_amount = df["amount_evidence"].isna().sum()
    missing_ml = df["ml_evidence"].isna().sum()
    if missing_amount or missing_ml:
        print(
            f"warning: {missing_amount} rows missing amount_evidence, {missing_ml} rows missing "
            "ml_evidence (txid not found in train/test.csv or isolation_forest_scores.csv) -- dropping them"
        )
        df = df.dropna(subset=["amount_evidence", "ml_evidence"]).reset_index(drop=True)

    df["investigative_confidence"] = (
        df["ml_evidence"] + df["network_evidence"] + df["temporal_evidence"]
        + df["wallet_evidence"] + df["amount_evidence"] + df["graph_evidence"]
    ).round(4)
    df["investigative_confidence_pct"] = (df["investigative_confidence"] / 6).round(4)
    df["confidence_level"] = df["investigative_confidence_pct"].apply(_confidence_level)
    df["explanation"] = df.apply(_explain, axis=1)

    return df[[
        "src_ip", "txid", "timestamp", "input_wallet", "entity_id", "output_wallets",
        "tx_amount_btc",
        "ml_evidence", "ml_source",
        "network_evidence", "temporal_evidence", "wallet_evidence",
        "amount_evidence", "graph_evidence",
        "investigative_confidence", "investigative_confidence_pct", "confidence_level",
        "explanation",
    ]]


if __name__ == "__main__":
    df = compute_investigative_confidence()
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH} ({len(df)} rows)")
    print(f"\nConfidence level distribution:\n{df['confidence_level'].value_counts().to_string()}")
    print("\n--- 3 highest-confidence investigations ---")
    for _, row in df.sort_values("investigative_confidence", ascending=False).head(3).iterrows():
        print(f"  {row['txid']}  {row['investigative_confidence']:.2f}/6.00 ({row['confidence_level']})")
    print("\n--- 3 lowest-confidence investigations ---")
    for _, row in df.sort_values("investigative_confidence", ascending=True).head(3).iterrows():
        print(f"  {row['txid']}  {row['investigative_confidence']:.2f}/6.00 ({row['confidence_level']})")
