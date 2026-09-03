"""
Explicit Network<->Blockchain Correlation Score
=================================================

Answers the exact question the PS asks for: "why do we believe this network
observation (an IP/port/ASN/country flow) is connected to this blockchain
entity (wallet)?" -- as a transparent, additive, rule-based score, NOT
another ML detector. Nothing here is fit, trained, or thresholded against
labels; every number is a plain ratio or count already sitting in this
pipeline's own state, combined with fixed, documented arithmetic.

    correlation_score = network_evidence + temporal_evidence
                       + wallet_evidence  + blockchain_evidence

Each of the four terms is in [0, 1], so correlation_score is in [0, 4]
(correlation_score_pct rescales to [0, 1] for convenience). Each term
answers a different half of "why":

  network_evidence   (IP-side)  -- does THIS IP look like it belongs almost
                                    exclusively to THIS entity, or is it
                                    shared across many unrelated entities
                                    (e.g. a hosting-provider ASN)?
                                    = avg(1/wallets_per_ip, ip_wallet_pair_count/tx_count_per_ip)

  wallet_evidence    (entity-side) -- does THIS entity look like it
                                    consistently operates from one
                                    identifiable IP, or does its network
                                    fingerprint keep changing?
                                    = avg(1/ips_per_wallet, 1 - wallet_churn)

  temporal_evidence  (timing)   -- is this observation part of a tight,
                                    ongoing burst of activity from this IP
                                    (strong live-session evidence), or a
                                    one-off separated by a long gap (weak)?
                                    = 1.0 if is_burst, else a linear decay
                                    of time_since_prev_tx_ip over 1h, else
                                    0.3 baseline if this IP has no prior
                                    history at all (neither confirms nor
                                    denies).

  blockchain_evidence (on-chain) -- independent of any network/IP data:
                                    does the blockchain graph ITSELF
                                    (common-input-ownership,
                                    likely-change-address -- see
                                    entity_clustering.py) already link this
                                    address to other addresses? Deliberately
                                    EXCLUDES entity_clustering's own
                                    shared_src_ip heuristic edges here, so
                                    this term never just echoes
                                    network_evidence back at itself.
                                    = min(1.0, n_qualifying_edges / 3)

Design note -- why this is NOT causal/leak-safe like split_dataset.py's
features, and why that's correct here: split_dataset.py's sender_* columns
feed a model that must never see the future relative to the row it's
scoring (training leakage). This script is the opposite kind of artifact --
a post-hoc INVESTIGATION score an analyst reads after the fact, using the
whole case file (the full entity_clusters.csv resolution, built from every
transaction in the dataset). Full-history context is appropriate and
desirable here, not a leak. `wallets_per_ip` / `ips_per_wallet` / churn
still accumulate chronologically as the FeatureEngine below processes
transactions in timestamp order (so "as of this observation, how much prior
evidence had built up" is preserved), but entity resolution itself is not
artificially restricted to the past.

Depends on (run first if missing):
  - output/transactions.csv        (generate_dataset.py)
  - output/entity_clusters.csv      (entity_clustering.py) -- address -> entity_id
  - output/entity_clusters_evidence.json (entity_clustering.py) -- why each
    entity was formed; this script filters OUT shared_src_ip edges from it
    (see blockchain_evidence above)

If entity_clusters.csv / entity_clusters_evidence.json aren't there yet,
this script says so and exits rather than silently scoring blockchain_evidence
as a fake zero.

Output: output/correlation_scores.csv, one row per transaction, with the 4
component scores, the summed correlation_score, and a plain-English
`explanation` string -- so every score is traceable back to "why", not just
a number.

Usage:
    python3 entity_clustering.py     # if entity_clusters*.{csv,json} missing
    python3 correlation_score.py
"""

from __future__ import annotations

import json
import os

import pandas as pd

from ingest import load_transactions
from ENGINE_network_blockchain_correlation import FeatureEngine

WINDOW_1H = 60 * 60

TX_PATH = "output/transactions.csv"
ENTITY_MEMBERSHIP_PATH = "output/entity_clusters.csv"
ENTITY_EVIDENCE_PATH = "output/entity_clusters_evidence.json"
OUT_PATH = "output/correlation_scores.csv"


def _require_entity_clustering_outputs():
    missing = [p for p in (ENTITY_MEMBERSHIP_PATH, ENTITY_EVIDENCE_PATH) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "correlation_score.py needs entity_clustering.py's output first "
            f"(missing: {missing}). Run `python3 entity_clustering.py` and retry -- "
            "blockchain_evidence has no honest value without it."
        )


def _load_qualifying_blockchain_edges(evidence_path: str) -> dict[str, int]:
    """
    Per entity_id, count DISTINCT edges from entity_clusters_evidence.json
    whose reason is NOT shared_src_ip. shared_src_ip is itself a network-layer
    heuristic (same IP broadcasting for two addresses) -- counting it here
    too would let network_evidence and blockchain_evidence double-count the
    same underlying fact. Only common_input and likely_change_address (pure
    on-chain structural heuristics) count as independent blockchain evidence.
    """
    with open(evidence_path) as f:
        evidence_by_entity = json.load(f)

    counts: dict[str, int] = {}
    for entity_id, edges in evidence_by_entity.items():
        qualifying = {
            (e["from"], e["to"], e["reason"])
            for e in edges
            if e["reason"] != "shared_src_ip"
        }
        counts[entity_id] = len(qualifying)
    return counts


def _temporal_evidence(is_burst: bool, time_since_prev_tx_ip) -> float:
    if is_burst:
        return 1.0
    if time_since_prev_tx_ip is None:
        return 0.3  # first-ever sighting of this IP: neither confirms nor denies
    return round(max(0.0, 1.0 - min(time_since_prev_tx_ip, WINDOW_1H) / WINDOW_1H), 4)


def _explain(net_ev, wal_ev, tmp_ev, chn_ev, wallets_per_ip, ips_per_wallet,
             pair_share, wallet_churn, is_burst, time_since_prev_tx_ip, n_chain_edges) -> str:
    def level(x):
        return "Strong" if x >= 0.66 else ("Moderate" if x >= 0.33 else "Weak")

    parts = []
    parts.append(
        f"{level(net_ev)} network binding (IP shared by {wallets_per_ip} entit"
        f"{'y' if wallets_per_ip == 1 else 'ies'}, {pair_share:.0%} of its traffic is this entity)"
    )
    parts.append(
        f"{level(wal_ev)} wallet-side stability ({ips_per_wallet} distinct IP"
        f"{'s' if ips_per_wallet != 1 else ''} used, churn={wallet_churn:.2f})"
    )
    if is_burst:
        parts.append("Strong temporal alignment (part of an active burst from this IP)")
    elif time_since_prev_tx_ip is None:
        parts.append("Weak temporal alignment (first-ever observation of this IP, no history)")
    else:
        parts.append(f"{level(tmp_ev)} temporal alignment ({time_since_prev_tx_ip/60:.1f} min since this IP's last activity)")
    if n_chain_edges > 0:
        parts.append(f"{level(chn_ev)} blockchain-layer corroboration ({n_chain_edges} on-chain structural link(s), independent of IP data)")
    else:
        parts.append("No independent blockchain-layer corroboration (singleton, or only linked via shared IP)")
    return "; ".join(parts) + "."


def compute_correlation_scores(tx_path: str = TX_PATH) -> pd.DataFrame:
    _require_entity_clustering_outputs()

    raw = load_transactions(tx_path)
    raw["timestamp_dt"] = pd.to_datetime(raw["timestamp"], format="mixed", utc=True)
    raw["first_input_addr"] = raw["input_addresses"].apply(lambda x: x[0])
    raw = raw.sort_values("timestamp_dt").reset_index(drop=True)

    membership = pd.read_csv(ENTITY_MEMBERSHIP_PATH)
    addr_to_entity = dict(zip(membership["address"], membership["entity_id"]))
    raw["entity_id"] = raw["first_input_addr"].map(addr_to_entity)
    unmapped = raw["entity_id"].isna().sum()
    if unmapped:
        print(f"warning: {unmapped} rows have no entity_id (address not in {ENTITY_MEMBERSHIP_PATH}) -- dropping them")
        raw = raw.dropna(subset=["entity_id"]).reset_index(drop=True)

    chain_edge_counts = _load_qualifying_blockchain_edges(ENTITY_EVIDENCE_PATH)

    engine = FeatureEngine()
    rows = []
    for _, tx in raw.iterrows():
        ip = tx["src_ip"]
        entity_id = tx["entity_id"]
        feats = engine.process(tx_id=tx["txid"], ip=ip, wallet=entity_id, ts=tx["timestamp_dt"].timestamp())

        ip_exclusivity = 1.0 / feats.wallets_per_ip if feats.wallets_per_ip else 0.0
        pair_share = (feats.ip_wallet_pair_count / feats.tx_count_per_ip) if feats.tx_count_per_ip else 0.0
        network_evidence = round((ip_exclusivity + pair_share) / 2, 4)

        wallet_exclusivity = 1.0 / feats.ips_per_wallet if feats.ips_per_wallet else 0.0
        stability = 1.0 - feats.wallet_churn
        wallet_evidence = round((wallet_exclusivity + stability) / 2, 4)

        temporal_evidence = _temporal_evidence(feats.is_burst, feats.time_since_prev_tx_ip)

        n_chain_edges = chain_edge_counts.get(entity_id, 0)
        blockchain_evidence = round(min(1.0, n_chain_edges / 3), 4)

        correlation_score = round(network_evidence + temporal_evidence + wallet_evidence + blockchain_evidence, 4)

        explanation = _explain(
            network_evidence, wallet_evidence, temporal_evidence, blockchain_evidence,
            feats.wallets_per_ip, feats.ips_per_wallet, pair_share, feats.wallet_churn,
            feats.is_burst, feats.time_since_prev_tx_ip, n_chain_edges,
        )

        rows.append({
            "txid": tx["txid"],
            "timestamp": tx["timestamp"],
            "src_ip": ip,
            "entity_id": entity_id,
            "network_evidence": network_evidence,
            "temporal_evidence": temporal_evidence,
            "wallet_evidence": wallet_evidence,
            "blockchain_evidence": blockchain_evidence,
            "correlation_score": correlation_score,
            "correlation_score_pct": round(correlation_score / 4, 4),
            "explanation": explanation,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = compute_correlation_scores()
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH} ({len(df)} rows)")
    print(f"\nMean correlation_score: {df['correlation_score'].mean():.3f} / 4.0")
    print("\n--- 3 highest-confidence observations ---")
    for _, row in df.sort_values("correlation_score", ascending=False).head(3).iterrows():
        print(f"  {row['txid']}  score={row['correlation_score']:.2f}  {row['explanation']}")
    print("\n--- 3 lowest-confidence observations ---")
    for _, row in df.sort_values("correlation_score", ascending=True).head(3).iterrows():
        print(f"  {row['txid']}  score={row['correlation_score']:.2f}  {row['explanation']}")
