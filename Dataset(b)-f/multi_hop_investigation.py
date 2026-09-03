"""
Multi-Hop Investigation
========================

Given a starting wallet/entity, traces how funds move onward across
MULTIPLE transaction hops -- entity_A --tx1--> entity_B --tx2--> entity_C --
answering the PS's investigation need directly: "where did this money go
after it left this wallet (or where did it come from before it arrived)?",
several hops deep, not just one transaction at a time.

This is NOT a new scoring model. Every number surfaced here (amount,
investigative_confidence, confidence_level) is read straight out of
output/correlation_scores.csv -- one row per hop, already computed by
correlation_score.py from FeatureEngine's final investigative confidence
(see ENGINE_network_blockchain_correlation.py and correlation_score.py's
docstrings). This script's only new job is:

  1. GRAPH CONSTRUCTION -- turn the flat transaction list into
     entity -> entity edges: for each transaction, the sender is
     entity_id (already resolved from first_input_addr by
     entity_clustering.py); each output_address is resolved to its own
     entity_id the same way (or, if that address was never clustered,
     treated as its own singleton pseudo-entity "addr:<address>" so the
     chain doesn't just vanish at an unclustered hop -- see
     entity_clustering.py's high-fan-in exclusion caveat).

  2. TRAVERSAL -- BFS from a starting entity, up to N hops, following
     the money `outgoing` (where funds went), `incoming` (where they came
     from), or `both`. No cycles within a single traced path (an entity
     already visited on this path is not revisited), so a wallet that
     launders funds back to itself doesn't spin the trace forever.

Why entity-level, not address-level: `first_input_addr` collapses every
transaction to one entity per side (via entity_clusters.csv), so within a
peeling chain the disposable "change" addresses that keep spinning up
don't each look like a separate hop -- what gets traced is real
owner-to-owner movement, matching how correlation_score.py already thinks
about wallets.

Path confidence: a multi-hop chain is only as trustworthy as its WEAKEST
single-hop link, so each path's `weakest_link_confidence` is the MIN of its
hops' investigative_confidence values -- not the average, which would let
one strong hop paper over a shaky one and overstate how solid the whole
trace is.

Depends on (run first if missing):
  - output/correlation_scores.csv   (correlation_score.py)
  - output/entity_clusters.csv      (entity_clustering.py)
  - output/transactions.csv         (generate_dataset.py) -- for output_addresses

Usage:
    python3 entity_clustering.py
    python3 correlation_score.py
    python3 multi_hop_investigation.py --entity E00028 --hops 4 --direction outgoing
"""

from __future__ import annotations

import argparse
import json
import os
from collections import deque

import pandas as pd

from ingest import load_transactions

CORR_SCORES_PATH = "output/correlation_scores.csv"
ENTITY_MEMBERSHIP_PATH = "output/entity_clusters.csv"
TX_PATH = "output/transactions.csv"
OUT_PATH = "output/multi_hop_investigation.json"


def _require_inputs():
    missing = [p for p in (CORR_SCORES_PATH, ENTITY_MEMBERSHIP_PATH, TX_PATH) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "multi_hop_investigation.py needs correlation_score.py's and "
            f"entity_clustering.py's output first (missing: {missing}). "
            "Run those two scripts and retry."
        )


def _pseudo_entity(address: str) -> str:
    """An output address with no entity mapping becomes its own singleton
    pseudo-entity, so it's still a traceable node instead of a dead end."""
    return f"addr:{address}"


def build_hop_edges() -> pd.DataFrame:
    """
    One row per (sending entity -> receiving entity) hop, carrying the
    txid, amount, and investigative_confidence that ALREADY exist in
    correlation_scores.csv for that transaction. A tx with several outputs
    to the SAME downstream entity collapses to one edge (de-duped) so it
    isn't double-counted as two separate hops; change back to the sender's
    own entity is dropped (not a hop to anywhere new).
    """
    _require_inputs()

    scores = pd.read_csv(CORR_SCORES_PATH)
    membership = pd.read_csv(ENTITY_MEMBERSHIP_PATH)
    addr_to_entity = dict(zip(membership["address"], membership["entity_id"]))

    raw = load_transactions(TX_PATH)[["txid", "output_addresses"]]
    merged = scores.merge(raw, on="txid", how="left")

    rows = []
    for _, r in merged.iterrows():
        from_entity = r["entity_id"]
        out_addrs = r["output_addresses"] or []
        to_entities = {addr_to_entity.get(a, _pseudo_entity(a)) for a in out_addrs}
        for to_entity in to_entities:
            if to_entity == from_entity:
                continue
            rows.append({
                "txid": r["txid"],
                "from_entity": from_entity,
                "to_entity": to_entity,
                "tx_amount_btc": r["tx_amount_btc"],
                "investigative_confidence": r["investigative_confidence"],
                "confidence_level": r["confidence_level"],
                "timestamp": r["timestamp"],
            })
    return pd.DataFrame(rows)


def trace_multi_hop(
    edges: pd.DataFrame,
    start_entity: str,
    hops: int = 3,
    direction: str = "outgoing",
    max_paths: int = 5000,
) -> dict:
    """
    BFS from `start_entity` up to `hops` hops, following the money in
    `direction` ('outgoing' = where funds went, 'incoming' = where they
    came from, 'both'). Returns every distinct path found (only MAXIMAL
    paths are kept -- a 2-hop path that's just a prefix of a 3-hop path
    already found is dropped), each annotated with its `weakest_link_confidence`.

    `max_paths` guards against combinatorial blowup at high-fan-out hub
    entities (e.g. a wallet_reuse_burst node with hundreds of counterparties
    -- on real datasets this branches multiplicatively per hop and can
    reach far more paths than an analyst can review anyway). Traversal
    stops early once this many paths have been explored; `truncated: true`
    is set in the result so that's visible rather than silently returning
    a partial trace.
    """
    if direction not in ("outgoing", "incoming", "both"):
        raise ValueError("direction must be 'outgoing', 'incoming', or 'both'")
    if start_entity not in set(edges["from_entity"]) | set(edges["to_entity"]):
        raise ValueError(f"Entity '{start_entity}' not found in the hop graph.")

    out_edges: dict[str, list[dict]] = {}
    in_edges: dict[str, list[dict]] = {}
    for _, e in edges.iterrows():
        out_edges.setdefault(e["from_entity"], []).append(e.to_dict())
        in_edges.setdefault(e["to_entity"], []).append(e.to_dict())

    def neighbors(entity):
        result = []
        if direction in ("outgoing", "both"):
            result += [(edge, "outgoing") for edge in out_edges.get(entity, [])]
        if direction in ("incoming", "both"):
            result += [(edge, "incoming") for edge in in_edges.get(entity, [])]
        return result

    all_paths: list[list[dict]] = []
    truncated = False
    queue = deque([(start_entity, [], frozenset({start_entity}))])
    while queue:
        if len(all_paths) >= max_paths:
            truncated = True
            break
        entity, path, visited = queue.popleft()
        if len(path) >= hops:
            continue
        for edge, edge_dir in neighbors(entity):
            next_entity = edge["to_entity"] if edge_dir == "outgoing" else edge["from_entity"]
            if next_entity in visited:
                continue  # no cycles within one traced path
            hop = {**edge, "hop_direction": edge_dir, "hop_number": len(path) + 1}
            new_path = path + [hop]
            all_paths.append(new_path)
            queue.append((next_entity, new_path, visited | {next_entity}))
            if len(all_paths) >= max_paths:
                truncated = True
                break

    def entity_sequence(path):
        return tuple(
            h["to_entity"] if h["hop_direction"] == "outgoing" else h["from_entity"] for h in path
        )

    sequences = [entity_sequence(p) for p in all_paths]
    maximal_paths = [
        p for p, seq in zip(all_paths, sequences)
        if not any(other != seq and other[: len(seq)] == seq for other in sequences)
    ]

    results = []
    for p in maximal_paths:
        confidences = [h["investigative_confidence"] for h in p]
        results.append({
            "hop_count": len(p),
            "path": p,
            "weakest_link_confidence": round(min(confidences), 4),
            "total_amount_btc": round(sum(h["tx_amount_btc"] for h in p), 8),
        })
    results.sort(key=lambda r: r["weakest_link_confidence"], reverse=True)

    return {
        "start_entity": start_entity,
        "hops_requested": hops,
        "direction": direction,
        "paths_found": len(results),
        "truncated": truncated,
        "paths": results,
    }


def _format_chain(start_entity: str, path: list[dict]) -> str:
    chain = [start_entity]
    for hop in path:
        chain.append(hop["to_entity"] if hop["hop_direction"] == "outgoing" else hop["from_entity"])
    return " -> ".join(chain)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Trace multi-hop fund flow from a starting entity/wallet.")
    ap.add_argument("--entity", required=True, help="Starting entity_id (e.g. E00028) to trace from.")
    ap.add_argument("--hops", type=int, default=3, help="Max hops to follow (default 3).")
    ap.add_argument("--direction", choices=["outgoing", "incoming", "both"], default="outgoing")
    ap.add_argument("--max-paths", type=int, default=5000, help="Cap on paths explored (default 5000).")
    args = ap.parse_args()

    edge_table = build_hop_edges()
    result = trace_multi_hop(
        edge_table, args.entity, hops=args.hops, direction=args.direction, max_paths=args.max_paths
    )

    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"Traced {result['paths_found']} path(s) from {args.entity} "
          f"(<= {args.hops} hops, direction={args.direction})"
          + (" [TRUNCATED -- narrow --direction/--hops or start closer to the target]" if result["truncated"] else ""))
    for p in result["paths"][:5]:
        print(f"  {_format_chain(args.entity, p['path'])}")
        print(f"    weakest_link_confidence={p['weakest_link_confidence']:.2f}  "
              f"total_amount={p['total_amount_btc']:.8f} BTC  hops={p['hop_count']}")
    print(f"Wrote {OUT_PATH}")
