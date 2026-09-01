"""
Transaction/entity graph builder for SIH PS 26146.

PS requirement (link-analysis angle): once transactions are ingested and
scored, an analyst needs to see *who is connected to whom* -- which wallets
send/receive with which other wallets, and where the flagged (alert) wallets
sit inside that flow. A flat alerts.csv row only shows one transaction at a
time; it can't show that wallet #376 shows up in five separate high-priority
alerts, or that two flagged wallets are two hops apart through a shared
counterparty. That's what this script builds.

Graph model
-----------
- Node  = a wallet (canonical_wallet_id from wallets_reference.csv, joined
  on address). Node attributes: country, asn, script_type, typical_amount_btc,
  n_alerts, max_priority_tier ('high' > 'medium-high' > 'worth reviewing' >
  none), is_alerted.
- Edge  = one transaction's money flow from an input address's wallet to an
  output address's wallet. A tx with multiple inputs/outputs fans out into
  one edge per (input_wallet, output_wallet) pair, sharing that tx's total
  output amount evenly across its output legs (this is a simplification --
  real input/output amounts aren't evenly split across pairs, but exact
  UTXO-level attribution isn't recoverable from this data shape, and this
  keeps edge weights proportional and non-fabricated in aggregate). Edge
  attributes: txid, timestamp, fee, script_type, priority_tier (if alerted).

Multiple transactions between the same wallet pair collapse into ONE edge
in the simplified weighted view (n_tx count, total_btc sum) -- 800 wallets
with 11k+ raw transactions is unreadable as a raw multigraph, so this is
the view actually worth looking at.

Outputs (both written to output/):
  - entity_graph_full.graphml       full weighted wallet graph, for Gephi/
                                     Neo4j/any GraphML-reading tool
  - entity_graph_risk.html          interactive subgraph: every alerted
                                     wallet plus its direct counterparties,
                                     colored by priority tier -- this is the
                                     one worth opening in a browser
  - entity_graph_summary.txt        node/edge counts, top-connected wallets,
                                     which alerted wallets are linked to
                                     each other (multi-alert clusters)

Usage:
    python3 entity_graph.py
"""
import json
from collections import defaultdict

import networkx as nx
import pandas as pd

from ingest import load_transactions

TX_PATH = "output/transactions.csv"
WALLETS_PATH = "output/wallets_reference.csv"
ALERTS_PATH = "output/alerts.csv"

TIER_RANK = {"high": 3, "medium-high": 2, "worth reviewing": 1}
TIER_COLOR = {
    "high": "#e63946",
    "medium-high": "#f4a261",
    "worth reviewing": "#e9c46a",
    None: "#a8b0b8",
}


def build_address_to_wallet(wallets_df):
    return dict(zip(wallets_df["address"], wallets_df["wallet_id"]))


def build_wallet_graph(tx_df, wallets_df, alerts_df):
    addr_to_wallet = build_address_to_wallet(wallets_df)
    wallet_info = wallets_df.set_index("wallet_id").to_dict("index")

    # per-wallet alert rollup: how many alerts, what's the worst tier
    alert_rollup = defaultdict(lambda: {"n_alerts": 0, "max_tier": None})
    for _, row in alerts_df.iterrows():
        wid = row["canonical_wallet_id"]
        rec = alert_rollup[wid]
        rec["n_alerts"] += 1
        if rec["max_tier"] is None or TIER_RANK[row["priority_tier"]] > TIER_RANK[rec["max_tier"]]:
            rec["max_tier"] = row["priority_tier"]

    G = nx.DiGraph()

    def ensure_node(wid):
        if wid in G:
            return
        info = wallet_info.get(wid, {})
        rollup = alert_rollup.get(wid, {"n_alerts": 0, "max_tier": None})
        G.add_node(
            wid,
            country=info.get("country", "?"),
            asn=info.get("asn", "?"),
            script_type=info.get("script_type", "?"),
            typical_amount_btc=info.get("typical_amount_btc", 0.0),
            n_alerts=rollup["n_alerts"],
            max_priority_tier=rollup["max_tier"] or "none",
            is_alerted=rollup["n_alerts"] > 0,
        )

    unmatched_addresses = 0
    skipped_no_endpoints = 0

    for _, tx in tx_df.iterrows():
        in_addrs = tx["input_addresses"]
        out_addrs = tx["output_addresses"]
        out_amounts = tx["output_amounts"]
        total_out = sum(out_amounts) if out_amounts else 0.0

        in_wallets = []
        for a in in_addrs:
            wid = addr_to_wallet.get(a)
            if wid is None:
                unmatched_addresses += 1
            else:
                in_wallets.append(wid)
        out_wallets = []
        for a in out_addrs:
            wid = addr_to_wallet.get(a)
            if wid is None:
                unmatched_addresses += 1
            else:
                out_wallets.append(wid)

        if not in_wallets or not out_wallets:
            skipped_no_endpoints += 1
            continue

        for wid in in_wallets + out_wallets:
            ensure_node(wid)

        n_pairs = len(in_wallets) * len(out_wallets)
        amount_per_pair = (total_out / n_pairs) if n_pairs else 0.0

        for src in in_wallets:
            for dst in out_wallets:
                if src == dst:
                    continue  # self-transfer between own addresses, not an entity link
                if G.has_edge(src, dst):
                    e = G[src][dst]
                    e["n_tx"] += 1
                    e["total_btc"] += amount_per_pair
                    e["total_fee"] += tx["fee"] / n_pairs
                    e["txids"].append(tx["txid"])
                    if tx["txid"] in alerted_txids and (
                        e["max_priority_tier"] is None
                        or TIER_RANK[tx_tier.get(tx["txid"], "worth reviewing")] > TIER_RANK[e["max_priority_tier"]]
                    ):
                        e["max_priority_tier"] = tx_tier.get(tx["txid"])
                else:
                    tier = tx_tier.get(tx["txid"]) if tx["txid"] in alerted_txids else None
                    G.add_edge(
                        src, dst,
                        n_tx=1,
                        total_btc=amount_per_pair,
                        total_fee=tx["fee"] / n_pairs,
                        txids=[tx["txid"]],
                        max_priority_tier=tier,
                    )

    return G, unmatched_addresses, skipped_no_endpoints


def build_risk_subgraph(G, radius=1):
    """Every alerted wallet plus its direct (radius-hop) neighbors, as an
    undirected view for layout purposes -- direction is kept in edge data."""
    alerted = [n for n, d in G.nodes(data=True) if d["is_alerted"]]
    keep = set(alerted)
    UG = G.to_undirected(as_view=True)
    for n in alerted:
        keep.update(nx.single_source_shortest_path_length(UG, n, cutoff=radius).keys())
    return G.subgraph(keep).copy(), alerted


def render_pyvis_html(G, alerted_nodes, out_path):
    from pyvis.network import Network

    net = Network(height="900px", width="100%", directed=True, notebook=False,
                  bgcolor="#111318", font_color="#e8e8e8", cdn_resources="in_line")
    net.barnes_hut(gravity=-2500, central_gravity=0.15, spring_length=140, spring_strength=0.02)

    for n, d in G.nodes(data=True):
        tier = d["max_priority_tier"] if d["max_priority_tier"] != "none" else None
        color = TIER_COLOR[tier]
        size = 14 + 4 * min(d["n_alerts"], 6)
        title = (
            f"wallet #{n}<br>country: {d['country']}  asn: {d['asn']}<br>"
            f"script: {d['script_type']}<br>alerts: {d['n_alerts']} "
            f"(worst: {d['max_priority_tier']})"
        )
        net.add_node(
            n, label=f"#{n}", title=title, color=color, size=size,
            borderWidth=3 if d["is_alerted"] else 1,
        )

    for u, v, d in G.edges(data=True):
        tier = d["max_priority_tier"]
        color = TIER_COLOR[tier] if tier else "#4a5058"
        width = 1 + min(d["n_tx"], 8)
        title = f"{d['n_tx']} tx, {d['total_btc']:.4f} BTC total"
        if tier:
            title += f"<br>flagged: {tier}"
        net.add_edge(u, v, color=color, width=width, title=title, arrows="to")

    net.set_options("""
    {
      "physics": {"stabilization": {"iterations": 150}},
      "interaction": {"hover": true, "tooltipDelay": 100}
    }
    """)
    net.write_html(out_path, notebook=False, open_browser=False)


def main():
    tx_df = load_transactions(TX_PATH)
    wallets_df = pd.read_csv(WALLETS_PATH)
    alerts_df = pd.read_csv(ALERTS_PATH)

    global alerted_txids, tx_tier
    alerted_txids = set(alerts_df["txid"])
    tx_tier = dict(zip(alerts_df["txid"], alerts_df["priority_tier"]))

    G, unmatched, skipped = build_wallet_graph(tx_df, wallets_df, alerts_df)

    print(f"Full entity graph: {G.number_of_nodes()} wallets, {G.number_of_edges()} distinct wallet-pair links")
    if unmatched:
        print(f"  ({unmatched} address references had no matching wallet -- excluded from graph)")
    if skipped:
        print(f"  ({skipped} transactions skipped -- missing input or output side)")

    # GraphML can't hold list-valued attributes -- flatten txids to a
    # comma-joined string for the export copy only (the in-memory G used
    # for the HTML view below keeps the real list).
    G_export = G.copy()
    for _, _, d in G_export.edges(data=True):
        d["txids"] = ",".join(d["txids"])
        if d["max_priority_tier"] is None:
            d["max_priority_tier"] = "none"
    for _, d in G_export.nodes(data=True):
        if d.get("max_priority_tier") is None:
            d["max_priority_tier"] = "none"
    nx.write_graphml(G_export, "output/entity_graph_full.graphml")
    print("Wrote output/entity_graph_full.graphml (open in Gephi / Neo4j / any GraphML tool)")

    risk_G, alerted = build_risk_subgraph(G, radius=1)
    print(f"\nRisk subgraph: {len(alerted)} alerted wallets + counterparties "
          f"= {risk_G.number_of_nodes()} nodes, {risk_G.number_of_edges()} links")
    render_pyvis_html(risk_G, alerted, "output/entity_graph_risk.html")
    print("Wrote output/entity_graph_risk.html (interactive -- open in a browser)")

    # ---- summary text: top-connected wallets + alert-to-alert links
    lines = []
    degree = dict(G.degree())
    top = sorted(degree.items(), key=lambda kv: kv[1], reverse=True)[:15]
    lines.append("Top 15 most-connected wallets (in+out degree):")
    for wid, deg in top:
        d = G.nodes[wid]
        flag = f" [ALERTED x{d['n_alerts']}, worst={d['max_priority_tier']}]" if d["is_alerted"] else ""
        lines.append(f"  wallet #{wid}: degree={deg}, country={d['country']}{flag}")

    lines.append("")
    lines.append("Direct links between two alerted wallets (both endpoints flagged):")
    alerted_set = set(alerted)
    found_any = False
    for u, v, d in G.edges(data=True):
        if u in alerted_set and v in alerted_set:
            found_any = True
            lines.append(
                f"  #{u} -> #{v}: {d['n_tx']} tx, {d['total_btc']:.4f} BTC"
                + (f", flagged link ({d['max_priority_tier']})" if d["max_priority_tier"] else "")
            )
    if not found_any:
        lines.append("  none -- flagged wallets in this dataset don't transact directly with each other")

    summary = "\n".join(lines)
    with open("output/entity_graph_summary.txt", "w") as f:
        f.write(summary + "\n")
    print("\n" + summary)
    print("\nWrote output/entity_graph_summary.txt")


if __name__ == "__main__":
    main()
