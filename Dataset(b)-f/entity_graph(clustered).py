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
- Node = an address. Two kinds:
    * profile wallet -- address is in wallets_reference.csv (has a
      canonical wallet_id). Attributes: country, asn, script_type,
      typical_amount_btc, n_alerts, max_priority_tier ('high' >
      'medium-high' > 'worth reviewing' > none), is_alerted,
      is_disposable=False.
    * disposable address -- everything else (peeling-chain hop addresses,
      one-off change/output addresses, etc.). These were PREVIOUSLY
      DROPPED ENTIRELY: the old version only created nodes for addresses
      it could resolve to a wallets_reference.csv row, so any address a
      peeling chain burned after hop 1 (see generate_dataset.py:
      current_addr = change_addr) had no node -- 100% of peeling_chain
      transactions were silently skipped as a result (verified: 35/35).
      Fixed by giving every address a node. Disposable nodes get
      is_disposable=True, script_type/country/asn pulled from the
      transaction itself (the closest available signal, since there's no
      wallet profile to look up), typical_amount_btc=0.0, and the same
      n_alerts/max_priority_tier/is_alerted rollup as profile wallets --
      which will usually be 0/none/False for them, since alerts.csv
      attributes peeling-chain hops back to the chain's canonical SOURCE
      wallet (see split_dataset.py's synthetic_chain_N_hopK regex), not
      to each disposable hop individually. That's correct, not a bug:
      the source wallet carries the alert badge, and the disposable hops
      are now visible as the trail leading away from it.
- Edge = one transaction's money flow from an input address's node to an
  output address's node. A tx with multiple inputs/outputs fans out into
  one edge per (input_node, output_node) pair, sharing that tx's total
  output amount evenly across its output legs (this is a simplification --
  real input/output amounts aren't evenly split across pairs, but exact
  UTXO-level attribution isn't recoverable from this data shape, and this
  keeps edge weights proportional and non-fabricated in aggregate). Edge
  attributes: txid, timestamp, fee, script_type, priority_tier (if alerted).

Multiple transactions between the same address pair collapse into ONE edge
in the simplified weighted view (n_tx count, total_btc sum) -- 11k+ raw
transactions across ~11k distinct addresses is unreadable as a raw
multigraph, so this is the view actually worth looking at. The full graph
is now much bigger than the old 800-node version (every address is a node,
not just profile wallets) -- that's expected and correct, not a
regression; it's why the risk subgraph (alerted wallets + their direct
neighbors) below remains the one meant for actually opening in a browser.

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
import sys
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


def build_wallet_graph(tx_df, wallets_df, alerts_df, alerted_txids, tx_tier):
    addr_to_wallet = build_address_to_wallet(wallets_df)
    wallet_info = wallets_df.set_index("wallet_id").to_dict("index")

    # per-wallet alert rollup: how many alerts, what's the worst tier.
    # Keyed by canonical_wallet_id, same identity space as wallets_reference's
    # wallet_id -- disposable addresses never appear here directly (see
    # module docstring: chain hops roll up to the chain's source wallet).
    alert_rollup = defaultdict(lambda: {"n_alerts": 0, "max_tier": None})
    for _, row in alerts_df.iterrows():
        wid = row["canonical_wallet_id"]
        rec = alert_rollup[wid]
        rec["n_alerts"] += 1
        if (
            rec["max_tier"] is None
            or TIER_RANK[row["priority_tier"]] > TIER_RANK[rec["max_tier"]]
        ):
            rec["max_tier"] = row["priority_tier"]

    G = nx.DiGraph()

    def node_key(address):
        """Every address gets a node. Profile wallets use their canonical
        wallet_id (int) as the key, matching alerts.csv's canonical_wallet_id
        and everything downstream that already keys off wallet_id. Disposable
        addresses use the address string itself -- there's no canonical id
        for them, and they were never dropped before this fix either way."""
        wid = addr_to_wallet.get(address)
        return wid if wid is not None else address

    def ensure_node(key, address, tx):
        if key in G:
            return
        is_wallet = isinstance(key, (int,)) or key in wallet_info
        if is_wallet and key in wallet_info:
            info = wallet_info[key]
            rollup = alert_rollup.get(key, {"n_alerts": 0, "max_tier": None})
            G.add_node(
                key,
                address=address,
                country=info.get("country", "?"),
                asn=info.get("asn", "?"),
                script_type=info.get("script_type", "?"),
                typical_amount_btc=info.get("typical_amount_btc", 0.0),
                n_alerts=rollup["n_alerts"],
                max_priority_tier=rollup["max_tier"] or "none",
                is_alerted=rollup["n_alerts"] > 0,
                is_disposable=False,
            )
        else:
            # Disposable address -- no wallets_reference.csv row. Use the
            # transaction's own geo/script fields as the closest available
            # signal instead of leaving it blank. n_alerts/max_priority_tier
            # will almost always be 0/none: alerts attach to the chain's
            # canonical source wallet, not to each disposable hop (see
            # module docstring) -- that's expected, not a gap.
            G.add_node(
                key,
                address=address,
                country=tx.get("geo_country", "?"),
                asn=tx.get("asn", "?"),
                script_type=tx.get("script_type", "?"),
                typical_amount_btc=0.0,
                n_alerts=0,
                max_priority_tier="none",
                is_alerted=False,
                is_disposable=True,
            )

    def ensure_ip_node(ip_key, ip, tx):
        if ip_key in G:
            return
        # PS requirement: "entity/transaction graph linking IPs, wallets,
        # and transactions" -- src_ip only, not dst_ip. Checked both against
        # the actual dataset before deciding: dst_ip is unique per
        # transaction (6570/6570 distinct in the reference run) -- a random
        # P2P broadcast peer, not a stable entity, so graphing it is pure
        # clutter. src_ip is a real per-wallet fingerprint: 790/800 wallets
        # use exactly one src_ip for their whole history, while the 10
        # ip_hopping-labeled wallets use 16-26 distinct src_ips each
        # (verified via groupby on transactions.csv) -- that's the actual
        # correlatable network-layer signal, and it's the same anomaly type
        # split_dataset.py's sender_distinct_ip_last10 feature targets.
        # dst_port is skipped too -- constant 8333 (Bitcoin's standard P2P
        # port) in every row, zero information.
        G.add_node(
            ip_key,
            is_ip=True,
            is_disposable=False,
            ip=ip,
            country=tx.get("geo_country", "?"),
            asn=tx.get("asn", "?"),
            script_type="?",
            typical_amount_btc=0.0,
            n_alerts=0,
            max_priority_tier="none",
            is_alerted=False,
        )

    skipped_no_endpoints = 0
    disposable_address_refs = 0

    for _, tx in tx_df.iterrows():
        in_addrs = tx["input_addresses"]
        out_addrs = tx["output_addresses"]
        out_amounts = tx["output_amounts"]
        total_out = sum(out_amounts) if out_amounts else 0.0

        # Every address becomes a node -- nothing is dropped for lacking a
        # wallets_reference.csv row. Only an actually-empty input/output
        # list (malformed row) skips the tx now.
        in_nodes = [node_key(a) for a in in_addrs]
        out_nodes = [node_key(a) for a in out_addrs]
        disposable_address_refs += sum(
            1 for a in in_addrs + out_addrs if a not in addr_to_wallet
        )

        if not in_nodes or not out_nodes:
            skipped_no_endpoints += 1
            continue

        for key, addr in zip(in_nodes, in_addrs):
            ensure_node(key, addr, tx)
        for key, addr in zip(out_nodes, out_addrs):
            ensure_node(key, addr, tx)

        # ---- IP node + network-layer edges (src_ip -> every input node of
        # this tx). Kept as edge_type="network", separate n_tx/n_txids
        # counters from the money-flow edges below -- an IP link isn't a
        # BTC transfer, and reusing that schema would misrepresent it as one.
        src_ip = tx.get("src_ip")
        if src_ip and not pd.isna(src_ip):
            ip_key = f"ip:{src_ip}"
            ensure_ip_node(ip_key, src_ip, tx)
            for dst_node in in_nodes:
                if G.has_edge(ip_key, dst_node):
                    e = G[ip_key][dst_node]
                    e["n_tx"] += 1
                    e["txids"].append(tx["txid"])
                else:
                    G.add_edge(
                        ip_key,
                        dst_node,
                        edge_type="network",
                        n_tx=1,
                        txids=[tx["txid"]],
                    )

        n_pairs = len(in_nodes) * len(out_nodes)
        amount_per_pair = (total_out / n_pairs) if n_pairs else 0.0

        for src in in_nodes:
            for dst in out_nodes:
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
                        or TIER_RANK[tx_tier.get(tx["txid"], "worth reviewing")]
                        > TIER_RANK[e["max_priority_tier"]]
                    ):
                        e["max_priority_tier"] = tx_tier.get(tx["txid"])
                else:
                    tier = (
                        tx_tier.get(tx["txid"]) if tx["txid"] in alerted_txids else None
                    )
                    G.add_edge(
                        src,
                        dst,
                        edge_type="money",
                        n_tx=1,
                        total_btc=amount_per_pair,
                        total_fee=tx["fee"] / n_pairs,
                        txids=[tx["txid"]],
                        max_priority_tier=tier,
                    )

    return G, disposable_address_refs, skipped_no_endpoints


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

    net = Network(
        height="900px",
        width="100%",
        directed=True,
        notebook=False,
        bgcolor="#111318",
        font_color="#e8e8e8",
        cdn_resources="in_line",
    )
    net.barnes_hut(
        gravity=-2500, central_gravity=0.15, spring_length=140, spring_strength=0.02
    )

    for n, d in G.nodes(data=True):
        is_ip = d.get("is_ip", False)
        is_disposable = d.get("is_disposable", False)
        tier = d["max_priority_tier"] if d["max_priority_tier"] != "none" else None

        if is_ip:
            # network-layer node -- visually distinct (square, blue) so it
            # reads as a different kind of entity from wallets/addresses,
            # not just another dot in the same color scale.
            color = "#4895ef"
            size = 16
            label = n.replace("ip:", "")
            shape = "square"
        elif is_disposable and tier is None:
            # disposable addresses (no wallets_reference.csv row) render
            # smaller, dimmer, and unlabeled by default -- they're the
            # trail, not the focus, unless they happen to also carry an alert
            color = "#5a616b"
            size = 8
            label = ""
            shape = "dot"
        else:
            color = TIER_COLOR[tier]
            size = 14 + 4 * min(d["n_alerts"], 6)
            label = f"#{n}"
            shape = "dot"

        if is_ip:
            title = f"src_ip {n.replace('ip:', '')}<br>country: {d['country']}  asn: {d['asn']}"
        else:
            node_desc = f"wallet #{n}" if not is_disposable else f"disposable address {n[:12]}..."
            title = (
                f"{node_desc}<br>country: {d['country']}  asn: {d['asn']}<br>"
                f"script: {d['script_type']}<br>alerts: {d['n_alerts']} "
                f"(worst: {d['max_priority_tier']})"
            )

        net.add_node(
            n,
            label=label,
            title=title,
            color=color,
            size=size,
            shape=shape,
            borderWidth=3 if d.get("is_alerted") else 1,
        )

    for u, v, d in G.edges(data=True):
        if d.get("edge_type") == "network":
            # IP -> wallet link: not a BTC transfer, drawn as a thin dashed
            # line so it's clearly a different relationship type from money
            # flow, not just a lighter-weight transaction edge.
            net.add_edge(
                u,
                v,
                color="#4895ef",
                width=1,
                dashes=True,
                title=f"{d['n_tx']} tx observed from this src_ip",
                arrows="to",
            )
            continue
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
    html = net.generate_html(notebook=False)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    tx_df = load_transactions(TX_PATH)
    wallets_df = pd.read_csv(WALLETS_PATH)
    alerts_df = pd.read_csv(ALERTS_PATH)

    alerted_txids = set(alerts_df["txid"])
    tx_tier = dict(zip(alerts_df["txid"], alerts_df["priority_tier"]))

    G, disposable_refs, skipped = build_wallet_graph(
        tx_df, wallets_df, alerts_df, alerted_txids, tx_tier
    )
    n_disposable_nodes = sum(1 for _, d in G.nodes(data=True) if d.get("is_disposable"))
    n_ip_nodes = sum(1 for _, d in G.nodes(data=True) if d.get("is_ip"))
    n_wallet_nodes = G.number_of_nodes() - n_disposable_nodes - n_ip_nodes
    n_network_edges = sum(1 for _, _, d in G.edges(data=True) if d.get("edge_type") == "network")

    print(
        f"Full entity graph: {G.number_of_nodes()} nodes ({n_wallet_nodes} profile wallets + "
        f"{n_disposable_nodes} disposable addresses + {n_ip_nodes} src_ip nodes), "
        f"{G.number_of_edges()} links ({G.number_of_edges() - n_network_edges} money-flow, "
        f"{n_network_edges} network/IP)"
    )
    if disposable_refs:
        print(
            f"  ({disposable_refs} address references were disposable/non-profile -- "
            f"now included as their own nodes, not dropped)"
        )
    if skipped:
        print(
            f"  ({skipped} transactions skipped -- malformed row with an empty input or output list)"
        )

    # GraphML can't hold list-valued attributes -- flatten txids to a
    # comma-joined string for the export copy only (the in-memory G used
    # for the HTML view below keeps the real list).
    G_export = G.copy()
    for _, _, d in G_export.edges(data=True):
        d["txids"] = ",".join(d["txids"])
        # network edges (IP -> wallet) don't carry max_priority_tier/total_btc/
        # total_fee -- they're not money-flow edges. Fill in neutral defaults
        # so GraphML export doesn't KeyError and every edge has the same key
        # set (GraphML needs consistent attribute schemas across edges).
        if d.get("max_priority_tier") is None:
            d["max_priority_tier"] = "none"
        d.setdefault("total_btc", 0.0)
        d.setdefault("total_fee", 0.0)
    for _, d in G_export.nodes(data=True):
        if d.get("max_priority_tier") is None:
            d["max_priority_tier"] = "none"
    nx.write_graphml(G_export, "output/entity_graph_full.graphml")
    print(
        "Wrote output/entity_graph_full.graphml (open in Gephi / Neo4j / any GraphML tool)"
    )

    risk_G, alerted = build_risk_subgraph(G, radius=1)
    print(
        f"\nRisk subgraph: {len(alerted)} alerted wallets + counterparties "
        f"= {risk_G.number_of_nodes()} nodes, {risk_G.number_of_edges()} links"
    )
    render_pyvis_html(risk_G, alerted, "output/entity_graph_risk.html")
    print("Wrote output/entity_graph_risk.html (interactive -- open in a browser)")

    # ---- summary text: top-connected wallets + alert-to-alert links
    lines = []
    degree = dict(G.degree())
    # profile wallets only here -- "most connected wallet" is a wallet-identity
    # stat; disposable addresses are single-use by construction, so ranking
    # them by degree wouldn't mean the same thing (see graph_full.graphml /
    # entity_graph_risk.html for the disposable-address trails themselves)
    wallet_degree = {
        n: deg
        for n, deg in degree.items()
        if not G.nodes[n].get("is_disposable") and not G.nodes[n].get("is_ip")
    }
    top = sorted(wallet_degree.items(), key=lambda kv: kv[1], reverse=True)[:15]
    lines.append("Top 15 most-connected profile wallets (in+out degree):")
    for wid, deg in top:
        d = G.nodes[wid]
        flag = (
            f" [ALERTED x{d['n_alerts']}, worst={d['max_priority_tier']}]"
            if d["is_alerted"]
            else ""
        )
        lines.append(f"  wallet #{wid}: degree={deg}, country={d['country']}{flag}")

    n_disposable = sum(1 for n in G.nodes if G.nodes[n].get("is_disposable"))
    lines.append("")
    lines.append(
        f"Disposable (non-profile) addresses in graph: {n_disposable} "
        f"-- mostly single-use peeling-chain hops and one-off outputs, "
        f"visible now as edges/trails but not ranked above"
    )

    lines.append("")
    lines.append("Direct links between two alerted wallets (both endpoints flagged):")
    alerted_set = set(alerted)
    found_any = False
    for u, v, d in G.edges(data=True):
        if u in alerted_set and v in alerted_set:
            found_any = True
            lines.append(
                f"  #{u} -> #{v}: {d['n_tx']} tx, {d['total_btc']:.4f} BTC"
                + (
                    f", flagged link ({d['max_priority_tier']})"
                    if d["max_priority_tier"]
                    else ""
                )
            )
    if not found_any:
        lines.append(
            "  none -- flagged wallets in this dataset don't transact directly with each other"
        )

    summary = "\n".join(lines)
    with open("output/entity_graph_summary.txt", "w") as f:
        f.write(summary + "\n")
    print("\n" + summary)
    print("\nWrote output/entity_graph_summary.txt")


if __name__ == "__main__":
    main()