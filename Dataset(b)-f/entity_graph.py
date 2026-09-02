"""
Entity/transaction graph builder for SIH PS 26146.

Builds two graph views:
1. Wallet-risk graph: wallet -> wallet money-flow relationships.
2. Full entity graph: IP -> wallet -> transaction -> wallet -> IP.

The full entity graph explicitly represents the three required entity types:
IPs, wallets, and transactions.
"""

import pandas as pd
import networkx as nx
from collections import defaultdict

from ingest import load_transactions

TX_PATH = "output/transactions.csv"
WALLETS_PATH = "output/wallets_reference.csv"
ALERTS_PATH = "output/alerts.csv"

TIER_RANK = {
    "high": 3,
    "medium-high": 2,
    "worth reviewing": 1,
}

TIER_COLOR = {
    "high": "#e63946",
    "medium-high": "#f4a261",
    "worth reviewing": "#e9c46a",
    None: "#a8b0b8",
}


def build_address_to_wallet(wallets_df):
    return dict(zip(wallets_df["address"], wallets_df["wallet_id"]))


def build_wallet_graph(tx_df, wallets_df, alerts_df):
    """Original compact wallet -> wallet risk graph."""
    addr_to_wallet = build_address_to_wallet(wallets_df)
    wallet_info = wallets_df.set_index("wallet_id").to_dict("index")

    alert_rollup = defaultdict(lambda: {"n_alerts": 0, "max_tier": None})

    for _, row in alerts_df.iterrows():
        wid = row["canonical_wallet_id"]
        rec = alert_rollup[wid]
        rec["n_alerts"] += 1

        tier = row["priority_tier"]
        if (
            rec["max_tier"] is None
            or TIER_RANK.get(tier, 0) > TIER_RANK.get(rec["max_tier"], 0)
        ):
            rec["max_tier"] = tier

    alerted_txids = set(alerts_df["txid"])
    tx_tier = dict(zip(alerts_df["txid"], alerts_df["priority_tier"]))

    G = nx.DiGraph()

    def ensure_node(wid):
        if wid in G:
            return

        info = wallet_info.get(wid, {})
        rollup = alert_rollup.get(
            wid,
            {"n_alerts": 0, "max_tier": None},
        )

        G.add_node(
            wid,
            country=info.get("country", "?"),
            asn=info.get("asn", "?"),
            script_type=info.get("script_type", "?"),
            typical_amount_btc=info.get("typical_amount_btc", 0.0),
            n_alerts=rollup["n_alerts"],
            max_priority_tier=rollup["max_tier"] or "none",
            is_alerted=rollup["n_alerts"] > 0,
            node_type="wallet",
        )

    unmatched_addresses = 0
    skipped_no_endpoints = 0

    for _, tx in tx_df.iterrows():
        in_addrs = tx["input_addresses"]
        out_addrs = tx["output_addresses"]
        out_amounts = tx["output_amounts"]
        total_out = sum(out_amounts) if out_amounts else 0.0

        in_wallets = []
        for address in in_addrs:
            wid = addr_to_wallet.get(address)
            if wid is None:
                unmatched_addresses += 1
            else:
                in_wallets.append(wid)

        out_wallets = []
        for address in out_addrs:
            wid = addr_to_wallet.get(address)
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
        amount_per_pair = total_out / n_pairs if n_pairs else 0.0

        for src in in_wallets:
            for dst in out_wallets:
                if src == dst:
                    continue

                tier = tx_tier.get(tx["txid"]) if tx["txid"] in alerted_txids else None

                if G.has_edge(src, dst):
                    edge = G[src][dst]
                    edge["n_tx"] += 1
                    edge["total_btc"] += amount_per_pair
                    edge["total_fee"] += tx["fee"] / n_pairs
                    edge["txids"].append(tx["txid"])

                    if (
                        tier
                        and (
                            edge["max_priority_tier"] is None
                            or TIER_RANK[tier]
                            > TIER_RANK.get(edge["max_priority_tier"], 0)
                        )
                    ):
                        edge["max_priority_tier"] = tier
                else:
                    G.add_edge(
                        src,
                        dst,
                        n_tx=1,
                        total_btc=amount_per_pair,
                        total_fee=tx["fee"] / n_pairs,
                        txids=[tx["txid"]],
                        max_priority_tier=tier,
                    )

    return G, unmatched_addresses, skipped_no_endpoints


def build_entity_graph(tx_df, wallets_df, alerts_df):
    """
    Build an explicit IP -> wallet -> transaction -> wallet -> IP graph.

    Node types:
      ip
      wallet
      transaction

    Edges:
      IP -> wallet          observed source IP
      wallet -> transaction input/sender relationship
      transaction -> wallet output relationship
      transaction -> IP     observed destination IP

    Multiple address references are preserved as separate relationships.
    """
    addr_to_wallet = build_address_to_wallet(wallets_df)
    wallet_info = wallets_df.set_index("wallet_id").to_dict("index")

    alert_tier = dict(
        zip(alerts_df["txid"], alerts_df["priority_tier"])
    )
    alerted_txids = set(alerts_df["txid"])

    G = nx.DiGraph()

    unmatched_addresses = 0
    skipped_transactions = 0

    for _, tx in tx_df.iterrows():
        txid = str(tx["txid"])

        input_addresses = tx["input_addresses"]
        output_addresses = tx["output_addresses"]

        if not input_addresses or not output_addresses:
            skipped_transactions += 1
            continue

        tx_node = f"tx:{txid}"
        tier = alert_tier.get(txid)
        is_alerted = txid in alerted_txids

        # Transaction node
        G.add_node(
            tx_node,
            node_type="transaction",
            label=txid[:12],
            txid=txid,
            timestamp=str(tx["timestamp"]),
            fee=float(tx["fee"]),
            script_type=str(tx.get("script_type", "?")),
            priority_tier=tier or "none",
            is_alerted=is_alerted,
        )

        # Source IP -> transaction
        src_ip = str(tx.get("src_ip", "UNRESOLVED"))
        if src_ip and src_ip != "nan":
            ip_node = f"ip:{src_ip}"
            G.add_node(
                ip_node,
                node_type="ip",
                label=src_ip,
                ip=src_ip,
            )
            G.add_edge(
                ip_node,
                tx_node,
                relation="observed_transaction",
            )

        # Destination IP -> transaction
        dst_ip = str(tx.get("dst_ip", "UNRESOLVED"))
        if dst_ip and dst_ip != "nan":
            ip_node = f"ip:{dst_ip}"
            G.add_node(
                ip_node,
                node_type="ip",
                label=dst_ip,
                ip=dst_ip,
            )
            G.add_edge(
                tx_node,
                ip_node,
                relation="destination_observation",
            )

        # Input wallet -> transaction
        for address in input_addresses:
            wid = addr_to_wallet.get(address)

            if wid is None:
                unmatched_addresses += 1
                continue

            wallet_node = f"wallet:{wid}"
            info = wallet_info.get(wid, {})

            if wallet_node not in G:
                G.add_node(
                    wallet_node,
                    node_type="wallet",
                    label=f"Wallet #{wid}",
                    wallet_id=wid,
                    country=info.get("country", "?"),
                    asn=info.get("asn", "?"),
                    script_type=info.get("script_type", "?"),
                )

            G.add_edge(
                wallet_node,
                tx_node,
                relation="input",
            )

        # Transaction -> output wallet
        for address in output_addresses:
            wid = addr_to_wallet.get(address)

            if wid is None:
                unmatched_addresses += 1
                continue

            wallet_node = f"wallet:{wid}"
            info = wallet_info.get(wid, {})

            if wallet_node not in G:
                G.add_node(
                    wallet_node,
                    node_type="wallet",
                    label=f"Wallet #{wid}",
                    wallet_id=wid,
                    country=info.get("country", "?"),
                    asn=info.get("asn", "?"),
                    script_type=info.get("script_type", "?"),
                )

            G.add_edge(
                tx_node,
                wallet_node,
                relation="output",
            )

    # Add alert metadata to wallet nodes.
    wallet_alerts = defaultdict(list)

    for _, row in alerts_df.iterrows():
        wallet_alerts[row["canonical_wallet_id"]].append(
            row["priority_tier"]
        )

    for node, data in G.nodes(data=True):
        if data.get("node_type") != "wallet":
            continue

        wid = data["wallet_id"]
        tiers = wallet_alerts.get(wid, [])

        worst = max(
            tiers,
            key=lambda x: TIER_RANK.get(x, 0),
            default="none",
        )

        data["n_alerts"] = len(tiers)
        data["max_priority_tier"] = worst
        data["is_alerted"] = len(tiers) > 0

    return G, unmatched_addresses, skipped_transactions


def build_entity_risk_subgraph(
    G,
    max_alerted_transactions=75,
    max_related_wallets=150,
):
    """
    Build a compact investigator/demo graph.

    Instead of taking a radius around every alerted wallet (which can explode
    into thousands of nodes), start with the highest-priority alerted
    transactions and retain their directly connected IPs and wallets.

    The full entity graph is still exported separately for analysis.
    """
    priority_rank = {
        "high": 3,
        "medium-high": 2,
        "worth reviewing": 1,
        "none": 0,
    }

    alerted_tx = [
        (node, data)
        for node, data in G.nodes(data=True)
        if data.get("node_type") == "transaction"
        and data.get("is_alerted")
    ]

    alerted_tx.sort(
        key=lambda item: (
            priority_rank.get(item[1].get("priority_tier"), 0),
            item[1].get("txid", ""),
        ),
        reverse=True,
    )

    selected_tx = [
        node
        for node, _ in alerted_tx[:max_alerted_transactions]
    ]

    keep = set(selected_tx)

    # Direct neighbors of selected transactions:
    # IP -> TX, wallet -> TX, TX -> wallet, TX -> IP.
    for tx_node in selected_tx:
        keep.update(G.predecessors(tx_node))
        keep.update(G.successors(tx_node))

    # If many wallets are connected, retain alerted wallets first and then
    # cap the remaining wallet nodes by number of incident selected TXs.
    wallet_nodes = [
        node
        for node in keep
        if G.nodes[node].get("node_type") == "wallet"
    ]

    wallet_nodes.sort(
        key=lambda node: (
            bool(G.nodes[node].get("is_alerted")),
            G.nodes[node].get("n_alerts", 0),
            G.degree(node),
        ),
        reverse=True,
    )

    allowed_wallets = set(wallet_nodes[:max_related_wallets])

    # Never remove transaction/IP nodes selected above.
    keep = {
        node
        for node in keep
        if G.nodes[node].get("node_type") != "wallet"
        or node in allowed_wallets
    }

    return G.subgraph(keep).copy()


def render_pyvis_html(G, out_path):
    """Render an explicit entity graph to browser HTML."""
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
        gravity=-2500,
        central_gravity=0.15,
        spring_length=140,
        spring_strength=0.02,
    )

    for node, data in G.nodes(data=True):
        node_type = data.get("node_type")

        if node_type == "wallet":
            tier = data.get("max_priority_tier")
            color = TIER_COLOR.get(tier)
            size = 24 if data.get("is_alerted") else 16
            label = data.get("label", str(node))

            title = (
                f"<b>{label}</b><br>"
                f"type: wallet<br>"
                f"country: {data.get('country', '?')}<br>"
                f"ASN: {data.get('asn', '?')}<br>"
                f"alerts: {data.get('n_alerts', 0)}"
            )

        elif node_type == "transaction":
            tier = data.get("priority_tier")
            color = TIER_COLOR.get(tier)
            size = 20 if data.get("is_alerted") else 12
            label = f"TX {data.get('label', '')}"

            title = (
                f"<b>Transaction</b><br>"
                f"txid: {data.get('txid')}<br>"
                f"timestamp: {data.get('timestamp')}<br>"
                f"fee: {data.get('fee')}<br>"
                f"priority: {data.get('priority_tier')}"
            )

        else:  # IP
            color = "#4dabf7"
            size = 13
            label = data.get("label", str(node))
            title = f"<b>IP</b><br>{data.get('ip', label)}"

        net.add_node(
            node,
            label=label,
            title=title,
            color=color,
            size=size,
            borderWidth=3 if data.get("is_alerted") else 1,
        )

    for u, v, data in G.edges(data=True):
        relation = data.get("relation", "")
        net.add_edge(
            u,
            v,
            title=relation,
            label=relation,
            arrows="to",
        )

    net.set_options(
        """
        {
          "physics": {
            "stabilization": {
              "iterations": 150
            }
          },
          "interaction": {
            "hover": true,
            "tooltipDelay": 100
          }
        }
        """
    )

    # Windows may default to cp1252. Explicit UTF-8 avoids UnicodeEncodeError.
    html = net.generate_html(notebook=False)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    tx_df = load_transactions(TX_PATH)
    wallets_df = pd.read_csv(WALLETS_PATH)
    alerts_df = pd.read_csv(ALERTS_PATH)

    print("=== Building original wallet-risk graph ===")

    wallet_G, unmatched, skipped = build_wallet_graph(
        tx_df,
        wallets_df,
        alerts_df,
    )

    print(
        f"Full wallet graph: {wallet_G.number_of_nodes()} wallets, "
        f"{wallet_G.number_of_edges()} wallet-pair links"
    )

    if unmatched:
        print(
            f"  ({unmatched} address references had no matching wallet)"
        )

    if skipped:
        print(
            f"  ({skipped} transactions skipped -- missing endpoint)"
        )

    # GraphML-safe export
    wallet_export = wallet_G.copy()

    for _, _, data in wallet_export.edges(data=True):
        data["txids"] = ",".join(data["txids"])
        if data["max_priority_tier"] is None:
            data["max_priority_tier"] = "none"

    nx.write_graphml(
        wallet_export,
        "output/entity_graph_full.graphml",
    )

    print(
        "Wrote output/entity_graph_full.graphml"
    )

    # Explicit IP/TX/wallet graph
    print("\n=== Building IP / Transaction / Wallet graph ===")

    entity_G, entity_unmatched, entity_skipped = build_entity_graph(
        tx_df,
        wallets_df,
        alerts_df,
    )

    print(
        f"Full entity graph: "
        f"{entity_G.number_of_nodes()} nodes, "
        f"{entity_G.number_of_edges()} links"
    )

    print(
        "Node types:",
        dict(
            pd.Series(
                [
                    d.get("node_type")
                    for _, d in entity_G.nodes(data=True)
                ]
            ).value_counts()
        ),
    )

    if entity_unmatched:
        print(
            f"  ({entity_unmatched} address references had no matching wallet)"
        )

    if entity_skipped:
        print(
            f"  ({entity_skipped} transactions skipped -- missing endpoint)"
        )

    # GraphML cannot safely store all Python/list attributes, but this
    # explicit graph only uses scalar attributes.
    nx.write_graphml(
        entity_G,
        "output/entity_graph_ip_tx_wallet.graphml",
    )

    print(
        "Wrote output/entity_graph_ip_tx_wallet.graphml"
    )

    risk_G = build_entity_risk_subgraph(
        entity_G,
        max_alerted_transactions=75,
        max_related_wallets=150,
    )

    print(
        f"Focused entity risk graph: "
        f"{risk_G.number_of_nodes()} nodes, "
        f"{risk_G.number_of_edges()} links"
    )

    focused_graphml = "output/entity_graph_ip_tx_wallet_focused.graphml"
    nx.write_graphml(
        risk_G,
        focused_graphml,
    )

    print(
        f"Wrote {focused_graphml}"
    )

    focused_html = "output/entity_graph_ip_tx_wallet_focused.html"

    render_pyvis_html(
        risk_G,
        focused_html,
    )

    print(
        f"Wrote {focused_html}"
    )


if __name__ == "__main__":
    main()
