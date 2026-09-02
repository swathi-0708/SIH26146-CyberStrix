"""
Entity/transaction graph builder for SIH PS 26146.

Graph views:
1. Wallet-risk graph
2. Full IP / Transaction / Wallet graph
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
    "none": "#a8b0b8",
}


def build_address_to_wallet(wallets_df):
    return dict(zip(wallets_df["address"], wallets_df["wallet_id"]))


# -------------------------------------------------------------------
# WALLET-RISK GRAPH
# -------------------------------------------------------------------


def build_wallet_graph(tx_df, wallets_df, alerts_df):
    """Build compact wallet -> wallet money-flow graph."""

    addr_to_wallet = build_address_to_wallet(wallets_df)
    wallet_info = wallets_df.set_index("wallet_id").to_dict("index")

    alert_rollup = defaultdict(lambda: {"n_alerts": 0, "max_tier": None})

    for _, row in alerts_df.iterrows():
        wid = row["canonical_wallet_id"]
        rec = alert_rollup[wid]
        rec["n_alerts"] += 1

        tier = row["priority_tier"]

        if rec["max_tier"] is None or TIER_RANK.get(tier, 0) > TIER_RANK.get(
            rec["max_tier"], 0
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
        out_wallets = []

        for address in in_addrs:
            wid = addr_to_wallet.get(address)

            if wid is None:
                unmatched_addresses += 1
            else:
                in_wallets.append(wid)

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

                txid = str(tx["txid"])
                tier = tx_tier.get(txid) if txid in alerted_txids else None

                if G.has_edge(src, dst):
                    edge = G[src][dst]

                    edge["n_tx"] += 1
                    edge["total_btc"] += amount_per_pair
                    edge["total_fee"] += float(tx["fee"]) / n_pairs
                    edge["txids"].append(txid)

                    if tier and TIER_RANK.get(tier, 0) > TIER_RANK.get(
                        edge["max_priority_tier"], 0
                    ):
                        edge["max_priority_tier"] = tier

                else:
                    G.add_edge(
                        src,
                        dst,
                        n_tx=1,
                        total_btc=amount_per_pair,
                        total_fee=float(tx["fee"]) / n_pairs,
                        txids=[txid],
                        max_priority_tier=tier or "none",
                    )

    return G, unmatched_addresses, skipped_no_endpoints


# -------------------------------------------------------------------
# FULL ENTITY GRAPH
# -------------------------------------------------------------------


def build_entity_graph(tx_df, wallets_df, alerts_df):
    """
    Build explicit:

        IP -> Transaction -> Wallet
                       |
                       -> Wallet
                       |
                       -> IP

    Unknown/disposable addresses are retained as wallet/address nodes.
    """

    addr_to_wallet = build_address_to_wallet(wallets_df)
    wallet_info = wallets_df.set_index("wallet_id").to_dict("index")

    alert_tier = dict(zip(alerts_df["txid"].astype(str), alerts_df["priority_tier"]))

    alerted_txids = set(alert_tier.keys())

    G = nx.DiGraph()

    unmatched_addresses = 0
    skipped_transactions = 0

    for _, tx in tx_df.iterrows():
        txid = str(tx["txid"])

        input_addresses = tx["input_addresses"]
        output_addresses = tx["output_addresses"]

        if not input_addresses and not output_addresses:
            skipped_transactions += 1
            continue

        tier = alert_tier.get(txid)
        is_alerted = txid in alerted_txids

        input_amounts = tx.get("input_amounts", [])
        output_amounts = tx.get("output_amounts", [])

        input_total = float(sum(input_amounts)) if input_amounts else 0.0

        output_total = float(sum(output_amounts)) if output_amounts else 0.0

        # -----------------------------------------------------------
        # TRANSACTION NODE
        # -----------------------------------------------------------

        tx_node = f"tx:{txid}"

        G.add_node(
            tx_node,
            node_type="transaction",
            label=txid[:12],
            txid=txid,
            timestamp=str(tx.get("timestamp", "")),
            fee=float(tx.get("fee", 0.0)),
            input_total_btc=input_total,
            output_total_btc=output_total,
            script_type=str(tx.get("script_type", "?")),
            priority_tier=tier or "none",
            is_alerted=is_alerted,
        )

        # -----------------------------------------------------------
        # NETWORK OBSERVATION
        # -----------------------------------------------------------

        src_ip = str(tx.get("src_ip", ""))
        dst_ip = str(tx.get("dst_ip", ""))

        src_port = tx.get("src_port", "")
        dst_port = tx.get("dst_port", "")

        geo_country = tx.get("geo_country", tx.get("country", "?"))

        asn = tx.get("asn", "?")

        if src_ip and src_ip != "nan":
            ip_node = f"ip:{src_ip}"

            if ip_node not in G:
                G.add_node(
                    ip_node,
                    node_type="ip",
                    label=src_ip,
                    ip=src_ip,
                    country=str(geo_country),
                    asn=str(asn),
                )

            G.add_edge(
                ip_node,
                tx_node,
                relation="observed_source",
                port=str(src_port),
                timestamp=str(tx.get("timestamp", "")),
            )

        if dst_ip and dst_ip != "nan":
            ip_node = f"ip:{dst_ip}"

            if ip_node not in G:
                G.add_node(
                    ip_node,
                    node_type="ip",
                    label=dst_ip,
                    ip=dst_ip,
                    country=str(geo_country),
                    asn=str(asn),
                )

            G.add_edge(
                tx_node,
                ip_node,
                relation="observed_destination",
                port=str(dst_port),
                timestamp=str(tx.get("timestamp", "")),
            )

        # -----------------------------------------------------------
        # INPUT ADDRESSES
        # -----------------------------------------------------------

        for i, address in enumerate(input_addresses):
            address = str(address)

            wid = addr_to_wallet.get(address)

            if wid is None:
                unmatched_addresses += 1

                # Keep unknown/disposable addresses
                wallet_node = f"wallet_address:{address}"

                if wallet_node not in G:
                    G.add_node(
                        wallet_node,
                        node_type="wallet",
                        wallet_type="disposable_or_unresolved",
                        label=f"Address {address[:12]}",
                        address=address,
                        wallet_id="unresolved",
                        country="?",
                        asn="?",
                        script_type="?",
                        n_alerts=0,
                        max_priority_tier="none",
                        is_alerted=False,
                    )

            else:
                wallet_node = f"wallet:{wid}"
                info = wallet_info.get(wid, {})

                if wallet_node not in G:
                    G.add_node(
                        wallet_node,
                        node_type="wallet",
                        wallet_type="canonical",
                        label=f"Wallet #{wid}",
                        wallet_id=wid,
                        address=address,
                        country=info.get("country", "?"),
                        asn=info.get("asn", "?"),
                        script_type=info.get("script_type", "?"),
                    )

            amount = float(input_amounts[i]) if i < len(input_amounts) else 0.0

            G.add_edge(
                wallet_node,
                tx_node,
                relation="input",
                address=address,
                amount_btc=amount,
                input_index=i,
            )

        # -----------------------------------------------------------
        # OUTPUT ADDRESSES
        # -----------------------------------------------------------

        for i, address in enumerate(output_addresses):
            address = str(address)

            wid = addr_to_wallet.get(address)

            if wid is None:
                unmatched_addresses += 1

                wallet_node = f"wallet_address:{address}"

                if wallet_node not in G:
                    G.add_node(
                        wallet_node,
                        node_type="wallet",
                        wallet_type="disposable_or_unresolved",
                        label=f"Address {address[:12]}",
                        address=address,
                        wallet_id="unresolved",
                        country="?",
                        asn="?",
                        script_type="?",
                        n_alerts=0,
                        max_priority_tier="none",
                        is_alerted=False,
                    )

            else:
                wallet_node = f"wallet:{wid}"
                info = wallet_info.get(wid, {})

                if wallet_node not in G:
                    G.add_node(
                        wallet_node,
                        node_type="wallet",
                        wallet_type="canonical",
                        label=f"Wallet #{wid}",
                        wallet_id=wid,
                        address=address,
                        country=info.get("country", "?"),
                        asn=info.get("asn", "?"),
                        script_type=info.get("script_type", "?"),
                    )

            amount = float(output_amounts[i]) if i < len(output_amounts) else 0.0

            G.add_edge(
                tx_node,
                wallet_node,
                relation="output",
                address=address,
                amount_btc=amount,
                output_index=i,
            )

    # ---------------------------------------------------------------
    # ALERT METADATA
    # ---------------------------------------------------------------

    wallet_alerts = defaultdict(list)

    for _, row in alerts_df.iterrows():
        wid = row["canonical_wallet_id"]

        wallet_alerts[wid].append(row["priority_tier"])

    for node, data in G.nodes(data=True):
        if data.get("node_type") != "wallet":
            continue

        wid = data.get("wallet_id")

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


# -------------------------------------------------------------------
# FOCUSED RISK GRAPH
# -------------------------------------------------------------------


def build_entity_risk_subgraph(
    G,
    max_alerted_transactions=75,
    max_related_wallets=150,
):
    """Build compact investigator graph."""

    priority_rank = {
        "high": 3,
        "medium-high": 2,
        "worth reviewing": 1,
        "none": 0,
    }

    alerted_tx = [
        (node, data)
        for node, data in G.nodes(data=True)
        if data.get("node_type") == "transaction" and data.get("is_alerted")
    ]

    alerted_tx.sort(
        key=lambda item: (
            priority_rank.get(
                item[1].get("priority_tier"),
                0,
            ),
            item[1].get("txid", ""),
        ),
        reverse=True,
    )

    selected_tx = [node for node, _ in alerted_tx[:max_alerted_transactions]]

    keep = set(selected_tx)

    for tx_node in selected_tx:
        keep.update(G.predecessors(tx_node))
        keep.update(G.successors(tx_node))

    wallet_nodes = [node for node in keep if G.nodes[node].get("node_type") == "wallet"]

    wallet_nodes.sort(
        key=lambda node: (
            bool(G.nodes[node].get("is_alerted")),
            G.nodes[node].get("n_alerts", 0),
            G.degree(node),
        ),
        reverse=True,
    )

    allowed_wallets = set(wallet_nodes[:max_related_wallets])

    keep = {
        node
        for node in keep
        if G.nodes[node].get("node_type") != "wallet" or node in allowed_wallets
    }

    return G.subgraph(keep).copy()


# -------------------------------------------------------------------
# PYVIS
# -------------------------------------------------------------------


def render_pyvis_html(G, out_path):
    """Render graph to interactive HTML."""

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
            tier = data.get(
                "max_priority_tier",
                "none",
            )

            color = TIER_COLOR.get(
                tier,
                "#a8b0b8",
            )

            size = 24 if data.get("is_alerted") else 16

            if data.get("wallet_type") == "disposable_or_unresolved":
                label = data.get(
                    "label",
                    str(node),
                )
            else:
                label = data.get(
                    "label",
                    str(node),
                )

            title = (
                f"<b>{label}</b><br>"
                f"type: wallet<br>"
                f"wallet type: "
                f"{data.get('wallet_type', '?')}<br>"
                f"address: "
                f"{data.get('address', '?')}<br>"
                f"country: "
                f"{data.get('country', '?')}<br>"
                f"ASN: "
                f"{data.get('asn', '?')}<br>"
                f"alerts: "
                f"{data.get('n_alerts', 0)}"
            )

        elif node_type == "transaction":
            tier = data.get(
                "priority_tier",
                "none",
            )

            color = TIER_COLOR.get(
                tier,
                "#a8b0b8",
            )

            size = 20 if data.get("is_alerted") else 12

            label = f"TX {data.get('label', '')}"

            title = (
                f"<b>Transaction</b><br>"
                f"txid: {data.get('txid')}<br>"
                f"timestamp: "
                f"{data.get('timestamp')}<br>"
                f"input total: "
                f"{data.get('input_total_btc', 0):.6f} BTC<br>"
                f"output total: "
                f"{data.get('output_total_btc', 0):.6f} BTC<br>"
                f"fee: "
                f"{data.get('fee', 0):.6f} BTC<br>"
                f"priority: "
                f"{data.get('priority_tier')}"
            )

        else:
            color = "#4dabf7"
            size = 13

            label = data.get(
                "label",
                str(node),
            )

            title = (
                f"<b>IP</b><br>"
                f"IP: {data.get('ip', label)}<br>"
                f"country: "
                f"{data.get('country', '?')}<br>"
                f"ASN: "
                f"{data.get('asn', '?')}"
            )

        net.add_node(
            node,
            label=label,
            title=title,
            color=color,
            size=size,
            borderWidth=(3 if data.get("is_alerted") else 1),
        )

    for u, v, data in G.edges(data=True):
        relation = data.get(
            "relation",
            "",
        )

        amount = data.get("amount_btc")

        port = data.get("port")

        edge_title = relation

        if amount is not None:
            edge_title += f"<br>amount: {float(amount):.6f} BTC"

        if port:
            edge_title += f"<br>port: {port}"

        net.add_edge(
            u,
            v,
            title=edge_title,
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

    html = net.generate_html(notebook=False)

    with open(
        out_path,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(html)


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------


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
        f"Full wallet graph: "
        f"{wallet_G.number_of_nodes()} wallets, "
        f"{wallet_G.number_of_edges()} wallet-pair links"
    )

    if unmatched:
        print(f"  ({unmatched} address references had no matching wallet)")

    if skipped:
        print(f"  ({skipped} transactions skipped -- missing endpoint)")

    wallet_export = wallet_G.copy()

    for _, _, data in wallet_export.edges(data=True):
        data["txids"] = ",".join(data["txids"])

        if data["max_priority_tier"] is None:
            data["max_priority_tier"] = "none"

    nx.write_graphml(
        wallet_export,
        "output/entity_graph_full.graphml",
    )

    print("Wrote output/entity_graph_full.graphml")

    # ---------------------------------------------------------------
    # ENTITY GRAPH
    # ---------------------------------------------------------------

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
                [d.get("node_type") for _, d in entity_G.nodes(data=True)]
            ).value_counts()
        ),
    )

    if entity_unmatched:
        print(f"  ({entity_unmatched} address references were unresolved and retained)")

    if entity_skipped:
        print(f"  ({entity_skipped} transactions skipped)")

    nx.write_graphml(
        entity_G,
        "output/entity_graph_ip_tx_wallet.graphml",
    )

    print("Wrote output/entity_graph_ip_tx_wallet.graphml")

    # ---------------------------------------------------------------
    # FOCUSED GRAPH
    # ---------------------------------------------------------------

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

    print(f"Wrote {focused_graphml}")

    focused_html = "output/entity_graph_ip_tx_wallet_focused.html"

    render_pyvis_html(
        risk_G,
        focused_html,
    )

    print(f"Wrote {focused_html}")


if __name__ == "__main__":
    main()
