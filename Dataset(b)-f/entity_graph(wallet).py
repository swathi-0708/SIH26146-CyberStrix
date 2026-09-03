"""
Wallet-to-Wallet Graph Builder for SIH PS 26146.

Shows canonical wallet relationships only:
    Wallet A  --->  Wallet B

Disposable addresses and IP nodes are excluded so the graph stays focused
on entity-level money flow.
"""

from collections import defaultdict

import networkx as nx
import pandas as pd

from ingest import load_transactions


# ---------------------------------------------------------
# INPUT FILES
# ---------------------------------------------------------

TX_PATH = "output/transactions.csv"
WALLETS_PATH = "output/wallets_reference.csv"
ALERTS_PATH = "output/alerts.csv"


# ---------------------------------------------------------
# RISK SETTINGS
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# ADDRESS -> WALLET MAPPING
# ---------------------------------------------------------

def build_address_to_wallet(wallets_df):
    return dict(
        zip(
            wallets_df["address"],
            wallets_df["wallet_id"]
        )
    )


# ---------------------------------------------------------
# BUILD WALLET GRAPH
# ---------------------------------------------------------

def build_wallet_graph(tx_df, wallets_df, alerts_df):

    addr_to_wallet = build_address_to_wallet(wallets_df)

    wallet_info = (
        wallets_df
        .set_index("wallet_id")
        .to_dict("index")
    )

    # -----------------------------------------------------
    # Roll up alerts to wallet level
    # -----------------------------------------------------

    alert_rollup = defaultdict(
        lambda: {
            "n_alerts": 0,
            "max_tier": None
        }
    )

    for _, row in alerts_df.iterrows():

        wid = row["canonical_wallet_id"]
        tier = row["priority_tier"]

        rec = alert_rollup[wid]

        rec["n_alerts"] += 1

        if (
            rec["max_tier"] is None
            or TIER_RANK[tier]
            > TIER_RANK[rec["max_tier"]]
        ):
            rec["max_tier"] = tier

    # -----------------------------------------------------
    # Directed wallet graph
    # -----------------------------------------------------

    G = nx.DiGraph()

    # Add ALL canonical wallets first.
    # This means wallets with no direct transaction
    # relationship still exist as nodes.

    for wid, info in wallet_info.items():

        rollup = alert_rollup.get(
            wid,
            {
                "n_alerts": 0,
                "max_tier": None
            }
        )

        G.add_node(
            wid,

            node_type="wallet",

            wallet_id=wid,

            address=info.get(
                "address",
                "?"
            ),

            country=info.get(
                "country",
                "?"
            ),

            asn=info.get(
                "asn",
                "?"
            ),

            script_type=info.get(
                "script_type",
                "?"
            ),

            typical_amount_btc=info.get(
                "typical_amount_btc",
                0.0
            ),

            n_alerts=rollup["n_alerts"],

            max_priority_tier=(
                rollup["max_tier"]
                or "none"
            ),

            is_alerted=(
                rollup["n_alerts"] > 0
            ),
        )

    # -----------------------------------------------------
    # Process transactions
    # -----------------------------------------------------

    skipped = 0
    wallet_transactions = 0

    for _, tx in tx_df.iterrows():

        input_addresses = tx["input_addresses"]
        output_addresses = tx["output_addresses"]
        output_amounts = tx["output_amounts"]

        # Resolve addresses to canonical wallets
        input_wallets = [
            addr_to_wallet[a]
            for a in input_addresses
            if a in addr_to_wallet
        ]

        output_wallets = [
            addr_to_wallet[a]
            for a in output_addresses
            if a in addr_to_wallet
        ]

        # Remove duplicates within the same transaction
        input_wallets = list(
            dict.fromkeys(input_wallets)
        )

        output_wallets = list(
            dict.fromkeys(output_wallets)
        )

        # No resolvable wallet endpoints
        if not input_wallets or not output_wallets:
            skipped += 1
            continue

        wallet_transactions += 1

        # -------------------------------------------------
        # Total output BTC
        # -------------------------------------------------

        total_output_btc = sum(
            output_amounts
        ) if output_amounts else 0.0

        # -------------------------------------------------
        # Create wallet -> wallet relationships
        # -------------------------------------------------

        n_pairs = (
            len(input_wallets)
            * len(output_wallets)
        )

        if n_pairs == 0:
            continue

        btc_per_pair = (
            total_output_btc / n_pairs
        )

        fee_per_pair = (
            float(tx["fee"]) / n_pairs
        )

        for src_wallet in input_wallets:

            for dst_wallet in output_wallets:

                # Ignore self-transfers
                if src_wallet == dst_wallet:
                    continue

                # -------------------------------------------------
                # Existing relationship
                # -------------------------------------------------

                if G.has_edge(
                    src_wallet,
                    dst_wallet
                ):

                    edge = G[
                        src_wallet
                    ][
                        dst_wallet
                    ]

                    edge["n_tx"] += 1

                    edge["total_btc"] += (
                        btc_per_pair
                    )

                    edge["total_fee"] += (
                        fee_per_pair
                    )

                    edge["txids"].append(
                        tx["txid"]
                    )

                    # Keep strongest alert tier
                    tx_tier = None

                    if tx["txid"] in alerts_df["txid"].values:

                        matching = alerts_df[
                            alerts_df["txid"]
                            == tx["txid"]
                        ]

                        if not matching.empty:
                            tx_tier = matching.iloc[0][
                                "priority_tier"
                            ]

                    if tx_tier:

                        current = edge.get(
                            "max_priority_tier"
                        )

                        if (
                            current is None
                            or current == "none"
                            or TIER_RANK[tx_tier]
                            > TIER_RANK[current]
                        ):
                            edge[
                                "max_priority_tier"
                            ] = tx_tier

                # -------------------------------------------------
                # New relationship
                # -------------------------------------------------

                else:

                    tx_tier = None

                    if tx["txid"] in alerts_df["txid"].values:

                        matching = alerts_df[
                            alerts_df["txid"]
                            == tx["txid"]
                        ]

                        if not matching.empty:
                            tx_tier = matching.iloc[0][
                                "priority_tier"
                            ]

                    G.add_edge(
                        src_wallet,
                        dst_wallet,

                        edge_type="money",

                        n_tx=1,

                        total_btc=btc_per_pair,

                        total_fee=fee_per_pair,

                        txids=[
                            tx["txid"]
                        ],

                        max_priority_tier=(
                            tx_tier
                            or "none"
                        ),
                    )

    return (
        G,
        skipped,
        wallet_transactions
    )


# ---------------------------------------------------------
# EXPORT GRAPHML
# ---------------------------------------------------------

def export_graphml(G, path):

    G_export = G.copy()

    # GraphML doesn't support Python lists.
    for _, _, data in G_export.edges(
        data=True
    ):

        data["txids"] = ",".join(
            str(x)
            for x in data["txids"]
        )

    nx.write_graphml(
        G_export,
        path
    )


# ---------------------------------------------------------
# INTERACTIVE PYVIS GRAPH
# ---------------------------------------------------------

def render_pyvis_html(
    G,
    out_path
):

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
        spring_length=160,
        spring_strength=0.02,
    )

    # -----------------------------------------------------
    # Nodes
    # -----------------------------------------------------

    for wallet_id, data in G.nodes(
        data=True
    ):

        tier = data.get(
            "max_priority_tier",
            "none"
        )

        color = TIER_COLOR.get(
            tier,
            TIER_COLOR["none"]
        )

        alerts = data.get(
            "n_alerts",
            0
        )

        # Bigger nodes for repeatedly alerted wallets
        size = 16 + (
            5 * min(alerts, 6)
        )

        label = f"#{wallet_id}"

        title = (
            f"<b>Wallet #{wallet_id}</b><br>"
            f"Address: {data.get('address', '?')}<br>"
            f"Country: {data.get('country', '?')}<br>"
            f"ASN: {data.get('asn', '?')}<br>"
            f"Script: {data.get('script_type', '?')}<br>"
            f"Alerts: {alerts}<br>"
            f"Worst tier: {tier}"
        )

        net.add_node(
            wallet_id,

            label=label,

            title=title,

            color=color,

            size=size,

            shape="dot",

            borderWidth=(
                4
                if data.get("is_alerted")
                else 1
            ),
        )

    # -----------------------------------------------------
    # Edges
    # -----------------------------------------------------

    for src, dst, data in G.edges(
        data=True
    ):

        tier = data.get(
            "max_priority_tier",
            "none"
        )

        color = TIER_COLOR.get(
            tier,
            "#4a5058"
        )

        n_tx = data.get(
            "n_tx",
            1
        )

        total_btc = data.get(
            "total_btc",
            0.0
        )

        width = 1 + min(
            n_tx,
            8
        )

        title = (
            f"<b>Wallet #{src} → Wallet #{dst}</b><br>"
            f"Transactions: {n_tx}<br>"
            f"Total BTC: {total_btc:.4f}<br>"
            f"Priority: {tier}"
        )

        net.add_edge(
            src,
            dst,

            arrows="to",

            width=width,

            color=color,

            title=title,
        )

    # -----------------------------------------------------
    # Physics / interaction
    # -----------------------------------------------------

    net.set_options("""
    {
        "physics": {
            "stabilization": {
                "iterations": 200
            }
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "navigationButtons": true,
            "keyboard": true
        }
    }
    """)

    # UTF-8 output — avoids Windows cp1252 problem
    html = net.generate_html(
        notebook=False
    )

    with open(
        out_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(html)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print(
        "Loading transaction data..."
    )

    tx_df = load_transactions(
        TX_PATH
    )

    wallets_df = pd.read_csv(
        WALLETS_PATH
    )

    alerts_df = pd.read_csv(
        ALERTS_PATH
    )

    print(
        f"Transactions: {len(tx_df)}"
    )

    print(
        f"Canonical wallets: "
        f"{len(wallets_df)}"
    )

    print(
        f"Alerts: {len(alerts_df)}"
    )

    # -----------------------------------------------------
    # Build
    # -----------------------------------------------------

    G, skipped, wallet_transactions = (
        build_wallet_graph(
            tx_df,
            wallets_df,
            alerts_df,
        )
    )

    # -----------------------------------------------------
    # Statistics
    # -----------------------------------------------------

    alerted_wallets = sum(
        1
        for _, data
        in G.nodes(data=True)
        if data.get("is_alerted")
    )

    print(
        "\nWallet-to-Wallet Graph:"
    )

    print(
        f"  Nodes: {G.number_of_nodes()}"
    )

    print(
        f"  Wallet relationships: "
        f"{G.number_of_edges()}"
    )

    print(
        f"  Alerted wallets: "
        f"{alerted_wallets}"
    )

    print(
        f"  Transactions with "
        f"wallet endpoints: "
        f"{wallet_transactions}"
    )

    print(
        f"  Transactions skipped: "
        f"{skipped}"
    )

    # -----------------------------------------------------
    # Output paths
    # -----------------------------------------------------

    graphml_path = (
        "output/"
        "entity_graph_wallet_wallet.graphml"
    )

    html_path = (
        "output/"
        "entity_graph_wallet_wallet.html"
    )

    # -----------------------------------------------------
    # Save GraphML
    # -----------------------------------------------------

    export_graphml(
        G,
        graphml_path
    )

    print(
        f"\nWrote {graphml_path}"
    )

    # -----------------------------------------------------
    # Save interactive HTML
    # -----------------------------------------------------

    render_pyvis_html(
        G,
        html_path
    )

    print(
        f"Wrote {html_path}"
    )

    print(
        "\nWallet-to-wallet graph "
        "generation complete."
    )


# ---------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":
    main()