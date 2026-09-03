import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
from pathlib import Path
import tempfile
import re


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

ALERTS_FILE = OUTPUT_DIR / "alerts.csv"
EXPLAINED_FILE = OUTPUT_DIR / "alerts_explained.csv"

# Main IP -> Transaction -> Wallet GraphML
GRAPHML_FILE = OUTPUT_DIR / "entity_graph_ip_tx_wallet.graphml"

# Fallback HTML
GRAPH_HTML_FILE = OUTPUT_DIR / "entity_graph_ip_tx_wallet_focused.html"

# Additional graph views
RISK_GRAPH_HTML_FILE = OUTPUT_DIR / "entity_graph_risk.html"
WALLET_GRAPHML_FILE = OUTPUT_DIR / "entity_graph_wallet_wallet.graphml"


# --------------------------------------------------------------
# Single source of truth for tier -> color, used everywhere a tier
# is drawn (metrics, table, graph nodes) so the same tier is never
# two different colors in two different places. Only the three
# tiers build_alerts.py actually produces -- no "critical", it
# doesn't exist in this pipeline.
# --------------------------------------------------------------
TIER_COLORS = {
    "high": "#d64545",
    "medium-high": "#c98a3e",
    "worth reviewing": "#b8a13c",
    "none": "#5b6472",
}
BG_COLOR = "#12141a"
PANEL_COLOR = "#181b22"
ACCENT_COLOR = "#4f8ff7"


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="CyberStrix Investigator",
    page_icon=None,
    layout="wide",
)

# --------------------------------------------------------------
# Theme: Unified dark ground with high-contrast, fully readable
# inputs, dropdowns, labels, and buttons across the entire dashboard.
# --------------------------------------------------------------
st.markdown(
    f"""
    <style>
    /* --------------------------------------------------------
       CyberStrix visual theme
       Professional dark dashboard with high-contrast, fully
       readable form controls, inputs, dropdowns, and buttons.
       -------------------------------------------------------- */
    .stApp {{
        background: {BG_COLOR};
        color: #e8ecf2;
    }}

    /* Sidebar container */
    [data-testid="stSidebar"] {{
        background: {PANEL_COLOR};
        border-right: 1px solid #2b303a;
    }}

    /* Typography */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4,
    .stApp h5, .stApp h6 {{
        color: #f5f7fa !important;
        font-weight: 600 !important;
    }}

    .stApp p, .stApp li {{
        color: #e8ecf2 !important;
    }}

    .stApp [data-testid="stCaptionContainer"] {{
        color: #94a3b8 !important;
    }}

    /* Metric cards */
    [data-testid="stMetric"] {{
        background: {PANEL_COLOR};
        border: 1px solid #2b303a;
        padding: 14px 18px;
        border-radius: 8px;
    }}

    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] * {{
        color: #94a3b8 !important;
        font-size: 13px !important;
        font-weight: 500 !important;
    }}

    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] * {{
        color: #f8fafc !important;
        font-weight: 600 !important;
    }}

    /* Widget labels - readable high-contrast on dark background */
    .stApp label,
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] *,
    [data-testid="stWidgetLabel"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] * {{
        color: #e2e8f0 !important;
        -webkit-text-fill-color: #e2e8f0 !important;
        font-weight: 500 !important;
        font-size: 14px !important;
    }}

    /* Native text inputs, number inputs, text areas */
    .stApp input[type="text"],
    .stApp input[type="number"],
    .stApp textarea,
    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input,
    [data-testid="stTextArea"] textarea {{
        background-color: #f1f5f9 !important;
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
        caret-color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 6px !important;
        font-size: 14px !important;
        font-weight: 500 !important;
    }}

    .stApp input::placeholder,
    .stApp textarea::placeholder {{
        color: #64748b !important;
        -webkit-text-fill-color: #64748b !important;
        opacity: 1 !important;
    }}

    /* BaseWeb selectboxes (closed state & input) */
    .stApp [data-baseweb="select"] > div,
    [data-testid="stSidebar"] [data-baseweb="select"] > div {{
        background-color: #f1f5f9 !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 6px !important;
    }}

    /* Ensure text inside selectbox is dark and clearly visible */
    .stApp [data-baseweb="select"] *,
    [data-testid="stSidebar"] [data-baseweb="select"] *,
    .stApp [data-baseweb="select"] span,
    .stApp [data-baseweb="select"] div,
    .stApp [data-baseweb="select"] input,
    .stApp [data-baseweb="select"] p {{
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
        fill: #0f172a !important;
        font-size: 14px !important;
    }}

    /* BaseWeb dropdown menus / popovers (open list state) */
    div[data-baseweb="popover"],
    div[data-baseweb="menu"],
    ul[role="listbox"],
    div[role="listbox"] {{
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 6px !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4) !important;
    }}

    li[role="option"],
    div[role="option"],
    [data-baseweb="menu"] li {{
        background-color: #ffffff !important;
        color: #0f172a !important;
    }}

    li[role="option"] *,
    div[role="option"] *,
    [data-baseweb="menu"] li * {{
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
    }}

    /* Option hover / active state */
    li[role="option"]:hover,
    li[role="option"][aria-selected="true"],
    div[role="option"]:hover,
    div[role="option"][aria-selected="true"] {{
        background-color: #e0e7ff !important;
    }}

    li[role="option"]:hover *,
    li[role="option"][aria-selected="true"] *,
    div[role="option"]:hover *,
    div[role="option"][aria-selected="true"] * {{
        color: #1e3a8a !important;
        -webkit-text-fill-color: #1e3a8a !important;
        font-weight: 600 !important;
    }}

    /* Multiselect tags */
    .stApp [data-baseweb="tag"],
    [data-testid="stSidebar"] [data-baseweb="tag"] {{
        background-color: #dbeafe !important;
        border: 1px solid #bfdbfe !important;
        border-radius: 4px !important;
    }}

    .stApp [data-baseweb="tag"] *,
    [data-testid="stSidebar"] [data-baseweb="tag"] * {{
        color: #1e3a8a !important;
        -webkit-text-fill-color: #1e3a8a !important;
        fill: #1e3a8a !important;
        font-weight: 500 !important;
    }}

    /* Buttons - dark slate with crisp, high-contrast readable text */
    .stButton > button {{
        background-color: #242936 !important;
        color: #f8fafc !important;
        -webkit-text-fill-color: #f8fafc !important;
        border: 1px solid #3b4354 !important;
        border-radius: 6px !important;
        padding: 8px 18px !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2) !important;
        transition: background-color 0.15s ease, border-color 0.15s ease !important;
    }}

    .stButton > button:hover {{
        background-color: #32394a !important;
        border-color: {ACCENT_COLOR} !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }}

    .stButton > button:active {{
        background-color: #1b1f2b !important;
    }}

    .stButton > button * {{
        color: #f8fafc !important;
        -webkit-text-fill-color: #f8fafc !important;
    }}

    /* Sliders */
    [data-testid="stSlider"] * {{
        color: #e2e8f0 !important;
    }}

    /* Tabs */
    .stApp [data-testid="stTabs"] [data-baseweb="tab-list"] {{
        background-color: transparent !important;
        border-bottom: 1px solid #2b303a !important;
    }}

    .stApp [data-testid="stTabs"] button {{
        color: #94a3b8 !important;
        -webkit-text-fill-color: #94a3b8 !important;
        font-size: 14px !important;
    }}

    .stApp [data-testid="stTabs"] button[aria-selected="true"] {{
        color: {ACCENT_COLOR} !important;
        -webkit-text-fill-color: {ACCENT_COLOR} !important;
        font-weight: 600 !important;
        border-bottom-color: {ACCENT_COLOR} !important;
    }}

    /* Dataframe container */
    div[data-testid="stDataFrame"] {{
        border: 1px solid #2b303a;
        border-radius: 8px;
    }}

    /* Clean iframe container - remove borders, blend with dark background */
    iframe {{
        border: none !important;
        background-color: {BG_COLOR} !important;
    }}

    [data-testid="stCustomComponentV1"] {{
        border: none !important;
        background-color: {BG_COLOR} !important;
    }}

    /* Clean spacing */
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 3rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD DATA
# ============================================================


@st.cache_data
def load_data():
    alerts = pd.read_csv(ALERTS_FILE)
    explained = pd.read_csv(EXPLAINED_FILE)

    # CSV reloads can differ in whitespace/object representation after a
    # pipeline rerun. Normalize TXIDs once so alert/explanation lookups
    # remain reliable.
    if "txid" in alerts.columns:
        alerts["txid"] = alerts["txid"].astype(str).str.strip()
    if "txid" in explained.columns:
        explained["txid"] = explained["txid"].astype(str).str.strip()

    return alerts, explained


try:
    alerts, explained = load_data()

except Exception as e:
    st.error(f"Could not load output files: {e}")
    st.stop()


@st.cache_resource
def load_graph():
    """Load the full IP -> Transaction -> Wallet GraphML once."""
    if not GRAPHML_FILE.exists():
        return None

    try:
        return nx.read_graphml(GRAPHML_FILE)
    except Exception as e:
        st.error(f"Could not load entity graph: {e}")
        return None


G = load_graph()


# ============================================================
# GRAPH FUNCTIONS
# ============================================================


def find_transaction_node(G, txid):
    """
    Find the transaction node corresponding to the selected TXID.
    Handles different node-id/label formats.
    """

    txid = str(txid)

    for node, data in G.nodes(data=True):
        node_text = str(node)

        label = str(data.get("label", ""))

        node_type = str(data.get("node_type", "")).lower()

        if txid == node_text:
            return node

        if txid in label:
            return node

        # Transaction nodes in our graph normally contain TX
        if node_type == "transaction" and txid in node_text:
            return node

    return None


def build_investigation_graph(txid, wallet_id):

    if not GRAPHML_FILE.exists():
        return None

    try:
        G = nx.read_graphml(GRAPHML_FILE)

    except Exception as e:
        st.error(f"Could not read entity graph: {e}")
        return None

    tx_node = find_transaction_node(G, txid)

    if tx_node is None:
        return None

    # --------------------------------------------------------
    # Find the wallet belonging to this alert
    # --------------------------------------------------------

    wallet_node = None
    wallet_id_text = str(wallet_id)

    for node, data in G.nodes(data=True):
        node_id = str(node)
        label = str(data.get("label", ""))

        node_type = str(data.get("node_type", data.get("type", ""))).lower()

        combined = (node_id + " " + label).lower()

        if "wallet" in node_type:
            # Prefer an exact Wallet #ID match.
            if (
                f"wallet #{wallet_id_text}".lower() in combined
                or f"wallet_{wallet_id_text}".lower() in combined
                or f"wallet-{wallet_id_text}".lower() in combined
            ):
                wallet_node = node
                break

    # --------------------------------------------------------
    # Build the investigation neighborhood
    # --------------------------------------------------------

    nodes_to_keep = {tx_node}

    # Direct transaction connections
    nodes_to_keep.update(G.neighbors(tx_node))

    # Explicitly include the wallet associated with the alert
    if wallet_node is not None:
        nodes_to_keep.add(wallet_node)

        # Wallet's direct connections
        nodes_to_keep.update(G.neighbors(wallet_node))

        # Incoming connections for directed graphs
        if G.is_directed():
            nodes_to_keep.update(G.predecessors(wallet_node))

    investigation_G = G.subgraph(nodes_to_keep).copy()

    # --------------------------------------------------------
    # Make TX -> Wallet relationship visible if it is absent
    # --------------------------------------------------------

    if wallet_node is not None:
        if not (
            investigation_G.has_edge(tx_node, wallet_node)
            or investigation_G.has_edge(wallet_node, tx_node)
        ):
            investigation_G.add_edge(tx_node, wallet_node, label="belongs_to_wallet")

    return investigation_G


def clean_graph_html(html_str):
    """
    Remove the default PyVis/Bootstrap white card border and white background,
    making the embedded graph container blend seamlessly into the dark theme.
    """
    if not html_str:
        return ""
    dark_style = f"""
    <style>
      html, body {{
        margin: 0 !important;
        padding: 0 !important;
        background-color: {BG_COLOR} !important;
        overflow: hidden !important;
      }}
      .card {{
        background-color: {BG_COLOR} !important;
        border: none !important;
        box-shadow: none !important;
        margin: 0 !important;
        padding: 0 !important;
      }}
      .card-body {{
        padding: 0 !important;
        margin: 0 !important;
        background-color: {BG_COLOR} !important;
      }}
      #mynetwork {{
        border: 1px solid #2b303a !important;
        border-radius: 8px !important;
        background-color: {BG_COLOR} !important;
      }}
      .vis-navigation {{
        background: transparent !important;
      }}
    </style>
    """
    if "</head>" in html_str:
        return html_str.replace("</head>", f"{dark_style}</head>", 1)
    return dark_style + html_str


def resolve_entity(G, entity):
    """
    Resolve user input (wallet ID, node ID, or label) to an exact node in G.
    Avoids greedy suffix/substring matching so short IDs resolve with 100% precision.
    """
    entity = str(entity).strip()
    if entity in G:
        return entity
    if f"wallet:{entity}" in G:
        return f"wallet:{entity}"
    if f"tx:{entity}" in G:
        return f"tx:{entity}"
    if f"ip:{entity}" in G:
        return f"ip:{entity}"
    for node, data in G.nodes(data=True):
        if (
            str(data.get("wallet_id", "")) == entity
            or str(data.get("label", "")) == entity
        ):
            return node
    return None


def n_hop_query(G, entity, hops=2):
    resolved = resolve_entity(G, entity)
    if resolved is None:
        return None

    lengths = nx.single_source_shortest_path_length(
        G.to_undirected(), resolved, cutoff=hops
    )

    return G.subgraph(lengths.keys()).copy()


def shortest_path_query(G, entity_a, entity_b):
    """Find shortest relationship path between two entities."""
    resolved_a = resolve_entity(G, entity_a)
    resolved_b = resolve_entity(G, entity_b)

    if resolved_a is None or resolved_b is None:
        return None

    try:
        return nx.shortest_path(G.to_undirected(), resolved_a, resolved_b)
    except nx.NetworkXNoPath:
        return None


def get_path_subgraph(G, path):
    """Create a graph containing only the nodes/edges in a path."""
    if not path or len(path) < 2:
        return None

    return G.subgraph(path).copy()


def fund_flow_query(G, entity, hops=2, direction="both"):
    resolved = resolve_entity(G, entity)
    if resolved is None:
        return None

    if direction == "outgoing":
        search_graph = G
    elif direction == "incoming":
        search_graph = G.reverse()
    else:
        search_graph = G.to_undirected()

    lengths = nx.single_source_shortest_path_length(search_graph, resolved, cutoff=hops)

    return G.subgraph(lengths.keys()).copy()


def render_investigation_graph(G, selected_txid):

    net = Network(
        height="650px",
        width="100%",
        bgcolor=BG_COLOR,
        font_color="#e6e8ec",
        directed=True,
    )

    net.set_options(
        """
        {
          "physics": {
            "enabled": true,
            "stabilization": {
              "iterations": 200
            }
          },
          "interaction": {
            "hover": true,
            "navigationButtons": true,
            "keyboard": true
          },
          "edges": {
            "arrows": {
              "to": {
                "enabled": true
              }
            },
            "smooth": {
              "enabled": true
            }
          }
        }
        """
    )

    # --------------------------------------------------------
    # Add nodes
    # --------------------------------------------------------

    # Computed ONCE per render call, not per-node inside the loop below --
    # rescanning the whole graph for every single node just to check "is
    # this the highlighted one" is O(n^2) and only gets away with it at
    # small focused-subgraph sizes.
    highlighted_node = str(find_transaction_node(G, selected_txid))

    for node, data in G.nodes(data=True):
        node_id = str(node)

        label = str(data.get("label", node_id))

        node_type = str(data.get("node_type", data.get("type", ""))).lower()

        # Determine node type from label/id if needed
        if not node_type:
            if "tx" in label.lower():
                node_type = "transaction"

            elif "wallet" in label.lower():
                node_type = "wallet"

            else:
                node_type = "ip"

        # ----------------------------------------------------
        # Colors
        # ----------------------------------------------------

        if node_id == highlighted_node:
            color = "#e8564f"
            size = 35

        elif node_type == "transaction":
            color = TIER_COLORS.get(
                str(data.get("priority_tier", "")).lower(), "#8a90a3"
            )
            size = 25

        elif node_type == "wallet":
            # Alerted wallets are colored by tier, normal counterparties
            # neutral grey-blue. NOTE: wallet nodes carry their tier under
            # "max_priority_tier" (a wallet can have several alerted
            # transactions; this is the worst one) -- entity_graph.py only
            # ever sets a plain "priority_tier" on TRANSACTION nodes.
            # Reading "priority_tier" here always misses on wallet nodes,
            # so every wallet silently fell through to the same color
            # regardless of alert status.
            priority = str(data.get("max_priority_tier", "")).lower()
            color = TIER_COLORS.get(priority, "#6f9bd1")
            size = 22

        else:
            # IP
            color = ACCENT_COLOR
            size = 17

        # ----------------------------------------------------
        # Tooltip
        # ----------------------------------------------------

        tooltip = f"""
        <b>{label}</b><br>
        Type: {node_type}
        """

        if "risk_score" in data:
            tooltip += f"<br>Risk score: {data['risk_score']}"

        if "priority_tier" in data:
            tooltip += f"<br>Priority: {data['priority_tier']}"

        if "max_priority_tier" in data:
            tooltip += f"<br>Worst alert tier: {data['max_priority_tier']}"

        net.add_node(
            node_id,
            label=label,
            title=tooltip,
            color=color,
            size=size,
        )

    # --------------------------------------------------------
    # Add edges
    # --------------------------------------------------------

    for source, target, data in G.edges(data=True):
        edge_label = str(
            data.get("label", data.get("relationship", data.get("type", "")))
        )

        net.add_edge(
            str(source),
            str(target),
            title=edge_label,
            label=edge_label if edge_label else "",
            color="#3a3f4b",
        )

    # --------------------------------------------------------
    # Generate HTML
    # --------------------------------------------------------

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as temp:
        temp_path = temp.name

    net.write_html(
        temp_path,
        notebook=False,
        open_browser=False,
    )

    html = Path(temp_path).read_text(encoding="utf-8")

    return clean_graph_html(html)


# ============================================================
# GRAPH VIEW SELECTOR
# ============================================================


@st.cache_resource
def load_wallet_graph():
    """Load the wallet -> wallet graph generated by entity_graph(wallet).py."""
    if not WALLET_GRAPHML_FILE.exists():
        return None
    try:
        return nx.read_graphml(WALLET_GRAPHML_FILE)
    except Exception as e:
        st.error(f"Could not load wallet graph: {e}")
        return None


def render_wallet_graph(G_wallet, selected_wallet=None):
    """Render the wallet-only graph, optionally focused on one wallet."""
    if G_wallet is None:
        return None

    graph = G_wallet

    if selected_wallet:
        selected_wallet = str(selected_wallet)
        if selected_wallet in graph:
            neighbors = set(graph.neighbors(selected_wallet))
            if graph.is_directed():
                neighbors.update(graph.predecessors(selected_wallet))
            keep = neighbors | {selected_wallet}
            graph = graph.subgraph(keep).copy()

    net = Network(
        height="650px",
        width="100%",
        bgcolor=BG_COLOR,
        font_color="#e6e8ec",
        directed=True,
    )

    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "stabilization": {"iterations": 200}
      },
      "interaction": {
        "hover": true,
        "navigationButtons": true,
        "keyboard": true
      },
      "edges": {
        "arrows": {"to": {"enabled": true}},
        "smooth": {"enabled": true}
      }
    }
    """)

    for node, data in graph.nodes(data=True):
        node_id = str(node)
        label = str(data.get("label", f"Wallet #{node_id}"))

        priority = str(data.get("max_priority_tier", "")).lower()
        if node_id == str(selected_wallet):
            color = "#e8564f"
            size = 34
        else:
            color = TIER_COLORS.get(priority, "#6f9bd1")
            size = 23

        tooltip = (
            f"<b>{label}</b><br>"
            f"Wallet ID: {data.get('wallet_id', node_id)}<br>"
            f"Country: {data.get('country', 'N/A')}<br>"
            f"ASN: {data.get('ASN', data.get('asn', 'N/A'))}<br>"
            f"Alerts: {data.get('n_alerts', 0)}"
        )

        net.add_node(
            node_id,
            label=label,
            title=tooltip,
            color=color,
            size=size,
        )

    for source, target, data in graph.edges(data=True):
        edge_title = (
            f"Transactions: {data.get('n_tx', 'N/A')}<br>"
            f"Total BTC: {data.get('total_btc', 'N/A')}<br>"
            f"Priority: {data.get('max_priority_tier', 'N/A')}"
        )

        net.add_edge(
            str(source),
            str(target),
            title=edge_title,
            label=str(data.get("n_tx", "")),
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as temp:
        temp_path = temp.name

    net.write_html(
        temp_path,
        notebook=False,
        open_browser=False,
    )

    return clean_graph_html(Path(temp_path).read_text(encoding="utf-8"))


def show_graph_explorer():
    """Top-level switch between the three investigation graph views."""
    st.divider()
    st.header("Graph Explorer")
    st.caption(
        "Switch between the three entity-graph views generated by the "
        "CyberStrix pipeline."
    )

    graph_choice = st.selectbox(
        "Graph View",
        [
            "IP → Transaction → Wallet",
            "Risk Cluster",
            "Wallet → Wallet",
        ],
        key="graph_view",
    )

    if graph_choice == "IP → Transaction → Wallet":
        if GRAPH_HTML_FILE.exists():
            html = GRAPH_HTML_FILE.read_text(encoding="utf-8")
            st.components.v1.html(clean_graph_html(html), height=720)
        elif GRAPHML_FILE.exists():
            st.info("Focused HTML not found; the GraphML exists.")
            st.caption("Run entity_graph_modified.py to regenerate the graph.")
        else:
            st.warning("IP → Transaction → Wallet graph not found.")

    elif graph_choice == "Risk Cluster":
        if RISK_GRAPH_HTML_FILE.exists():
            html = RISK_GRAPH_HTML_FILE.read_text(encoding="utf-8")
            st.components.v1.html(clean_graph_html(html), height=720)
        else:
            st.warning(
                "Risk Cluster graph not found. Run entity_graph(clustered).py first."
            )

    else:
        wallet_graph = load_wallet_graph()

        if wallet_graph is None:
            st.warning(
                "Wallet → Wallet graph not found. Run entity_graph(wallet).py first."
            )
            return

        wallet_nodes = sorted(
            [str(n) for n in wallet_graph.nodes()],
            key=lambda x: int(float(x)) if x.replace(".", "", 1).isdigit() else x,
        )

        selected = st.selectbox(
            "Select Wallet",
            wallet_nodes,
            key="wallet_graph_selector",
        )

        if selected:
            outgoing = (
                list(wallet_graph.successors(selected))
                if wallet_graph.is_directed()
                else list(wallet_graph.neighbors(selected))
            )
            incoming = (
                list(wallet_graph.predecessors(selected))
                if wallet_graph.is_directed()
                else []
            )

            connected = sorted(
                {str(x) for x in outgoing + incoming},
                key=lambda x: int(float(x)) if x.replace(".", "", 1).isdigit() else x,
            )

            c1, c2, c3 = st.columns(3)
            c1.metric("Selected Wallet", f"#{selected}")
            c2.metric("Connected Wallets", len(connected))
            c3.metric("Transactions", wallet_graph.degree(selected))

            if connected:
                st.caption(
                    "Showing the selected wallet and its directly connected wallets."
                )
                st.write(
                    "**Connected wallets:** " + ", ".join(f"#{x}" for x in connected)
                )
            else:
                st.info("No direct wallet-to-wallet connections found.")

            wallet_html = render_wallet_graph(wallet_graph, selected)
            st.components.v1.html(wallet_html, height=720)


show_graph_explorer()


# ============================================================
# TITLE
# ============================================================

st.title("CyberStrix Investigator")

st.caption(
    "Offline Bitcoin transaction anomaly detection and investigation dashboard "
    "· runs fully offline, no network calls · synthetic demonstration data, "
    "not real seized evidence"
)


# ============================================================
# METRICS
# ============================================================

total_alerts = len(alerts)

# "critical" is not a tier build_alerts.py ever produces (only high /
# medium-high / worth reviewing) -- counting it in here was dead code.
high_alerts = len(alerts[alerts["priority_tier"] == "high"])

medium_alerts = len(alerts[alerts["priority_tier"] == "medium-high"])

both_alerts = len(alerts[alerts["detector"] == "both"])


col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Alerts", total_alerts)

col2.metric("High Priority", high_alerts)

col3.metric("Medium-High", medium_alerts)

col4.metric("Flagged by Both Detectors", both_alerts)


st.divider()


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header("Investigation Filters")

priority_options = sorted(alerts["priority_tier"].dropna().unique())

selected_priority = st.sidebar.multiselect(
    "Priority",
    priority_options,
    default=priority_options,
)

detector_options = sorted(alerts["detector"].dropna().unique())

selected_detector = st.sidebar.multiselect(
    "Detector",
    detector_options,
    default=detector_options,
)


filtered = alerts[
    alerts["priority_tier"].isin(selected_priority)
    & alerts["detector"].isin(selected_detector)
].copy()


# ============================================================
# ALERT TABLE
# ============================================================

st.header("Investigation Alerts")

st.write(f"Showing **{len(filtered)}** of **{len(alerts)}** alerts.")

display_columns = [
    "txid",
    "canonical_wallet_id",
    "detector",
    "priority_tier",
    "xgb_proba",
    "if_score",
]

table = filtered[display_columns].copy()

table["xgb_proba"] = table["xgb_proba"].round(4)

table["if_score"] = table["if_score"].round(4)

st.dataframe(
    table,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# INVESTIGATE ALERT
# ============================================================

st.divider()

st.header("Investigate Alert")

if len(filtered) == 0:
    st.warning("No alerts match the selected filters.")

else:
    selected_wallet_id = st.selectbox(
        "Select a wallet",
        sorted(
            filtered["canonical_wallet_id"].dropna().astype(str).unique(),
            key=lambda x: int(float(x)) if x.replace(".", "", 1).isdigit() else x,
        ),
    )

    wallet_filtered = filtered[
        filtered["canonical_wallet_id"].astype(str) == selected_wallet_id
    ].copy()

    st.caption(
        f"Wallet **#{selected_wallet_id}** has "
        f"**{len(wallet_filtered)} alert(s)** matching the current filters."
    )

    selected_txid = st.selectbox(
        "Select a transaction for this wallet", wallet_filtered["txid"].tolist()
    )

    # Match explanations by normalized TXID. The pipeline can regenerate
    # alerts.csv independently of the explanation file, so never treat a
    # missing SHAP row as if the alert itself has disappeared.
    selected_txid = str(selected_txid).strip()
    selected = explained[explained["txid"].astype(str).str.strip() == selected_txid]

    # Always use the alert row as the primary source of truth.
    alert_row_match = alerts[alerts["txid"].astype(str).str.strip() == selected_txid]
    row = (
        alert_row_match.iloc[0].copy()
        if not alert_row_match.empty
        else wallet_filtered.iloc[0].copy()
    )

    # Overlay explanation columns when an explanation exists.
    if not selected.empty:
        explanation_row = selected.iloc[0]
        for col in explanation_row.index:
            if col not in row.index or pd.isna(row.get(col)):
                row[col] = explanation_row[col]

    explanation_available = (
        "top_reasons" in row.index
        and pd.notna(row.get("top_reasons"))
        and str(row.get("top_reasons")).strip()
    )

    if not explanation_available:
        st.info(
            "SHAP explanation is not available for this alert yet. "
            "The alert details below are still available."
        )

    # ----------------------------------------------------
    # Alert information
    # ----------------------------------------------------

    c1, c2, c3 = st.columns(3)

    c1.metric("Wallet ID", str(row["canonical_wallet_id"]))

    c2.metric("XGBoost Probability", f"{row['xgb_proba']:.2%}")

    c3.metric("Detector", str(row["detector"]))

    # ----------------------------------------------------
    # Priority
    # ----------------------------------------------------

    st.subheader("Alert Priority")

    priority = str(row["priority_tier"])

    if priority == "high":
        st.error("HIGH PRIORITY")

    elif priority == "medium-high":
        st.warning("MEDIUM-HIGH PRIORITY")

    else:
        st.info(priority.upper())

    # ----------------------------------------------------
    # Explanation
    # ----------------------------------------------------

    st.subheader("Why was it flagged?")

    # Show clean, human-readable reasons first. Keep the numerical
    # feature values as supporting evidence underneath.
    reason_feature_map = {
        "n_inputs": "Unusually few inputs",
        "n_outputs": "Unusual number of outputs",
        "n_unique_input_addresses": "Unusual number of unique input addresses",
        "n_unique_output_addresses": "Unusual number of unique output addresses",
        "total_input_btc": "Large total BTC moved",
        "input_output_ratio": "Unusual input-to-output amount ratio",
        "fan_in_5plus": "Multiple inputs feeding into the transaction",
        "output_min_max_ratio": "Unusual distribution of output amounts",
        "fee": "Unusually high transaction fee",
        "fee_ratio": "Unusual fee relative to transaction value",
        "input_addr_is_recent_output": "Input address was recently used as an output",
        "minutes_since_addr_last_output": "Unusual time since the input address was last active",
        "sender_tx_count_1h": "Unusually high transaction activity in the last hour",
        "sender_tx_count_24h": "Unusually high transaction activity in the last 24 hours",
        "sender_time_since_last_tx_min": "Unusual time since the sender's previous transaction",
        "sender_distinct_asn_last10": "Activity observed across multiple network providers",
        "sender_distinct_ip_last10": "Activity observed across multiple IP addresses",
        "sender_amount_zscore": "Transaction amount differs from the sender's usual pattern",
    }

    reasons = []
    supporting_data = []

    for i in range(1, 4):
        feature = row.get(f"top_feature_{i}", None)
        if pd.isna(feature):
            continue

        feature = str(feature)
        readable_reason = reason_feature_map.get(
            feature, feature.replace("_", " ").capitalize()
        )

        if readable_reason not in reasons:
            reasons.append(readable_reason)

        value = row.get(feature, None)
        if pd.isna(value) or value is None:
            value = row.get(f"feature_value_{i}", None)

        if not pd.isna(value) if value is not None else False:
            supporting_data.append(
                {
                    "Reason": readable_reason,
                    "Supporting value": value,
                }
            )

    # Fallback for older explanation files where top_feature columns
    # are unavailable.
    if not reasons:
        raw_reason = str(row.get("top_reasons", ""))
        reasons = [
            re.sub(r"\s*\([^)]*\)", "", part).strip()
            for part in raw_reason.split(";")
            if part.strip()
        ]

    if reasons:
        for reason in reasons:
            st.markdown(f"- **{reason}**")
    else:
        st.info("No explanation is available for this alert.")

    if supporting_data:
        with st.expander("Supporting evidence"):
            evidence_df = pd.DataFrame(supporting_data)
            st.dataframe(
                evidence_df,
                use_container_width=True,
                hide_index=True,
            )

    # ----------------------------------------------------
    # SHAP
    # ----------------------------------------------------

    st.subheader("Top contributing features")

    feature_data = []

    for i in range(1, 4):
        feature = row.get(f"top_feature_{i}", None)

        shap_value = row.get(f"shap_{i}", None)

        if pd.notna(feature) and pd.notna(shap_value):
            feature_data.append(
                {
                    "Feature": feature,
                    "SHAP Impact": float(shap_value),
                }
            )

    if feature_data:
        shap_df = pd.DataFrame(feature_data)

        shap_df["SHAP Impact"] = shap_df["SHAP Impact"].round(4)

        st.dataframe(
            shap_df,
            use_container_width=True,
            hide_index=True,
        )

    # ----------------------------------------------------
    # Raw data
    # ----------------------------------------------------

    with st.expander("View raw alert data"):
        st.json({str(k): (None if pd.isna(v) else str(v)) for k, v in row.items()})

    # ====================================================
    # WALLET DOSSIER
    # ====================================================

st.divider()
st.header("Wallet Dossier")
all_wallets = sorted(
    alerts["canonical_wallet_id"].dropna().astype(str).unique(),
    key=lambda x: int(x) if x.isdigit() else x,
)

default_dossier_wallet = (
    str(selected_wallet_id)
    if "selected_wallet_id" in locals() and selected_wallet_id in all_wallets
    else all_wallets[0]
    if all_wallets
    else None
)

if default_dossier_wallet is not None:
    dossier_wallet = st.selectbox(
        "Select Wallet",
        all_wallets,
        index=all_wallets.index(default_dossier_wallet),
        key="dossier_wallet",
    )

    wallet_tx = alerts[
        alerts["canonical_wallet_id"].astype(str) == dossier_wallet
    ].copy()
    dossier_row = wallet_tx.iloc[0]

    st.caption(f"Investigation summary for Wallet #{dossier_wallet}")

    st.subheader("Investigation Summary")

    total_transactions = len(wallet_tx)

    unique_ips = set()

    for col in ["src_ip", "dst_ip"]:
        if col in wallet_tx.columns:
            unique_ips.update(wallet_tx[col].dropna().astype(str))

    d1, d2, d3, d4 = st.columns(4)

    d1.metric("Transactions", total_transactions)

    d2.metric("Unique IPs", len(unique_ips))

    d3.metric("Risk Score", f"{float(dossier_row.get('xgb_proba', 0)):.2%}")

    d4.metric("Detector", str(dossier_row.get("detector", "N/A")))

    st.write("**Associated IPs:**")

    if unique_ips:
        st.write(", ".join(list(unique_ips)[:10]))
    else:
        st.write("No IP data available")
    st.subheader("Transaction History")

    display_cols = [
        col
        for col in [
            "txid",
            "canonical_wallet_id",
            "sender",
            "receiver",
            "total_input_btc",
            "total_output_btc",
            "xgb_proba",
            "detector",
            "priority_tier",
        ]
        if col in wallet_tx.columns
    ]

    st.dataframe(wallet_tx[display_cols], use_container_width=True, hide_index=True)
    st.subheader("Network Associations")

    if unique_ips:
        network_data = pd.DataFrame({"IP Address": sorted(unique_ips)})

        st.dataframe(network_data, use_container_width=True, hide_index=True)
    else:
        st.info("No network associations found for this wallet.")

    if "selected_txid" in locals() and "row" in locals():
        # ====================================================
        # ALERT-SPECIFIC GRAPH
        # ====================================================

        st.divider()

        st.header("Investigation Graph")

        st.caption(
            "Showing the selected transaction "
            "and its directly connected IPs and wallets."
        )

        investigation_graph = build_investigation_graph(
            selected_txid, row["canonical_wallet_id"]
        )

        if investigation_graph is not None:
            st.write(
                f"**{len(investigation_graph.nodes)} "
                f"entities** connected to this transaction."
            )

            graph_html = render_investigation_graph(investigation_graph, selected_txid)

            st.components.v1.html(
                graph_html,
                height=680,
            )

        elif GRAPH_HTML_FILE.exists():
            st.info(
                "The selected transaction could not "
                "be isolated from the GraphML file. "
                "Showing the focused graph instead."
            )

            graph_html = GRAPH_HTML_FILE.read_text(encoding="utf-8")

            st.components.v1.html(
                clean_graph_html(graph_html),
                height=680,
            )

        else:
            st.warning("Entity graph not found. Run entity_graph_modified.py first.")
else:
    st.info("No wallet data is available for the dossier.")

# ============================================================
# ADVANCED INVESTIGATION QUERIES
# ============================================================

st.divider()

st.header("Advanced Investigation")

if G is None:
    st.warning("Entity graph could not be loaded. Run entity_graph_modified.py first.")
else:
    # Build a clean, sorted list of available canonical wallets from the graph and dataset
    available_wallets = []
    for node, data in G.nodes(data=True):
        if data.get("node_type") == "wallet":
            wid = str(data.get("wallet_id", "")).strip()
            if wid and wid not in ("unresolved", "None"):
                available_wallets.append(wid)

    available_wallets = sorted(
        set(available_wallets),
        key=lambda x: int(x) if x.isdigit() else x,
    )
    if not available_wallets:
        available_wallets = sorted(
            alerts["canonical_wallet_id"].dropna().astype(str).unique(),
            key=lambda x: int(x) if x.isdigit() else x,
        )

    tab1, tab2, tab3 = st.tabs(["N-Hop Analysis", "Wallet Path", "Fund Flow"])

    # --------------------------------------------------------
    # N-HOP ANALYSIS
    # --------------------------------------------------------

    with tab1:
        st.subheader("N-Hop Neighborhood")

        default_hop_index = (
            available_wallets.index(str(selected_wallet_id))
            if "selected_wallet_id" in locals()
            and str(selected_wallet_id) in available_wallets
            else (
                available_wallets.index(str(row.get("canonical_wallet_id", "")))
                if "row" in locals()
                and str(row.get("canonical_wallet_id", "")) in available_wallets
                else 0
            )
        )

        hop_entity = st.selectbox(
            "Wallet / Entity",
            available_wallets,
            format_func=lambda w: f"Wallet #{w}",
            index=default_hop_index,
            key="hop_entity",
        )

        hop_count = st.slider(
            "Number of hops", min_value=1, max_value=5, value=2, key="hop_count"
        )

        if st.button("Run N-Hop Analysis", key="run_hop"):
            hop_graph = n_hop_query(G, hop_entity, hop_count)

            if hop_graph is None:
                st.error("Wallet entity not found in graph.")
            else:
                st.success(
                    f"Found {hop_graph.number_of_nodes()} entities "
                    f"within {hop_count} hops."
                )

                hop_html = render_investigation_graph(
                    hop_graph, selected_txid if "selected_txid" in locals() else ""
                )

                st.components.v1.html(hop_html, height=680)

    # --------------------------------------------------------
    # SHORTEST PATH
    # --------------------------------------------------------

    with tab2:
        st.subheader("Shortest Relationship Path")

        col_pa, col_pb = st.columns(2)
        with col_pa:
            default_a_index = (
                available_wallets.index(str(selected_wallet_id))
                if "selected_wallet_id" in locals()
                and str(selected_wallet_id) in available_wallets
                else 0
            )
            wallet_a = st.selectbox(
                "Source Wallet",
                available_wallets,
                format_func=lambda w: f"Wallet #{w}",
                index=default_a_index,
                key="path_wallet_a",
            )
        with col_pb:
            default_b_index = 1 if len(available_wallets) > 1 else 0
            wallet_b = st.selectbox(
                "Destination Wallet",
                available_wallets,
                format_func=lambda w: f"Wallet #{w}",
                index=default_b_index,
                key="path_wallet_b",
            )

        if st.button("Find Shortest Path", key="run_path"):
            path = shortest_path_query(G, wallet_a, wallet_b)

            if path is None:
                st.warning("No relationship path found between these two wallets.")
            else:
                st.success(f"Path found: {len(path) - 1} hops")

                st.write(" → ".join(str(G.nodes[n].get("label", n)) for n in path))

                path_graph = get_path_subgraph(G, path)

                path_html = render_investigation_graph(
                    path_graph, selected_txid if "selected_txid" in locals() else ""
                )

                st.components.v1.html(path_html, height=680)

    # --------------------------------------------------------
    # FUND FLOW
    # --------------------------------------------------------

    with tab3:
        st.subheader("Fund Flow Tracing")

        default_flow_index = (
            available_wallets.index(str(selected_wallet_id))
            if "selected_wallet_id" in locals()
            and str(selected_wallet_id) in available_wallets
            else (
                available_wallets.index(str(row.get("canonical_wallet_id", "")))
                if "row" in locals()
                and str(row.get("canonical_wallet_id", "")) in available_wallets
                else 0
            )
        )

        flow_entity = st.selectbox(
            "Wallet / Entity",
            available_wallets,
            format_func=lambda w: f"Wallet #{w}",
            index=default_flow_index,
            key="flow_entity",
        )

        flow_hops = st.slider(
            "Tracing depth", min_value=1, max_value=5, value=2, key="flow_hops"
        )

        flow_direction = st.selectbox(
            "Direction", ["both", "outgoing", "incoming"], key="flow_direction"
        )

        if st.button("Trace Fund Flow", key="run_flow"):
            flow_graph = fund_flow_query(G, flow_entity, flow_hops, flow_direction)

            if flow_graph is None:
                st.error("Wallet entity not found in graph.")
            else:
                st.success(
                    f"Found {flow_graph.number_of_nodes()} entities "
                    f"in the traced network."
                )

                flow_html = render_investigation_graph(
                    flow_graph, selected_txid if "selected_txid" in locals() else ""
                )

                st.components.v1.html(flow_html, height=680)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "CyberStrix · Offline AI/ML investigation prototype · "
    "XGBoost + Isolation Forest + SHAP + Entity Graph"
)
