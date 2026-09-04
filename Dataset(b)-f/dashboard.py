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
# two different colors in two different places.
# --------------------------------------------------------------
# --------------------------------------------------------------
# Single source of truth for tier -> color, used everywhere a tier
# is drawn (metrics, table, graph nodes) so the same tier is never
# two different colors in two different places.
# --------------------------------------------------------------
# --------------------------------------------------------------
# Single source of truth for tier -> color, used everywhere a tier
# is drawn (metrics, table, graph nodes) so the same tier is never
# two different colors in two different places.
# --------------------------------------------------------------
TIER_COLORS = {
    "high": "#C4473A",
    "medium-high": "#C1843D",
    "worth reviewing": "#A89538",
    "none": "#62806B",
}

# Theme Configurations: Professional Forensic Analyst Console Theme
THEME_CONFIG = {
    "Dark": {
        "BG_INK": "#142F35",         # Page background
        "PANEL_PAPER": "#1B3D44",    # Report panel background
        "SECONDARY_PANEL": "#244D56",# Secondary panel / input
        "LINE_HAIRLINE": "#365E66",  # All rules/dividers
        "TEXT_PRIMARY": "#E9EEED",   # Primary body text
        "TEXT_MUTED": "#A7B9BC",     # Muted labels/secondary text
        "PRIMARY_TEAL": "#5A8995",   # Primary teal
        "DEEP_TEAL": "#234B54",      # Deep teal
        "ACCENT_TEAL": "#6D9CA6",    # Accent teal
        "INPUT_BG": "#244D56",
        "INPUT_HOVER": "#2D5B66",
        "TAG_BG": "#244D56",
        "BUTTON_BG": "#244D56",
        "BUTTON_HOVER": "#2E606B",
        "BUTTON_ACTIVE": "#1D3F47",
        "ACCENT_STAMP": "#5A8995",   # Primary teal for active indicators
        "ACCENT_SIGNAL": "#6D9CA6",  # Accent teal for technical lines
        "GRAPH_BG": "#142F35",
        "GRAPH_FONT": "#E9EEED",
        "GRAPH_EDGE": "#52747D",
    },
    "Light": {
        "BG_INK": "#E9EEED",         # Page background (deliberately not pure white)
        "PANEL_PAPER": "#F7F8F6",    # Crisp paper panel surface
        "SECONDARY_PANEL": "#DEE6E6",# Secondary panel / input
        "LINE_HAIRLINE": "#C4D0D1",  # Neutral hairline rule & border
        "TEXT_PRIMARY": "#19383F",   # Primary body text
        "TEXT_MUTED": "#5E7074",     # Muted secondary text label
        "INPUT_BG": "#DEE6E6",       # Input container background
        "INPUT_HOVER": "#D5E0E0",    # Hover state for dropdown options
        "TAG_BG": "#DEE6E6",         # Multiselect tag background
        "BUTTON_BG": "#DEE6E6",      # Button default background
        "BUTTON_HOVER": "#C8D7D8",   # Button hover background
        "BUTTON_ACTIVE": "#B6C8C9",  # Button active background
        "PRIMARY_TEAL": "#386D7A",   # Primary teal
        "DEEP_TEAL": "#234B54",      # Deep teal
        "ACCENT_TEAL": "#5A8995",    # Accent teal
        "ACCENT_STAMP": "#386D7A",   # Primary teal accent
        "ACCENT_SIGNAL": "#5A8995",  # Technical accent
        "GRAPH_BG": "#E9EEED",       # Graph canvas background
        "GRAPH_FONT": "#19383F",     # Font for node labels in light mode
        "GRAPH_EDGE": "#789196",     # Graph edge color for light mode
    },
}


# ============================================================
# PAGE CONFIGURATION & THEME INITIALIZATION
# ============================================================

st.set_page_config(
    page_title="CyberStrix Investigator",
    page_icon=None,
    layout="wide",
)

# --------------------------------------------------------------
# Session State & Query Parameter Theme Management
# Default initialization state is "Light" mode.
# --------------------------------------------------------------
query_theme = st.query_params.get("theme", None)

if query_theme in ["Light", "Dark"]:
    st.session_state["theme_mode"] = query_theme
elif "theme_mode" not in st.session_state:
    st.session_state["theme_mode"] = "Light"

theme_mode = st.session_state["theme_mode"]
active_theme = THEME_CONFIG[theme_mode]

BG_INK = active_theme["BG_INK"]
PANEL_PAPER = active_theme["PANEL_PAPER"]
LINE_HAIRLINE = active_theme["LINE_HAIRLINE"]
TEXT_PRIMARY = active_theme["TEXT_PRIMARY"]
TEXT_MUTED = active_theme["TEXT_MUTED"]
INPUT_BG = active_theme["INPUT_BG"]
INPUT_HOVER = active_theme["INPUT_HOVER"]
TAG_BG = active_theme["TAG_BG"]
BUTTON_BG = active_theme["BUTTON_BG"]
BUTTON_HOVER = active_theme["BUTTON_HOVER"]
BUTTON_ACTIVE = active_theme["BUTTON_ACTIVE"]
ACCENT_STAMP = active_theme["ACCENT_STAMP"]
ACCENT_SIGNAL = active_theme["ACCENT_SIGNAL"]
PRIMARY_TEAL = active_theme["PRIMARY_TEAL"]
DEEP_TEAL = active_theme["DEEP_TEAL"]
ACCENT_TEAL = active_theme["ACCENT_TEAL"]
GRAPH_BG = active_theme["GRAPH_BG"]
GRAPH_FONT = active_theme["GRAPH_FONT"]
GRAPH_EDGE = active_theme["GRAPH_EDGE"]

# --------------------------------------------------------------
# Native Three-Dot Menu Theme Control Injection
# Inject theme item directly into Streamlit's native menu container
# --------------------------------------------------------------
st.components.v1.html(
    f"""
    <script>
    (function() {{
        const parentDoc = window.parent.document;
        const currentTheme = "{theme_mode}";
        const primaryTeal = "{PRIMARY_TEAL}";
        const panelPaper = "{PANEL_PAPER}";
        
        function injectThemeToNativeMenu() {{
            const menuList = parentDoc.querySelector('ul[data-testid="main-menu-list"], ul[role="menu"], [data-testid="stMainMenuPopover"] ul');
            if (!menuList) return;
            if (parentDoc.getElementById('native-theme-menu-item')) return;
            
            const li = parentDoc.createElement('li');
            li.id = 'native-theme-menu-item';
            li.setAttribute('role', 'menuitem');
            li.style.cssText = 'padding: 10px 16px; display: flex; align-items: center; justify-content: space-between; font-family: "IBM Plex Sans", sans-serif; font-size: 13px; border-bottom: 1px solid rgba(128,128,128,0.2); cursor: default; margin-bottom: 4px;';
            
            const labelSpan = parentDoc.createElement('span');
            labelSpan.innerText = 'THEME';
            labelSpan.style.cssText = 'font-weight: 600; font-size: 11px; letter-spacing: 0.05em; opacity: 0.75; font-family: "IBM Plex Sans", sans-serif;';
            
            const btnContainer = parentDoc.createElement('div');
            btnContainer.style.cssText = 'display: flex; gap: 4px; border-radius: 2px; padding: 2px; background: rgba(128,128,128,0.15);';
            
            ['Light', 'Dark'].forEach(mode => {{
                const btn = parentDoc.createElement('button');
                btn.type = 'button';
                btn.innerText = mode;
                const isActive = currentTheme === mode;
                btn.style.cssText = `border: none; padding: 3px 10px; border-radius: 2px; font-size: 12px; font-weight: 500; font-family: "IBM Plex Sans", sans-serif; cursor: pointer; transition: all 0.15s ease; ${{isActive ? `background: ${primaryTeal}; color: #ffffff;` : 'background: transparent; color: inherit; opacity: 0.75;'}}`;
                
                btn.onclick = function(e) {{
                    e.preventDefault();
                    e.stopPropagation();
                    if (currentTheme !== mode) {{
                        const searchParams = new URLSearchParams(window.parent.location.search);
                        searchParams.set('theme', mode);
                        window.parent.location.search = searchParams.toString();
                    }}
                }};
                btnContainer.appendChild(btn);
            }});
            
            li.appendChild(labelSpan);
            li.appendChild(btnContainer);
            menuList.insertBefore(li, menuList.firstChild);
        }}
        
        const observer = new MutationObserver(function() {{
            injectThemeToNativeMenu();
        }});
        
        observer.observe(parentDoc.body, {{ childList: true, subtree: true }});
        injectThemeToNativeMenu();
    }})();
    </script>
    """,
    height=0,
    width=0,
)

# --------------------------------------------------------------
# Theme CSS Injection
# --------------------------------------------------------------
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&display=swap');

    /* Main application background & body font */
    .stApp {
        background-color: {BG_INK} !important;
        color: {TEXT_PRIMARY} !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
    }

    /* Sidebar container */
    [data-testid="stSidebar"] {
        background-color: {PANEL_PAPER} !important;
        border-right: 1px solid {LINE_HAIRLINE} !important;
    }

    /* Headings: Source Serif 4, serif, weight 600 */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4,
    .stApp h5, .stApp h6 {
        color: {TEXT_PRIMARY} !important;
        font-family: 'Source Serif 4', serif !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em !important;
    }

    .stApp p, .stApp li {
        color: {TEXT_PRIMARY} !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
    }

    .stApp [data-testid="stCaptionContainer"],
    .stApp [data-testid="stCaptionContainer"] * {
        color: {TEXT_MUTED} !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-size: 13px !important;
    }

    /* Hairline rules */
    hr, [data-testid="stDivider"] {
        border-color: {LINE_HAIRLINE} !important;
    }

    /* Analyst metric readout card styling */
    [data-testid="stMetric"] {
        background-color: {PANEL_PAPER} !important;
        border: 1px solid {LINE_HAIRLINE} !important;
        border-radius: 2px !important;
        box-shadow: none !important;
        padding: 12px 16px !important;
        margin: 0 !important;
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] * {
        color: {TEXT_MUTED} !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-size: 11px !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }

    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] * {
        color: {TEXT_PRIMARY} !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 24px !important;
        font-weight: 600 !important;
    }

    /* Flat form controls, 2px radius, thin borders */
    .stApp label,
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] *,
    [data-testid="stWidgetLabel"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] * {
        color: {TEXT_PRIMARY} !important;
        -webkit-text-fill-color: {TEXT_PRIMARY} !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-weight: 500 !important;
        font-size: 13px !important;
    }

    /* Native text inputs, number inputs, text areas */
    .stApp input[type="text"],
    .stApp input[type="number"],
    .stApp textarea,
    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input,
    [data-testid="stTextArea"] textarea {
        background-color: {INPUT_BG} !important;
        color: {TEXT_PRIMARY} !important;
        -webkit-text-fill-color: {TEXT_PRIMARY} !important;
        caret-color: {TEXT_PRIMARY} !important;
        border: 1px solid {LINE_HAIRLINE} !important;
        border-radius: 2px !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 13px !important;
    }

    .stApp input::placeholder,
    .stApp textarea::placeholder {
        color: {TEXT_MUTED} !important;
        -webkit-text-fill-color: {TEXT_MUTED} !important;
        opacity: 0.8 !important;
    }

    /* BaseWeb selectboxes */
    .stApp [data-baseweb="select"] > div,
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background-color: {INPUT_BG} !important;
        border: 1px solid {LINE_HAIRLINE} !important;
        border-radius: 2px !important;
    }

    .stApp [data-baseweb="select"] *,
    [data-testid="stSidebar"] [data-baseweb="select"] *,
    .stApp [data-baseweb="select"] span,
    .stApp [data-baseweb="select"] div,
    .stApp [data-baseweb="select"] input,
    .stApp [data-baseweb="select"] p {
        color: {TEXT_PRIMARY} !important;
        -webkit-text-fill-color: {TEXT_PRIMARY} !important;
        fill: {TEXT_PRIMARY} !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-size: 13px !important;
    }

    /* Dropdown menus / popovers */
    div[data-baseweb="popover"],
    div[data-baseweb="menu"],
    ul[role="listbox"],
    div[role="listbox"] {
        background-color: {PANEL_PAPER} !important;
        border: 1px solid {LINE_HAIRLINE} !important;
        border-radius: 2px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
    }

    li[role="option"],
    div[role="option"],
    [data-baseweb="menu"] li {
        background-color: {PANEL_PAPER} !important;
        color: {TEXT_PRIMARY} !important;
    }

    li[role="option"] *,
    div[role="option"] *,
    [data-baseweb="menu"] li * {
        color: {TEXT_PRIMARY} !important;
        -webkit-text-fill-color: {TEXT_PRIMARY} !important;
    }

    li[role="option"]:hover,
    li[role="option"][aria-selected="true"],
    div[role="option"]:hover,
    div[role="option"][aria-selected="true"] {
        background-color: {INPUT_HOVER} !important;
    }

    li[role="option"]:hover *,
    li[role="option"][aria-selected="true"] *,
    div[role="option"]:hover *,
    div[role="option"][aria-selected="true"] * {
        color: {PRIMARY_TEAL} !important;
        -webkit-text-fill-color: {PRIMARY_TEAL} !important;
        font-weight: 500 !important;
    }

    /* Multiselect tags */
    .stApp [data-baseweb="tag"],
    [data-testid="stSidebar"] [data-baseweb="tag"] {
        background-color: {TAG_BG} !important;
        border: 1px solid {LINE_HAIRLINE} !important;
        border-radius: 2px !important;
    }

    .stApp [data-baseweb="tag"] *,
    [data-testid="stSidebar"] [data-baseweb="tag"] * {
        color: {TEXT_PRIMARY} !important;
        -webkit-text-fill-color: {TEXT_PRIMARY} !important;
        fill: {TEXT_PRIMARY} !important;
        font-size: 12px !important;
    }

    /* Buttons: understated & technical */
    .stButton > button {
        background-color: {BUTTON_BG} !important;
        color: {TEXT_PRIMARY} !important;
        -webkit-text-fill-color: {TEXT_PRIMARY} !important;
        border: 1px solid {LINE_HAIRLINE} !important;
        border-radius: 2px !important;
        padding: 6px 16px !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-weight: 500 !important;
        font-size: 13px !important;
        box-shadow: none !important;
        transition: background-color 0.15s ease, border-color 0.15s ease !important;
    }

    .stButton > button:hover {
        background-color: {BUTTON_HOVER} !important;
        border-color: {PRIMARY_TEAL} !important;
        color: {TEXT_PRIMARY} !important;
        -webkit-text-fill-color: {TEXT_PRIMARY} !important;
    }

    .stButton > button:active {
        background-color: {BUTTON_ACTIVE} !important;
    }

    .stButton > button * {
        color: {TEXT_PRIMARY} !important;
        -webkit-text-fill-color: {TEXT_PRIMARY} !important;
    }

    /* Sliders */
    [data-testid="stSlider"] * {
        color: {TEXT_PRIMARY} !important;
    }

    /* Tabs: technical understated tabs */
    .stApp [data-testid="stTabs"] [data-baseweb="tab-list"] {
        background-color: transparent !important;
        border-bottom: 1px solid {LINE_HAIRLINE} !important;
        gap: 16px !important;
    }

    .stApp [data-testid="stTabs"] button {
        color: {TEXT_MUTED} !important;
        -webkit-text-fill-color: {TEXT_MUTED} !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-size: 13px !important;
        border-radius: 0px !important;
        padding: 8px 12px !important;
    }

    .stApp [data-testid="stTabs"] button[aria-selected="true"] {
        color: {PRIMARY_TEAL} !important;
        -webkit-text-fill-color: {PRIMARY_TEAL} !important;
        font-weight: 600 !important;
        border-bottom: 2px solid {PRIMARY_TEAL} !important;
    }

    /* Dataframe container */
    div[data-testid="stDataFrame"] {
        border: 1px solid {LINE_HAIRLINE} !important;
        border-radius: 2px !important;
        background-color: {PANEL_PAPER} !important;
    }

    /* Monospace for technical labels/values */
    code, pre, [data-testid="stJson"] {
        font-family: 'IBM Plex Mono', monospace !important;
    }

    /* Instrument panel framing & corner ticks */
    .instrument-panel {
        position: relative;
        border: 1px solid {LINE_HAIRLINE} !important;
        background-color: {PANEL_PAPER} !important;
        padding: 10px;
        margin: 8px 0;
        border-radius: 2px;
    }

    .instrument-panel::before {
        content: '';
        position: absolute;
        top: -4px;
        left: -4px;
        width: 10px;
        height: 10px;
        border-top: 1px solid {ACCENT_TEAL};
        border-left: 1px solid {ACCENT_TEAL};
        pointer-events: none;
        z-index: 10;
    }

    .instrument-panel::after {
        content: '';
        position: absolute;
        bottom: -4px;
        right: -4px;
        width: 10px;
        height: 10px;
        border-bottom: 1px solid {ACCENT_TEAL};
        border-right: 1px solid {ACCENT_TEAL};
        pointer-events: none;
        z-index: 10;
    }

    .corner-tick-tr {
        position: absolute;
        top: -4px;
        right: -4px;
        width: 10px;
        height: 10px;
        border-top: 1px solid {ACCENT_TEAL};
        border-right: 1px solid {ACCENT_TEAL};
        pointer-events: none;
        z-index: 10;
    }

    .corner-tick-bl {
        position: absolute;
        bottom: -4px;
        left: -4px;
        width: 10px;
        height: 10px;
        border-bottom: 1px solid {ACCENT_TEAL};
        border-left: 1px solid {ACCENT_TEAL};
        pointer-events: none;
        z-index: 10;
    }

    /* Expander treatment */
    [data-testid="stExpander"] {
        border: 1px solid {LINE_HAIRLINE} !important;
        border-radius: 2px !important;
        background-color: {PANEL_PAPER} !important;
        position: relative !important;
    }

    [data-testid="stExpander"]::before {
        content: '';
        position: absolute;
        top: -4px;
        left: -4px;
        width: 10px;
        height: 10px;
        border-top: 1px solid {ACCENT_TEAL};
        border-left: 1px solid {ACCENT_TEAL};
        pointer-events: none;
    }

    [data-testid="stExpander"]::after {
        content: '';
        position: absolute;
        bottom: -4px;
        right: -4px;
        width: 10px;
        height: 10px;
        border-bottom: 1px solid {ACCENT_TEAL};
        border-right: 1px solid {ACCENT_TEAL};
        pointer-events: none;
    }

    /* Technical header style */
    .tech-title {
        font-family: 'IBM Plex Mono', monospace !important;
        color: {ACCENT_TEAL} !important;
        font-size: 13px !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase;
        margin-bottom: 6px;
    }

    /* Clean iframe container */
    iframe {
        border: none !important;
        background-color: {BG_INK} !important;
    }

    [data-testid="stCustomComponentV1"] {
        border: none !important;
        background-color: {BG_INK} !important;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
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
    making the embedded graph container blend seamlessly into the workstation theme.
    """
    if not html_str:
        return ""
    dark_style = f"""
    <style>
      html, body {{
        margin: 0 !important;
        padding: 0 !important;
        background-color: {GRAPH_BG} !important;
        overflow: hidden !important;
        font-family: 'IBM Plex Mono', monospace !important;
      }}
      .card {{
        background-color: {GRAPH_BG} !important;
        border: none !important;
        box-shadow: none !important;
        margin: 0 !important;
        padding: 0 !important;
      }}
      .card-body {{
        padding: 0 !important;
        margin: 0 !important;
        background-color: {GRAPH_BG} !important;
      }}
      #mynetwork {{
        border: 1px solid {LINE_HAIRLINE} !important;
        border-radius: 2px !important;
        background-color: {GRAPH_BG} !important;
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
        bgcolor=GRAPH_BG,
        font_color=GRAPH_FONT,
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

    # Computed ONCE per render call, not per-node inside the loop below
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
            color = "#C4473A"
            size = 35

        elif node_type == "transaction":
            color = TIER_COLORS.get(
                str(data.get("priority_tier", "")).lower(), TIER_COLORS["none"]
            )
            size = 25

        elif node_type == "wallet":
            priority = str(data.get("max_priority_tier", "")).lower()
            color = TIER_COLORS.get(priority, PRIMARY_TEAL)
            size = 22

        else:
            # IP
            color = "#5A8995"
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
            color=GRAPH_EDGE,
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
        bgcolor=GRAPH_BG,
        font_color=GRAPH_FONT,
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
            color = "#C4473A"
            size = 34
        else:
            color = TIER_COLORS.get(priority, PRIMARY_TEAL)
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
            color=GRAPH_EDGE,
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
    st.markdown('<div class="tech-title">Entity Graph Instrumentation</div>', unsafe_allow_html=True)
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
            st.markdown('<div class="instrument-panel"><div class="corner-tick-tr"></div><div class="corner-tick-bl"></div>', unsafe_allow_html=True)
            st.components.v1.html(clean_graph_html(html), height=720)
            st.markdown('</div>', unsafe_allow_html=True)
        elif GRAPHML_FILE.exists():
            st.info("Focused HTML not found; the GraphML exists.")
            st.caption("Run entity_graph_modified.py to regenerate the graph.")
        else:
            st.warning("IP → Transaction → Wallet graph not found.")

    elif graph_choice == "Risk Cluster":
        if RISK_GRAPH_HTML_FILE.exists():
            html = RISK_GRAPH_HTML_FILE.read_text(encoding="utf-8")
            st.markdown('<div class="instrument-panel"><div class="corner-tick-tr"></div><div class="corner-tick-bl"></div>', unsafe_allow_html=True)
            st.components.v1.html(clean_graph_html(html), height=720)
            st.markdown('</div>', unsafe_allow_html=True)
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
            st.markdown('<div class="instrument-panel"><div class="corner-tick-tr"></div><div class="corner-tick-bl"></div>', unsafe_allow_html=True)
            st.components.v1.html(wallet_html, height=720)
            st.markdown('</div>', unsafe_allow_html=True)


show_graph_explorer()


# ============================================================
# TITLE
# ============================================================

st.title("CyberStrix Investigator")

st.caption(
    "Offline Bitcoin transaction anomaly detection and investigation dashboard "
    "running fully offline using synthetic demonstration data."
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
        st.markdown(f'<div style="color: {TIER_COLORS["high"]}; font-weight: 600; font-family: \'IBM Plex Sans\', sans-serif; font-size: 13px; padding: 6px 12px; border: 1px solid {LINE_HAIRLINE}; background-color: {PANEL_PAPER}; border-radius: 2px;">High priority alert</div>', unsafe_allow_html=True)

    elif priority == "medium-high":
        st.markdown(f'<div style="color: {TIER_COLORS["medium-high"]}; font-weight: 600; font-family: \'IBM Plex Sans\', sans-serif; font-size: 13px; padding: 6px 12px; border: 1px solid {LINE_HAIRLINE}; background-color: {PANEL_PAPER}; border-radius: 2px;">Medium-high priority alert</div>', unsafe_allow_html=True)

    else:
        st.markdown(f'<div style="color: {TIER_COLORS.get(priority, TEXT_MUTED)}; font-weight: 600; font-family: \'IBM Plex Sans\', sans-serif; font-size: 13px; padding: 6px 12px; border: 1px solid {LINE_HAIRLINE}; background-color: {PANEL_PAPER}; border-radius: 2px;">{priority.capitalize()} priority alert</div>', unsafe_allow_html=True)

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

        st.markdown('<div class="instrument-panel"><div class="corner-tick-tr"></div><div class="corner-tick-bl"></div>', unsafe_allow_html=True)
        st.dataframe(
            shap_df,
            use_container_width=True,
            hide_index=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

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

            st.markdown('<div class="instrument-panel"><div class="corner-tick-tr"></div><div class="corner-tick-bl"></div>', unsafe_allow_html=True)
            st.components.v1.html(
                graph_html,
                height=680,
            )
            st.markdown('</div>', unsafe_allow_html=True)

        elif GRAPH_HTML_FILE.exists():
            st.info(
                "The selected transaction could not "
                "be isolated from the GraphML file. "
                "Showing the focused graph instead."
            )

            graph_html = GRAPH_HTML_FILE.read_text(encoding="utf-8")

            st.markdown('<div class="instrument-panel"><div class="corner-tick-tr"></div><div class="corner-tick-bl"></div>', unsafe_allow_html=True)
            st.components.v1.html(
                clean_graph_html(graph_html),
                height=680,
            )
            st.markdown('</div>', unsafe_allow_html=True)

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

                st.markdown('<div class="instrument-panel"><div class="corner-tick-tr"></div><div class="corner-tick-bl"></div>', unsafe_allow_html=True)
                st.components.v1.html(hop_html, height=680)
                st.markdown('</div>', unsafe_allow_html=True)

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

                st.markdown('<div class="instrument-panel"><div class="corner-tick-tr"></div><div class="corner-tick-bl"></div>', unsafe_allow_html=True)
                st.components.v1.html(path_html, height=680)
                st.markdown('</div>', unsafe_allow_html=True)

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

                st.markdown('<div class="instrument-panel"><div class="corner-tick-tr"></div><div class="corner-tick-bl"></div>', unsafe_allow_html=True)
                st.components.v1.html(flow_html, height=680)
                st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "CyberStrix · Offline AI/ML investigation prototype · "
    "XGBoost + Isolation Forest + SHAP + Entity Graph"
)
