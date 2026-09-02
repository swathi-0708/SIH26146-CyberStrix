"""
Clean, demo-sized link-analysis view for ONE entity, instead of the full
1000+ node risk subgraph entity_graph.py produces (that one's real evidence,
but unreadable as a first thing to show a judge).

Uses investigation_queries.py's n_hop_query -- already wired, already
verified against the real graph -- to pull just one wallet's local
neighborhood, then renders it with pyvis using the same visual language as
entity_graph.py (tier colors, IP nodes as blue squares, disposable
addresses dimmed) so the two views look like one consistent system, not two
different tools bolted together.

Usage:
    python3 render_investigation_graph.py --entity 674 --hops 2
    python3 render_investigation_graph.py                     # auto-picks
                                                                # the highest-
                                                                # degree
                                                                # alerted
                                                                # wallet
"""

import argparse

from pyvis.network import Network

from investigation_queries import load_graph, n_hop_query, _pick_sample_entities

TIER_COLOR = {
    "high": "#e63946",
    "medium-high": "#f4a261",
    "worth reviewing": "#e9c46a",
    "none": "#a8b0b8",
}


def render(sub, focus_entity, out_path):
    net = Network(
        height="800px",
        width="100%",
        directed=True,
        notebook=False,
        bgcolor="#111318",
        font_color="#e8e8e8",
        cdn_resources="in_line",
    )
    net.barnes_hut(
        gravity=-3000, central_gravity=0.2, spring_length=120, spring_strength=0.03
    )

    for n, d in sub.nodes(data=True):
        is_ip = str(d.get("is_ip")) == "True"
        is_disposable = str(d.get("is_disposable")) == "True"
        tier = d.get("max_priority_tier", "none")
        tier = tier if tier != "none" else None
        is_focus = n == focus_entity

        if is_ip:
            color, size, shape = "#4895ef", 16, "square"
            label = str(d.get("ip", n)).replace("ip:", "")
        elif is_disposable and tier is None:
            color, size, shape = "#5a616b", 8, "dot"
            label = ""
        else:
            color = TIER_COLOR[tier] if tier else TIER_COLOR["none"]
            size = 14 + 4 * min(int(float(d.get("n_alerts", 0))), 6)
            shape = "dot"
            label = f"#{n}"

        if is_focus:
            size += 10  # focus entity is visibly the biggest node in the view
            color = "#00e5ff"  # bright cyan ring, distinct from every tier color
            shape = "star" if not is_ip else shape

        title = f"{'src_ip ' if is_ip else 'wallet #'}{n}"
        if not is_ip:
            title += (
                f"<br>country: {d.get('country', '?')}  asn: {d.get('asn', '?')}"
                f"<br>alerts: {d.get('n_alerts', 0)} (worst: {d.get('max_priority_tier', 'none')})"
            )

        net.add_node(
            n,
            label=label,
            title=title,
            color=color,
            size=size,
            shape=shape,
            borderWidth=4 if is_focus else (3 if str(d.get("is_alerted")) == "True" else 1),
        )

    for u, v, d in sub.edges(data=True):
        if d.get("edge_type") == "network":
            net.add_edge(
                u, v, color="#4895ef", width=1, dashes=True,
                title=f"{d.get('n_tx', '?')} tx observed from this src_ip", arrows="to",
            )
            continue
        tier = d.get("max_priority_tier", "none")
        color = TIER_COLOR[tier] if tier and tier != "none" else "#4a5058"
        width = 1 + min(int(float(d.get("n_tx", 1))), 8)
        title = f"{d.get('n_tx', '?')} tx"
        if "total_btc" in d:
            title += f", {float(d['total_btc']):.4f} BTC total"
        net.add_edge(u, v, color=color, width=width, title=title, arrows="to")

    net.set_options("""
    { "physics": {"stabilization": {"iterations": 150}},
      "interaction": {"hover": true, "tooltipDelay": 100} }
    """)
    net.write_html(out_path, notebook=False, open_browser=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", type=str, default=None,
                     help="Wallet id or src_ip:x.x.x.x node to center on. Default: auto-pick highest-degree alerted wallet.")
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--out", type=str, default="output/investigation_graph.html")
    args = ap.parse_args()

    G = load_graph()
    entity = args.entity or _pick_sample_entities(G, n=1)[0]
    sub = n_hop_query(G, entity, hops=args.hops)

    render(sub, entity, args.out)
    print(f"Entity {entity}: {args.hops}-hop neighborhood -> {sub.number_of_nodes()} nodes, {sub.number_of_edges()} edges")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()