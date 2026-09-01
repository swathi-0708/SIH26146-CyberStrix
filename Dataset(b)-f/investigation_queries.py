"""
Phase 5 — Investigation Queries
SIH 2026 PS 26146

Uses the entity graph built by entity_graph.py.

Core investigation queries:
1. N-hop neighborhood
2. Fund flow tracing (incoming / outgoing / both)
3. Shortest path between two entities
"""

import json

import networkx as nx

# -------------------------------------------------------------------
# 1. N-HOP NEIGHBORHOOD
# -------------------------------------------------------------------


def n_hop_query(G, entity, hops=2):
    """
    Find all entities within N hops of a selected wallet/address.
    Distance is calculated ignoring edge direction so the analyst can
    see the complete local relationship network.
    """
    if entity not in G:
        raise ValueError(f"Entity '{entity}' not found in graph.")

    UG = G.to_undirected(as_view=True)
    nodes = nx.single_source_shortest_path_length(UG, entity, cutoff=hops)
    return G.subgraph(nodes.keys()).copy()


# -------------------------------------------------------------------
# 2. FUND FLOW TRACE
# -------------------------------------------------------------------


def fund_flow_query(G, entity, hops=2, direction="both"):
    """
    Trace money-flow relationships around an entity.

    direction:
        incoming  -> entities that sent funds toward this entity
        outgoing  -> entities that received funds from this entity
        both      -> complete local flow network
    """
    if entity not in G:
        raise ValueError(f"Entity '{entity}' not found in graph.")

    if direction == "outgoing":
        nodes = nx.single_source_shortest_path_length(G, entity, cutoff=hops)
    elif direction == "incoming":
        nodes = nx.single_source_shortest_path_length(
            G.reverse(copy=False), entity, cutoff=hops
        )
    elif direction == "both":
        UG = G.to_undirected(as_view=True)
        nodes = nx.single_source_shortest_path_length(UG, entity, cutoff=hops)
    else:
        raise ValueError("direction must be 'incoming', 'outgoing', or 'both'")

    return G.subgraph(nodes.keys()).copy()


# -------------------------------------------------------------------
# 3. SHORTEST PATH
# -------------------------------------------------------------------


def shortest_path_query(G, entity_a, entity_b, directed=False):
    """
    Find the shortest connection between two entities.

    directed=False:
        Finds the shortest relationship path regardless of transaction
        direction.

    directed=True:
        Finds a valid money-flow path from entity_a -> entity_b.
    """
    if entity_a not in G:
        raise ValueError(f"Entity '{entity_a}' not found in graph.")

    if entity_b not in G:
        raise ValueError(f"Entity '{entity_b}' not found in graph.")

    graph = G if directed else G.to_undirected(as_view=True)

    try:
        path = nx.shortest_path(graph, source=entity_a, target=entity_b)
        return path
    except nx.NetworkXNoPath:
        return None


# -------------------------------------------------------------------
# HELPER: BUILD A PATH SUBGRAPH
# -------------------------------------------------------------------


def get_path_subgraph(G, path):
    """
    Convert a shortest-path result into a graph that can be displayed
    in PyVis/Streamlit.
    """
    if not path:
        return nx.DiGraph()

    return G.subgraph(path).copy()


# -------------------------------------------------------------------
# LOADER + CLI DRIVER — runs the three queries against the real graph
# -------------------------------------------------------------------

GRAPH_PATH = "output/entity_graph_full.graphml"
SUMMARY_PATH = "output/entity_graph_summary.txt"
OUT_PATH = "output/investigation_results.json"


def load_graph(path=GRAPH_PATH):
    """
    Load the graph entity_graph.py wrote to disk.

    IMPORTANT: entity_graph.py builds nodes with int wallet_ids in memory
    (e.g. 674), but GraphML node ids are strings on disk. After
    nx.read_graphml, `674 in G` is False and `"674" in G` is True --
    verified directly against output/entity_graph_full.graphml (674 in G
    -> False, '674' in G -> True). Every entity id passed into the query
    functions below MUST be a str for this reason. This loader does not
    convert ids back to int to avoid masking the same bug for any other
    caller that loads this file with plain nx.read_graphml().

    Also note: `txids` on edges is a Python list in entity_graph.py's
    in-memory graph but becomes a single comma-joined string after the
    GraphML round-trip (GraphML has no native list type). Split on ','
    if you need the individual txids back.
    """
    return nx.read_graphml(path)


def _pick_sample_entities(G, n=3):
    """Highest-degree alerted wallets, for a demo run with no CLI args."""
    alerted = [(node, G.degree(node)) for node, d in G.nodes(data=True) if d.get("is_alerted")]
    alerted.sort(key=lambda x: x[1], reverse=True)
    return [node for node, _ in alerted[:n]]


def run_demo(entities=None, hops=2):
    G = load_graph()
    if not entities:
        entities = _pick_sample_entities(G, n=3)
    if len(entities) < 1:
        raise RuntimeError("No alerted entities found in graph to query.")

    results = {"graph_nodes": G.number_of_nodes(), "graph_edges": G.number_of_edges(), "queries": []}

    for entity in entities:
        entry = {"entity": entity}
        neigh = n_hop_query(G, entity, hops=hops)
        entry["n_hop"] = {
            "hops": hops,
            "subgraph_nodes": neigh.number_of_nodes(),
            "subgraph_edges": neigh.number_of_edges(),
        }
        flow = fund_flow_query(G, entity, hops=hops, direction="both")
        entry["fund_flow"] = {
            "direction": "both",
            "subgraph_nodes": flow.number_of_nodes(),
            "subgraph_edges": flow.number_of_edges(),
        }
        results["queries"].append(entry)

    if len(entities) >= 2:
        path = shortest_path_query(G, entities[0], entities[1], directed=False)
        results["shortest_path_sample"] = {
            "from": entities[0],
            "to": entities[1],
            "path": path,
            "hop_count": (len(path) - 1) if path else None,
        }

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Loaded graph: {results['graph_nodes']} nodes, {results['graph_edges']} edges")
    for entry in results["queries"]:
        print(
            f"  entity {entry['entity']}: {hops}-hop -> "
            f"{entry['n_hop']['subgraph_nodes']} nodes / "
            f"{entry['n_hop']['subgraph_edges']} edges | "
            f"fund-flow(both) -> {entry['fund_flow']['subgraph_nodes']} nodes"
        )
    if "shortest_path_sample" in results:
        sp = results["shortest_path_sample"]
        print(f"  shortest path {sp['from']} -> {sp['to']}: {sp['path']}")
    print(f"Wrote {OUT_PATH}")
    return results


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Run investigation queries against the saved entity graph.")
    ap.add_argument("--entity", action="append", help="Entity (wallet id or address) to query. Repeatable.")
    ap.add_argument("--hops", type=int, default=2)
    args = ap.parse_args()
    run_demo(entities=args.entity, hops=args.hops)