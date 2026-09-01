"""
Phase 5 — Investigation Queries
SIH 2026 PS 26146

Uses the entity graph built by entity_graph.py.

Core investigation queries:
1. N-hop neighborhood
2. Fund flow tracing (incoming / outgoing / both)
3. Shortest path between two entities
"""

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
