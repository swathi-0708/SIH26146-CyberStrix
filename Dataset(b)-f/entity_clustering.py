"""
Entity clustering for SIH PS 26146.

PS requirement: "...correlates network-layer (IP/port/timing) observations
with blockchain-layer (wallet/TXID/amount) data, and applies AI/ML to
detect anomalies, CLUSTER ENTITIES, and generate prioritized, explainable
investigative leads." `entity_graph.py` draws the link-analysis picture
(who sent to whom) but stops at individual addresses -- it never answers
the actual investigative question: "how many distinct real-world actors
are in this dataset, and which addresses/wallets does each one control?"
That's entity resolution, and this script is what closes that gap.

Why this is a separate script, not a patch to entity_graph.py: entity_graph
answers "who is connected to whom" (a graph question); this answers "which
nodes are the SAME actor" (a partition/union-find question). They share
inputs and this script optionally enriches the graph's exports, but they
are conceptually different operations and entity_graph.py is already
verified end-to-end -- safer to build clustering alongside it than to
risk regressing something working.

Heuristics implemented (each is a standard, published Bitcoin entity-
resolution technique -- see Meiklejohn et al. 2013, "A Fistful of
Bitcoins" -- plus one network-layer extension specific to this PS):

1. Common-input-ownership. All inputs spent together in one transaction
   must be signed by whoever controls them, i.e. the same entity. This is
   the classical heuristic used by every production blockchain-analytics
   tool. CAVEAT (and why it's guarded below): CoinJoins / mixing services
   break this assumption by construction -- a mixer's fan-in transaction
   pools deposits from many UNRELATED depositors into one transaction, so
   naively unioning all of that transaction's inputs would merge strangers
   into one false "entity". This dataset's fan_out_mixer pattern does
   exactly that (25 unrelated depositor addresses in one fan-in tx -- see
   generate_dataset.py:inject_fan_out_mixer). Mitigation: transactions
   with more than MAX_COMMON_INPUT_SIZE inputs are excluded from this
   heuristic and reported separately as likely pooling/mixing points
   instead of silently mis-clustering their depositors. Normal wallet
   behaviour here never exceeds 3 inputs (generate_dataset.py's n_in
   distribution), so the threshold has real headroom before it risks
   cutting off genuine multi-input spends.

2. Likely-change-address linking. For a 2-output transaction, the
   convention (and this dataset's own peeling_chain generator -- see
   inject_peeling_chain) is one output pays a counterparty and the other
   returns change to the sender. Change never leaves the sender's control,
   so the larger-value output of a 2-output tx is treated as probably
   change and unioned with the sender -- but ONLY when the two outputs are
   meaningfully asymmetric (min/max amount ratio below
   CHANGE_RATIO_THRESHOLD). Verified against this dataset: a genuine
   peeling-chain hop has ratio ~0.02-0.09 (small peel vs. large change --
   see inject_peeling_chain's `peel`/`remaining` split), while this
   dataset's normal 2-output transactions split the amount EVENLY (ratio
   ~1.0 -- see make_normal_transaction's `out_amounts` formula, identical
   value per output). Without the ratio guard, an even split makes "larger
   output" an arbitrary coin-flip between the real receiver and a disposable
   filler address, and applying it dataset-wide false-links most of the
   wallet population into one giant component through recurring
   counterparties (verified: 794/800 wallets collapsed into a single
   "entity" before this guard was added). This is what actually stitches a
   peeling chain's disposable hop addresses into one entity (each hop's
   "remaining" output becomes next hop's input -- see entity_graph.py's
   module docstring on why those addresses were previously invisible
   entirely) without also swallowing unrelated normal traffic.

3. Shared-broadcast-IP correlation. This is the PS's own network-layer
   angle, not a classic blockchain-only heuristic: if the *same* src_ip
   is the broadcasting origin for transactions whose primary sender
   (first input address) differs, that's a strong signal both addresses
   are operated from the same machine/node, i.e. the same entity -- even
   if they never appear as co-inputs on any single transaction. This is
   exactly how a peeling chain is fully reconstructed here even on hops
   where heuristic 2 alone would miss a hop (e.g. a hop with >2 outputs):
   every hop broadcasts from the same source.home_ip regardless of which
   disposable address is spending at that hop (see inject_peeling_chain).
   Guard: only the primary sender of each transaction is considered (not
   every input), for the same false-merge reason as heuristic 1 -- a
   mixer's fan-in tx has one src_ip but 25 unrelated depositor inputs,
   and we must not let IP co-location merge those depositors either.

Each union records a reason + evidence (txid or ip + timestamp), so every
cluster can be explained: "why are these N addresses one entity?" is
answerable, not just asserted -- matching the PS's explainability
requirement at the entity level the same way explain_alerts.py does it at
the transaction level.

Outputs (all written to output/):
  - entity_clusters.csv          address -> entity_id membership (+ wallet
                                  id, disposable flag, per-address alert info)
  - entity_clusters_summary.csv  one row per entity: size, alert rollup,
                                  risk_score, country/ASN/IP spread flags --
                                  sorted by risk_score, this IS the ranked
                                  investigative-lead list at entity level
  - entity_clusters_evidence.json  per entity, the union-find edges that
                                  built it (heuristic + txid/ip) -- the
                                  "why" behind each cluster
  - entity_clusters_summary.txt  human-readable top clusters + excluded
                                  high-fan-in pooling points
  - entity_clusters_risk.html    interactive view of the top-risk clusters,
                                  nodes coloured/grouped by entity_id

Usage:
    python3 entity_clustering.py
"""

import json
from collections import defaultdict

import pandas as pd

from ingest import load_transactions

TX_PATH = "output/transactions.csv"
WALLETS_PATH = "output/wallets_reference.csv"
ALERTS_PATH = "output/alerts.csv"

MAX_COMMON_INPUT_SIZE = 6  # normal wallets never exceed 3 inputs; mixer fan-in uses 25
CHANGE_RATIO_THRESHOLD = 0.3  # min/max output ratio below this = probable peel+change
TIER_WEIGHT = {"high": 3, "medium-high": 2, "worth reviewing": 1}
TOP_N_FOR_REPORT = 25


class UnionFind:
    """Standard union-find with path compression + union by rank."""

    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank.get(ra, 0) < self.rank.get(rb, 0):
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank.get(ra, 0) == self.rank.get(rb, 0):
            self.rank[ra] = self.rank.get(ra, 0) + 1


def build_key_fn(wallets_df):
    """Same identity-key convention as entity_graph.py: profile wallets key
    off their canonical (int) wallet_id, disposable addresses key off the
    address string itself. Keeps this script's entity_id space directly
    comparable with entity_graph.py / alerts.csv (canonical_wallet_id)."""
    addr_to_wallet = dict(zip(wallets_df["address"], wallets_df["wallet_id"]))

    def key(address):
        wid = addr_to_wallet.get(address)
        return wid if wid is not None else address

    return key, addr_to_wallet


def cluster_entities(tx_df, wallets_df):
    key, addr_to_wallet = build_key_fn(wallets_df)
    uf = UnionFind()
    evidence = defaultdict(list)  # canonical union-target key -> list of edge dicts
    high_fanin_txids = []

    def record(a, b, reason, ref):
        uf.union(a, b)
        # store on both ends so lookups after re-rooting still find it
        evidence[a].append({"other": b, "reason": reason, "ref": ref})
        evidence[b].append({"other": a, "reason": reason, "ref": ref})

    # ---- Heuristic 1: common-input-ownership (guarded against mixer pooling)
    for _, tx in tx_df.iterrows():
        in_addrs = tx["input_addresses"]
        if len(in_addrs) < 2:
            continue
        if len(in_addrs) > MAX_COMMON_INPUT_SIZE:
            high_fanin_txids.append((tx["txid"], len(in_addrs)))
            continue
        in_keys = [key(a) for a in in_addrs]
        base = in_keys[0]
        for other in in_keys[1:]:
            if other != base:
                record(base, other, "common_input", tx["txid"])

    # ---- Heuristic 2: likely-change-address (2-output tx, larger = change,
    # only when outputs are meaningfully asymmetric -- see module docstring)
    for _, tx in tx_df.iterrows():
        out_addrs = tx["output_addresses"]
        out_amts = tx["output_amounts"]
        in_addrs = tx["input_addresses"]
        if len(out_addrs) != 2 or len(out_amts) != 2 or not in_addrs:
            continue
        lo, hi = min(out_amts), max(out_amts)
        if hi <= 0 or (lo / hi) >= CHANGE_RATIO_THRESHOLD:
            continue  # near-even split -- not a reliable peel/change signal
        change_idx = 0 if out_amts[0] >= out_amts[1] else 1
        sender_key = key(in_addrs[0])
        change_key = key(out_addrs[change_idx])
        if sender_key != change_key:
            record(sender_key, change_key, "likely_change_address", tx["txid"])

    # ---- Heuristic 3: shared broadcast IP for each tx's primary sender
    ip_to_senders = defaultdict(set)  # src_ip -> {(sender_key, txid)}
    for _, tx in tx_df.iterrows():
        src_ip = tx.get("src_ip")
        if not src_ip or (isinstance(src_ip, float) and pd.isna(src_ip)):
            continue
        in_addrs = tx["input_addresses"]
        if not in_addrs:
            continue
        ip_to_senders[src_ip].add((key(in_addrs[0]), tx["txid"]))

    for ip, senders in ip_to_senders.items():
        senders = list(senders)
        if len(senders) < 2:
            continue
        base_key, base_txid = senders[0]
        for other_key, other_txid in senders[1:]:
            if other_key != base_key:
                record(base_key, other_key, "shared_src_ip", ip)

    return uf, evidence, key, addr_to_wallet, high_fanin_txids


def build_alert_rollup(alerts_df):
    rollup = defaultdict(lambda: {"n_alerts": 0, "max_tier": None, "risk": 0})
    for _, row in alerts_df.iterrows():
        wid = row["canonical_wallet_id"]
        rec = rollup[wid]
        rec["n_alerts"] += 1
        rec["risk"] += TIER_WEIGHT.get(row["priority_tier"], 0)
        if rec["max_tier"] is None or TIER_WEIGHT[row["priority_tier"]] > TIER_WEIGHT.get(
            rec["max_tier"], 0
        ):
            rec["max_tier"] = row["priority_tier"]
    return rollup


def summarize(tx_df, wallets_df, alerts_df, uf, evidence, key, addr_to_wallet):
    wallet_info = wallets_df.set_index("wallet_id").to_dict("index")
    alert_rollup = build_alert_rollup(alerts_df)

    # every key ever seen (profile wallets + every address touched by a tx)
    all_keys = set(wallets_df["wallet_id"])
    for _, tx in tx_df.iterrows():
        for a in tx["input_addresses"] + tx["output_addresses"]:
            all_keys.add(key(a))

    # per-key attributes: country/asn/is_disposable + primary-sender src_ips
    # and volume, collected from whichever transactions actually feature it
    key_country = {}
    key_asn = {}
    key_ips = defaultdict(set)
    key_volume = defaultdict(float)
    key_is_wallet = {}
    for wid, info in wallet_info.items():
        key_country[wid] = info.get("country", "?")
        key_asn[wid] = info.get("asn", "?")
        key_is_wallet[wid] = True

    for _, tx in tx_df.iterrows():
        in_addrs = tx["input_addresses"]
        if not in_addrs:
            continue
        sender_key = key(in_addrs[0])
        key_is_wallet.setdefault(sender_key, sender_key in wallet_info)
        if sender_key not in key_country:
            key_country[sender_key] = tx.get("geo_country", "?")
            key_asn[sender_key] = tx.get("asn", "?")
        src_ip = tx.get("src_ip")
        if src_ip and not (isinstance(src_ip, float) and pd.isna(src_ip)):
            key_ips[sender_key].add(src_ip)
        in_amts = tx["input_amounts"]
        key_volume[sender_key] += sum(in_amts) if in_amts else 0.0
        for a in tx["output_addresses"]:
            key_is_wallet.setdefault(key(a), key(a) in wallet_info)

    # group keys by union-find root
    groups = defaultdict(list)
    for k in all_keys:
        groups[uf.find(k)].append(k)

    # stable, risk-sorted entity ids assigned after we know each group's risk
    rows = []
    per_entity_evidence = {}
    for root, members in groups.items():
        wallet_members = [m for m in members if isinstance(m, int)]
        disposable_members = [m for m in members if not isinstance(m, int)]

        n_alerts = sum(alert_rollup[w]["n_alerts"] for w in wallet_members)
        risk_score = sum(alert_rollup[w]["risk"] for w in wallet_members)
        tiers = [alert_rollup[w]["max_tier"] for w in wallet_members if alert_rollup[w]["max_tier"]]
        max_tier = max(tiers, key=lambda t: TIER_WEIGHT[t]) if tiers else None

        countries = sorted({key_country.get(m, "?") for m in members} - {"?"})
        asns = sorted({key_asn.get(m, "?") for m in members} - {"?"})
        ips = set()
        for m in members:
            ips |= key_ips.get(m, set())
        volume = sum(key_volume.get(m, 0.0) for m in members)

        heuristics_seen = set()
        for m in members:
            for e in evidence.get(m, []):
                heuristics_seen.add(e["reason"])

        rows.append(
            {
                "entity_root": root,
                "n_addresses": len(members),
                "n_profile_wallets": len(wallet_members),
                "n_disposable_addresses": len(disposable_members),
                "wallet_ids": ";".join(str(w) for w in sorted(wallet_members)),
                "n_alerts": n_alerts,
                "max_priority_tier": max_tier or "none",
                "risk_score": risk_score,
                "countries": ";".join(countries),
                "n_countries": len(countries),
                "asns": ";".join(asns),
                "n_asns": len(asns),
                "n_distinct_src_ips": len(ips),
                "cross_border": len(countries) > 1,
                "multi_asn": len(asns) > 1,
                "multi_ip_operator": len(ips) > 1,
                "approx_volume_btc": round(volume, 8),
                "heuristics_used": ";".join(sorted(heuristics_seen)) or "singleton",
            }
        )
        per_entity_evidence[root] = members

    df = pd.DataFrame(rows).sort_values(
        ["risk_score", "n_alerts", "n_addresses"], ascending=False
    ).reset_index(drop=True)
    df.insert(0, "entity_id", [f"E{idx:05d}" for idx in range(len(df))])
    root_to_entity_id = dict(zip(df["entity_root"], df["entity_id"]))
    df = df.drop(columns=["entity_root"])

    return df, root_to_entity_id, per_entity_evidence


def write_address_membership(wallets_df, tx_df, uf, key, root_to_entity_id, addr_to_wallet):
    seen = {}
    rows = []

    def add_row(address, k):
        if address in seen:
            return
        seen[address] = True
        entity_id = root_to_entity_id[uf.find(k)]
        wid = addr_to_wallet.get(address)
        rows.append(
            {
                "address": address,
                "entity_id": entity_id,
                "wallet_id": wid if wid is not None else "",
                "is_disposable": wid is None,
            }
        )

    for _, w in wallets_df.iterrows():
        add_row(w["address"], w["wallet_id"])
    for _, tx in tx_df.iterrows():
        for a in tx["input_addresses"] + tx["output_addresses"]:
            add_row(a, key(a))

    return pd.DataFrame(rows)


def render_html(summary_df, membership_df, evidence_by_entity, out_path, top_n=40):
    """Interactive view of the top-risk clusters: each cluster's addresses
    drawn as a small connected component, colour-coded per entity, with
    hover text explaining WHY each address is in that cluster (heuristic +
    evidence). Complements entity_graph_risk.html (which shows the full
    transaction flow) with a clustering-focused view (which nodes are the
    SAME actor)."""
    import colorsys

    from pyvis.network import Network

    top = summary_df.head(top_n)
    net = Network(
        height="900px",
        width="100%",
        directed=False,
        notebook=False,
        bgcolor="#111318",
        font_color="#e8e8e8",
        cdn_resources="in_line",
    )
    net.barnes_hut(gravity=-3000, central_gravity=0.2, spring_length=120, spring_strength=0.03)

    def color_for(i, n):
        h = (i / max(n, 1)) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.55, 0.95)
        return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

    mem_by_entity = defaultdict(list)
    for _, row in membership_df.iterrows():
        mem_by_entity[row["entity_id"]].append(row)

    added = set()
    for i, (_, erow) in enumerate(top.iterrows()):
        entity_id = erow["entity_id"]
        color = color_for(i, len(top))
        members = mem_by_entity.get(entity_id, [])
        for m in members:
            node_id = f"{entity_id}:{m['address']}"
            if node_id in added:
                continue
            added.add(node_id)
            label = f"#{m['wallet_id']}" if m["wallet_id"] != "" else m["address"][:8]
            title = (
                f"entity {entity_id}<br>address {m['address'][:16]}...<br>"
                f"{'profile wallet #' + str(m['wallet_id']) if m['wallet_id'] != '' else 'disposable address'}<br>"
                f"cluster risk_score={erow['risk_score']}, tier={erow['max_priority_tier']}"
            )
            net.add_node(
                node_id,
                label=label,
                title=title,
                color=color,
                size=12 + (6 if m["wallet_id"] != "" else 0),
            )
        # draw a simple chain through the entity's members as a visual grouping
        # cue (exact heuristic-edge topology is in entity_clusters_evidence.json;
        # this view is for "which addresses are the same actor", not edge detail)
        for j in range(len(members) - 1):
            a, b = members[j], members[j + 1]
            net.add_edge(
                f"{entity_id}:{a['address']}",
                f"{entity_id}:{b['address']}",
                color=color,
                width=1.5,
                title="same entity (see entity_clusters_evidence.json for exact heuristic)",
            )

    net.set_options(
        """
    {
      "physics": {"stabilization": {"iterations": 150}},
      "interaction": {"hover": true, "tooltipDelay": 100}
    }
    """
    )
    net.write_html(out_path, notebook=False, open_browser=False)


def main():
    tx_df = load_transactions(TX_PATH)
    wallets_df = pd.read_csv(WALLETS_PATH)
    try:
        alerts_df = pd.read_csv(ALERTS_PATH)
    except FileNotFoundError:
        print(f"warning: {ALERTS_PATH} not found -- clustering will run without risk scoring")
        alerts_df = pd.DataFrame(columns=["txid", "canonical_wallet_id", "priority_tier"])

    uf, evidence, key, addr_to_wallet, high_fanin = cluster_entities(tx_df, wallets_df)
    summary_df, root_to_entity_id, per_entity_evidence = summarize(
        tx_df, wallets_df, alerts_df, uf, evidence, key, addr_to_wallet
    )
    membership_df = write_address_membership(
        wallets_df, tx_df, uf, key, root_to_entity_id, addr_to_wallet
    )

    n_singletons = (summary_df["n_addresses"] == 1).sum()
    n_multi = len(summary_df) - n_singletons
    print(
        f"Clustered {membership_df['address'].nunique()} addresses into "
        f"{len(summary_df)} entities ({n_multi} multi-address entities, "
        f"{n_singletons} singletons -- an address with no clustering edge "
        f"stays its own entity, which is correct, not a gap)"
    )
    print(
        f"Excluded {len(high_fanin)} high-fan-in transactions (>{MAX_COMMON_INPUT_SIZE} "
        f"inputs) from common-input clustering -- reported as pooling/mixing "
        f"points below instead of merged"
    )

    summary_df.to_csv("output/entity_clusters_summary.csv", index=False)
    membership_df.to_csv("output/entity_clusters.csv", index=False)
    print("Wrote output/entity_clusters_summary.csv, output/entity_clusters.csv")

    evidence_by_entity = {}
    for root, members in per_entity_evidence.items():
        entity_id = root_to_entity_id[root]
        edges = []
        seen_pairs = set()
        for m in members:
            for e in evidence.get(m, []):
                pair = tuple(sorted([str(m), str(e["other"])])) + (e["reason"],)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                edges.append({"from": str(m), "to": str(e["other"]), "reason": e["reason"], "ref": str(e["ref"])})
        if edges:
            evidence_by_entity[entity_id] = edges[:25]

    with open("output/entity_clusters_evidence.json", "w") as f:
        json.dump(evidence_by_entity, f, indent=2)
    print("Wrote output/entity_clusters_evidence.json (why each cluster was formed)")

    # ---- human-readable report
    lines = []
    lines.append(f"Total entities: {len(summary_df)} ({n_multi} multi-address, {n_singletons} singleton)")
    lines.append("")
    lines.append(f"Top {min(TOP_N_FOR_REPORT, len(summary_df))} entities by risk_score:")
    for _, row in summary_df.head(TOP_N_FOR_REPORT).iterrows():
        flags = []
        if row["cross_border"]:
            flags.append(f"cross-border({row['countries']})")
        if row["multi_asn"]:
            flags.append("multi-ASN")
        if row["multi_ip_operator"]:
            flags.append(f"{row['n_distinct_src_ips']} distinct src_ips")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        lines.append(
            f"  {row['entity_id']}: risk={row['risk_score']}, {row['n_alerts']} alerts "
            f"(worst={row['max_priority_tier']}), {row['n_addresses']} addresses "
            f"({row['n_profile_wallets']} profile wallets, wallet_ids=[{row['wallet_ids']}]), "
            f"~{row['approx_volume_btc']:.4f} BTC via primary-sender legs, "
            f"heuristics=[{row['heuristics_used']}]{flag_str}"
        )
    if high_fanin:
        lines.append("")
        lines.append(
            f"Excluded high-fan-in transactions (likely mixer/pooling points, "
            f"NOT clustered as common ownership -- see heuristic 1 caveat in module docstring):"
        )
        for txid, n in sorted(high_fanin, key=lambda x: -x[1])[:15]:
            lines.append(f"  {txid}: {n} inputs")

    report = "\n".join(lines)
    with open("output/entity_clusters_summary.txt", "w") as f:
        f.write(report + "\n")
    print("\n" + report)
    print("\nWrote output/entity_clusters_summary.txt")

    try:
        render_html(summary_df, membership_df, evidence_by_entity, "output/entity_clusters_risk.html")
        print("Wrote output/entity_clusters_risk.html (interactive -- open in a browser)")
    except Exception as e:
        print(f"warning: could not render entity_clusters_risk.html ({e})")


if __name__ == "__main__":
    main()
