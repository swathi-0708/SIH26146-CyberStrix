"""
Network Correlation Feature Module
===================================

Closes the gap flagged in RUN_ORDER.md under "Still open":

    "Network-layer work beyond `src_ip` -- `dst_ip`/`dst_port` are ingested
    but unused in features or the graph."

`ENGINE_network_blockchain_correlation.py` (despite its name) only covers
temporal-per-IP and IP<->wallet correlation -- it never touches dst_ip,
ports, ASNs, or countries, and isn't wired into the run order at all. This
module is the actual **network correlation** layer: per-sender network-flow
diversity, computed the same causal/leak-safe way as the rest of
`split_dataset.py`'s per-sender features (grouped by `first_input_addr`,
sorted by time, each row scored from history strictly BEFORE it).

Produces 7 columns per transaction row, one per requirement:

    sender_unique_src_ips          unique source IPs      this sender has used
    sender_unique_dst_ips          unique destination IPs this sender has used
    sender_unique_ports            unique ports (src+dst)  this sender has used
    sender_unique_asns             unique ASNs             this sender has used
    sender_unique_countries        unique countries        this sender has used
    sender_cross_country_activity  1 if >1 distinct country seen so far, else 0
    sender_multi_asn_activity      1 if >1 distinct ASN seen so far, else 0

All five "unique_*" counts and both activity flags are CUMULATIVE and
CAUSAL: they reflect everything seen for that sender strictly before the
current row's timestamp, then the current row's own network attributes are
folded in for future rows. This mirrors the sender_tx_count_1h /
sender_distinct_asn_last10 loop already in split_dataset.py -- same
sender_col ("first_input_addr"), same ts_col ("timestamp_dt"), same
before-then-append ordering -- so a per-transaction model gets the same
signal a wallet-level network-graph rollup would show, without leaking the
current row's own label into its own features.

Note on this dataset specifically: `dst_port` is a constant (8333, the
standard Bitcoin P2P port) in every row `generate_dataset.py` emits, so
`sender_unique_ports` here is really tracking `src_port` diversity (which is
drawn fresh per transaction -- expect it to climb roughly 1:1 with
transaction count, which is itself a legitimate "how much port/connection
churn does this sender show" signal, just not a strong standalone one). If
real-world dst_port data (multiple services/protocols) is substituted in
later, this column starts carrying more signal for free -- no code change
needed.

Usage:
    from network_correlation import add_network_correlation_features, NETWORK_CORR_COLS

    raw = add_network_correlation_features(raw)   # raw needs first_input_addr + timestamp_dt already
    feature_cols += NETWORK_CORR_COLS
"""

from __future__ import annotations

import pandas as pd

NETWORK_CORR_COLS = [
    "sender_unique_src_ips",
    "sender_unique_dst_ips",
    "sender_unique_ports",
    "sender_unique_asns",
    "sender_unique_countries",
    "sender_cross_country_activity",
    "sender_multi_asn_activity",
]


def add_network_correlation_features(
    raw: pd.DataFrame,
    sender_col: str = "first_input_addr",
    ts_col: str = "timestamp_dt",
    src_ip_col: str = "src_ip",
    dst_ip_col: str = "dst_ip",
    src_port_col: str = "src_port",
    dst_port_col: str = "dst_port",
    asn_col: str = "asn",
    country_col: str = "geo_country",
) -> pd.DataFrame:
    """
    Adds the 7 NETWORK_CORR_COLS to `raw` in place (and returns it).

    Requires `sender_col` and `ts_col` to already exist (split_dataset.py
    computes both before this is called: first_input_addr from n_inputs/
    input_list, timestamp_dt via pd.to_datetime). Requires the network flow
    columns (src_ip/dst_ip/src_port/dst_port/asn/country) to already be
    present -- ingest.py guarantees this for every supported input format
    (CSV/JSON/XML).
    """
    for col in NETWORK_CORR_COLS:
        raw[col] = 0

    for addr, idxs in raw.groupby(sender_col).groups.items():
        idxs = sorted(idxs, key=lambda i: raw.at[i, ts_col])

        seen_src_ips: set = set()
        seen_dst_ips: set = set()
        seen_ports: set = set()
        seen_asns: set = set()
        seen_countries: set = set()

        for i in idxs:
            # --- score this row from history strictly BEFORE it (causal) ---
            raw.at[i, "sender_unique_src_ips"] = len(seen_src_ips)
            raw.at[i, "sender_unique_dst_ips"] = len(seen_dst_ips)
            raw.at[i, "sender_unique_ports"] = len(seen_ports)
            raw.at[i, "sender_unique_asns"] = len(seen_asns)
            raw.at[i, "sender_unique_countries"] = len(seen_countries)
            raw.at[i, "sender_cross_country_activity"] = int(len(seen_countries) > 1)
            raw.at[i, "sender_multi_asn_activity"] = int(len(seen_asns) > 1)

            # --- NOW fold this row's network attrs into history, so they can
            # only affect FUTURE rows for this sender ---
            seen_src_ips.add(raw.at[i, src_ip_col])
            seen_dst_ips.add(raw.at[i, dst_ip_col])
            seen_ports.add(raw.at[i, src_port_col])
            seen_ports.add(raw.at[i, dst_port_col])
            seen_asns.add(raw.at[i, asn_col])
            seen_countries.add(raw.at[i, country_col])

    return raw


if __name__ == "__main__":
    # Small standalone smoke test -- no real dataset required.
    demo = pd.DataFrame(
        [
            # wallet "A": stable single IP/ASN/country -- should stay quiet
            {"first_input_addr": "A", "timestamp_dt": pd.Timestamp("2024-01-01 00:00:00", tz="UTC"),
             "src_ip": "1.1.1.1", "dst_ip": "9.9.9.9", "src_port": 5000, "dst_port": 8333,
             "asn": "AS111", "geo_country": "US"},
            {"first_input_addr": "A", "timestamp_dt": pd.Timestamp("2024-01-01 00:05:00", tz="UTC"),
             "src_ip": "1.1.1.1", "dst_ip": "9.9.9.9", "src_port": 5001, "dst_port": 8333,
             "asn": "AS111", "geo_country": "US"},
            # wallet "B": ip-hopping across ASNs/countries -- should light up
            {"first_input_addr": "B", "timestamp_dt": pd.Timestamp("2024-01-01 00:00:00", tz="UTC"),
             "src_ip": "2.2.2.2", "dst_ip": "9.9.9.9", "src_port": 6000, "dst_port": 8333,
             "asn": "AS222", "geo_country": "DE"},
            {"first_input_addr": "B", "timestamp_dt": pd.Timestamp("2024-01-01 00:05:00", tz="UTC"),
             "src_ip": "3.3.3.3", "dst_ip": "9.9.9.9", "src_port": 6001, "dst_port": 8333,
             "asn": "AS333", "geo_country": "VN"},
            {"first_input_addr": "B", "timestamp_dt": pd.Timestamp("2024-01-01 00:10:00", tz="UTC"),
             "src_ip": "4.4.4.4", "dst_ip": "9.9.9.9", "src_port": 6002, "dst_port": 8333,
             "asn": "AS444", "geo_country": "RO"},
        ]
    )
    out = add_network_correlation_features(demo)
    print(out[["first_input_addr", "timestamp_dt"] + NETWORK_CORR_COLS].to_string(index=False))
