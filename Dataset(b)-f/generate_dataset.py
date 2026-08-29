#!/usr/bin/env python3
"""
Synthetic Bitcoin transaction + network metadata generator for SIH PS 26146.

Produces data matching the exact schema in the problem statement:
    timestamp, src_ip, dst_ip, src_port, dst_port, txid,
    input_addresses[], output_addresses[], input_amounts[], output_amounts[],
    fee, script_type, geo_country, asn

Design:
  - A population of "wallet profiles" generates NORMAL transaction behavior
    (regular counterparties, stable amount range, stable IP/ASN, steady cadence).
  - Six distinct ANOMALY PATTERNS are injected on top, each with a ground-truth
    label kept in a SEPARATE file (ground_truth_labels.csv) so your model never
    sees the labels during training/inference -- only you use them, to score
    precision/recall for your demo and write-up.

Anomaly patterns injected (map these to your research write-up):
  1. peeling_chain      - repeated small "peel" outputs + one large change output,
                           chained across several hops (classic laundering pattern)
  2. fan_out_mixer       - one wallet receives from many inputs then rapidly fans
                           out to many outputs in a short window (mixer-like)
  3. wallet_reuse_burst  - a normally-quiet wallet suddenly transacts dozens of
                           times in a short burst after long dormancy (temporal anomaly)
  4. structuring         - many transactions with amounts just under a round
                           threshold, to a similar set of destinations (smurfing)
  5. ip_hopping          - a single wallet's transactions get broadcast from many
                           distinct IPs/ASNs in a short time (proxy/mixing signal,
                           this is the network-layer anomaly specific to this PS)
  6. profile_deviation   - a wallet suddenly transacts a value/counterparty wildly
                           outside its own historical profile (account-takeover-like)

Usage:
    python3 generate_dataset.py --n-wallets 800 --n-normal-tx 6000 --seed 42
Outputs (in ./output/):
    transactions.csv          - full dataset, CSV (list fields JSON-encoded)
    transactions.json         - same dataset, JSON array
    transactions_sample.xml   - small XML sample (first 25 tx) to test XML ingestion
    ground_truth_labels.csv   - txid -> is_anomaly, anomaly_type, wallet(s) involved
    wallets_reference.csv     - wallet profile reference (for your own debugging)
"""

import argparse
import json
import random
import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker

fake = Faker()

# Real, GeoIP-verified IPs for each ASN (found via find_real_asn_ips.py by
# actually resolving random public IPv4 space through the same offline
# GeoIP database our system uses -- see real_asn_ips.json). This makes every
# IP in the dataset resolve, via independent GeoIP lookup, to the same
# country/ASN the wallet profile intends, instead of using unrelated random
# public IPs with a fictional label bolted on.
with open("real_asn_ips.json") as _f:
    REAL_ASN_IPS = json.load(_f)

SCRIPT_TYPES = ["P2PKH", "P2SH", "P2WPKH", "P2WSH", "P2TR"]

# A handful of ASN "flavors" so IP-diversity / hosting-provider anomalies are meaningful
ASN_POOL = [
    {"asn": "AS7922", "org": "Comcast", "country": "US", "kind": "residential"},
    {"asn": "AS3320", "org": "Deutsche Telekom", "country": "DE", "kind": "residential"},
    {"asn": "AS4713", "org": "NTT", "country": "JP", "kind": "residential"},
    {"asn": "AS45899", "org": "VNPT", "country": "VN", "kind": "residential"},
    {"asn": "AS9829", "org": "Mahanagar Telephone Nigam Limited", "country": "IN", "kind": "residential"},
    {"asn": "AS24560", "org": "Airtel", "country": "IN", "kind": "residential"},
    {"asn": "AS16509", "org": "Amazon AWS", "country": "US", "kind": "hosting"},
    {"asn": "AS14061", "org": "DigitalOcean", "country": "US", "kind": "hosting"},
    {"asn": "AS16276", "org": "OVH", "country": "FR", "kind": "hosting"},
    {"asn": "AS9009", "org": "M247 (VPN-heavy)", "country": "RO", "kind": "hosting"},
    {"asn": "AS60068", "org": "Datacamp Limited / CDN77 (VPN-heavy)", "country": "NL", "kind": "hosting"},
]


def rand_ip(asn_entry):
    """Return a REAL public IP that genuinely resolves (via our offline GeoIP
    database) to this ASN. Draws from the ~15 verified anchor IPs per ASN,
    with the last octet jittered to avoid literally repeating the same
    address thousands of times -- jittering only the last octet keeps the
    IP inside the same /24, which almost always stays within the same
    allocated block and therefore still resolves to the same ASN."""
    pool = REAL_ASN_IPS.get(asn_entry["asn"])
    if not pool:
        return fake.ipv4_public()  # fallback, shouldn't happen
    base = random.choice(pool)
    octets = base.split(".")
    octets[3] = str(random.randint(1, 254))
    return ".".join(octets)


def new_wallet_address():
    prefix = random.choice(["1", "3", "bc1q"])
    tail = uuid.uuid4().hex[:30]
    return f"{prefix}{tail}"


def new_txid():
    return uuid.uuid4().hex


class WalletProfile:
    """A normal wallet's stable behavioral profile."""

    def __init__(self, wid, kind="regular"):
        self.id = wid
        self.kind = kind  # "regular" | "merchant" | "exchange" -- hard negatives live here
        self.address = new_wallet_address()
        self.asn_entry = random.choice(ASN_POOL)
        self.home_ip = rand_ip(self.asn_entry)
        if kind == "regular":
            self.typical_amount = round(np.random.lognormal(mean=-3.0, sigma=1.0), 8)
            self.avg_gap_minutes = random.choice([30, 60, 180, 720, 1440, 4320])
        else:
            # merchants/exchanges: legitimately high-frequency, high-fan-out, many
            # counterparties -- feature-wise this OVERLAPS the injected anomalies
            # on purpose, so the model can't just learn "high activity = anomaly"
            self.typical_amount = round(np.random.lognormal(mean=-1.5, sigma=1.2), 8)
            self.avg_gap_minutes = random.choice([2, 5, 10, 20])
        self.typical_counterparties = []
        self.script_type = random.choices(SCRIPT_TYPES, weights=[35, 15, 30, 10, 10])[0]
        self.last_tx_time = None


def make_normal_transaction(wallets, t, wid_from=None):
    """One organic transaction between two (usually recurring) wallet profiles."""
    if wid_from is None:
        sender = random.choice(wallets)
    else:
        sender = wallets[wid_from]

    # merchants/exchanges legitimately have far more counterparties and larger fan-in/out
    max_counterparties = 5 if sender.kind == "regular" else 40
    if sender.typical_counterparties and random.random() < 0.7:
        receiver = random.choice(sender.typical_counterparties)
    else:
        receiver = random.choice(wallets)
        while receiver.id == sender.id:
            receiver = random.choice(wallets)
        if len(sender.typical_counterparties) < max_counterparties:
            sender.typical_counterparties.append(receiver)

    if sender.kind == "regular":
        n_in = np.random.choice([1, 1, 1, 2, 3], p=[0.55, 0.2, 0.15, 0.06, 0.04])
        n_out = np.random.choice([1, 2, 2, 3], p=[0.5, 0.3, 0.15, 0.05])
    else:
        # legitimate high-volume wallets: naturally high fan-in/out, same shape
        # a fan_out_mixer anomaly would have -- this is the deliberate overlap
        n_in = np.random.choice([1, 2, 3, 5, 8], p=[0.3, 0.25, 0.2, 0.15, 0.1])
        n_out = np.random.choice([1, 2, 4, 8, 15], p=[0.3, 0.25, 0.2, 0.15, 0.1])

    amount = max(0.00001, np.random.normal(sender.typical_amount, sender.typical_amount * 0.25))
    fee = round(amount * random.uniform(0.0005, 0.004) + 0.00001, 8)

    input_addrs = [sender.address] + [new_wallet_address() for _ in range(n_in - 1)]
    output_addrs = [receiver.address] + [new_wallet_address() for _ in range(n_out - 1)]
    in_amounts = [round(amount / n_in, 8) for _ in range(n_in)]
    out_amounts = [round((amount - fee) / n_out, 8) for _ in range(n_out)]

    row = {
        "timestamp": t.isoformat() + "Z",
        "src_ip": sender.home_ip,
        "dst_ip": rand_ip(receiver.asn_entry),
        "src_port": random.randint(1024, 65535),
        "dst_port": 8333,  # standard Bitcoin P2P port
        "txid": new_txid(),
        "input_addresses": input_addrs,
        "output_addresses": output_addrs,
        "input_amounts": in_amounts,
        "output_amounts": out_amounts,
        "fee": fee,
        "script_type": sender.script_type,
        "geo_country": sender.asn_entry["country"],
        "asn": sender.asn_entry["asn"],
    }
    sender.last_tx_time = t
    return row, sender.id, receiver.id


def inject_peeling_chain(wallets, start_time, chain_len=8):
    rows, labels = [], []
    source = random.choice(wallets)
    current_amount = round(np.random.uniform(2.0, 8.0), 6)  # start with a "big" balance
    t = start_time
    current_addr = source.address
    for hop in range(chain_len):
        peel = round(current_amount * np.random.uniform(0.02, 0.08), 8)  # small peeled-off amount
        change_addr = new_wallet_address()
        peel_addr = new_wallet_address()
        fee = round(current_amount * 0.0008, 8)
        remaining = round(current_amount - peel - fee, 8)
        txid = new_txid()
        row = {
            "timestamp": t.isoformat() + "Z",
            "src_ip": source.home_ip,
            "dst_ip": rand_ip(source.asn_entry),
            "src_port": random.randint(1024, 65535),
            "dst_port": 8333,
            "txid": txid,
            "input_addresses": [current_addr],
            "output_addresses": [peel_addr, change_addr],
            "input_amounts": [current_amount],
            "output_amounts": [peel, remaining],
            "fee": fee,
            "script_type": source.script_type,
            "geo_country": source.asn_entry["country"],
            "asn": source.asn_entry["asn"],
        }
        rows.append(row)
        labels.append({"txid": txid, "is_anomaly": 1, "anomaly_type": "peeling_chain",
                        "wallet_ids": f"synthetic_chain_{source.id}_hop{hop}"})
        current_addr = change_addr
        current_amount = remaining
        t += timedelta(minutes=random.randint(5, 45))
    return rows, labels


def inject_fan_out_mixer(wallets, start_time, n_in=25, n_out=25):
    rows, labels = [], []
    mixer = random.choice(wallets)
    t = start_time
    # fan-in
    txid_in = new_txid()
    input_addrs = [new_wallet_address() for _ in range(n_in)]
    total_in = round(sum(np.random.uniform(0.05, 0.5) for _ in range(n_in)), 8)
    rows.append({
        "timestamp": t.isoformat() + "Z", "src_ip": mixer.home_ip,
        "dst_ip": rand_ip(mixer.asn_entry), "src_port": random.randint(1024, 65535),
        "dst_port": 8333, "txid": txid_in,
        "input_addresses": input_addrs, "output_addresses": [mixer.address],
        "input_amounts": [round(total_in / n_in, 8)] * n_in, "output_amounts": [total_in * 0.999],
        "fee": round(total_in * 0.001, 8), "script_type": mixer.script_type,
        "geo_country": mixer.asn_entry["country"], "asn": mixer.asn_entry["asn"],
    })
    labels.append({"txid": txid_in, "is_anomaly": 1, "anomaly_type": "fan_out_mixer",
                    "wallet_ids": f"mixer_{mixer.id}_fanin"})
    t += timedelta(minutes=random.randint(2, 10))
    # rapid fan-out, split across several *different* source IPs (mixer-like laundering)
    per_out = round((total_in * 0.995) / n_out, 8)
    for i in range(n_out):
        out_asn = random.choice(ASN_POOL)
        txid_out = new_txid()
        rows.append({
            "timestamp": t.isoformat() + "Z", "src_ip": rand_ip(out_asn),
            "dst_ip": fake.ipv4_public(), "src_port": random.randint(1024, 65535),
            "dst_port": 8333, "txid": txid_out,
            "input_addresses": [mixer.address], "output_addresses": [new_wallet_address()],
            "input_amounts": [per_out], "output_amounts": [round(per_out * 0.998, 8)],
            "fee": round(per_out * 0.002, 8), "script_type": random.choice(SCRIPT_TYPES),
            "geo_country": out_asn["country"], "asn": out_asn["asn"],
        })
        labels.append({"txid": txid_out, "is_anomaly": 1, "anomaly_type": "fan_out_mixer",
                        "wallet_ids": f"mixer_{mixer.id}_fanout{i}"})
        t += timedelta(seconds=random.randint(30, 240))
    return rows, labels


def inject_wallet_reuse_burst(wallets, start_time, burst_size=30):
    rows, labels = [], []
    w = random.choice(wallets)
    t = start_time
    for i in range(burst_size):
        row, _, _ = make_normal_transaction(wallets, t, wid_from=w.id)
        row["timestamp"] = t.isoformat() + "Z"
        rows.append(row)
        labels.append({"txid": row["txid"], "is_anomaly": 1, "anomaly_type": "wallet_reuse_burst",
                        "wallet_ids": f"burst_{w.id}"})
        t += timedelta(seconds=random.randint(20, 90))  # far faster than its normal cadence
    return rows, labels


def inject_structuring(wallets, start_time, threshold=10.0, n_tx=12):
    rows, labels = [], []
    w = random.choice(wallets)
    t = start_time
    for i in range(n_tx):
        amount = round(threshold - np.random.uniform(0.01, 0.35), 8)  # just under threshold
        receiver_addr = new_wallet_address()
        txid = new_txid()
        rows.append({
            "timestamp": t.isoformat() + "Z", "src_ip": w.home_ip,
            "dst_ip": rand_ip(w.asn_entry), "src_port": random.randint(1024, 65535),
            "dst_port": 8333, "txid": txid,
            "input_addresses": [w.address], "output_addresses": [receiver_addr],
            "input_amounts": [amount], "output_amounts": [round(amount * 0.999, 8)],
            "fee": round(amount * 0.001, 8), "script_type": w.script_type,
            "geo_country": w.asn_entry["country"], "asn": w.asn_entry["asn"],
        })
        labels.append({"txid": txid, "is_anomaly": 1, "anomaly_type": "structuring",
                        "wallet_ids": f"structuring_{w.id}"})
        t += timedelta(hours=random.randint(4, 30))
    return rows, labels


def inject_ip_hopping(wallets, start_time, n_tx=15):
    rows, labels = [], []
    w = random.choice(wallets)
    t = start_time
    for i in range(n_tx):
        hop_asn = random.choice(ASN_POOL)  # different ASN each time -> unstable network origin
        receiver_addr = new_wallet_address()
        amount = round(np.random.uniform(0.01, 0.3), 8)
        txid = new_txid()
        rows.append({
            "timestamp": t.isoformat() + "Z", "src_ip": rand_ip(hop_asn),
            "dst_ip": fake.ipv4_public(), "src_port": random.randint(1024, 65535),
            "dst_port": 8333, "txid": txid,
            "input_addresses": [w.address], "output_addresses": [receiver_addr],
            "input_amounts": [amount], "output_amounts": [round(amount * 0.998, 8)],
            "fee": round(amount * 0.002, 8), "script_type": w.script_type,
            "geo_country": hop_asn["country"], "asn": hop_asn["asn"],
        })
        labels.append({"txid": txid, "is_anomaly": 1, "anomaly_type": "ip_hopping",
                        "wallet_ids": f"iphop_{w.id}"})
        t += timedelta(minutes=random.randint(3, 20))
    return rows, labels


def inject_profile_deviation(wallets, start_time, n_events=10):
    rows, labels = [], []
    t = start_time
    for i in range(n_events):
        w = random.choice(wallets)
        huge_amount = round(w.typical_amount * np.random.uniform(15, 60) + 1.0, 8)
        receiver_addr = new_wallet_address()  # brand new, never-seen counterparty
        txid = new_txid()
        rows.append({
            "timestamp": t.isoformat() + "Z", "src_ip": w.home_ip,
            "dst_ip": fake.ipv4_public(), "src_port": random.randint(1024, 65535),
            "dst_port": 8333, "txid": txid,
            "input_addresses": [w.address], "output_addresses": [receiver_addr],
            "input_amounts": [huge_amount], "output_amounts": [round(huge_amount * 0.999, 8)],
            "fee": round(huge_amount * 0.001, 8), "script_type": w.script_type,
            "geo_country": w.asn_entry["country"], "asn": w.asn_entry["asn"],
        })
        labels.append({"txid": txid, "is_anomaly": 1, "anomaly_type": "profile_deviation",
                        "wallet_ids": f"deviation_{w.id}"})
        t += timedelta(hours=random.randint(1, 72))
    return rows, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-wallets", type=int, default=800)
    # n-normal-tx tuned so the overall anomaly rate lands near the Elliptic/Elliptic++
    # benchmark's real-world illicit rate (~2% of all transactions, ~9.8% of LABELED
    # transactions -- see calibration_notes.md). ~11,000 normal + ~565 injected anomalies
    # gives ~4.9%, deliberately a bit above the raw 2% so a hackathon-scale demo still has
    # enough positive examples to train/evaluate on -- documented as an intentional
    # deviation in calibration_notes.md, not an oversight.
    ap.add_argument("--n-normal-tx", type=int, default=11000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=str, default="output")
    ap.add_argument("--realism", choices=["clean", "realistic"], default="realistic",
                     help="'clean': textbook anomaly signatures, no hard negatives, no "
                          "capture noise -- good for initial dev/debugging (this is "
                          "'Dataset A'). 'realistic': adds legitimate high-activity "
                          "merchant/exchange wallets that overlap anomaly features, plus "
                          "missing-field/timestamp-jitter capture noise -- use this for "
                          "your real evaluation numbers (this is 'Dataset B').")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    Faker.seed(args.seed)

    import os
    os.makedirs(args.out_dir, exist_ok=True)
    tag = "_clean" if args.realism == "clean" else ""

    n_merchants = max(1, int(args.n_wallets * 0.03)) if args.realism == "realistic" else 0
    n_exchanges = max(1, int(args.n_wallets * 0.01)) if args.realism == "realistic" else 0
    kinds = ["regular"] * args.n_wallets
    special_idx = random.sample(range(args.n_wallets), n_merchants + n_exchanges)
    for i, idx in enumerate(special_idx):
        kinds[idx] = "merchant" if i < n_merchants else "exchange"
    wallets = [WalletProfile(i, kind=kinds[i]) for i in range(args.n_wallets)]

    start = datetime(2026, 6, 1, 0, 0, 0)
    all_rows = []
    all_labels = []

    # --- normal traffic, spread over ~60 days ---
    t = start
    for i in range(args.n_normal_tx):
        t += timedelta(minutes=np.random.exponential(12))
        row, _, _ = make_normal_transaction(wallets, t)
        all_rows.append(row)
        all_labels.append({"txid": row["txid"], "is_anomaly": 0, "anomaly_type": "normal", "wallet_ids": ""})

    end_time = t

    # --- inject anomaly patterns at random points within the timeline ---
    def rand_time():
        return start + timedelta(seconds=random.randint(0, int((end_time - start).total_seconds())))

    for _ in range(6):
        r, l = inject_peeling_chain(wallets, rand_time(), chain_len=random.randint(5, 10))
        all_rows += r; all_labels += l
    for _ in range(4):
        r, l = inject_fan_out_mixer(wallets, rand_time())
        all_rows += r; all_labels += l
    for _ in range(8):
        r, l = inject_wallet_reuse_burst(wallets, rand_time())
        all_rows += r; all_labels += l
    for _ in range(6):
        r, l = inject_structuring(wallets, rand_time())
        all_rows += r; all_labels += l
    for _ in range(6):
        r, l = inject_ip_hopping(wallets, rand_time())
        all_rows += r; all_labels += l
    for _ in range(1):
        r, l = inject_profile_deviation(wallets, rand_time(), n_events=12)
        all_rows += r; all_labels += l

    # sort everything by timestamp, like a real capture would be ordered
    combined = list(zip(all_rows, all_labels))
    combined.sort(key=lambda x: x[0]["timestamp"])
    all_rows = [c[0] for c in combined]
    all_labels = [c[1] for c in combined]

    df = pd.DataFrame(all_rows)
    labels_df = pd.DataFrame(all_labels)

    # --- realistic capture noise (only in 'realistic' mode): real network taps have
    # gaps, not a perfect feed. None of this touches labels. ---
    if args.realism == "realistic":
        rng = np.random.default_rng(args.seed)
        n = len(df)
        geo_gap_idx = rng.choice(n, size=int(n * 0.015), replace=False)
        df.loc[geo_gap_idx, ["geo_country", "asn"]] = None
        port_gap_idx = rng.choice(n, size=int(n * 0.01), replace=False)
        df.loc[port_gap_idx, "dst_port"] = None
        jitter_seconds = rng.integers(-4, 5, size=n)
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601") + pd.to_timedelta(jitter_seconds, unit="s")
        df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    # CSV: JSON-encode the list-type columns (standard practice for list fields in CSV)
    df_csv = df.copy()
    for col in ["input_addresses", "output_addresses", "input_amounts", "output_amounts"]:
        df_csv[col] = df_csv[col].apply(json.dumps)
    df_csv.to_csv(f"{args.out_dir}/transactions{tag}.csv", index=False)

    # JSON: native list fields
    df.to_json(f"{args.out_dir}/transactions{tag}.json", orient="records", indent=2)

    # small XML sample so the team can test their XML parser path too
    import xml.etree.ElementTree as ET
    root = ET.Element("transactions")
    for row in all_rows[:25]:
        tx_el = ET.SubElement(root, "transaction")
        for k, v in row.items():
            child = ET.SubElement(tx_el, k)
            if isinstance(v, list):
                child.text = ",".join(str(x) for x in v)
            else:
                child.text = str(v)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(f"{args.out_dir}/transactions_sample{tag}.xml", encoding="utf-8", xml_declaration=True)

    labels_df.to_csv(f"{args.out_dir}/ground_truth_labels{tag}.csv", index=False)

    wallets_ref = pd.DataFrame([{
        "wallet_id": w.id, "address": w.address, "home_ip": w.home_ip,
        "asn": w.asn_entry["asn"], "country": w.asn_entry["country"],
        "typical_amount_btc": w.typical_amount, "script_type": w.script_type,
    } for w in wallets])
    wallets_ref.to_csv(f"{args.out_dir}/wallets_reference{tag}.csv", index=False)

    n_anom = labels_df["is_anomaly"].sum()
    print(f"Generated {len(df)} transactions ({n_anom} anomalous, {len(df) - n_anom} normal, "
          f"{n_anom/len(df)*100:.2f}% anomaly rate)")
    print(f"Anomaly type breakdown:\n{labels_df[labels_df.is_anomaly==1].anomaly_type.value_counts()}")
    print(f"Files written to ./{args.out_dir}/")


if __name__ == "__main__":
    main()
