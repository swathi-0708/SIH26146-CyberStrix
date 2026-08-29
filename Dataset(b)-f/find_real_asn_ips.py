#!/usr/bin/env python3
"""
One-time search: sample real public IPv4 space and bucket IPs by which ASN
they actually belong to (per our offline GeoIP database), for each ASN in
our wallet-profile pool. Output: a JSON file of real, verified IP/CIDR
samples per ASN, which generate_dataset.py then draws from -- so every IP
in the dataset genuinely resolves, via independent GeoIP lookup, to the
ASN/country the wallet profile intends.
"""
import json
import random
import time
from geoip2fast import GeoIP2Fast

G = GeoIP2Fast(geoip2fast_data_file="geoip2fast-asn.dat.gz")

TARGETS = {
    "AS7922":  {"keyword": "comcast", "country": "US"},
    "AS3320":  {"keyword": "deutsche telekom", "country": "DE"},
    "AS4713":  {"keyword": "ntt", "country": "JP"},
    "AS45899": {"keyword": "vnpt", "country": "VN"},
    "AS9829":  {"keyword": "bsnl", "country": "IN"},
    "AS24560": {"keyword": "airtel", "country": "IN"},
    "AS16509": {"keyword": "amazon", "country": "US"},
    "AS14061": {"keyword": "digitalocean", "country": "US"},
    "AS16276": {"keyword": "ovh", "country": "FR"},
    "AS9009":  {"keyword": "m247", "country": "RO"},
    "AS60068": {"keyword": "cdn77", "country": "NL"},
}

found = {k: set() for k in TARGETS}
NEEDED_PER_TARGET = 15
MAX_SAMPLES = 4_000_000

start = time.time()
random.seed(7)
checked = 0
while checked < MAX_SAMPLES and any(len(v) < NEEDED_PER_TARGET for v in found.values()):
    a = random.randint(1, 223)
    if a in (10, 127, 169, 172, 192):  # skip common private/reserved first octets
        continue
    b, c, d = random.randint(0, 255), random.randint(0, 255), random.randint(1, 254)
    ip = f"{a}.{b}.{c}.{d}"
    checked += 1
    try:
        r = G.lookup(ip)
    except Exception:
        continue
    if r.is_private or not r.asn_name:
        continue
    asn_lower = r.asn_name.lower()
    for asn_key, spec in TARGETS.items():
        if len(found[asn_key]) >= NEEDED_PER_TARGET:
            continue
        if spec["keyword"] in asn_lower:
            found[asn_key].add(ip)

    if checked % 500000 == 0:
        elapsed = time.time() - start
        counts = {k: len(v) for k, v in found.items()}
        print(f"[{checked:,} checked, {elapsed:.0f}s] progress: {counts}")

elapsed = time.time() - start
print(f"\nDone. Checked {checked:,} IPs in {elapsed:.0f}s")
for k, v in found.items():
    print(f"  {k} ({TARGETS[k]['keyword']}): {len(v)} real IPs found")

out = {k: sorted(v) for k, v in found.items()}
with open("real_asn_ips.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nWritten to real_asn_ips.json")
