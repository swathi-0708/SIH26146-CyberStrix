"""
GeoIP enrichment for SIH PS 26146.

PS requirement: "integrate open source downloadable Geo IP database" -- this
resolves src_ip (and optionally dst_ip) to country/ASN via an actual offline
lookup, independent of the dataset's own baked-in geo_country/asn columns
(which are synthetic ground-truth labels the generator used to shape wallet
behavior, not derived from the IP itself).

Backend: geoip2fast (pip install geoip2fast). Its .dat.gz database ships
INSIDE the pip package -- one `pip install geoip2fast` while online gets you
everything; after that, lookups are 100% offline, no network calls, no
separate database download step. Satisfies the offline-Linux requirement
cleanly.

Note on fields: geoip2fast gives country_code (2-letter, e.g. "US") and
asn_name (an organization name string, e.g. "GOOGLE", "CLOUDFLARENET") --
NOT a numeric "AS1234"-style ASN code. That's a real, independently-resolved
value, just a different shape than the dataset's own synthetic `asn` column
("AS7922" style). Don't conflate the two -- they answer different questions
(this module: "what does the IP actually resolve to", dataset's own column:
"what ASN did the generator intend for this wallet profile").
"""

import pandas as pd
from geoip2fast import GeoIP2Fast

DEFAULT_DATA_FILE = "geoip2fast-asn.dat.gz"


class GeoEnricher:
    def __init__(self, data_file=DEFAULT_DATA_FILE):
        self._geo = GeoIP2Fast(geoip2fast_data_file=data_file)
        self._cache = {}  # ip -> (country_code, asn_name) -- most datasets
        # reuse the same small pool of IPs across many rows, so caching avoids
        # redundant lookups. geoip2fast is already fast (~0.00003s/lookup per
        # its own docs), but this makes repeated IPs effectively free.

    def _resolve_one(self, ip):
        if ip in self._cache:
            return self._cache[ip]
        try:
            r = self._geo.lookup(ip)
            if r.is_private:
                result = ("PRIVATE", "PRIVATE")
            else:
                result = (r.country_code or "UNRESOLVED", r.asn_name or "UNRESOLVED")
        except Exception:
            result = ("UNRESOLVED", "UNRESOLVED")
        self._cache[ip] = result
        return result

    def enrich_dataframe(self, df, ip_col, country_col, asn_col):
        """Adds two new columns to df: country_col and asn_col, resolved from
        ip_col via offline GeoIP lookup. Returns the same df (mutated + returned,
        matching how train_anomaly_model.py already calls this)."""
        resolved = df[ip_col].apply(self._resolve_one)
        df[country_col] = resolved.apply(lambda x: x[0])
        df[asn_col] = resolved.apply(lambda x: x[1])
        return df


if __name__ == "__main__":
    # quick self-test / demo
    import sys

    test_df = pd.DataFrame(
        {"src_ip": ["8.8.8.8", "1.1.1.1", "192.168.1.1", "not-an-ip"]}
    )
    geo = GeoEnricher()
    out = geo.enrich_dataframe(
        test_df, "src_ip", "geo_country_resolved", "asn_resolved"
    )
    print(out)
