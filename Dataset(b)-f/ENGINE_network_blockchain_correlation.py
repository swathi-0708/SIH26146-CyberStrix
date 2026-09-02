"""
Temporal + IP<->Wallet Correlation Feature Engine
==================================================

For every incoming transaction, computes:

  Temporal correlation (per IP):
    - count_1m, count_5m, count_1h   : rolling tx counts in each window
    - time_since_prev_tx_ip          : seconds since this IP's last tx
    - is_burst                       : bool, count_1m >= BURST_THRESHOLD
    - first_seen_ip, last_seen_ip    : epoch seconds

  IP <-> wallet correlation:
    - wallets_per_ip        : distinct wallets ever seen from this IP
    - tx_count_per_ip       : all-time tx count from this IP
    - ip_wallet_pair_count  : times THIS (ip, wallet) pair has occurred
    - ips_per_wallet        : distinct IPs ever seen for this wallet
    - ip_churn              : new-wallet ratio for this IP in the window
    - wallet_churn          : new-IP ratio for this wallet in the window

Design notes
------------
- State is abstracted behind a `Store` interface so you can back it with
  Redis in production (sorted sets give you windowed counts + first/last
  seen for free) or with the in-memory implementation below for local
  dev/tests.
- "Churn" is not a standardized term, so a concrete definition is used:
  the fraction of an entity's associated-entity events in the trailing
  window that are *first-time* pairings. Tune WINDOW_CHURN_SECONDS and
  the formula to match your fraud model.
- All timestamps are epoch seconds (float). Swap in datetime as needed.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Optional


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

WINDOW_1M = 60
WINDOW_5M = 5 * 60
WINDOW_1H = 60 * 60

BURST_THRESHOLD = 5          # count_1m >= this => is_burst
CHURN_WINDOW_SECONDS = WINDOW_1H  # window used for ip_churn / wallet_churn


# --------------------------------------------------------------------------
# Storage abstraction
# --------------------------------------------------------------------------

class Store(ABC):
    """State backend. Implement this over Redis for production."""

    @abstractmethod
    def record_ip_tx(self, ip: str, ts: float) -> None: ...

    @abstractmethod
    def count_ip_tx_since(self, ip: str, since_ts: float) -> int: ...

    @abstractmethod
    def first_last_seen_ip(self, ip: str) -> tuple[Optional[float], Optional[float]]: ...

    @abstractmethod
    def prev_tx_ts_ip(self, ip: str) -> Optional[float]: ...

    @abstractmethod
    def record_ip_wallet(self, ip: str, wallet: str, ts: float) -> None: ...

    @abstractmethod
    def wallets_for_ip(self, ip: str) -> set[str]: ...

    @abstractmethod
    def ips_for_wallet(self, wallet: str) -> set[str]: ...

    @abstractmethod
    def pair_count(self, ip: str, wallet: str) -> int: ...

    @abstractmethod
    def ip_tx_total(self, ip: str) -> int: ...

    @abstractmethod
    def wallet_first_seen_for_ip(self, ip: str, wallet: str) -> Optional[float]: ...

    @abstractmethod
    def ip_first_seen_for_wallet(self, wallet: str, ip: str) -> Optional[float]: ...


class InMemoryStore(Store):
    """
    Simple in-memory implementation, good for local dev / unit tests.
    Not safe for multi-process deployments -- port the same logic to
    Redis sorted sets / hashes for production (see notes at bottom).
    """

    def __init__(self) -> None:
        self._ip_tx_times: dict[str, list[float]] = defaultdict(list)
        self._ip_wallet_first_seen: dict[tuple[str, str], float] = {}
        self._wallet_ip_first_seen: dict[tuple[str, str], float] = {}
        self._pair_counts: dict[tuple[str, str], int] = defaultdict(int)
        self._ip_wallets: dict[str, dict[str, float]] = defaultdict(dict)   # ip -> {wallet: last_seen}
        self._wallet_ips: dict[str, dict[str, float]] = defaultdict(dict)   # wallet -> {ip: last_seen}

    def record_ip_tx(self, ip: str, ts: float) -> None:
        self._ip_tx_times[ip].append(ts)

    def count_ip_tx_since(self, ip: str, since_ts: float) -> int:
        times = self._ip_tx_times[ip]
        idx = bisect_left(times, since_ts)
        return len(times) - idx

    def first_last_seen_ip(self, ip: str) -> tuple[Optional[float], Optional[float]]:
        times = self._ip_tx_times[ip]
        if not times:
            return None, None
        return times[0], times[-1]

    def prev_tx_ts_ip(self, ip: str) -> Optional[float]:
        times = self._ip_tx_times[ip]
        # second-to-last, since the current tx has already been recorded
        # by the time this is called in the pipeline below -- caller
        # handles ordering; here we just expose "the one before the last".
        if len(times) < 2:
            return None
        return times[-2]

    def record_ip_wallet(self, ip: str, wallet: str, ts: float) -> None:
        if (ip, wallet) not in self._ip_wallet_first_seen:
            self._ip_wallet_first_seen[(ip, wallet)] = ts
        if (wallet, ip) not in self._wallet_ip_first_seen:
            self._wallet_ip_first_seen[(wallet, ip)] = ts
        self._pair_counts[(ip, wallet)] += 1
        self._ip_wallets[ip][wallet] = ts
        self._wallet_ips[wallet][ip] = ts

    def wallets_for_ip(self, ip: str) -> set[str]:
        return set(self._ip_wallets[ip].keys())

    def ips_for_wallet(self, wallet: str) -> set[str]:
        return set(self._wallet_ips[wallet].keys())

    def pair_count(self, ip: str, wallet: str) -> int:
        return self._pair_counts[(ip, wallet)]

    def ip_tx_total(self, ip: str) -> int:
        return len(self._ip_tx_times[ip])

    def wallet_first_seen_for_ip(self, ip: str, wallet: str) -> Optional[float]:
        return self._ip_wallet_first_seen.get((ip, wallet))

    def ip_first_seen_for_wallet(self, wallet: str, ip: str) -> Optional[float]:
        return self._wallet_ip_first_seen.get((wallet, ip))


# --------------------------------------------------------------------------
# Feature record
# --------------------------------------------------------------------------

@dataclass
class TransactionFeatures:
    tx_id: str
    ip: str
    wallet: str
    ts: float

    # temporal
    count_1m: int = 0
    count_5m: int = 0
    count_1h: int = 0
    time_since_prev_tx_ip: Optional[float] = None
    is_burst: bool = False
    first_seen_ip: Optional[float] = None
    last_seen_ip: Optional[float] = None

    # ip <-> wallet correlation
    wallets_per_ip: int = 0
    tx_count_per_ip: int = 0
    ip_wallet_pair_count: int = 0
    ips_per_wallet: int = 0
    ip_churn: float = 0.0
    wallet_churn: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------

class FeatureEngine:
    def __init__(self, store: Optional[Store] = None):
        self.store = store or InMemoryStore()

    def process(self, tx_id: str, ip: str, wallet: str, ts: Optional[float] = None) -> TransactionFeatures:
        """
        Call this once per incoming transaction, in arrival order.
        Updates internal state AND returns the computed feature row.
        """
        ts = ts if ts is not None else time.time()

        # --- record raw events first, so counts include the current tx ---
        prev_ts = self.store.prev_tx_ts_ip(ip)  # "before" this tx is recorded, per InMemoryStore semantics
        self.store.record_ip_tx(ip, ts)
        self.store.record_ip_wallet(ip, wallet, ts)

        # --- temporal correlation ---
        count_1m = self.store.count_ip_tx_since(ip, ts - WINDOW_1M)
        count_5m = self.store.count_ip_tx_since(ip, ts - WINDOW_5M)
        count_1h = self.store.count_ip_tx_since(ip, ts - WINDOW_1H)
        first_seen, last_seen = self.store.first_last_seen_ip(ip)

        time_since_prev = (ts - prev_ts) if prev_ts is not None else None
        is_burst = count_1m >= BURST_THRESHOLD

        # --- ip <-> wallet correlation ---
        wallets_for_ip = self.store.wallets_for_ip(ip)
        ips_for_wallet = self.store.ips_for_wallet(wallet)
        pair_count = self.store.pair_count(ip, wallet)
        tx_count_per_ip = self.store.ip_tx_total(ip)

        ip_churn = self._churn_ratio(
            entity_new_at=self.store.wallet_first_seen_for_ip,
            entity_key=ip,
            associated_set=wallets_for_ip,
            now=ts,
        )
        wallet_churn = self._churn_ratio(
            entity_new_at=self.store.ip_first_seen_for_wallet,
            entity_key=wallet,
            associated_set=ips_for_wallet,
            now=ts,
        )

        return TransactionFeatures(
            tx_id=tx_id, ip=ip, wallet=wallet, ts=ts,
            count_1m=count_1m, count_5m=count_5m, count_1h=count_1h,
            time_since_prev_tx_ip=time_since_prev,
            is_burst=is_burst,
            first_seen_ip=first_seen, last_seen_ip=last_seen,
            wallets_per_ip=len(wallets_for_ip),
            tx_count_per_ip=tx_count_per_ip,
            ip_wallet_pair_count=pair_count,
            ips_per_wallet=len(ips_for_wallet),
            ip_churn=ip_churn,
            wallet_churn=wallet_churn,
        )

    def _churn_ratio(self, entity_new_at, entity_key: str, associated_set: set[str], now: float) -> float:
        """
        Fraction of `entity_key`'s associated entities whose FIRST pairing
        with it happened inside the trailing CHURN_WINDOW_SECONDS window.
        0.0  => nothing new lately (stable association)
        1.0  => everything associated with this entity is brand new
        """
        if not associated_set:
            return 0.0
        new_count = 0
        for other in associated_set:
            first_seen_ts = entity_new_at(entity_key, other)
            if first_seen_ts is not None and (now - first_seen_ts) <= CHURN_WINDOW_SECONDS:
                new_count += 1
        return round(new_count / len(associated_set), 4)


# --------------------------------------------------------------------------
# Example usage
# --------------------------------------------------------------------------

if __name__ == "__main__":
    engine = FeatureEngine()

    demo_txs = [
        ("tx1", "10.0.0.1", "walletA", 1000.0),
        ("tx2", "10.0.0.1", "walletB", 1005.0),   # same IP, new wallet -> burst-ish, churn signal
        ("tx3", "10.0.0.1", "walletC", 1008.0),
        ("tx4", "10.0.0.2", "walletA", 1010.0),   # walletA now seen from a 2nd IP
        ("tx5", "10.0.0.1", "walletA", 4000.0),   # much later, repeat pair
    ]

    for tx_id, ip, wallet, ts in demo_txs:
        feats = engine.process(tx_id, ip, wallet, ts)
        print(feats.to_dict())
