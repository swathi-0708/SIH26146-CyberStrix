# CyberStrix -- working pipeline (dataset + detection + graph + explainability)

This is the transaction-level XGBoost + Isolation Forest pipeline, hybrid
layer removed. Everything here is verified working end to end.

## Setup

```bash
pip install -r requirements.txt --break-system-packages   # or use a venv
mkdir -p output
```

## Run order

```bash
python3 generate_dataset.py --n-wallets 800 --n-normal-tx 6000 --seed 42
python3 split_dataset.py            # feature engineering + leak-safe train/test split
python3 check.py                    # AUC leak scan on train.csv -- confirm no SUSPECT features before continuing
python3 train_compare.py            # LogisticRegression / RandomForest / XGBoost comparison
python3 train_isolation_forest.py   # blind unsupervised secondary detector
python3 build_alerts.py             # merge XGBoost + Isolation Forest into tiered output/alerts.csv
python3 entity_graph.py             # wallet + disposable-address graph, pyvis + GraphML export
python3 explain_alerts.py           # SHAP explainability -> output/alerts_explained.csv
```

Each script reads/writes `output/`, so run them from the directory containing
that folder (or adjust paths). `split_dataset.py` accepts `--input <path>` if
you're testing against a `.json`/`.xml` transactions file instead of the
default `output/transactions.csv`.

## What changed vs. the previous version

`entity_graph.py` was fixed: it used to only create graph nodes for
addresses that matched a `wallets_reference.csv` row (the 800 profile
wallets), silently dropping every disposable/throwaway address a
transaction touched otherwise. Since peeling chains are made almost
entirely of disposable addresses after hop 1, **100% of peeling_chain
transactions (35/35) were being dropped from the graph** -- verified before
and after the fix. Now every address gets a node (profile wallets keep
their full metadata; disposable addresses get `is_disposable=True` and
metadata pulled from the transaction itself). Verified after the fix:
35/35 peeling_chain transactions now appear as edges, 0 transactions are
skipped (was 313), and the risk subgraph correctly shows the hop trail
leading out of a flagged source wallet. See the comments in
`entity_graph.py`'s module docstring and `build_wallet_graph()` for the
full explanation.

Nothing else changed -- `generate_dataset.py`, `ingest.py`,
`geoip_enrich.py`, `split_dataset.py`, `check.py`, `train_compare.py`,
`train_isolation_forest.py`, `build_alerts.py`, and `explain_alerts.py` are
all unmodified from what was already verified working.

## Deliberately excluded from this package (for now)

- **`find_real_asn_ips.py`** -- mislabeled, contains an old draft of the
  split script rather than an ASN-IP finder. Don't run it. `split_dataset.py`
  already does everything this script's name implies via `geoip_enrich.py`.
- **The wallet-level hybrid pipeline** (`feature_extractor.py`,
  `transaction_anomaly_detector.py`, `aggregate_transaction_anomalies.py`,
  `detect_peeling_chains.py`, `detect_profile_deviations.py`,
  `train_xgboost.py`, `hybrid_risk_engine.py` / `evaluate_hybrid.py`,
  `stress_test_xgboost.py`, `train_hybrid.py`) -- by team decision, parked
  for now. It's a separate, parallel system that nothing in the run order
  above reads, so leaving it out doesn't affect any of these 8 scripts. Two
  real issues need fixing before it's demo-ready: (1) `detect_peeling_chains.py`
  and `detect_profile_deviations.py`'s thresholds are literal copies of
  `generate_dataset.py`'s own injection parameters rather than organic
  detection logic; (2) `hybrid_risk_engine.py` evaluates the wallet-level
  XGBoost model on all 800 wallets including the ~600 it was trained/
  validated on, instead of a held-out test set. Both inflate the apparent
  100%/100% precision/recall in `hybrid_risk_results.csv`.
- **`real_asn_ips.json`** -- not currently read by any script in this
  pipeline (`generate_dataset.py` doesn't load external files at all). Kept
  aside as a resource in case it gets wired in later; no need to ship it
  with a run that doesn't use it.

## Still open (not done in this pass, per team split)

- **Structuring anomaly detection** -- recall stuck at ~10-13% vs 85%+ for
  the other five types; the injection signal (near-threshold clustering)
  needs to be recovered in the feature set or model.
- **Network-layer work beyond `src_ip`** -- `dst_ip`/`dst_port` are ingested
  but unused in features or the graph.
- **Investigation queries** (Phase 5, spec undefined).
- **Streamlit dashboard** (Phase 10).
- **Offline-Linux packaging** -- pin `requirements.txt` from a clean venv,
  test with network disabled.
