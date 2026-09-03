"""
Single entrypoint for the whole Dataset(b)-f pipeline. Run this instead of
running generate_dataset.py / split_dataset.py / check.py / train_compare.py /
train_isolation_forest.py / build_alerts.py one at a time by hand.

Usage:
    python3 run_pipeline.py                    # full run, default CSV
    python3 run_pipeline.py --input file.json  # ingest a specific file (any of CSV/JSON/XML)
    python3 run_pipeline.py --skip-generate     # reuse existing output/transactions.* instead of regenerating
    python3 run_pipeline.py --skip-check        # skip the leakage audit (NOT recommended -- see note below)

Stops immediately on the first stage that fails -- won't silently continue
training on bad data if generation or the leakage check breaks. Exit code is
non-zero if any stage fails, so this is also safe to wire into a CI step or
a pre-demo smoke test.

Why check.py isn't skipped by default: check.py doesn't fail the pipeline
even when it finds SUSPECT features -- it just prints its verdict. Skipping
it entirely means you'd never see that warning at all. --skip-check exists
for fast dev iteration only; don't use it before a real run you plan to trust.
"""

import argparse
import subprocess
import sys
import time

STAGES = [
    ("generate_dataset.py", [], "generate"),
    ("split_dataset.py", [], "split"),
    ("check.py", [], "check"),
    ("train_compare.py", [], "train"),
    ("train_isolation_forest.py", [], "train"),
    ("build_alerts.py", [], "alerts"),
    ("entity_graph.py", [], "graph"),
    ("explain_alerts.py", [], "explain"),
]


def run_stage(script, extra_args, label):
    print(f"\n{'=' * 60}")
    print(f"  {script}")
    print("=" * 60)
    t0 = time.time()
    result = subprocess.run([sys.executable, script] + extra_args)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(
            f"\n!!! {script} FAILED (exit code {result.returncode}, {elapsed:.1f}s) -- stopping pipeline."
        )
        sys.exit(result.returncode)
    print(f"--- {script} done in {elapsed:.1f}s ---")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=str,
        default=None,
        help="Transactions file to ingest (.csv/.json/.xml). Passed through to split_dataset.py.",
    )
    ap.add_argument(
        "--skip-generate",
        action="store_true",
        help="Reuse existing output/transactions.* instead of regenerating.",
    )
    ap.add_argument(
        "--skip-check",
        action="store_true",
        help="Skip the leakage audit. Dev/iteration use only, not recommended before a real run.",
    )
    args = ap.parse_args()

    t_start = time.time()

    for script, extra, label in STAGES:
        if label == "generate" and args.skip_generate:
            print(f"\n(skipping {script} -- reusing existing output/)")
            continue
        if label == "check" and args.skip_check:
            print(f"\n(skipping {script} -- leakage audit NOT run this pass)")
            continue
        stage_args = list(extra)
        if label == "split" and args.input:
            stage_args += ["--input", args.input]
        run_stage(script, stage_args, label)

    total = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"  Pipeline complete in {total:.1f}s")
    print(f"  Alerts: output/alerts.csv (demo, no ground truth)")
    print(f"          output/alerts_eval.csv (dev, with ground truth -- not for demo)")
    print("=" * 60)


if __name__ == "__main__":
    main()
