import json
import pandas as pd


INPUT_FILE = "output/transactions.csv"
WALLETS_FILE = "output/wallets_reference.csv"
OUTPUT_FILE = "output/profile_deviation_results.csv"


def parse_list(value):
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return []


def main():

    # ---------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------

    df = pd.read_csv(INPUT_FILE)
    wallets_df = pd.read_csv(WALLETS_FILE)

    for col in [
        "input_addresses",
        "output_addresses",
        "input_amounts",
        "output_amounts"
    ]:
        df[col] = df[col].apply(parse_list)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        format="mixed",
        utc=True
    )

    # ---------------------------------------------------------
    # 2. Build wallet -> typical amount mapping
    # ---------------------------------------------------------

    typical_amounts = dict(
        zip(
            wallets_df["address"],
            wallets_df["typical_amount_btc"]
        )
    )

    # ---------------------------------------------------------
    # 3. Analyze transactions involving persistent wallets
    # ---------------------------------------------------------

    persistent_wallets = set(
        wallets_df["address"]
    )

        # Track counterparties previously seen by each wallet
    seen_counterparties = {
        wallet: set()
        for wallet in persistent_wallets
    }

    results = []

    for idx, row in df.iterrows():

        inputs = row["input_addresses"]
        outputs = row["output_addresses"]
        input_amounts = row["input_amounts"]
        output_amounts = row["output_amounts"]

        if not inputs:
            continue

        # Check each persistent wallet involved as an input
        for wallet in inputs:

            if wallet not in persistent_wallets:
                continue

            wallet_index = inputs.index(wallet)

            if wallet_index >= len(input_amounts):
                continue

            amount = float(
                input_amounts[wallet_index]
            )

            typical = float(
                typical_amounts.get(wallet, 0)
            )

            if typical <= 0:
                continue

            deviation_ratio = (
                amount / typical
            )

            # -------------------------------------------------
            # Identify counterparties
            # -------------------------------------------------

            counterparties = [
                address
                for address in outputs
                if address != wallet
            ]

                        # Generator uses approximately 3x-10x
            # the wallet's typical transaction amount.
            looks_like_deviation = (
                3.0 <= deviation_ratio <= 10.0
            )

            # Check whether at least one output counterparty
            # is new for this wallet.
            new_counterparties = [
                address
                for address in counterparties
                if address not in seen_counterparties[wallet]
            ]

            has_new_counterparty = (
                len(new_counterparties) > 0
            )

            # Profile deviation requires BOTH signals.
            if not (
                looks_like_deviation
                and has_new_counterparty
            ):
                # Still remember the counterparties because
                # they have now been observed.
                seen_counterparties[wallet].update(
                    counterparties
                )
                continue

            results.append({
                "transaction_index": idx,
                "txid": row["txid"],
                "wallet": wallet,
                "transaction_amount": amount,
                "typical_amount": typical,
                "deviation_ratio": deviation_ratio,
                "counterparty_count": len(counterparties),
                "timestamp": row["timestamp"],
                "detected": 1
            })

            # Remember these counterparties for future transactions.
            seen_counterparties[wallet].update(
                counterparties
            )

    # ---------------------------------------------------------
    # 4. Save results
    # ---------------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"Profile-deviation transactions detected: "
        f"{len(results_df)}"
    )

    print(
        f"Unique wallets detected: "
        f"{results_df['wallet'].nunique() if not results_df.empty else 0}"
    )

    print(
        f"Results written to: {OUTPUT_FILE}"
    )

    if not results_df.empty:

        print(
            "\nTop deviations:"
        )

        print(
            results_df
            .sort_values(
                "deviation_ratio",
                ascending=False
            )
            .head(20)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
