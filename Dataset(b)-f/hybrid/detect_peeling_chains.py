import json
import pandas as pd


INPUT_FILE = "output/transactions.csv"
OUTPUT_FILE = "output/peeling_chain_results.csv"


def parse_list(value):
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return []


def main():

    df = pd.read_csv(INPUT_FILE)

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

    df = df.sort_values("timestamp").reset_index(drop=True)

    # ---------------------------------------------------------
    # Build lookup:
    # address -> transactions where it appears as an input
    # ---------------------------------------------------------

    input_lookup = {}

    for idx, row in df.iterrows():

        for address in row["input_addresses"]:

            input_lookup.setdefault(
                address,
                []
            ).append(idx)

    results = []

    # ---------------------------------------------------------
    # Examine every transaction for peeling behavior
    # ---------------------------------------------------------

    for idx, row in df.iterrows():

        inputs = row["input_addresses"]
        outputs = row["output_addresses"]
        input_amounts = row["input_amounts"]
        output_amounts = row["output_amounts"]

        # Peeling pattern requires one input and two outputs
        if len(inputs) != 1 or len(outputs) != 2:
            continue

        if len(input_amounts) != 1 or len(output_amounts) != 2:
            continue

        input_wallet = inputs[0]
        input_amount = float(input_amounts[0])

        if input_amount <= 0:
            continue

        output1 = float(output_amounts[0])
        output2 = float(output_amounts[1])

        # Identify small output and change output
        if output1 <= output2:
            peel_amount = output1
            change_amount = output2
            change_address = outputs[1]
        else:
            peel_amount = output2
            change_amount = output1
            change_address = outputs[0]

        peel_ratio = peel_amount / input_amount

        # Generator uses approximately 2%-8%
        looks_like_peel = (
            0.01 <= peel_ratio <= 0.15
            and change_amount > peel_amount
        )

        if not looks_like_peel:
            continue

        # -----------------------------------------------------
        # Does the change address become an input later?
        # -----------------------------------------------------

        next_transactions = []

        for next_idx in input_lookup.get(
            change_address,
            []
        ):

            if next_idx <= idx:
                continue

            time_gap = (
                df.loc[next_idx, "timestamp"]
                - row["timestamp"]
            ).total_seconds() / 60

            # Generator uses 5-45 minutes between hops
            if 1 <= time_gap <= 60:
                next_transactions.append(
                    next_idx
                )

        if not next_transactions:
            continue

        # -----------------------------------------------------
        # Follow the chain
        # -----------------------------------------------------

        chain = [idx]
        current_idx = next_transactions[0]

        while True:

            if current_idx in chain:
                break

            chain.append(current_idx)

            current_row = df.loc[current_idx]

            current_outputs = (
                current_row["output_addresses"]
            )

            current_output_amounts = (
                current_row["output_amounts"]
            )

            current_inputs = (
                current_row["input_addresses"]
            )

            current_input_amounts = (
                current_row["input_amounts"]
            )

            if (
                len(current_inputs) != 1
                or len(current_outputs) != 2
                or len(current_input_amounts) != 1
                or len(current_output_amounts) != 2
            ):
                break

            current_input_amount = float(
                current_input_amounts[0]
            )

            if current_input_amount <= 0:
                break

            amounts = [
                float(x)
                for x in current_output_amounts
            ]

            small_index = (
                0 if amounts[0] <= amounts[1]
                else 1
            )

            change_index = 1 - small_index

            ratio = (
                amounts[small_index]
                / current_input_amount
            )

            if not (
                0.01 <= ratio <= 0.15
                and amounts[change_index] > amounts[small_index]
            ):
                break

            next_change_address = (
                current_outputs[change_index]
            )

            candidates = []

            for candidate in input_lookup.get(
                next_change_address,
                []
            ):

                if candidate <= current_idx:
                    continue

                gap = (
                    df.loc[candidate, "timestamp"]
                    - current_row["timestamp"]
                ).total_seconds() / 60

                if 1 <= gap <= 60:
                    candidates.append(candidate)

            if not candidates:
                break

            current_idx = candidates[0]

        # Only call it a chain if there are at least 2 hops
        if len(chain) < 2:
            continue

        results.append({
            "start_transaction": df.loc[
                chain[0], "txid"
            ],
            "input_wallet": input_wallet,
            "chain_length": len(chain),
            "peeling_transactions": len(chain),
            "detected": 1,
            "start_time": df.loc[
                chain[0], "timestamp"
            ],
            "end_time": df.loc[
                chain[-1], "timestamp"
            ],
        })

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"Peeling chains detected: "
        f"{len(results_df)}"
    )

    print(
        f"Results written to: {OUTPUT_FILE}"
    )

    if not results_df.empty:

        print("\nTop detected chains:")

        print(
            results_df
            .sort_values(
                "chain_length",
                ascending=False
            )
            .head(20)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
