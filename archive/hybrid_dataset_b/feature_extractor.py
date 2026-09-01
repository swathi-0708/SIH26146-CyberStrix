import json
import os
import pandas as pd


INPUT_FILE = "output/transactions.csv"
WALLETS_FILE = "output/wallets_reference.csv"
OUTPUT_FILE = "output/ml_features.csv"

def parse_list(value):
    """Convert a JSON-encoded list from the CSV into a Python list."""
    if isinstance(value, list):
        return value

    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return []


def main():
    # Load raw transaction data
    df = pd.read_csv(INPUT_FILE)
    wallets_df = pd.read_csv(WALLETS_FILE)
    persistent_wallets = set(wallets_df["address"])

    # Parse list columns
    list_columns = [
        "input_addresses",
        "output_addresses",
        "input_amounts",
        "output_amounts",
    ]

    for col in list_columns:
        df[col] = df[col].apply(parse_list)

    # Convert timestamps
    df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    format="mixed",
    utc=True
)

    # Store all wallets we encounter
    wallets = persistent_wallets

    features = []

    for wallet in wallets:

        # Transactions involving this wallet
        involved = df[
            df["input_addresses"].apply(lambda x: wallet in x)
            | df["output_addresses"].apply(lambda x: wallet in x)
        ].copy()

        incoming = involved[
            involved["output_addresses"].apply(lambda x: wallet in x)
        ]

        outgoing = involved[
            involved["input_addresses"].apply(lambda x: wallet in x)
        ]

        # Basic transaction counts
        transaction_count = len(involved)
        incoming_count = len(incoming)
        outgoing_count = len(outgoing)

        # Amounts
        total_incoming = 0.0
        total_outgoing = 0.0

        for _, row in incoming.iterrows():
            for address, amount in zip(
                row["output_addresses"],
                row["output_amounts"]
            ):
                if address == wallet:
                    total_incoming += float(amount)

        for _, row in outgoing.iterrows():
            for address, amount in zip(
                row["input_addresses"],
                row["input_amounts"]
            ):
                if address == wallet:
                    total_outgoing += float(amount)

                # Small outgoing transaction ratio
        # Measures how often outgoing transfers are much smaller
        # than the wallet's typical outgoing transaction.

        outgoing_amounts = []

        for _, row in outgoing.iterrows():
            for address, amount in zip(
                row["input_addresses"],
                row["input_amounts"]
            ):
                if address == wallet:
                    outgoing_amounts.append(float(amount))

        if outgoing_amounts:
            median_outgoing = pd.Series(
                outgoing_amounts
            ).median()

            small_outgoing_count = sum(
                1
                for amount in outgoing_amounts
                if amount < 0.5 * median_outgoing
            )

            outgoing_small_amount_ratio = (
                small_outgoing_count / len(outgoing_amounts)
                if median_outgoing > 0
                else 0.0
            )
        else:
            outgoing_small_amount_ratio = 0.0

        # Peeling-chain behavior:
        # Measures how consistently the wallet sends a large
        # portion of its outgoing value onward in repeated transfers.

        peeling_ratios = []

        if len(outgoing_amounts) >= 2:

            sorted_outgoing = outgoing.sort_values("timestamp")

            previous_amount = None

            for _, row in sorted_outgoing.iterrows():

                wallet_amount = 0.0

                for address, amount in zip(
                    row["input_addresses"],
                    row["input_amounts"]
                ):
                    if address == wallet:
                        wallet_amount = float(amount)
                        break

                if previous_amount is not None and previous_amount > 0:
                    ratio = wallet_amount / previous_amount
                    peeling_ratios.append(ratio)

                previous_amount = wallet_amount

        peeling_chain_consistency = (
            sum(
                1 for ratio in peeling_ratios
                if 0.5 <= ratio <= 1.0
            ) / len(peeling_ratios)
            if peeling_ratios
            else 0.0
        )

        # Average transaction amount
        all_amounts = []

        for _, row in involved.iterrows():

            if wallet in row["input_addresses"]:
                for address, amount in zip(
                    row["input_addresses"],
                    row["input_amounts"]
                ):
                    if address == wallet:
                        all_amounts.append(float(amount))

            if wallet in row["output_addresses"]:
                for address, amount in zip(
                    row["output_addresses"],
                    row["output_amounts"]
                ):
                    if address == wallet:
                        all_amounts.append(float(amount))

        avg_transaction_amount = (
            sum(all_amounts) / len(all_amounts)
            if all_amounts
            else 0.0
        )

        # Unique counterparties
        counterparties = set()

        for _, row in involved.iterrows():

            if wallet in row["input_addresses"]:
                for address in row["output_addresses"]:
                    if address != wallet:
                        counterparties.add(address)

            if wallet in row["output_addresses"]:
                for address in row["input_addresses"]:
                    if address != wallet:
                        counterparties.add(address)

        # IP and ASN diversity
        unique_ips = set()
        unique_asns = set()

        for _, row in involved.iterrows():
            unique_ips.add(row["src_ip"])
            unique_ips.add(row["dst_ip"])
            unique_asns.add(row["asn"])

        # Fan-in / fan-out
        # Fan-in: unique wallets that send money TO this wallet
        incoming_counterparties = set()

        for _, row in incoming.iterrows():
            for address in row["input_addresses"]:
                if address != wallet:
                    incoming_counterparties.add(address)

        # Fan-out: unique wallets that receive money FROM this wallet
        outgoing_counterparties = set()

        for _, row in outgoing.iterrows():
            for address in row["output_addresses"]:
                if address != wallet:
                    outgoing_counterparties.add(address)

        fan_in = len(incoming_counterparties)
        fan_out = len(outgoing_counterparties)

        # Time gaps
        times = sorted(involved["timestamp"].tolist())

        gaps = []

        for i in range(1, len(times)):
            gap = (
                times[i] - times[i - 1]
            ).total_seconds() / 60.0

            gaps.append(gap)

        avg_time_gap = (
            sum(gaps) / len(gaps)
            if gaps
            else 0.0
        )

        # Rapid transfer ratio
        rapid_count = sum(1 for gap in gaps if gap <= 5)

        rapid_transfer_ratio = (
            rapid_count / len(gaps)
            if gaps
            else 0.0
        )

                # Burst intensity:
        # Maximum number of transactions occurring within a 5-minute window.

        sorted_times = sorted(involved["timestamp"].tolist())

        max_transactions_in_window = 0
        window_minutes = 5

        for i in range(len(sorted_times)):
            window_end = sorted_times[i] + pd.Timedelta(
                minutes=window_minutes
            )

            count = sum(
                1 for t in sorted_times
                if sorted_times[i] <= t <= window_end
            )

            max_transactions_in_window = max(
                max_transactions_in_window,
                count
            )

        # ASN hopping behavior:
        # Measures how frequently the source ASN changes
        # between consecutive transactions.

        sorted_involved = involved.sort_values("timestamp")

        asns = sorted_involved["asn"].tolist()

        asn_changes = 0

        for i in range(1, len(asns)):
            if asns[i] != asns[i - 1]:
                asn_changes += 1

        asn_change_ratio = (
            asn_changes / (len(asns) - 1)
            if len(asns) > 1
            else 0.0
        )
        # Split behavior:
        # How often this wallet sends transactions with multiple outputs.
        split_transactions = 0

        for _, row in outgoing.iterrows():
            if len(row["output_addresses"]) > 1:
                split_transactions += 1

        split_ratio = (
            split_transactions / outgoing_count
            if outgoing_count > 0
            else 0.0
        )

        # Merge behavior:
        # How often this wallet receives transactions with multiple inputs.
        merge_transactions = 0

        for _, row in incoming.iterrows():
            if len(row["input_addresses"]) > 1:
                merge_transactions += 1

        merge_ratio = (
            merge_transactions / incoming_count
            if incoming_count > 0
            else 0.0
        )

                # Profile deviation:
        # Compare the wallet's largest transaction to its typical amount.
        wallet_info = wallets_df[
            wallets_df["address"] == wallet
        ]

        if not wallet_info.empty:
            typical_amount = float(
                wallet_info.iloc[0]["typical_amount_btc"]
            )
        else:
            typical_amount = 0.0

        max_amount_ratio = 0.0

        if typical_amount > 0 and all_amounts:
            max_amount_ratio = max(all_amounts) / typical_amount
        
        features.append({
            "wallet": wallet,
            "transaction_count": transaction_count,
            "incoming_count": incoming_count,
            "outgoing_count": outgoing_count,
            "total_incoming_btc": total_incoming,
            "total_outgoing_btc": total_outgoing,
            "avg_transaction_amount": avg_transaction_amount,
            "avg_time_gap_minutes": avg_time_gap,
            "unique_counterparties": len(counterparties),
            "fan_in": fan_in,
            "fan_out": fan_out,
            "unique_ips": len(unique_ips),
            "unique_asns": len(unique_asns),
            "rapid_transfer_ratio": rapid_transfer_ratio,
            "split_ratio": split_ratio,
            "merge_ratio": merge_ratio,
            "max_amount_ratio": max_amount_ratio,
            "outgoing_small_amount_ratio": outgoing_small_amount_ratio,
            "asn_change_ratio": asn_change_ratio,
            "max_transactions_in_5min": max_transactions_in_window,
            "peeling_chain_consistency": peeling_chain_consistency,
        })

    features_df = pd.DataFrame(features)

    os.makedirs("output", exist_ok=True)

    features_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Created: {OUTPUT_FILE}")
    print(f"Wallets processed: {len(features_df)}")
    print("\nFeature columns:")
    print(list(features_df.columns))


if __name__ == "__main__":
    main()
