import json
import os
import pandas as pd


INPUT_FILE = "output/transactions.csv"
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
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Store all wallets we encounter
    wallets = set()

    for _, row in df.iterrows():
        wallets.update(row["input_addresses"])
        wallets.update(row["output_addresses"])

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
        fan_in = len(counterparties) if incoming_count > 0 else 0
        fan_out = len(counterparties) if outgoing_count > 0 else 0

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
