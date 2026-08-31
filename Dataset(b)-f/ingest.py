"""
Universal ingestion loader for SIH PS 26146.

PS requirement: "Ingest & parse a bulk metadata dataset ... in CSV/JSON/XML".
Every other script in this pipeline currently hardcodes pd.read_csv() -- if
judges hand you a .json or .xml file, everything breaks. This is the single
entry point that fixes that: one function, format detected by extension,
output always normalized to the SAME shape regardless of source format --
matching what generate_dataset.py's own CSV already produces (input/output
addresses and amounts as real Python lists in the dataframe, not strings).

Verified against generate_dataset.py's actual output formats, not assumed:
  - CSV:  list-fields are JSON-encoded strings, e.g. '["addr1","addr2"]'
  - JSON: list-fields are native JSON arrays already
  - XML:  list-fields are comma-joined into a single text node per element,
          e.g. <output_addresses>addr1,addr2</output_addresses>

Usage:
    from ingest import load_transactions
    df = load_transactions("output/transactions.csv")   # or .json / .xml
"""
import json
import os
import xml.etree.ElementTree as ET

import pandas as pd

LIST_COLS = ["input_addresses", "output_addresses", "input_amounts", "output_amounts"]
FLAT_COLS = ["timestamp", "src_ip", "dst_ip", "src_port", "dst_port", "txid",
             "fee", "script_type", "geo_country", "asn"]
AMOUNT_LIST_COLS = ["input_amounts", "output_amounts"]


def _load_csv(path):
    df = pd.read_csv(path)
    missing = [c for c in LIST_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required list-columns: {missing}")
    for col in LIST_COLS:
        df[col] = df[col].apply(json.loads)
    return df


def _load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON array of transaction objects, got {type(data).__name__}")
    return pd.DataFrame(data)


def _load_xml(path):
    tree = ET.parse(path)
    root = tree.getroot()
    rows = []
    for txn_el in root.findall("transaction"):
        row = {}
        for child in txn_el:
            tag, text = child.tag, (child.text or "")
            if tag in LIST_COLS:
                parts = [p for p in text.split(",") if p != ""]
                if tag in AMOUNT_LIST_COLS:
                    parts = [float(p) for p in parts]
                row[tag] = parts
            else:
                row[tag] = text
        rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no <transaction> elements found -- check the XML structure")
    return pd.DataFrame(rows)


def _normalize_dtypes(df):
    """Bring all three formats to the same dtypes regardless of source quirks
    (e.g. JSON's dst_port arriving as a float, XML's numeric fields arriving
    as plain text)."""
    for col in ["src_port", "dst_port"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    if "fee" in df.columns:
        df["fee"] = pd.to_numeric(df["fee"], errors="coerce")
    return df


def load_transactions(path):
    """Detects format by extension (.csv / .json / .xml), parses it, and
    returns a dataframe with input_addresses/output_addresses/input_amounts/
    output_amounts as real Python lists and consistent dtypes -- identical
    shape no matter which format the file arrived in.

    Raises a clear error (not a silent empty dataframe) on missing files,
    unsupported extensions, or malformed content.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    ext = os.path.splitext(path)[1].lower()

    try:
        if ext == ".csv":
            df = _load_csv(path)
        elif ext == ".json":
            df = _load_json(path)
        elif ext == ".xml":
            df = _load_xml(path)
        else:
            raise ValueError(
                f"Unsupported file extension '{ext}' for {path} -- "
                f"expected .csv, .json, or .xml (per PS ingestion requirement)"
            )
    except (json.JSONDecodeError, ET.ParseError) as e:
        raise ValueError(f"Failed to parse {path} as {ext}: {e}") from e

    missing = [c for c in FLAT_COLS + LIST_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns after parsing: {missing}")

    df = _normalize_dtypes(df)
    return df


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "output/transactions.csv"
    df = load_transactions(path)
    print(f"Loaded {len(df)} rows from {path}")
    print(df.dtypes)
    print(df.iloc[0])