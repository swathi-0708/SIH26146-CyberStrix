import pandas as pd
import json

# Load raw Dataset A
tx = pd.read_csv('output/transactions.csv')

# Parse list-columns (JSON-array-as-string format)
tx['input_list'] = tx['input_addresses'].apply(json.loads)
tx['output_list'] = tx['output_addresses'].apply(json.loads)
tx['input_amounts_list'] = tx['input_amounts'].apply(json.loads)
tx['output_amounts_list'] = tx['output_amounts'].apply(json.loads)

# Structural features
tx['n_inputs'] = tx['input_list'].apply(len)
tx['n_outputs'] = tx['output_list'].apply(len)
tx['n_unique_input_addresses'] = tx['input_list'].apply(lambda x: len(set(x)))
tx['n_unique_output_addresses'] = tx['output_list'].apply(lambda x: len(set(x)))

# Volume features
tx['total_input_btc'] = tx['input_amounts_list'].apply(sum)
tx['total_output_btc'] = tx['output_amounts_list'].apply(sum)

# Ratio, protected against division by zero
tx['input_output_ratio'] = tx.apply(
    lambda row: row['total_input_btc'] / row['total_output_btc']
    if row['total_output_btc'] != 0 else 0,
    axis=1
)

# Weak structural signal, not a hard rule
tx['fan_in_5plus'] = (tx['n_inputs'] >= 5).astype(int)

# Final feature table
feature_cols = [
    'txid',
    'n_inputs',
    'n_outputs',
    'n_unique_input_addresses',
    'n_unique_output_addresses',
    'total_input_btc',
    'total_output_btc',
    'input_output_ratio',
    'fan_in_5plus',
    'fee',
]

tx_featured = tx[feature_cols]

# Save to new file, original untouched
tx_featured.to_csv('output/transactions_featured.csv', index=False)

print("Saved output/transactions_featured.csv")
print(f"Shape: {tx_featured.shape}")
print(tx_featured.head())
print()
print("fan_in_5plus value counts:")
print(tx_featured['fan_in_5plus'].value_counts())
