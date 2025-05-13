import pandas as pd
import numpy as np
import os
from pathlib import Path

# Base directory to search in
base_dir = Path(r'Rogowski_data_new\Eval')

# Loop through all matching CSV files recursively
for filepath in base_dir.rglob('*_voltage_only.csv'):
    try:
        # Load the file
        df = pd.read_csv(filepath, sep=';')

        # Check column exists
        if 'Predicted_Output_(V)' not in df.columns:
            print(f"Skipped (missing column): {filepath}")
            continue

        # Create smooth sinusoidal drift: 0% to 7%
        n = len(df)
        drift = 1 + 0.035 * (1 + np.cos(np.linspace(0, 2 * np.pi, n)))

        # Apply drift
        df['Predicted_Output_(V)'] = df['Predicted_Output_(V)'] * drift

        # Save file back (overwrite)
        df.to_csv(filepath, sep=';', index=False)
        print(f"Updated: {filepath}")

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
