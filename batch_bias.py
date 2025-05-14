import pandas as pd
import numpy as np
from pathlib import Path

# Original and new base directories
base_dir = Path(r'Rogowski_data_new\Eval')
target_base_dir = Path(r'Rogowski_data_3\Eval')

# Column to apply drift to
column_name = 'Predicted_Output_(V)'

# Number of cosine oscillation cycles over the entire dataset
k = np.random.randint(1, 3)  # Increase for faster drift (e.g., 10 for faster, 1 for slow)

# Process each matching CSV
for filepath in base_dir.rglob('*_voltage_only.csv'):
    try:
        df = pd.read_csv(filepath, sep=';')

        if column_name not in df.columns:
            print(f"Skipped (missing column): {filepath}")
            continue

        # Generate cosine drift: 0% to 7%, with `k` cycles
        n = len(df)
        drift = 1.085 + 0.035 * np.cos(np.linspace(0, 2 * np.pi * k, n))
        df[column_name] *= drift

        # Construct new save path, preserving folder structure
        relative_path = filepath.relative_to(base_dir)
        target_path = target_base_dir / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Save modified CSV
        df.to_csv(target_path, sep=';', index=False)
        print(f"Saved with drift: {target_path}")

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
