import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load CSV
file_path = r'Rogowski_data_new\Eval\50C\50C_250Hz_20A_voltage_only.csv'
df = pd.read_csv(file_path, sep=';')

# Create smooth sinusoidal drift between 0% and 7% (multiplier: 1.00 to 1.07)
n = len(df)
drift = 1 + 0.035 * (1 + np.cos(np.linspace(0, 2 * np.pi, n)))  # range: [1.00, 1.07]
df['Predicted_Output_With_Error'] = df['Predicted_Output_(V)'] * drift

# Plot Real Output (Normalized) and Smoothed Error-Affected Predicted Output
plt.figure(figsize=(10, 5))
plt.plot(df['Real_Output_Normalized'], label='Real Output (Normalized)', color='green')
plt.plot(df['Predicted_Output_(V)'], label='Predicted Output with 0–7% Drift', color='orange', linestyle='--')

plt.title('Predicted vs Real Output Voltage (with Smooth 0–7% Error Drift)')
plt.xlabel('Sample Index')
plt.ylabel('Voltage (V)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
