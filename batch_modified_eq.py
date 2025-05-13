import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob

# Constants
tr_50 = 100 / 1000 / 1000 
f = 50 
w_50 = 2 * np.pi * f 
alpha = 4 
T_0 = 20 
T = 55 
M_0 = tr_50 / w_50 
x = -(5 / 200)
M = M_0 * (1 + alpha * (T - T_0))  # Adjust M based on temperature
time = 50000 

# Output voltage function
def u_s(t, I_p):
    dI_p_dt = np.gradient(I_p, t)
    return -M * dI_p_dt * x

# Normalization function
def normalize(data):
    min_val = np.min(data)
    max_val = np.max(data)
    return (2 * (data - min_val) / (max_val - min_val)) - 1

# Base folder
base_folder = r'Rogowski_data_new\Eval'

# Recursively find all .csv files excluding ones ending in '_voltage_only.csv'
csv_files = [f for f in glob.glob(os.path.join(base_folder, '**', '*.csv'), recursive=True)
             if not f.endswith('_voltage_only.csv')]

# Process each CSV file
for file_path in csv_files:
    try:
        df = pd.read_csv(file_path, sep=';', header=None)

        t = np.linspace(0, 1, time)
        I_p = df.iloc[0, :time]
        real_output = df.iloc[1, :time]
        output = u_s(t, I_p)

        normalized_output = normalize(output)
        normalized_real_output = normalize(real_output)

        # Prepare DataFrame
        voltage_data = pd.DataFrame({
            'Input_Current_(A)': I_p,
            'Predicted_Output_(V)': normalized_output,
            'Real_Output_(V)': real_output,
            'Real_Output_Normalized': normalized_real_output
        })

        # Construct new file name
        new_file_path = file_path.replace('.csv', '_voltage_only.csv')

        # Save to CSV
        voltage_data.to_csv(new_file_path, index=False, sep=';')
        print(f"Processed and saved: {new_file_path}")

    except Exception as e:
        print(f"Failed to process {file_path}: {e}")
