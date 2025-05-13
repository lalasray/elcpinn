import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

def u_s(t, I_p):
    dI_p_dt = np.gradient(I_p, t)  
    u_s = -M * dI_p_dt * x     
    return u_s

file_path = r'Rogowski_data_new\Eval\-10C\-10C_50Hz_50A.csv'
df = pd.read_csv(file_path, sep=';', header=None)

t = np.linspace(0, 1, time) 
I_p = df.iloc[0, :time] 
real_output = df.iloc[1, :time] 
output = u_s(t, I_p) 

def normalize(data):
    min_val = np.min(data)
    max_val = np.max(data)
    return (2 * (data - min_val) / (max_val - min_val)) - 1

output = normalize(output)
normalized_real_output = normalize(real_output)

# Create a DataFrame with predicted and real output voltages
voltage_data = pd.DataFrame({
    'input': I_p,
    'Predicted_Output': output,
    'Real_Output': real_output, # scaling as in plot
    'Real_Output_Normalized': normalized_real_output
})

# Construct the new file name
new_file_path = file_path.replace('.csv', '_voltage_only.csv')

# Save to CSV
voltage_data.to_csv(new_file_path, index=False, sep=';')

fig, axs = plt.subplots(3, 1, figsize=(10, 8))

axs[0].plot(t, I_p, label='Input Current $I_p(t)$')
axs[0].set_title('Input Current $I_p(t)$')
axs[0].set_ylabel('Current (A)')
axs[0].grid(True)
axs[0].legend()

axs[1].plot(t,output, label='Calculated Output Voltage $u_s(t)$', color='orange')
axs[1].set_title('Output Voltage $u_s(t)$')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Voltage (V)')
axs[1].grid(True)
axs[1].legend()

axs[2].plot(t, real_output, label='Real Output Voltage $u_s(t)$', color='green')
axs[2].set_title('Real Output Voltage $u_s(t)$')
axs[2].set_xlabel('Time (s)')
axs[2].set_ylabel('Voltage (V)')
axs[2].grid(True)
axs[2].legend()

plt.tight_layout()
plt.show()