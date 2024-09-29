import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Constants
tr_50 = 100 / 1000 / 1000  # Transducer ratio
f = 50  # Frequency in Hz
w_50 = 2 * np.pi * f  # Angular frequency
alpha = 0.01  # Temperature coefficient
T_0 = 20  # Reference temperature in °C
T = 20  # Current temperature in °C
M_0 = tr_50 / w_50  # Nominal mutual inductance
x =-1

# Calculate mutual inductance based on temperature
M = M_0 * (1 + alpha * (T - T_0))  # Adjust M based on temperature
time = 2000  # Number of time points

def u_s(t, I_p):
    dI_p_dt = np.gradient(I_p, t)  # Time derivative of the current
    u_s = -M * dI_p_dt*x  # Output voltage considering mutual inductance
    
    return u_s

# Load data
file_path = r'Rogowski_data\Rogowski_data\20C\20C_50Hz_50A.csv'
df = pd.read_csv(file_path, sep=';', header=None)

t = np.linspace(0, 1, time)  # Time array
I_p = df.iloc[0, :time]  # Input current
real_output = df.iloc[1, :time]  # Real output voltage
output = u_s(t, I_p)  # Calculated output voltage

# Plotting
fig, axs = plt.subplots(3, 1, figsize=(10, 8))

# Plot Input Current
axs[0].plot(t, I_p, label='Input Current $I_p(t)$')
axs[0].set_title('Input Current $I_p(t)$')
axs[0].set_ylabel('Current (A)')
axs[0].grid(True)
axs[0].legend()

# Plot Output Voltage
axs[1].plot(t, output, label='Calculated Output Voltage $u_s(t)$', color='orange')
axs[1].set_title('Output Voltage $u_s(t)$')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Voltage (V)')
axs[1].grid(True)
axs[1].legend()

# Plot Real Output Voltage
axs[2].plot(t, real_output, label='Real Output Voltage $u_s(t)$', color='green')
axs[2].set_title('Real Output Voltage $u_s(t)$')
axs[2].set_xlabel('Time (s)')
axs[2].set_ylabel('Voltage (V)')
axs[2].grid(True)
axs[2].legend()

plt.tight_layout()
plt.show()