import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

tr_50 = 100 / 1000 / 1000 
w_50 = 2 * np.pi * 50  
M = tr_50 / w_50  
time = 2000 

def u_s(t, I_p):
    dI_p_dt = np.gradient(I_p, t)
    u_s = -M * dI_p_dt / (-1*10**(-8))
    
    return u_s

file_path = '/Users/virginianegri/Desktop/Rogowski/Rogowski_data/20C/20C_50Hz_50A.csv'
df = pd.read_csv(file_path, sep=';', header=None)

t = np.linspace(0, 1, time)
I_p = df.iloc[0, :time]
real_output = df.iloc[1, :time]
output = u_s(t, I_p)

fig, axs = plt.subplots(3, 1, figsize=(10, 8))

axs[0].plot(t, I_p, label='Input Current $I_p(t)$')
axs[0].set_title('Input Current $I_p(t)$')
axs[0].set_ylabel('Current (A)')
axs[0].grid(True)
axs[0].legend()

axs[1].plot(t, output, label='Output Voltage $u_s(t)$', color='orange')
axs[1].set_title('Output Voltage $u_s(t)$')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Voltage (V)')
axs[1].grid(True)
axs[1].legend()

axs[2].plot(t, real_output, label='Real Output Voltage $u_s(t)$', color='green')
axs[2].set_title('Output Voltage $u_s(t)$')
axs[2].set_xlabel('Time (s)')
axs[2].set_ylabel('Voltage (V)')
axs[2].grid(True)
axs[2].legend()

plt.tight_layout()
plt.show()
