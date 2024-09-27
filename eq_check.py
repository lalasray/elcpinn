import numpy as np
import matplotlib.pyplot as plt

tr_50 = 100 / 1000 / 1000 
w_50 = 2 * np.pi * 50  
M = tr_50 / w_50  

def u_s(t, I_p):
    dI_p_dt = np.gradient(I_p, t)
    u_s = -M * dI_p_dt
    
    return u_s

t = np.linspace(0, 1, 1000)
I_p = np.sin(2 * np.pi * 50 * t) 
output = u_s(t, I_p)

fig, axs = plt.subplots(2, 1, figsize=(10, 8))

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
plt.tight_layout()
plt.show()