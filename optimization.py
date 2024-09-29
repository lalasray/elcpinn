import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

# Constants
tr_50 = 100 / 1000 / 1000  # Time constant or related constant
w_50 = 2 * np.pi * 50  # Angular frequency for 50Hz
M = tr_50 / w_50  # Parameter M based on tr_50 and w_50
time = 2000  # Duration or time samples

# Modified function with X and C as parameters
def u_s(t, I_p, X=1, C=0):
    dI_p_dt = np.gradient(I_p, t)  # Calculate the derivative
    u_s = -M * dI_p_dt * (X + C)  # Adjust formula with X and C
    return u_s

# Load the CSV data
file_path = r'Rogowski_data\Rogowski_data\20C\20C_50Hz_50A.csv'
df = pd.read_csv(file_path, sep=';', header=None)

# Extract time and current data
t = np.linspace(0, 1, time)
I_p = df.iloc[0, :time].to_numpy()
real_output = df.iloc[1, :time].to_numpy()

# Example use of the function (with default X and C)
output = u_s(t, I_p)

# Optional: Plot the results
plt.plot(t, real_output, label='Real Output')
plt.plot(t, output, label='Calculated Output')
plt.legend()
plt.show()
