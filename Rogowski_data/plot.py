import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

file_path = 'Rogowski_data/-15C_150Hz_10A.csv'
file_path_alt = 'Rogowski_data/55C_150Hz_10A.csv'
file_path_alt_2 = 'Rogowski_data/-15C_2500Hz_10A.csv'
file_path_alt_3 = 'Rogowski_data/-15C_150Hz_20A.csv'

data = pd.read_csv(file_path,sep=';',header=None)
data_alt = pd.read_csv(file_path_alt,sep=';',header=None)
data_alt_2 = pd.read_csv(file_path_alt_2,sep=';',header=None)
data_alt_3 = pd.read_csv(file_path_alt_3,sep=';',header=None)

data = data.T
data_alt = data_alt.T
data_alt_2 = data_alt_2.T
data_alt_3 = data_alt_3.T

data.iloc[0:999].plot(legend=True) #plot
data_alt.iloc[0:999].plot(legend=True) #plot
data_alt_2.iloc[0:999].plot(legend=True) #plot
data_alt_3.iloc[0:999].plot(legend=True) #plot
plt.show()