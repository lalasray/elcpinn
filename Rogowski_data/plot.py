import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

file_path = '/Users/virginianegri/Desktop/Rogowski_data/40C/40C_350Hz_20A.csv'
file_path = '/Users/virginianegri/Desktop/Rogowski_data/40C/40C_650Hz_20A.csv'
# file_path = '/Users/virginianegri/Desktop/Rogowski_data/40C/40C_sinc_100A.csv'

data = pd.read_csv(file_path,sep=';',header=None)
print(data)
data = data.T

data.iloc[0:999].plot(legend=True) #plot
# data.iloc[130:230].plot(legend=True) #plot for the sinc
plt.show()