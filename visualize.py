import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


#file_path = '/home/lala/other/Repos/git/elcpinn/Rogowski Model/Rogowski coils/Power quality Rogowski/1_20C_30%/rog1_50Hz_20C_30%.txt'
#file_path = '/home/lala/other/Repos/git/elcpinn/Rogowski Model/Rogowski coils/Power quality Rogowski/2_40C_30%/rog1_50Hz_40C_30%.txt'
#file_path = '/home/lala/other/Repos/git/elcpinn/Rogowski Model/Rogowski coils/Power quality Rogowski/3_-5C_30%/rog1_50Hz_-5C_30%.txt'
#file_path = '/home/lala/other/Repos/git/elcpinn/Rogowski Model/Rogowski coils/Power quality Rogowski/4_20Cend_30%/rog1_50Hz_20Cend_30%.txt'

#file_path = '/home/lala/other/Repos/git/elcpinn/Rogowski Model/Rogowski coils/Power quality Rogowski/1_20C_30%/rog1_150Hz_20C_30%.txt'
#file_path = '/home/lala/other/Repos/git/elcpinn/Rogowski Model/Rogowski coils/Power quality Rogowski/2_40C_30%/rog1_150Hz_40C_30%.txt'
#file_path = '/home/lala/other/Repos/git/elcpinn/Rogowski Model/Rogowski coils/Power quality Rogowski/3_-5C_30%/rog1_150Hz_-5C_30%.txt'
#file_path = '/home/lala/other/Repos/git/elcpinn/Rogowski Model/Rogowski coils/Power quality Rogowski/4_20Cend_30%/rog1_150Hz_20Cend_30%.txt'

#file_path = '/home/lala/other/Repos/git/elcpinn/Rogowski Model/Rogowski coils/Power quality Rogowski/1_20C_30%/rog1_2500Hz_20C_30%.txt'
file_path = f'Rogowski Model/Rogowski coils/Power quality Rogowski/2_40C_30%/rog1_2500Hz_40C_30%.txt'
#file_path = '/home/lala/other/Repos/git/elcpinn/Rogowski Model/Rogowski coils/Power quality Rogowski/3_-5C_30%/rog1_150Hz_-5C_30%.txt'
#file_path = '/home/lala/other/Repos/git/elcpinn/Rogowski Model/Rogowski coils/Power quality Rogowski/4_20Cend_30%/rog1_150Hz_20Cend_30%.txt'

with open(file_path, 'r') as file:
    data = file.readlines()

data = np.array([float(line.strip()) for line in data])

data = data[:int(data.shape[0]/3)] #take first 1/3 of the signal

data = data[:5000] # only first 5000 values

time_interval = 1
time = np.arange(0, len(data) * time_interval, time_interval)

plt.plot(time, data)
plt.title("Plot")
plt.xlabel("Time")
plt.ylabel("Value")
plt.grid(True)
plt.show()


'''
# Read the data from the text file
#file_path = '/home/lala/other/Repos/git/elcpinn/Rogowski Model/Rogowski coils/Resistance Rogowski/1_rog3_tamb_40_-5_s.txt'
#file_path = '/home/lala/other/Repos/git/elcpinn/Rogowski Model/Rogowski coils/Resistance Rogowski/1_rog4_tamb_40_-5_s.txt'
#file_path = '/home/lala/other/Repos/git/elcpinn/Rogowski Model/Rogowski coils/Resistance Rogowski/2_rog1_-5_40_s.txt'
#file_path = '/home/lala/other/Repos/git/elcpinn/Rogowski Model/Rogowski coils/Resistance Rogowski/2_rog1_-5_40_s.txt'
# Read the data from the text file

#df = pd.read_csv(file_path, sep="\t", header=None, names=["time", "first_value", "second_value"], parse_dates=["time"])

#print(df['first_value'])
#plt.figure(figsize=(15, 5))
 
#plt.subplot(1, 2, 1)
#plt.plot(df['time'], df['first_value'], linestyle='-')
#plt.plot(df['first_value'], linestyle='-')
#plt.xlabel('Time')
#plt.ylabel('First Value')
#plt.title('Time vs First Value')
#plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
#plt.plot(df['time'], df['second_value'], linestyle='-')
plt.plot(df['second_value'], linestyle='-')
plt.xlabel('Time')
plt.ylabel('Second Value')
plt.title('Time vs Second Value')
plt.xticks(rotation=45)

plt.subplot(1, 3, 3)
plt.plot(df['time'], linestyle='-')
plt.xlabel('Time')
plt.ylabel('Time')
plt.title('Time vs Second Value')
plt.xticks(rotation=45)


plt.tight_layout()
plt.show()
'''

