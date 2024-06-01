import matplotlib.pyplot as plt
import pandas as pd

# Read the data from the text file
#file_path = '/home/lala/other/Repos/git/elcpinn/Rogowski Model/Rogowski coils/Resistance Rogowski/1_rog3_tamb_40_-5_s.txt'
#file_path = '/home/lala/other/Repos/git/elcpinn/Rogowski Model/Rogowski coils/Resistance Rogowski/1_rog4_tamb_40_-5_s.txt'
#file_path = '/home/lala/other/Repos/git/elcpinn/Rogowski Model/Rogowski coils/Resistance Rogowski/2_rog1_-5_40_s.txt'
file_path = '/home/lala/other/Repos/git/elcpinn/Rogowski Model/Rogowski coils/Resistance Rogowski/2_rog1_-5_40_s.txt'

# Read the data from the text file
df = pd.read_csv(file_path, sep="\t", header=None, names=["time", "first_value", "second_value"], parse_dates=["time"])

# Apply smoothing using a rolling window (moving average)
window_size = 20  # Adjust the window size as needed
df['first_value_smoothed'] = df['first_value'].rolling(window=window_size).mean()
df['second_value_smoothed'] = df['second_value'].rolling(window=window_size).mean()

# Plot the data
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(df['time'], df['first_value_smoothed'], linestyle='-')
plt.xlabel('Time')
plt.ylabel('First Value (Smoothed)')
plt.title('Time vs First Value (Smoothed)')
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
plt.plot(df['time'], df['second_value_smoothed'], linestyle='-')
plt.xlabel('Time')
plt.ylabel('Second Value (Smoothed)')
plt.title('Time vs Second Value (Smoothed)')
plt.xticks(rotation=45)

#plt.subplot(1, 3, 3)
#plt.plot(df['time'], linestyle='-')
#plt.xlabel('Time')
#plt.ylabel('Time')
#plt.title('Time')
#plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
