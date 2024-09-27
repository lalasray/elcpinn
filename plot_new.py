import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pandas as pd

file_path = '20C_50Hz_50A.csv'
df = pd.read_csv(file_path, sep=';')
print(df.shape)


plt.figure(figsize=(10, 6))
for i in range(3):
    plt.plot(df.iloc[i, :2000], label=f'Val {i+1}')

plt.gca().set_xticks([])
plt.title('Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()