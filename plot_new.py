import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pandas as pd

file_path = r'Rogowski_data\Rogowski_data\20C\20C_50Hz_50A.csv'
df = pd.read_csv(file_path, sep=';', header=None)
print(df.shape)


plt.figure(figsize=(10, 6))
for i in range(4):
    plt.plot(df.iloc[i, :5000], label=f'Val {i+1}')

plt.gca().set_xticks([])
plt.title('Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()