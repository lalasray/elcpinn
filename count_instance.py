import pandas as pd
import re
from collections import Counter
import os
import glob

def extract_items(s):
    return re.findall(r'\[([^\[\]]+)\]', str(s))

def count_sublabels_in_file(file_path):
    df = pd.read_csv(file_path)
    df['sublabel_items'] = df['sublabel'].apply(extract_items)
    all_sublabels = [sublabel for sublist in df['sublabel_items'] for sublabel in sublist]
    return Counter(all_sublabels)

def count_sublabels_in_folder(folder_path):
    all_counts = Counter()
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    for file_path in csv_files:
        print(f"Processing: {os.path.basename(file_path)}")
        file_counts = count_sublabels_in_file(file_path)
        all_counts.update(file_counts)
    
    return all_counts

# Replace with your actual folder path
folder_path = r"C:\Users\lalas\Desktop\object"

# Get total sublabel counts across all CSVs
total_sublabel_counts = count_sublabels_in_folder(folder_path)

# Print results
print("\nTotal Sublabel Counts Across All CSV Files:")
for sublabel, count in total_sublabel_counts.items():
    print(f"{sublabel}: {count*3}")
