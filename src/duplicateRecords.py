import pandas as pd
import os

df = pd.read_csv('dataset (missing + split)/train.csv')


# Define the output directory
output_dir_dv = os.path.join(os.path.dirname(os.path.abspath(__file__ or '.')), "Duplicate Records in Dataset")

# Create the output directory if it doesn't exist
os.makedirs(output_dir_dv, exist_ok=True)
file_name = os.path.join(output_dir_dv, 'duplicateRecords.txt')
duplicate_records = df[df.duplicated(keep=False)]
with open(file_name, 'w') as file:
    file.write(f"Duplicate records:\n")
    if(duplicate_records.size) == 0:
        file.write(f"None")
    else:
        duplicate_records.to_csv(file_name, header=False, mode='a')