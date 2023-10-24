import pandas as pd
import os
df = pd.read_csv('dataset (missing + split)/train.csv')


# Define the output directory
output_dir_dt = os.path.join(os.path.dirname(os.path.abspath(__file__ or '.')), "Data Types  in Dataset")

# Create the output directory if it doesn't exist
os.makedirs(output_dir_dt, exist_ok=True)
file_name = os.path.join(output_dir_dt, 'dataTypes.txt')
data_types = df.dtypes
with open(file_name, 'w') as file:
    file.write(f"Data types of columns:\n")
    data_types.to_csv(file_name, header=True, mode='a')