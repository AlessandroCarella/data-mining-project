import pandas as pd
import os

df = pd.read_csv('dataset (missing + split)/train.csv')


# Define the output directory
output_dir_mv = os.path.join(os.path.dirname(os.path.abspath(__file__ or '.')), "Missing Values in Dataset")

# Create the output directory if it doesn't exist
os.makedirs(output_dir_mv, exist_ok=True)
file_name = os.path.join(output_dir_mv, 'missingValues.txt')
missing_values = df.isnull().sum()
with open(file_name, 'w') as file:
    file.write(f"Missing values in each column:\n")

missing_values.to_csv(file_name, header=False, mode='a')
