import pandas as pd
import os

df = pd.read_csv('dataset (missing + split)/train.csv')


# Define the output directory
output_dir_unique = os.path.join(os.path.dirname(os.path.abspath(__file__ or '.')), "Unique Values In Columns in Dataset")

# Create the output directory if it doesn't exist
os.makedirs(output_dir_unique, exist_ok=True)
file_name = os.path.join(output_dir_unique, 'uniqueValuesInCols.txt')
categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    file_name = os.path.join(output_dir_unique, f'uniqueValuesIn{col.capitalize()}.txt')
    unique_values = df[col].unique()
    with open(file_name, 'w') as file:
        file.write(f"Unique values in {col}:\n")
        for value in unique_values:
            file.write(f"{value}\n")
