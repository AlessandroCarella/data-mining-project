import pandas as pd
import matplotlib.pyplot as plt
import os
import os.path as path
import numpy as np

df = pd.read_csv('dataset (missing + split)/train.csv')

# Define the output directory
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__ or '.')), "First 9 Columns Latex Descriptions")

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Create a dictionary to map column names to their LaTeX labels
latex_labels = {
    'duration_ms': 'Duration ms',
    'popularity': 'Popularity',
    'danceability': 'Danceability',
    'energy': 'Energy',
    'key' : 'Key'
}
selected_columns = latex_labels.keys()
df_selected_columns=df[selected_columns]

# Function to format the float value as a string
def format_float(value):
    formatted_value = f"{value:.6f}"
    return formatted_value.rstrip('0').rstrip('.')

# Describe the selected columns
column_description = df_selected_columns.describe()

# Calculate additional statistics
range_data = df_selected_columns.max() - df_selected_columns.min()
variance_data = df_selected_columns.var()
std_deviation_data = df_selected_columns.std()
mode_data = df_selected_columns.mode().iloc[0]


# Calculate mode separately using the mode() function
mode_data = df_selected_columns.mode().iloc[0]

# Function to format the float value as a string
def format_float(value):
    formatted_value = f"{value:.6f}"
    return formatted_value.rstrip('0').rstrip('.')

# Generate LaTeX-formatted output and save to individual files
for column in latex_labels.keys():
    file_name = os.path.join(output_dir, f'{column}_description.txt')
    with open(file_name, 'w') as file:
        file.write(f"\\item \\textbf{{{latex_labels[column]}}}\n")
        file.write("\\textit{Description of the attribute's central and dispersion tendencies: }\n")
        file.write(f"Count: {int(column_description.loc['count'][column])}\n")
        file.write(f"Mean: {format_float(column_description.loc['mean'][column])}\n")
        # Note: Mode is already a string and doesn't require formatting
        file.write(f"Mode: {mode_data[column]}\n")
        file.write(f"Median: {format_float(column_description.loc['50%'][column])}\n\n")
        file.write(f"Min: {format_float(column_description.loc['min'][column])}\n")
        file.write(f"25\\%: {format_float(column_description.loc['25%'][column])}\n")
        file.write(f"50\\%: {format_float(column_description.loc['50%'][column])}\n")
        file.write(f"75\\%: {format_float(column_description.loc['75%'][column])}\n")
        file.write(f"Max: {format_float(column_description.loc['max'][column])}\n")
        file.write(f"Range: {format_float(range_data[column])}\n")
        file.write(f"Variance: {format_float(variance_data[column])}\n")
        file.write(f"Standard Deviation: {format_float(std_deviation_data[column])}\n\n")
        file.write("\\textit{Histogram}\n")
        file.write("\includegraphics[width=0.9\textwidth]{template/histograms/"+ latex_labels[column] + "_hist.png}~\\[1cm]")
        file.write("\n")


# Create a list to store the content of each file
file_contents = []

# Read the content of each file and store it in the list
for column in latex_labels.keys():
    file_name = os.path.join(output_dir, f'{column}_description.txt')
    with open(file_name, 'r') as file:
        file_contents.append(file.read())

# Merge the file contents in the specified order
merged_output = '\n'.join(file_contents)

# Write the merged output to a single file
merged_file_path = os.path.join(output_dir, 'merged_description.txt')
with open(merged_file_path, 'w') as merged_file:
    merged_file.write(merged_output)

print(f"Files merged and saved to {merged_file_path}")