import pandas as pd
import matplotlib.pyplot as plt
import os
import os.path as path
import numpy as np

# Replace 'your_dataset.csv' with the path to your dataset file
df = pd.read_csv(path.join(path.dirname(__file__), "../../dataset (missing + split)/train.csv"))

# Define the output directory
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__ or '.')), "latex generated")

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Select the columns you want to describe
selected_columns = ['loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness']
df_selected_columns=df[selected_columns]

# Integrate the code for histograms of the last 9 columns
for col in df_selected_columns.columns:
    if pd.api.types.is_numeric_dtype(df_selected_columns[col]):
        plt.figure(figsize=(8, 6))
        bin_width = 3.49 * df[col].std() / (len(df[col])**(1/3))
        plt.hist(
            df[col],
            bins=np.arange(min(df[col]), max(df[col]) + bin_width, bin_width),
            color='skyblue',
            edgecolor='black'
        )
        mean_value = df[col].mean()
        median_value = df[col].median()
        plt.ticklabel_format(style='plain', axis='x')
        plt.title(f'{col.capitalize()} Histogram')
        plt.xlabel(col.capitalize())
        plt.ylabel('Frequency')
        plt.axvline(mean_value, color='blue', linestyle='dashed')
        plt.axvline(median_value, color='red', linestyle='--')
        plt.grid(axis='y', alpha=0.75)
        # Save the bar chart as an image file
        output_file = os.path.join(output_dir, f'{col.capitalize()}_hist.png')
        plt.savefig(output_file)

# Close all plot windows (if you prefer not to display them)
plt.close('all')




# Describe the selected columns
column_description = df[selected_columns].describe()

# Calculate additional statistics
range_data = df[selected_columns].max() - df[selected_columns].min()
variance_data = df[selected_columns].var()
std_deviation_data = df[selected_columns].std()

# Create a dictionary to map column names to their LaTeX labels
latex_labels = {
    'loudness': 'Loudness',
    'mode': 'Mode',
    'speechiness': 'Speechiness',
    'acousticness': 'Acousticness',
    'instrumentalness': 'Instrumentalness',
    'liveness': 'Liveness'
}

# Calculate mode separately using the mode() function
mode_data = df[selected_columns].mode().iloc[0]

# Function to format the float value as a string
def format_float(value):
    formatted_value = f"{value:.6f}"
    return formatted_value.rstrip('0').rstrip('.')

# Generate LaTeX-formatted output and save to individual files
for column in selected_columns:
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
        #file.write(f"In the figure below, we observe a histogram illustrating the distribution of the '{latex_labels[column]}' attribute within the dataset. In the histogram, the frequency of data points falling into distinct bins is visualized, where the x-axis delineates the range of '{latex_labels[column]}' values partitioned into 18 bins, as determined by Sturge's rule.\n")
        file.write("\includegraphics[width=0.9\textwidth]{template/histograms/"+ latex_labels[column] + "_hist.png}~\\[1cm]")
        file.write("\n")

import os

# Define the directory where the individual description files are located
#output_dir = os.path.dirname(os.path.abspath(__file__))

# Define the order of columns
selected_columns = ['loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness']

# Create a list to store the content of each file
file_contents = []

# Read the content of each file and store it in the list
for column in selected_columns:
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