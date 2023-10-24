import pandas as pd
import matplotlib.pyplot as plt
import os
import os.path as path
import numpy as np

# Replace 'your_dataset.csv' with the path to your dataset file
df = pd.read_csv('dataset (missing + split)/train.csv')

# Define the output directory
output_hist_dir = os.path.join(os.path.dirname(os.path.abspath(__file__ or '.')), "First 9 Columns Histograms")

# Create the output directory if it doesn't exist
os.makedirs(output_hist_dir, exist_ok=True)

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

# Integrate the code for histograms of the first 9 columns
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
        mode_value = df[col].mode()[0]
        plt.ticklabel_format(style='plain', axis='x')
        plt.title(f'{col.capitalize()} Histogram')
        plt.xlabel(col.capitalize())
        plt.ylabel('Frequency')
        plt.axvline(mean_value, color = 'b', linestyle = 'dashed')
        plt.axvline(mode_value, color = 'm', linestyle = 'dashdot')
        plt.axvline(median_value, color = 'r', linestyle = 'dotted')
        plt.grid(axis='y', alpha=0.75)
        # Save the histogram as an image file
        output_file = os.path.join(output_hist_dir, f'{col.capitalize()}_hist.png')
        plt.savefig(output_file)

# Close all plot windows (if you prefer not to display them)
plt.close('all')