import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
import os
import os.path as path

# Replace 'your_dataset.csv' with the path to your dataset file
df = pd.read_csv(path.join(path.dirname(__file__), "../../dataset (missing + split)/train.csv"))

# Define the output directory
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__ or '.')), "allScatterPlots")

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Get a list of all combinations of features (columns) (without 'name', 'artists', 'album_name')
feature_combinations = [comb for comb in (list(set(tuple(sorted(comb)) for comb in combinations(df.columns, 2)))) if not any(word in comb for word in ['name', 'artists', 'album_name'])]

# Create scatter plots for each combination and save them as images
for combo in feature_combinations:
    feature1, feature2 = combo
   
    plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
    plt.scatter(df[feature1], df[feature2], alpha=0.5)
   
    plt.title(f'Scatter Plot of {feature1} vs {feature2}')
   
    plt.xlabel(feature1)
    plt.ylabel(feature2)
   
    plt.grid(True)
    

    # Generate the file name using feature names
    image_filename = path.join(output_dir, f'scatter_plot_{feature1}_vs_{feature2}.png')
    
    # Save the scatter plot as an image
    plt.savefig(image_filename)
    
    # Close the current plot to free up resources
    plt.close()


# To ensure all plots are saved successfully
print("Scatter plots saved as individual images.")
