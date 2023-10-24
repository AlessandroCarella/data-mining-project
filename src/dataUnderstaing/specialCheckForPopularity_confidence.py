import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import os
import os.path as path
import os

# Replace 'your_dataset.csv' with the path to your dataset file
df = pd.read_csv(path.join(path.dirname(__file__), "../../dataset (missing + split)/train.csv"))

df = df.dropna(subset=['popularity_confidence'])

# Step 2: Define the list of features for which you want to generate bar charts
featuresOutOfScatterPlots = ['name', 'artists', 'album_name', 'genre']

for feature in featuresOutOfScatterPlots:
    df[feature], _ = pd.factorize(df[feature])

# Define the output directory
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__ or '.')), "specialCheckForPopularity_confidence")
# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)



# Step 3: Generate bar charts for each feature
for feature in featuresOutOfScatterPlots:
    plt.figure(figsize=(8, 6))  # Adjust the figure size as needed

    # Count the occurrences of each unique value in the column and order them from most frequent to least frequent
    value_counts = df[feature].value_counts()

    # Plot the bar chart using the sorted value_counts
    plt.bar(value_counts.index, value_counts)
    
    plt.title(f'Bar Chart for {feature}')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.xticks(rotation=45)  # Rotate x-axis labels if needed
    plt.savefig(os.path.join(output_dir, f'{feature}_bar.png'))

# Step 4: Save the value counts to CSV files

for feature in featuresOutOfScatterPlots:
    # Count the occurrences of each unique value in the column and order them from most frequent to least frequent
    value_counts = dict(sorted(df[feature].value_counts().to_dict().items(), key=lambda item: item[1], reverse=True))

    # Define the path where you want to save the CSV file
    output_csv_file = f'{feature}_value_counts.csv'
    output_csv_file_path = os.path.join (output_dir, output_csv_file)

    # Save the dictionary to a CSV file
    with open(output_csv_file_path, 'w', encoding='utf-8') as f:
        f.write(f"{feature}, value_counts\n")
        for key, value in value_counts.items():
            f.write(f"{str(key).replace(',',';')}, {value}\n")