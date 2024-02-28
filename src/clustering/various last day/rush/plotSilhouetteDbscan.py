import pandas as pd
import matplotlib.pyplot as plt
import re
import os.path as path

# Load the data from the CSV file
df = pd.read_csv("metrics/silhouette dbscan.csv")

def extract_valuesForDBSCAN (input_string):
    # Define a regular expression pattern to extract values
    pattern = r'min_samples=(\d+) eps=([0-9eE.-]+)'

    # Use re.search to find the pattern in the string
    match = re.search(pattern, input_string)

    # Extract values from the matched groups
    min_samples = match.group(1)
    eps = match.group(2)

    return min_samples, eps

minSamplesAndEps = {}
names = list(df["clustering type"])
values = list(df["value"])
for name, value in zip(names, values):
    min_samples, eps = extract_valuesForDBSCAN (name)
    if not float(eps) in minSamplesAndEps:
        minSamplesAndEps [float(eps)] = []
    minSamplesAndEps [float(eps)].append ([int(min_samples), value])
    
"""
for eps, group in dbscan_data.groupby('eps'):
    plt.plot(group['value'], label=f"EPS {eps}", marker='o')
"""

for eps, min_samplesValue in minSamplesAndEps.items():
    # Plot the results
    plt.figure(figsize=(21, 9))
    
    x_values = [item[0] for item in min_samplesValue]
    y_values = [item[1] for item in min_samplesValue]

    plt.plot (x_values, y_values, marker='o')

    plt.title('DBSCAN Clustering Results with eps=' + str (eps))
    plt.xlabel('Min samples')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig (path.join(path.dirname(__file__), "plots", "dbscan", "dbscan with eps=" + str (eps) + ".png"))