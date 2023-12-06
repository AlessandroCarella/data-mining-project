from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import itertools
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
os.chdir("../clustering/Methods")

df = pd.read_csv("../dataset (missing + split)/R_Norm_Prueba_Dataset_nomissing_nofeatur_noutlier.csv", skip_blank_lines=True)

eps_values = [0.3,0.5, 1, 2.5,3,4,5,6,7]
min_samples_values = [3,5,7,10,15, 25,35, 50]

results_labels = {}
results_silhouettes = {}

for eps_val, min_samples_val in itertools.product(eps_values, min_samples_values):
    dbscan_model = DBSCAN(eps=eps_val, min_samples=min_samples_val)
    cluster_labels = dbscan_model.fit_predict(df)
    col_name = f"DBSCAN_eps{eps_val}_minpts{min_samples_val}"
    results_labels[col_name] = cluster_labels

    # Calculate silhouette score excluding points with label -1 (noise/outliers)
    valid_indices = np.where(cluster_labels != -1)[0]
    valid_data = df.iloc[valid_indices]
    valid_labels = cluster_labels[valid_indices]
    
    if len(np.unique(valid_labels)) > 1:  # At least 2 clusters required for silhouette
        silhouette = silhouette_score(valid_data, valid_labels)
    else:
        silhouette = None
    
    col_name = f"DBSCAN_eps{eps_val}_minpts{min_samples_val}"
    results_silhouettes[col_name] = {'silhouette_score': silhouette}

# Convert results to a DataFrame
labels_df = pd.DataFrame(results_labels)
silhouettes_df = pd.DataFrame(results_silhouettes)

# Accessing the DataFrame
print("Labels DataFrame:")
print(labels_df.head())

print(labels_df.nunique().to_markdown())


# Concatenate multiple columns into a single series
concatenated = pd.concat([labels_df[col] for col in labels_df.columns])

unique_values = concatenated.unique()
print(unique_values)

silhouettes_df

labels_df["DBSCAN_eps0.5_minpts5"].value_counts()