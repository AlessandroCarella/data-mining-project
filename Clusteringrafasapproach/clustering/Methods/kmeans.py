import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score


os.chdir("../clustering/Methods")


df = pd.read_csv("../dataset (missing + split)/R_Norm_Prueba_Dataset_nomissing_nofeatur_noutlier.csv", skip_blank_lines=True)


def kmeans(df):
  silhouette_values=[]
  # Create a list of k values to iterate over
  k_values = range(2, 201)

  # Create an empty DataFrame to store the cluster labels
  kmeans_df = pd.DataFrame()

  # Iterate over the k values
  for k in k_values:
    # Create a k-means model with the current k value
    kmeans_model = KMeans(n_clusters=k)

    # Fit the model to the data
    kmeans_model.fit(df)
    sse_values.append(kmeans_model.inertia_)

    # Get the cluster labels
    cluster_labels = kmeans_model.predict(df)

    
    try:
            silhouette = silhouette_score(df, cluster_labels)
            silhouette_values.append(silhouette)  # Append silhouette score for k clusters
    except ValueError as e:
            print(f"Silhouette calculation failed for k={k}: {e}")
            silhouette_values.append(None)
    
    # Create a new column in the DataFrame to store the cluster labels
    kmeans_df["kmeans_" + str(k)] = cluster_labels

  # Return the DataFrame with all the cluster labels
  return kmeans_df, sse_values, silhouette_values

#kmeans labels
kmeans_df, sse_values, silhouette_values = kmeans(df)


sns.lineplot(x=range(len(sse_values)), y=sse_values, marker='o')
plt.ylabel('SSE')
plt.xlabel('k')
plt.show()


sns.lineplot(x=range(len(silhouette_values)), y=silhouette_values, marker='o')
plt.ylabel('Silhouette')
plt.xlabel('k')
plt.show()

#Entropy
def calculate_entropy(kmeans_df):
    # Initialize an empty list to store entropy values
    entropy_values = []

    # Iterate over each column representing cluster labels
    for cluster_label_column in kmeans_df.columns[1:]:
        # Extract cluster labels for the current cluster
        cluster_labels = kmeans_df[cluster_label_column]

        # Check if the cluster is empty
        if cluster_labels.empty:
            # Assign entropy as 0 for an empty cluster
            entropy = 0
        else:
            # Calculate class probabilities for non-empty clusters
            class_probabilities = cluster_labels.value_counts(normalize=True)

            # Calculate entropy using the Shannon entropy formula
            entropy = 0
            for probability in class_probabilities:
                entropy -= probability * np.log2(probability)

        entropy_values.append(entropy)

    return entropy_values

# Calculate entropy for each cluster
entropy_values = calculate_entropy(kmeans_df)

sns.lineplot(x=range(len(entropy_values)), y=entropy_values, marker='o')
plt.ylabel('Entropy')
plt.xlabel('k')
plt.show()

