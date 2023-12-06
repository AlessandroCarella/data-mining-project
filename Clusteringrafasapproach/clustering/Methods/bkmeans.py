import pandas as pd
import numpy as np
from sklearn.cluster import BisectingKMeans
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score

os.chdir("../clustering/Methods")


df = pd.read_csv("../dataset (missing + split)/R_Norm_Prueba_Dataset_nomissing_nofeatur_noutlier.csv", skip_blank_lines=True)

def bisecting_kmeans(df):
    sse_values = []
    silhouette_values = []
    k_values = range(2, 201)

    bkmeans_df = pd.DataFrame()

    for k in k_values:
        bkmeans_model = BisectingKMeans(n_clusters=k)
        bkmeans_model.fit(df)
        sse_values.append(bkmeans_model.inertia_)

        cluster_labels = bkmeans_model.predict(df)
        bkmeans_df["bkmeans_" + str(k)] = cluster_labels

        silhouette = silhouette_score(df, cluster_labels)
        silhouette_values.append(silhouette)

    return bkmeans_df, sse_values, silhouette_values

bkmeans_df, sse_values, silhouette_values = bisecting_kmeans(df)


sns.lineplot(x=range(len(sse_values)), y=sse_values, marker='o')
plt.ylabel('SSE')
plt.xlabel('k')
plt.show()


sns.lineplot(x=range(len(silhouette_values)), y=silhouette_values, marker='o')
plt.ylabel('Silhouette')
plt.xlabel('k')
plt.show()

#Entropy
def calculate_entropy(bkmeans_df):
    # Initialize an empty list to store entropy values
    entropy_values = []

    # Iterate over each column representing cluster labels
    for cluster_label_column in bkmeans_df.columns[1:]:
        # Extract cluster labels for the current cluster
        cluster_labels = bkmeans_df[cluster_label_column]

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
entropy_values = calculate_entropy(bkmeans_df)

sns.lineplot(x=range(len(entropy_values)), y=entropy_values, marker='o')
plt.ylabel('Entropy')
plt.xlabel('k')
plt.show()