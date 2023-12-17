import pandas as pd
import numpy as np
from kmodes.kmodes import KModes
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from collections import Counter


os.chdir("Methods")


df = pd.read_csv("../dataset (missing + split)/R_Cat_Prueba_Dataset_nomissing_nofeatur_noutlier.csv", skip_blank_lines=True)
plot_folder = '../results/plots/'

def calculate_sse(kmeans_model, data):
    cluster_labels = kmeans_model.fit_predict(data)
    unique_clusters = np.unique(cluster_labels)
    sse_values = []
    kmodes_df['genre'] = df['genre']

    for cluster_index in unique_clusters:
        cluster_data = data[cluster_labels == cluster_index]
        cluster_centroid = kmeans_model.cluster_centroids_[cluster_index]

        # Calculate squared distances from data points to centroid
        squared_distances = np.sum((cluster_data - cluster_centroid)**2, axis=1)

        # Calculate SSE for the cluster
        cluster_sse = np.sum(squared_distances)

        sse_values.append(cluster_sse)

    
    sse_values = sum(sse_values)
    print(sse_values)
    return sse_values


def kmodes(df):

    kmodes_df = pd.DataFrame()
    silhouette_values = []
    sse_values = []
    k_values = range(2, 201)  # Example: evaluating up to 10 clusters
    entropy_values = []

    for k in k_values:
        kmodes_model = KModes(n_clusters=k,verbose=0)
        cluster_labels = kmodes_model.fit_predict(df, categorical=[0,1,2,3])
        print(cluster_labels)
        kmodes_df["kmodes_" + str(k)] = cluster_labels

        # Calculate SSE for each cluster
        sse_value = calculate_sse(kmodes_model, df)
        sse_values.append(sse_value)
        silhouette = silhouette_score(df, cluster_labels)  # silhouette might not work directly with categorical data
        silhouette_values.append(silhouette)
        print(k)
    return  kmodes_df, silhouette_values, sse_values

kmodes_df, silhouette_values, sse_values = kmodes(df)

sns.lineplot(x=range(len(sse_values)), y=sse_values, marker='o')
plt.ylabel('SSE')
plt.xlabel('k')
plt.suptitle('sse_kmodes')
plt.savefig(plot_folder+'kmodes_sse.png')  # Specify the desired filename and extension
plt.show()

sns.lineplot(x=range(len(silhouette_values)), y=silhouette_values, marker='o')
plt.ylabel('Silhouette')
plt.xlabel('k')
plt.suptitle('silhouette_kmodes')
plt.savefig(plot_folder+'kmodes_silhouette.png')  # Specify the desired filename and extension
plt.show()


#Entropy
def calculate_entropy(kmodes_df):
    # Initialize an empty list to store entropy values
    entropy_values = []

    # Iterate over each column representing cluster labels
    for cluster_label_column in kmodes_df.columns[1:]:
        # Extract cluster labels for the current cluster
        cluster_labels = kmodes_df[cluster_label_column]

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
entropy_values = calculate_entropy(kmodes_df)

sns.lineplot(x=range(len(entropy_values)), y=entropy_values, marker='o')
plt.ylabel('Entropy')
plt.xlabel('k')
plt.suptitle('entropy_kmodes')
plt.savefig(plot_folder+'kmodes_entropy.png')  # Specify the desired filename and extension
plt.show()

#Let's save the results
folder_path_labs = ('../results/labels/')
folder_path_sse = ('../results/sse/')
folder_path_silh = ('../results/silhouette/')
folder_path_entropy = ('../results/entropy/')


df_sse = pd.DataFrame(sse_values, columns=['sse'])
df_silhouette = pd.DataFrame(silhouette_values, columns=['silhouette'])
df_entropy = pd.DataFrame(entropy_values, columns=['entropy'])

#Labels
kmodes_df.to_csv(folder_path_labs + 'kmodes.csv', index=False)
#SSE
df_sse.to_csv(folder_path_sse + 'kmodes.csv', index=False)
#Silhouette
df_silhouette.to_csv(folder_path_silh + 'kmodes.csv', index=False)
#Entropy
df_entropy.to_csv(folder_path_entropy + 'kmodes.csv', index=False)


grouped = kmodes_df.groupby(['genre', 'kmodes_4']).size().unstack()

# Plotting the grouped bar chart
colors = sns.color_palette('tab20', 19)
grouped.plot(kind='bar', stacked=False, color=colors)
plt.xlabel('Genre')
plt.ylabel('Count')
plt.title('Grouped Bar Chart of Genre by kmodes')
plt.legend(title='kmodes')
plt.savefig(plot_folder+'kmodes_grouped_bar_chart.png')  # Specify the desired filename and extension
plt.show()