import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score


os.chdir("Methods")


df = pd.read_csv("../dataset (missing + split)/R_Norm_Prueba_Dataset_nomissing_nofeatur_noutlier.csv", skip_blank_lines=True)
df_id =  pd.read_csv("../dataset (missing + split)/R_Cat_Prueba_Dataset_nomissing_nofeatur_noutlier.csv", skip_blank_lines=True)
plot_folder = '../results/plots/'


def kmeans(df):
  silhouette_values=[]
  sse_values=[]
  # Create a list of k values to iterate over
  k_values = range(2, 201)

  # Create an empty DataFrame to store the cluster labels
  kmeans_df = pd.DataFrame()
  kmeans_df['genre'] = df_id['genre']

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
plt.suptitle('sse_kmeans')
plt.savefig(plot_folder+'kmeans_sse.png')  # Specify the desired filename and extension
plt.show()

sns.lineplot(x=range(len(silhouette_values)), y=silhouette_values, marker='o')
plt.ylabel('Silhouette')
plt.xlabel('k')
plt.suptitle('silhouette_kmeans')
plt.savefig(plot_folder+'kmeans_silhouette.png')  # Specify the desired filename and extension
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
plt.suptitle('entropy_kmeans')
plt.savefig(plot_folder+'kmeans_entropy.png')  # Specify the desired filename and extension
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
kmeans_df.to_csv(folder_path_labs + 'kmeans.csv', index=False)
#SSE
df_sse.to_csv(folder_path_sse + 'kmeans.csv', index=False)
#Silhouette
df_silhouette.to_csv(folder_path_silh + 'kmeans.csv', index=False)
#Entropy
df_entropy.to_csv(folder_path_entropy + 'kmeans.csv', index=False)


grouped = kmeans_df.groupby(['genre', 'kmeans_5']).size().unstack()

# Plotting the grouped bar chart
colors = sns.color_palette('tab20', 19)
grouped.plot(kind='bar', stacked=False, color=colors)
plt.xlabel('Genre')
plt.ylabel('Count')
plt.title('Grouped Bar Chart of Genre by Kmeans')
plt.legend(title='Kmeans')
plt.savefig(plot_folder+'kmeans_grouped_bar_chart.png')  # Specify the desired filename and extension
plt.show()