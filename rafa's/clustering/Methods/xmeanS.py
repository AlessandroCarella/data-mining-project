from pyclustering.cluster.xmeans import xmeans
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy, warnings
numpy.warnings = warnings
os.chdir("Methods")


df = pd.read_csv("../dataset (missing + split)/R_Norm_Prueba_Dataset_nomissing_nofeatur_noutlier.csv", skip_blank_lines=True)
df_id =  pd.read_csv("../dataset (missing + split)/R_Cat_Prueba_Dataset_nomissing_nofeatur_noutlier.csv", skip_blank_lines=True)
plot_folder = '../results/plots/'

def calculate_sse(model, data):
    cluster_labels = model.predict(data)
    unique_clusters = np.unique(cluster_labels)
    sse_values = []

    for cluster_label in unique_clusters:
        cluster_data = data[cluster_labels == cluster_label]

        if len(cluster_data) > 1:
            cluster_center = np.mean(cluster_data, axis=0)  # Calculate mean as cluster center
        else:
            cluster_center = cluster_data[0]  # Single point as center

        # Calculate squared distances from data points to the cluster center
        squared_distances = np.sum((cluster_data - cluster_center) ** 2, axis=1)

        # Calculate SSE for the cluster
        cluster_sse = np.sum(squared_distances)
        sse_values.append(cluster_sse)

    total_sse = sum(sse_values)
    return total_sse

# Function to perform XMeans clustering and compute metrics
def xmeans_clustering(data):
    silhouette_scores = []
    entropy_values = []
    sse_values = []
    cluster_results = pd.DataFrame()
    cluster_results['genre'] = df_id['genre']

    for k in range(2, 201):  # Change the range as needed
        xmeans_instance = xmeans(data, kmax=k, random_state=42)
        xmeans_instance.process()
        clusters = xmeans_instance.get_clusters()
        cluster_labels = np.zeros(len(data))

        for cluster_index, cluster in enumerate(clusters):
            cluster_labels[cluster] = cluster_index

        cluster_results[f'xmeans_{k}'] = cluster_labels

        # Calculate silhouette score
        silhouette = silhouette_score(data, cluster_labels)
        silhouette_scores.append(silhouette)

        # Calculate SSE
        sse = calculate_sse(xmeans_instance, data)
        sse_values.append(sse)

        # Calculate entropy (you need to define your own entropy function based on your data)
        #entropy = calculate_entropy(cluster_labels, data)
        #entropy_values.append(entropy)

    return cluster_results, silhouette_scores, entropy_values, sse_values

# Example usage
# Assuming 'your_data' is your dataset in numpy array format
cluster_results, silhouette_scores, entropy_values, sse_values = xmeans_clustering(df)

cluster_results
sns.lineplot(x=range(len(sse_values)), y=sse_values, marker='o')
plt.ylabel('SSE')
plt.xlabel('k')
plt.suptitle('see_xmeans')
plt.savefig(plot_folder+'xmeans_see.png')  # Specify the desired filename and extension
plt.show()

sns.lineplot(x=range(len(silhouette_scores)), y=silhouette_scores, marker='o')
plt.ylabel('Silhouette')
plt.xlabel('k')
plt.suptitle('silhouette_xmeans')
plt.savefig(plot_folder+'xmeans_silhouette.png')  # Specify the desired filename and extension
plt.show()

#Let's save the results
folder_path_labs = ('../results/labels/')
folder_path_sse = ('../results/sse/')
folder_path_silh = ('../results/silhouette/')
folder_path_entropy = ('../results/entropy/')

xmeans_df = pd.DataFrame(cluster_results)
df_sse = pd.DataFrame(sse_values, columns=['sse'])
df_silhouette = pd.DataFrame(silhouette_scores, columns=['silhouette'])
df_entropy = pd.DataFrame(entropy_values, columns=['entropy'])

#Labels
xmeans_df.to_csv(folder_path_labs + 'xmeans.csv', index=False)
#SSE
df_sse.to_csv(folder_path_sse + 'xmeans.csv', index=False)
#Silhouette
df_silhouette.to_csv(folder_path_silh + 'xmeans.csv', index=False)
#Entropy
df_entropy.to_csv(folder_path_entropy + 'xmeans.csv', index=False)


grouped = xmeans_df.groupby(['genre','xmeans_11']).size().unstack()

# Plotting the grouped bar chart
colors = sns.color_palette('tab20', 19)
grouped.plot(kind='bar', stacked=False, color=colors)
plt.xlabel('Genre')
plt.ylabel('Count')
plt.title('Grouped Bar Chart of Genre by xmeans')
plt.legend(title='xmean')
plt.savefig(plot_folder+'xmeans_grouped_bar_chart.png')  # Specify the desired filename and extension
plt.show()