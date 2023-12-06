import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_blobs
import os

os.chdir("../clustering/Methods")


df = pd.read_csv("../dataset (missing + split)/R_Norm_Prueba_Dataset_nomissing_nofeatur_noutlier.csv", skip_blank_lines=True)

#Apparently complete and ward are the best to methods to implement hierarchical clustering in our dataset still I need to check with the whole ds.

def hierarchical_clustering(data, linkage_method, n_clusters=None, distance_threshold=None):
    # Create an AgglomerativeClustering object
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method, distance_threshold=distance_threshold)

    # Fit the clustering model to the data
    clustering.fit(data)

    try:
        # Calculate the silhouette score
        silhouette_avg = silhouette_score(data, clustering.labels_)
        print(f'Silhouette Score: {silhouette_avg}')
    except ValueError as e:
        if "Number of labels is 1" in str(e):
            print("Silhouette score cannot be calculated with only one cluster.")
            silhouette_avg = None
        else:
            raise

    # Plot dendrogram
    plt.figure(figsize=(10, 4))
    dendrogram(linkage(data, method=linkage_method), labels=data.index)
    plt.xlabel('Sample index')
    plt.ylabel('Distance')
    plt.title(f'Dendrogram for Linkage: {linkage_method}')
    plt.show()

    return silhouette_avg

# Try different configurations
configurations = [
    {'linkage_method': 'ward', 'n_clusters': 3, 'distance_threshold': None},
    {'linkage_method': 'complete', 'n_clusters': 3, 'distance_threshold': None},
    {'linkage_method': 'average', 'n_clusters': 3, 'distance_threshold': None},
    {'linkage_method': 'single', 'n_clusters': 3, 'distance_threshold': None},
    {'linkage_method': 'ward', 'n_clusters': None, 'distance_threshold': 20},
    {'linkage_method': 'complete', 'n_clusters': None, 'distance_threshold': 20},
    {'linkage_method': 'average', 'n_clusters': None, 'distance_threshold': 20},
    {'linkage_method': 'single', 'n_clusters': None, 'distance_threshold': 20},
]

linkage_methods = ['ward','complete','average','single']
#distance_threshold_values = [2,3,4,5,6,7]
n_clusters = [2,5,7,15,23,33,55,77,101]
configurationswide = []
for linkage_method in linkage_methods:
    for n_cluster in n_clusters:
        #for distance_threshold in distance_threshold_values:
            config = {
                'linkage_method': linkage_method,
                'n_clusters': n_cluster,
                'distance_threshold': None
            }
            configurationswide.append(config)

for config in configurationswide:
    configurations

for config in configurationswide:
    print(f'\nConfiguration: {config}')
    hierarchical_clustering(df, config['linkage_method'], config['n_clusters'], config['distance_threshold'])
    
#hierarchical_clustering(df, 'single', 2, None)

