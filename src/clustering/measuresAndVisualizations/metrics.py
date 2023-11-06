import numpy as np
import os.path as path 

from measuresAndVisualizations.measuresAndVisualizationsUtils import saveDictToFile, getMetricsFolderPath, checkSubFoldersExists

def entropy(df, clustering_columns):
    #@AlessandoCarella
    results = {}
    for clustering_type in clustering_columns:
        for column in clustering_type:
            labels = df[column]

            # Initialize a list to store cluster entropies
            cluster_entropies = []
            
            # Get the unique cluster labels
            unique_clusters = np.unique(labels)
            
            for cluster in unique_clusters:
                # Find the indices of data points in the current cluster
                cluster_indices = np.where(labels == cluster)[0]
                
                cluster_size = len(cluster_indices)
                cluster_entropy = entropy([cluster_size, len(labels) - cluster_size])
                
                cluster_entropies.append(cluster_entropy)
            
            # Calculate the weighted average entropy (considering cluster sizes)
            total_entropy = np.sum([entropy([len(np.where(labels == cluster)[0]), len(column)]) * len(np.where(labels == cluster)[0]) / len(labels) for cluster in unique_clusters])
            
            results[column] = total_entropy
    
    filePath = path.join (getMetricsFolderPath (), "entropy.csv")
    checkSubFoldersExists (filePath)
    saveDictToFile(results, filePath, custom_headers=["clustering type", "value"])

def sse(df, clustering_columns):
    #@SaraHoxha
    # TODO: Write the method sse
    for clustering_type in clustering_columns:
        pass
        # with open("""path""", 'w') as file:
        #     file.write(metric_result)

def clustersCohesionAndSeparation(df, clustering_columns):
    #@SaraHoxha
    # TODO: Write the method clustersCohesionAndSeparation
    for clustering_type in clustering_columns:
        pass
        # with open("""path""", 'w') as file:
        #     file.write(metric_result)

def silhouette(df, clustering_columns):
    #@RafaelUrbina
    # TODO: Write the method silhouette
    for clustering_type in clustering_columns:
        pass
        # with open("""path""", 'w') as file:
        #     file.write(metric_result)

def kthNeighborDistance(df, clustering_columns):
    #@AlessandroCarella
    # TODO: Write the method kthNeighborDistance
    for clustering_type in clustering_columns:
        pass
        # with open("""path""", 'w') as file:
        #     file.write(metric_result)
