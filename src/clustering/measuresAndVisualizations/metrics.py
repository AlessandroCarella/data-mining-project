import numpy as np
import pandas as pd
import os.path as path 
from sklearn.metrics import silhouette_score, pairwise_distances_argmin_min, davies_bouldin_score
from measuresAndVisualizations.measuresAndVisualizationsUtils import saveDictToFile, getMetricsFolderPath, checkSubFoldersExists
from clusteringMethods.clusteringUtility import copyAndScaleDataset, getMidRunObject
from clusteringMain import continuousFeatures, categoricalFeatures

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
    sse_results = []
    for clusteringType in clustering_columns:
        for column in clusteringType:
            cluster_centers = df.groupby(column).mean()
            cluster_assignments, _ = pairwise_distances_argmin_min(df.drop(columns=clustering_columns).values, cluster_centers.values)
            sse = np.sum((df.drop(columns=clustering_columns).values - cluster_centers.values[cluster_assignments]) ** 2)
            sse_results.append((column, sse))
        
    filePath = path.join (getMetricsFolderPath (), "sse.csv")
    checkSubFoldersExists (filePath)
    saveDictToFile(sse_results, filePath, custom_headers=["clustering type", "sse value"])

def clustersCohesionAndSeparation(df, clustering_columns):
    #@SaraHoxha
    cohesion_separation_results = []

    for clustering_type in clustering_columns:
        for column in clustering_type:
        # Calculate Cohesion
            cluster_centers = df.groupby(column).mean()
            cluster_assignments, _ = pairwise_distances_argmin_min(df.drop(columns=clustering_columns).values, cluster_centers.values)
            sse = np.sum((df.drop(columns=clustering_columns).values - cluster_centers.values[cluster_assignments]) ** 2)
            cohesion = sse

        # Calculate Separation for each cluster
        cluster_davies_bouldin_scores = []
        unique_clusters = np.unique(df[column])    
        for cluster in unique_clusters:
            cluster_data = df[df[column] == cluster]
            separation = davies_bouldin_score(cluster_data.drop(columns=clustering_columns).values)
            cluster_davies_bouldin_scores.append(separation)

        metric_result = {
            "clustering type": column,
            "cohesion": cohesion,
            "separation": cluster_davies_bouldin_scores
        }

        cohesion_separation_results.append(metric_result)

    filePath = path.join(getMetricsFolderPath(), "cohesion_and_separation.csv")
    checkSubFoldersExists(filePath)
    custom_headers = ["clustering type", "cohesion (SSE)", "separation (Davies-Bouldin Index)"]
    saveDictToFile(cohesion_separation_results, filePath, custom_headers=custom_headers)


def silhouette(df, clustering_columns):
    tempDfScal = copyAndScaleDataset (df, continuousFeatures)
    tempDfScalCat = copyAndScaleDataset (df, categoricalFeatures)
    #@RafaelUrbina
    # TODO: Write the method silhouette
    silhouettes = {}
    for clusteringType in clustering_columns:#[["", ""],["", ""]]
        for clusteringColumn in clusteringType:#["", ""]
            if "modes" in clustering_columns:
                labels = getMidRunObject ("(just object) Labels "+clusteringColumn)
                silhouette = silhouette_score(tempDfScalCat[categoricalFeatures], labels)
                #silhouette = silhouette_score(tempDfScalCat[tempDfScalCat[clusteringType] != -1][categoricalFeatures], labels[labels != -1]) #ASK if this could work
                silhouette[clusteringColumn] = silhouette
            if "hdscan" or "dbscan" or "optics" in clustering_columns:
                labels = getMidRunObject ("(just object) Labels "+clusteringColumn)
                silhouette = silhouette_score(tempDfScal[tempDfScal[clusteringType] != -1][continuousFeatures], labels[labels != -1]) #ASK if this could work
                #silhouette_score(X_minmax[dbscan.labels_ != -1], dbscan.labels_[dbscan.labels_ != -1]))
                silhouette[clusteringColumn] = silhouette
            else:
                labels = getMidRunObject ("(just object) Labels"+ +clusteringColumn)
                silhouette = silhouette_score(tempDfScal[continuousFeatures], labels)
                #silhouette = silhouette_score(tempDfScal[tempDfScal[clusteringType] != -1][continuousFeatures], labels[labels != -1]) #ASK if this could work
                silhouettes[clusteringColumn] = silhouette
    
    df_silhouettes = pd.DataFrame.from_dict(silhouettes, orient="index", columns=["Silhouette Score"])

    filePath = path.join (getMetricsFolderPath (), "silhouette.csv")
    
    checkSubFoldersExists (filePath)
    saveDictToFile(silhouettes, filePath, custom_headers=["clustering type", "value"])

    

            


def kthNeighborDistance(df, clustering_columns):
    #@AlessandroCarella
    # TODO: Write the method kthNeighborDistance
    for clustering_type in clustering_columns:
        pass
        # with open("""path""", 'w') as file:
        #     file.write(metric_result)
