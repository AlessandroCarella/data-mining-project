import numpy as np
import pandas as pd
import os.path as path 
from sklearn.metrics import silhouette_score, pairwise_distances_argmin_min, davies_bouldin_score
from scipy.stats import entropy

from measuresAndVisualizations.measuresAndVisualizationsUtils import saveDictToFile, getMetricsFolderPath, checkSubFoldersExists
from clusteringMethods.clusteringUtility import copyAndScaleDataset
categoricalFeatures = [
    "name",
    "explicit",
    "artists",
    "album_name",
    "key",
    "genre",
    "time_signature"
]
categoricalFeaturesForPlotting = [
    #"name",
    #"explicit",
    #"artists",
    #"album_name",
    "key",
    "genre",
    "time_signature"
]
continuousFeatures = [
    "duration_ms",
    "popularity",
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "n_beats"
]


def entropyMetric(df, clustering_columns):
    #@AlessandoCarella
    filePath = path.join (getMetricsFolderPath (), "entropy.csv")
    if not path.exists (filePath):
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
        
        checkSubFoldersExists (filePath)
        saveDictToFile(results, filePath, custom_headers=["clustering type", "value"])

def sse(df, clustering_columns):
    #@SaraHoxha
    #sse_results = []
    #changing to dict cuz of the saveDictToFile method you call at the end
    filePath = path.join(getMetricsFolderPath(), "sse.csv")
    if not path.exists (filePath):
        sse_results = {}
        """dfContinuous = df[continuousFeatures]
        for clusteringType in clustering_columns:
            for column in clusteringType:
                cluster_centers = dfContinuous.groupby(column).mean()
                cluster_assignments, _ = pairwise_distances_argmin_min(df.drop(columns=clustering_columns).values, cluster_centers.values)
                sse = np.sum((df.drop(columns=clustering_columns).values - cluster_centers.values[cluster_assignments]) ** 2)
                #sse_results.append((column, sse))
                sse_results[column] = sse
        
        checkSubFoldersExists (filePath)
        saveDictToFile(sse_results, filePath, custom_headers=["clustering type", "sse value"])"""

def clustersCohesionAndSeparation(df, clustering_columns):
    #@SaraHoxha
    filePath = path.join(getMetricsFolderPath(), "cohesion_and_separation.csv")
    if not path.exists(filePath):
        cohesion_separation_results = {}
        """dfContinuous = df[continuousFeatures]
        for clustering_type in clustering_columns:
            for column in clustering_type:
            # Calculate Cohesion
                cluster_centers = dfContinuous.groupby(column).mean()
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
                #"clustering type": column,
                "cohesion": cohesion,
                "separation": cluster_davies_bouldin_scores
            }

            cohesion_separation_results[column] = metric_result

        checkSubFoldersExists(filePath)
        custom_headers = ["clustering type", "cohesion (SSE)", "separation (Davies-Bouldin Index)"]
        saveDictToFile(cohesion_separation_results, filePath, custom_headers=custom_headers)"""

def silhouette(df, clustering_columns):
    #@RafaelUrbina
    filePath = path.join (getMetricsFolderPath (), "silhouette.csv")
    if not path.exists (filePath):
        tempDfScal = copyAndScaleDataset (df, continuousFeatures)
        tempDfScalCat = df.copy() [categoricalFeatures]

        silhouettes = {}
        for clusteringType in clustering_columns:#[["", ""],["", ""]]
            for clusteringColumn in clusteringType:#["", ""]
                labels = df[clusteringColumn]
                if "modes" in clustering_columns:
                    silhouette = silhouette_score(tempDfScalCat[categoricalFeatures], df[clusteringColumn])
                    silhouette[clusteringColumn] = silhouette
                if "hdbscan" or "dbscan" or "optics" in clustering_columns:
                    pass
                #    silhouette = silhouette_score(tempDfScal[tempDfScal[clusteringType] != -1][continuousFeatures], df[clusteringColumn][df[clusteringColumn] != -1]) #ASK if this could work
                #    silhouette[clusteringColumn] = silhouette
                else:
                    silhouette = silhouette_score(tempDfScal[continuousFeatures], labels)
                    silhouettes[clusteringColumn] = silhouette
        
        df_silhouettes = pd.DataFrame.from_dict(silhouettes, orient="index", columns=["Silhouette Score"])
        
        checkSubFoldersExists (filePath)
        saveDictToFile(silhouettes, filePath, custom_headers=["clustering type", "value"])

