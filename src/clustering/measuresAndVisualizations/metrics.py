import numpy as np
import pandas as pd
import os.path as path 
from clusteringMethods.clusteringUtility import getMidRunObject
from sklearn.metrics import silhouette_score, davies_bouldin_score
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
        dfContinuous = df[continuousFeatures]
        for clusteringType in clustering_columns:
            for column in clusteringType:
                X = df[column].values.reshape(-1, 1)
                
                centersDict = getMidRunObject ("kMeansCenters")
                cluster_centers = centersDict[column]
                
                dfContinuous['centroid'] = df[column].apply(lambda x: cluster_centers[x])
                sse = np.sum((X - dfContinuous['centroid'].values.reshape(-1, 1)) ** 2)
                dfContinuous.drop(['centroid'], axis=1, inplace=True)
                sse_results[column] = sse
        
        checkSubFoldersExists (filePath)
        saveDictToFile(sse_results, filePath, custom_headers=["clustering type", "sse value"])

def clustersCohesionAndSeparation(df, clustering_columns):
    #@SaraHoxha
    """filePath = path.join(getMetricsFolderPath(), "cohesion_and_separation.csv")
    if not path.exists(filePath):
        cohesion_separation_results = {}
        dfContinuous = df[continuousFeatures]
        for clustering_type in clustering_columns:
            for column in clustering_type:
            # Calculate Cohesion
                X = dfContinuous[column].values.reshape(-1, 1)
                
                centersDict= getMidRunObject ("kMeansCenters")
                cluster_centers = centersDict[column]
                
                dfContinuous['centroid'] = dfContinuous[column].apply(lambda x: cluster_centers[x])
                cohesion = np.sum((X - dfContinuous['centroid'].values.reshape(-1, 1)) ** 2)

            # Calculate Separation for each cluster
            cluster_davies_bouldin_scores = []
            unique_clusters = np.unique(df[column])    
            features = df[continuousFeatures].values
            clusters = dfContinuous[column].values
            davies_bouldin = davies_bouldin_score(features, clusters)

            metric_result = {
                #"clustering type": column,
                "cohesion": cohesion,
                "separation": davies_bouldin
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

