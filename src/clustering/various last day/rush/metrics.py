import numpy as np
import pandas as pd
import os.path as path 
from clusteringUtility import getMidRunObject
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler

from measuresAndVisualizationsUtils import saveDictToFile, getMetricsFolderPath, checkSubFoldersExists
from clusteringUtility import copyAndScaleDataset

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

def sse(df, clustering_columns, typeClust):
    filePath = path.join (getMetricsFolderPath (), "sse " + typeClust + ".csv")
    tempDf = copyAndScaleDataset (df, continuousFeatures)
    silhouettes = {}
    for clusteringColumn in clustering_columns:
        silhouettes [clusteringColumn] = silhouette_score (tempDf, df[clusteringColumn])
    
    checkSubFoldersExists (filePath)
    saveDictToFile(silhouettes, filePath, custom_headers=["clustering type", "value"])

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

from tqdm import tqdm

def silhouette(df, clustering_columns, typeClust):
    filePath = path.join (getMetricsFolderPath (), "silhouette " + typeClust + ".csv")
    tempDf = copyAndScaleDataset (df, continuousFeatures)
    silhouettes = {}
    for clusteringColumn in tqdm(clustering_columns):
        #try:
        silhouettes [clusteringColumn] = silhouette_score (tempDf, df[clusteringColumn])
        """except: 
            silhouettes [clusteringColumn] = -1"""
    
    checkSubFoldersExists (filePath)
    saveDictToFile(silhouettes, filePath, custom_headers=["clustering type", "value"])