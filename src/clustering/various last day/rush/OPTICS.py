from dbscan import dbscan, optics, hdbscan
import pandas as pd
import os.path as path
from sklearn.metrics import silhouette_score

from measuresAndVisualizationsUtils import getMetricsFolderPath, checkSubFoldersExists, saveDictToFile
from clusteringUtility import copyAndScaleDataset

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
def silhouette (df, clustering_columns, typeClust):
    filePath = path.join (getMetricsFolderPath (), "silhouette " + typeClust + ".csv")
    tempDf = copyAndScaleDataset (df, continuousFeatures)
    silhouettes = {}
    for clusteringColumn in clustering_columns:
        try:
            silhouettes [clusteringColumn] = silhouette_score (tempDf, df[clusteringColumn])
        except: 
            silhouettes [clusteringColumn] = -1
    
    checkSubFoldersExists (filePath)
    saveDictToFile(silhouettes, filePath, custom_headers=["clustering type", "value"])

def main ():
    df = pd.read_csv ("Dataset_nomissing_nofeatur_noutlier_noinconsistencies with optics.csv")
    optics_columns = [col for col in df.columns if 'optics' in col.lower()]

    silhouette (df, optics_columns, "optics")

main()