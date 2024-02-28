import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans, BisectingKMeans
from sklearn.metrics import silhouette_score

from clusteringUtility import getMidRunObject, copyAndScaleDataset, saveMidRunObjectToFile
from measuresAndVisualizationsUtils import saveDictToFile
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

def kMeans(df, Krange=[2, 3], random_state=69):
    tempDfScal = copyAndScaleDataset (df, continuousFeatures)
    centersDict = {}
    clusteringColumnsNames = []
    for k in range (Krange[0], Krange[1] + 1):
        newColumnName = 'kMeans=' + str (k)
        clusteringColumnsNames.append (newColumnName)
        
        """# Initialize the K-means model
        kmeans = KMeans(n_clusters=k, random_state=random_state)

        # Fit the model to your data
        kmeans.fit(tempDfScal)

        # Get the cluster centers
        centers = kmeans.cluster_centers_
        centersDict [newColumnName] = centers

        # Add the cluster assignment as a new column in the DataFrame
        df[newColumnName] = kmeans.labels_
        """
    #saveMidRunObjectToFile (centersDict, "midRunObj/centersDict.pickle")

    return df, clusteringColumnsNames, centersDict

def calculateSSE (df, clusteringColumnsNames, centersDict):
    sse_dict = {}

    for col in clusteringColumnsNames:
        # Extract the cluster centers for the current K value
        centers = centersDict[col]

        # Extract the cluster assignments for the current K value
        labels = df[col]

        # Calculate the squared distances from each point to its assigned cluster center
        squared_distances = np.sum((df[continuousFeatures].values - centers[labels]) ** 2, axis=1)

        # Sum up the squared distances to get the SSE for the current K value
        sse = np.sum(squared_distances)

        # Save the SSE in the dictionary
        sse_dict[col] = sse

    saveDictToFile (sse_dict, "metrics/sseKMeans.csv", ["clusteringColumn", "value"])

def calculateSilhouette (df, cluster_labelsList):
    dfStd = copyAndScaleDataset (df, continuousFeatures)
    silhouettes = {}
    for cluster_label in cluster_labelsList:
        print (cluster_label)
        silhouettes [cluster_label]= silhouette_score(dfStd, df[cluster_label])
    saveDictToFile (silhouettes, "metrics/silhouetteKMeans.csv", ["clusteringColumn", "value"])

def main ():
    df = pd.read_csv("trainFinalWithClustering.csv")
    df, clusteringColumnsNames, centersDict = kMeans (df, [2, 200])
    df.to_csv("dfs/dfWithKmeansClustering.csv")
    #calculateSSE (df, clusteringColumnsNames, centersDict)
    calculateSilhouette (df, clusteringColumnsNames)

main ()
