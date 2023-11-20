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

def bisectingKMeans(df, Krange=[2, 3], random_state=69):
    tempDfScal = copyAndScaleDataset(df, continuousFeatures)
    centersDict = {}
    clusteringColumnsNames = []

    for k in range(Krange[0], Krange[1] + 1):
        newColumnName = 'bisectingKmeans=' + str(k)
        print(newColumnName)
        clusteringColumnsNames.append(newColumnName)

        """# Initialize the Bisecting K-means model
        bkmeans = BisectingKMeans(n_clusters=k, random_state=random_state)

        # Fit the model to your data
        bkmeans.fit(tempDfScal)

        # Get the cluster centers
        centers = bkmeans.cluster_centers_
        centersDict[newColumnName] = centers

        # Add the cluster assignment as a new column in the DataFrame
        df[newColumnName] = bkmeans.labels_
        """

    #saveMidRunObjectToFile(centersDict, "midRunObj/centersDict_bisectingKMeans.pickle")

    return df, clusteringColumnsNames, centersDict

def calculateSSE(df, clusteringColumnsNames, centersDict):
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

    saveDictToFile(sse_dict, "metrics/sseBisectingKMeans.csv", ["clusteringColumn", "value"])

def calculateSilhouette (df, cluster_labelsList):
    dfStd = copyAndScaleDataset (df, continuousFeatures)
    silhouettes = {}
    for cluster_label in cluster_labelsList:
        print (cluster_label)
        silhouettes [cluster_label]= silhouette_score(dfStd, df[cluster_label])
    saveDictToFile (silhouettes, "metrics/silhouetteBisectingKMeans.csv", ["clusteringColumn", "value"])

def main():
    df = pd.read_csv("trainFinalWithClustering.csv")
    df, clusteringColumnsNames, centersDict = bisectingKMeans(df, [2, 200])
    df.to_csv("dfs/dfWithBisectingKMeansClustering.csv")
    #calculateSSE(df, clusteringColumnsNames, centersDict)
    calculateSilhouette (df, clusteringColumnsNames)

main()
