from sklearn.cluster import KMeans, BisectingKMeans
from kmodes.kmodes import KModes
from clusteringMethods.clusteringUtility import columnAlreadyInDf, copyAndScaleDataset, getMidRunObjectFolderPath, saveMidRunObjectToFile
import os.path as path

# The parameter 'K' controls the number of clusters in the K-means algorithm, allowing you to customize the granularity of your data segmentation.
def kMeans(df, columnsToUse, Krange=[2, 3], random_state=69):
    #@AlessandroCarella
    tempDfScal = copyAndScaleDataset (df, columnsToUse)

    centersDict = {}
    columnsNames = []
    for k in range (Krange[0], Krange[1] + 1):
        newColumnName = 'kMeans=' + str (k)
        columnsNames.append (newColumnName)
        if not columnAlreadyInDf (newColumnName, df):
            # Initialize the K-means model
            kmeans = KMeans(n_clusters=k, random_state=random_state)

            # Fit the model to your data
            kmeans.fit(tempDfScal)

            # Get the cluster assignments for each data point
            labels = kmeans.labels_

            # Get the cluster centers
            centers = kmeans.cluster_centers_
            centersDict [newColumnName] = centers

            # Add the cluster assignment as a new column in the DataFrame
            df[newColumnName] = labels

    saveMidRunObjectToFile (centersDict, path.join(getMidRunObjectFolderPath(), "kMeansCenters"))

    return df, columnsNames

def bisectingKmeans(df, columnsToUse,Krange=[2, 3]):
    #@SaraHoxha
    df_subset = df[columnsToUse]
    
    # Initial cluster assignment
    df['bisectingKmeans'] = 0
    
    columnsNames = []
    for k in range (Krange[0], Krange[1] + 1):
        newColumnName = 'bisectingKmeans=' + str (k)
        columnsNames.append (newColumnName)
        if not columnAlreadyInDf (newColumnName, df):
            clusters = BisectingKMeans(n_clusters = k).fit(df_subset)

            df[newColumnName] = clusters.labels_

    return df, columnsNames

def xMeans(df, originalDatasetColumnsToUse):
    #@RafaelUrbina
    # TODO: Implement the xMeans method
    return df, ["xMeans"]

def kModes(df, columnsToUse, Krange = [2,3]):
    df_subset = df[columnsToUse]
    
    columnsNames = []
    for k in range (Krange[0], Krange[1] + 1):
        newColumnName = 'kModes=' + str (k)
        columnsNames.append (newColumnName)
        if not columnAlreadyInDf (newColumnName, df):
            clusters = KModes(n_clusters = k).fit(df_subset)
            df[newColumnName] = clusters.labels_
            
    return df, columnsNames