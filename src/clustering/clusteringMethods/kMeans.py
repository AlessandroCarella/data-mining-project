from sklearn.cluster import KMeans

from clusteringUtility import columnAlreadyInDf, copyAndScaleDataset

# The parameter 'K' controls the number of clusters in the K-means algorithm, allowing you to customize the granularity of your data segmentation.
def kMeans(df, columnsToUse, Krange=[2, 3], random_state=69):
    #@AlessandroCarella
    tempDfScal = copyAndScaleDataset (df, columnsToUse)

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

            # Add the cluster assignment as a new column in the DataFrame
            df[newColumnName] = labels

    return df, columnsNames

def bisectingKmeans(df, originalDatasetColumnsToUse):
    #@SaraHoxha
    # TODO: Implement the bisectingKmeans method
    return df, "bisectingKmeans"

def xMeans(df, originalDatasetColumnsToUse):
    #@RafaelUrbina
    # TODO: Implement the xMeans method
    return df, "xMeans"

def kModes(df, originalDatasetColumnsToUse):
    #@SaraHoxha
    # TODO: Implement the kModes method
    return df, "kModes"
