from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler

# The parameter 'K' controls the number of clusters in the K-means algorithm, allowing you to customize the granularity of your data segmentation.
def kMeans(df, columnsToUse, Krange=[2, 3]):
    #@AlessandroCarella
    #create a copy of the dataset to select only certain features
    tempDf = df.copy()
    tempDf = tempDf [columnsToUse]

    #scale the temp dataset to use the clustering algorithm
    scaler = StandardScaler()
    scaler.fit(tempDf)
    tempDfScal = scaler.transform(tempDf)

    columnsNames = []
    for k in range (Krange[0], Krange[1] + 1):
        # Initialize the K-means model
        kmeans = KMeans(n_clusters=k)

        # Fit the model to your data
        kmeans.fit(tempDfScal)

        # Get the cluster assignments for each data point
        labels = kmeans.labels_

        # Get the cluster centers
        centers = kmeans.cluster_centers_

        # Add the cluster assignment as a new column in the DataFrame
        df['kMeans=' + str (k)] = labels
        columnsNames.append ('kMeans=' + str (k))

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
