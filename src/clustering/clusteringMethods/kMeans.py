from sklearn.cluster import KMeans, BisectingKMeans
from kmodes.kmodes import KModes
from clusteringMethods.clusteringUtility import columnAlreadyInDf, copyAndScaleDataset, getMidRunObjectFolderPath, saveMidRunObjectToFile
import os.path as path
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer

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

    if centersDict != {}:
        saveMidRunObjectToFile (centersDict, path.join(getMidRunObjectFolderPath(), "kMeansCenters"))

    return df, columnsNames

def bisectingKmeans(df, columnsToUse,Krange=[2, 3]):
    #@SaraHoxha
    df_subset = copyAndScaleDataset (df, columnsToUse)
    
    # Initial cluster assignment
    #df['bisectingKmeans'] = 0
    
    columnsNames = []
    for k in range (Krange[0], Krange[1] + 1):
        newColumnName = 'bisectingKmeans=' + str (k)
        columnsNames.append (newColumnName)
        if not columnAlreadyInDf (newColumnName, df):
            clusters = BisectingKMeans(n_clusters = k).fit(df_subset)

            df[newColumnName] = clusters.labels_
            
    return df, columnsNames

# Function to perform X-Means clustering and add cluster assignments
def xMeans(df, columnsToUse, Krange=[2, 3], random_state=69):
    #@RafaelUrbina
    data = df[columnsToUse].values
    
    columnsNames = []
    for k in range(Krange[0], Krange[1] + 1):
        newColumnName = 'xMeans=' + str(k)
        columnsNames.append(newColumnName)

        if not columnAlreadyInDf (newColumnName, df):
            # Initialize the X-Means instance with K-Means initialization
            initial_centers = kmeans_plusplus_initializer(data, k).initialize()
            xmeans_instance = xmeans(data, initial_centers, kmax=k, ccore=False)

            # Run X-Means
            xmeans_instance.process()

            # Get cluster centers and cluster assignments
            cluster_centers = xmeans_instance.get_centers()
            clusters = xmeans_instance.get_clusters()

            # Determine the cluster assignments for each data point
            cluster_assignments = [-1] * len(data)
            for i, cluster_indices in enumerate(clusters):
                for idx in cluster_indices:
                    cluster_assignments[idx] = i

            # Add the cluster assignment as a new column in the DataFrame
            df[newColumnName] = cluster_assignments

    return df, columnsNames

def kModes(df, columnsToUse, Krange = [2,3]):
    #@SaraHoxha
    #df_subset = copyAndScaleDataset (df, columnsToUse)
    tempDf = df.copy()
    tempDf = tempDf [columnsToUse]

    columnsNames = []
    for k in range (Krange[0], Krange[1] + 1):
        newColumnName = 'kModes=' + str (k)
        columnsNames.append (newColumnName)
        if not columnAlreadyInDf (newColumnName, df):
            clusters = KModes(n_clusters = k).fit(tempDf)#(df_subset)
            df[newColumnName] = clusters.labels_
            
    return df, columnsNames