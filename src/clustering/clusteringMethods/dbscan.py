from sklearn.cluster import OPTICS, DBSCAN
from scipy.spatial.distance import pdist, squareform
from hdbscan import HDBSCAN

#from clusteringMethods.clusteringUtility import columnAlreadyInDf, copyAndScaleDataset
from clusteringUtility import columnAlreadyInDf, copyAndScaleDataset #just for optics testing


def dbscan(df, columnsToUse, eps = [0.5, 3], min_samples=[1, 10]):
    #@SaraHoxha
    df_subset = copyAndScaleDataset (df, columnsToUse)
    
    columnsNames = []
    for min_sample in range (min_samples[0], min_samples[1]):
        for radius in eps:#range(eps[0], eps[1]):
                newColumnName = 'dbscan' + ' ' + 'min_samples=' + str(min_sample) + ' ' + 'eps=' + str(radius)
                columnsNames.append (newColumnName)
                if not columnAlreadyInDf (newColumnName, df):
                    
                    cluster = DBSCAN(eps=radius, min_samples=min_sample, n_jobs=-1)

                    # Fit the model to your data
                    cluster.fit(df_subset)

                    # Predict the cluster labels
                    cluster_labels = cluster.labels_

                    df[newColumnName] = cluster_labels

    return df, columnsNames

#min_samples (int, optional): The minimum number of samples required to form a core 
# point.
#xi (float, optional): The core distance scaling parameter. Larger values result in 
# more emphasis on global clustering structure.
#min_cluster_size (float, optional): The minimum size of a cluster, as a fraction 
# of the dataset size. (the number is a percentage of the dataset size)
def optics(df, columnsToUse, min_samples=[1, 10], xi=[0.05], min_cluster_size=[0.05]):
    print ("optics with " + str(len(df)) + " samples")
    #@AlessandroCarella
    tempDfScal = copyAndScaleDataset (df, columnsToUse)

    # Calculate pairwise distances using pdist
    distance_vector = pdist(tempDfScal, metric='euclidean')

    # Convert the distance vector to a square distance matrix
    distance_matrix = squareform(distance_vector)
    
    from datetime import datetime
    
    columnsNames = []
    for min_sample in range (min_samples[0], min_samples[1] + 1):
        for singleXi in xi:
            for single_min_cluster_size in min_cluster_size:
                newColumnName = 'optics' + ' ' + 'min_samples=' + str(min_sample) + ' ' + 'xi=' + str(singleXi) + ' ' + 'min_cluster_size=' + str(single_min_cluster_size)
                print (newColumnName)
                print (datetime.now().time().strftime("%H:%M:%S"))
                columnsNames.append (newColumnName)
                if not columnAlreadyInDf (newColumnName, df):
                    # Create an Optics clustering model                                                               #using all possible cores
                    clustering = OPTICS(min_samples=min_sample, xi=singleXi, min_cluster_size=single_min_cluster_size, n_jobs=-1, metric='precomputed')

                    cluster_labels = clustering.fit_predict(distance_matrix)

                    # The cluster labels are stored in 'cluster_labels' variable
                    # You can add them to your DataFrame if needed
                    df[newColumnName] = cluster_labels

    return df, columnsNames

def hdbscan(df, columnsToUse, min_cluster_size=[50,200]):
    #@RafaelUrbina

    df_subset = copyAndScaleDataset (df, columnsToUse)

    columnsNames = []
    for min_sample in range(min_cluster_size[0], min_cluster_size[1] + 1):
        newColumnName = f'hdbscan min_samples{min_sample}'
        columnsNames.append(newColumnName)

        if not columnAlreadyInDf(newColumnName, df):

            # Initialize the HDBSCAN clusterer with the desired parameters
            clusterer = HDBSCAN(min_cluster_size=min_sample)

            # Fit the model to the data
            clusterer.fit(df_subset)

            # Extract cluster labels
            cluster_labels = clusterer.labels_

            df[newColumnName] = cluster_labels

    return df, columnsNames
