from scipy.cluster.hierarchy import linkage, fcluster
import os.path as path

from clusteringMethods.clusteringUtility import columnAlreadyInDf, copyAndScaleDataset, getMidRunObjectFolderPath, saveMidRunObjectToFile

def hierarchicalCentroidLinkage(df, columnsToUse,linkage_thresholds=[1, 100]):
    #@RafaelUrbina
    tempDfScal = copyAndScaleDataset (df, columnsToUse)

    # Calculate the linkage matrix using centroid linkage
    linkage_matrix = linkage(tempDfScal, method='centroid')
    saveMidRunObjectToFile (linkage_matrix, path.join(getMidRunObjectFolderPath(), "(just object) hierarchicalCentroidLinkageLinkageMatrix"))
    
    columnsNames = []
    for threshold in range (linkage_thresholds[0], linkage_thresholds[1]):
        newColumnName = 'hierarchicalCentroidLinkage' + ' '  + 'threshold' + str(threshold)
        columnsNames.append(newColumnName)
        if not columnAlreadyInDf(newColumnName, df):
            # Determine the cluster assignments using a distance or number of clusters threshold
            clusters = fcluster(linkage_matrix, threshold, criterion='distance')
            # Add the cluster assignments to the dataset
            df[newColumnName] = clusters

    return df, columnsNames

def hierarchicalCompleteLinkage(df, columnsToUse, linkage_thresholds=[1, 100]):
    #@SaraHoxha
    df_subset = df[columnsToUse]

    # Calculate the linkage matrix using complete linkage
    linkage_matrix = linkage(df_subset, method='complete')
    saveMidRunObjectToFile (linkage_matrix, path.join(getMidRunObjectFolderPath(), "(just object) hierarchicalCompleteLinkageLinkageMatrix"))

    columnsNames=[]
    for threshold in range (linkage_thresholds[0], linkage_thresholds[1]):
        newColumnName = 'hierarchicalCompleteLinkage' + ' '  + 'threshold' + str (threshold)
        columnsNames.append (newColumnName)
        if not columnAlreadyInDf (newColumnName, df):
            # Determine the cluster assignments using a distance or number of clusters threshold
            clusters = fcluster(linkage_matrix, threshold, criterion='distance')
            # Add the cluster assignments to the dataset
            df[newColumnName] = clusters
            
    return df, columnsNames


def hierarchicalSingleLinkage(df, columnsToUse, linkage_thresholds=[1, 100]):
    #@RafaelUrbina
    tempDfScal = copyAndScaleDataset (df, columnsToUse)

    # Calculate the linkage matrix using single linkage
    linkage_matrix = linkage(tempDfScal, method='single')
    saveMidRunObjectToFile (linkage_matrix, path.join(getMidRunObjectFolderPath(), "(just object) hierarchicalSingleLinkageLinkageMatrix"))
    
    columnsNames = []
    for threshold in range (linkage_thresholds[0], linkage_thresholds[1]):
        newColumnName = 'hierarchicalSingleLinkage' + ' '  + 'threshold' + str(threshold)
        columnsNames.append(newColumnName)
        if not columnAlreadyInDf(newColumnName, df):
            # Determine the cluster assignments using a distance or number of clusters threshold
            clusters = fcluster(linkage_matrix, threshold, criterion='distance')
            # Add the cluster assignments to the dataset
            df[newColumnName] = clusters
            
    return df, columnsNames

# threshold: The distance threshold for clustering. Data points with distances below thisthreshold will be assigned to the same cluster.
def hierarchicalGroupAverage(df, columnsToUse, criterion, linkage_thresholds=[1, 100]):
    #@AlessandroCarella
    tempDfScal = copyAndScaleDataset (df, columnsToUse)

    # Perform hierarchical clustering using the 'average' linkage method
    linkage_matrix = linkage(tempDfScal, method='average')
    saveMidRunObjectToFile (linkage_matrix, path.join(getMidRunObjectFolderPath(), "(just object) hierarchicalGroupAverageLinkageMatrix"))
    
    columnsNames = []
    for threshold in range (linkage_thresholds[0], linkage_thresholds[1]):
        newColumnName = 'hierarchicalGroupAverage' + ' ' + 'threshold' + str (threshold) + ' ' + 'criterion=' + criterion
        columnsNames.append (newColumnName)
        if not columnAlreadyInDf (newColumnName, df):
            # Determine the cluster assignments using a distance or number of clusters threshold
            clusters = fcluster(linkage_matrix, threshold, criterion=criterion)

            # Add the cluster assignments to the dataset
            df[newColumnName] = clusters

    return df, columnsNames
