from scipy.cluster.hierarchy import linkage, fcluster
from clusteringMethods.clusteringUtility import columnAlreadyInDf, copyAndScaleDataset

def hierarchical(df, originalDatasetColumnsToUse):
    #@RafaelUrbina
    # TODO: Implement the hierarchical method
    return df, ["hierarchical"]

def hierarchicalCompleteLinkage(df, columnsToUse, linkage_thresholds=[1, 100]):
    #@SaraHoxha
    df_subset = df[columnsToUse]

    # Calculate the linkage matrix using complete linkage
    linkage_matrix = linkage(df_subset, method='complete')
    
    columnsNames=[]
    for threshold in linkage_thresholds:
        newColumnName = 'hierarchicalCompleteLinkage' + ' '  + 'threshold' + str (threshold)
        columnsNames.append (newColumnName)
        if not columnAlreadyInDf (newColumnName, df):
            # Determine the cluster assignments using a distance or number of clusters threshold
            clusters = fcluster(linkage_matrix, threshold, criterion='distance')
            # Add the cluster assignments to the dataset
            df[newColumnName] = clusters
            
    return df, columnsNames


def hierarchicalSingleLink(df, originalDatasetColumnsToUse):
    #@RafaelUrbina
    # TODO: Implement the hierarchicalSingleLink method
    return df, ["hierarchicalSingleLink"]

# threshold: The distance threshold for clustering. Data points with distances below thisthreshold will be assigned to the same cluster.
def hierarchicalGroupAverage(df, columnsToUse, criterion, thresholds=[1, 100]):
    #@AlessandroCarella
    tempDfScal = copyAndScaleDataset (df, columnsToUse)

    # Perform hierarchical clustering using the 'average' linkage method
    linkage_matrix = linkage(tempDfScal, method='average')

    columnsNames = []
    for threshold in thresholds:
        newColumnName = 'hierarchicalGroupAverage' + ' ' + 'criterion=' + criterion + ' ' + 'threshold' + str (threshold)
        columnsNames.append (newColumnName)
        if not columnAlreadyInDf (newColumnName, df):
            # Determine the cluster assignments using a distance or number of clusters threshold
            clusters = fcluster(linkage_matrix, threshold, criterion=criterion)

            # Add the cluster assignments to the dataset
            df[newColumnName] = clusters

    return df, columnsNames
