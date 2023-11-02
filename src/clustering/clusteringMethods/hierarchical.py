from sklearn.preprocessing import StandardScaler

def hierarchical(df, originalDatasetColumnsToUse):
    #@RafaelUrbina
    # TODO: Implement the hierarchical method
    return df, "hierarchical"

def hierarchicalCompleteLinkage(df, originalDatasetColumnsToUse):
    #@SaraHoxha
    # TODO: Implement the hierarchicalCompleteLinkage method
    return df, "hierarchicalCompleteLinkage"

def hierarchicalSingleLink(df, originalDatasetColumnsToUse):
    #@RafaelUrbina
    # TODO: Implement the hierarchicalSingleLink method
    return df, "hierarchicalSingleLink"

# threshold: The distance threshold for clustering. Data points with distances below thisthreshold will be assigned to the same cluster.
def hierarchicalGroupAverage(df, columnsToUse, criterion, thresholds=[1, 100]):
    #@AlessandroCarella
    from scipy.cluster.hierarchy import linkage, fcluster
    import matplotlib.pyplot as plt

    #create a copy of the dataset to select only certain features
    tempDf = df.copy()
    tempDf = tempDf [columnsToUse]

    #scale the temp dataset to use the clustering algorithm
    scaler = StandardScaler()
    scaler.fit(tempDf)
    tempDfScal = scaler.transform(tempDf)

    # Perform hierarchical clustering using the 'average' linkage method
    linkage_matrix = linkage(tempDfScal, method='average')

    columnsNames = []
    for threshold in thresholds:
        # Determine the cluster assignments using a distance or number of clusters threshold
        clusters = fcluster(linkage_matrix, threshold, criterion=criterion)

        # Add the cluster assignments to the dataset
        df['hierarchicalGroupAverage' + ' ' + 'criterion=' + criterion + ' ' + 'threshold' + str (threshold)] = clusters
        columnsNames.append ('hierarchicalGroupAverage' + ' ' + 'criterion=' + criterion + ' ' + 'threshold' + str (threshold))

    return df, columnsNames
