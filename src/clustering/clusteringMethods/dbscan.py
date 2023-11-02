
def dbscan(df, originalDatasetColumnsToUse):
    #@SaraHoxha
    # TODO: Implement the dbscan method
    return df, "dbscan"

#min_samples (int, optional): The minimum number of samples required to form a core 
# point.
#xi (float, optional): The core distance scaling parameter. Larger values result in 
# more emphasis on global clustering structure.
#min_cluster_size (float, optional): The minimum size of a cluster, as a fraction 
# of the dataset size. (the number is a percentage of the dataset size)
def optics(df, columnsToUse, min_samples=[1, 10], xi=[0.05], min_cluster_size=[0.05]):
    #@AlessandroCarella
    import numpy as np
    import pandas as pd
    from sklearn.cluster import OPTICS
    from sklearn.preprocessing import StandardScaler

    #create a copy of the dataset to select only certain features
    tempDf = df.copy()
    tempDf = tempDf [columnsToUse]

    #scale the temp dataset to use the clustering algorithm
    scaler = StandardScaler()
    scaler.fit(tempDf)
    tempDfScal = scaler.transform(tempDf)

    columnsNames = []
    for min_sample in range (min_samples[0], min_samples[1]):
        for singleXi in xi:
            for single_min_cluster_size in min_cluster_size:
                # Create an Optics clustering model                                                   #using all possible cores
                clustering = OPTICS(min_samples=min_sample, xi=singleXi, min_cluster_size=single_min_cluster_size, n_jobs=-1)

                # Fit the model to your data
                clustering.fit(tempDfScal)

                # Predict the cluster labels
                cluster_labels = clustering.labels_

                # The cluster labels are stored in 'cluster_labels' variable
                # You can add them to your DataFrame if needed
                df['optics' + ' ' + 'min_samples=' + min_sample + ' ' + 'xi=' + singleXi + ' ' + 'min_cluster_size=' + single_min_cluster_size] = cluster_labels
                columnsNames.append ('optics' + ' ' + 'min_samples=' + min_sample + ' ' + 'xi=' + singleXi + ' ' + 'min_cluster_size=' + single_min_cluster_size)

    return df, columnsNames

def hdbscan(df, originalDatasetColumnsToUse):
    #@RafaelUrbina
    # TODO: Implement the hdbscan method
    return df, "hdbscan"
