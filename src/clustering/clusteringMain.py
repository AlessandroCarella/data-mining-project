import pandas as pd
import os.path as path

#all the methods we're gonna use (unless they're not in there should be from those two libraries (cause the teahers were using)):
import sklearn #usually you import the modules from this library and not the whole thing
import scipy as sp 

from clusteringMethods.clusteringUtility import saveDfToFile

datasetPath = path.join(path.abspath(path.dirname(__file__)), "../../dataset (missing + split)/trainFinalWithClustering.csv")
if not path.exists (datasetPath):
    datasetPath = path.join(path.abspath(path.dirname(__file__)), "../../dataset (missing + split)/Dataset_nomissing_nofeatur_noutlier_noinconsistencies.csv") 

#those are all the names of the columns in the dataset right now
#we should select the ones that we can use for the clustering before proceeding
originalDatasetColumnsToUse = [
    "name",
    "duration_ms",
    "explicit",
    "popularity",
    "artists",
    "album_name",
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature",
    "n_beats",
    "genre"
]

categoricalFeatures = [
    "name",
    "explicit",
    "artists",
    "album_name",
    "key",
    "genre",
    "time_signature"
]

categoricalFeaturesForPlotting = [
    #"name",
    #"explicit",
    #"artists",
    #"album_name",
    "key",
    "genre",
    "time_signature"
]

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

#Range for the hierarchical methods
min_sample_range = [5, 10, 15, 20, 25, 30, 
                    35, 40, 50]
#all the method imported in the following methods should get 
#as INPUT:
#the dataset
#the originalDatasetColumnsToUse 
#as OUTPUT:
#the same dataset with an added column named after the method of clustering we're using and the relative values in it
#the name of the column used
def clusterings (df):
    from clusteringMethods.hierarchical import hierarchicalCentroidLinkage, hierarchicalCompleteLinkage, hierarchicalSingleLinkage, hierarchicalGroupAverage
    df, hierarchicalColumnName = hierarchicalCentroidLinkage(df, continuousFeatures,linkage_thresholds=min_sample_range)
    df, hierarchicalCompleteLinkageColumnName = hierarchicalCompleteLinkage(df, continuousFeatures, linkage_thresholds=min_sample_range)
    df, hierarchicalSingleLinkColumnName = hierarchicalSingleLinkage(df, continuousFeatures,linkage_thresholds=min_sample_range)
    df, hierarchicalGroupAverageColumnName = hierarchicalGroupAverage(df, continuousFeatures, criterion="distance", linkage_thresholds=min_sample_range)
    df, hierarchicalGroupAverageColumnName = hierarchicalGroupAverage(df, continuousFeatures, criterion="maxclust", linkage_thresholds=min_sample_range)
    saveDfToFile (df)

    from clusteringMethods.kMeans import kMeans, bisectingKmeans, xMeans, kModes
    df, kMeansColumnName = kMeans(df, continuousFeatures, Krange=[2, 200])
    df, bisectingKmeansColumnName = bisectingKmeans(df, continuousFeatures, Krange=[2, 200])
    #df, xMeansColumnName = xMeans(df, originalDatasetColumnsToUse,Krange=[2,200])
    df, kModesColumnName = kModes(df, categoricalFeatures, Krange=[2, 200])
    saveDfToFile (df)

    from clusteringMethods.mixtureGuassian import mixtureGuassian
    df, mixtureGuassianColumnName = mixtureGuassian(df, continuousFeatures, n_components=[2, 20], tols=[1e-2, 1e-3, 1e-4, 1e-5])
    saveDfToFile (df)

    from clusteringMethods.dbscan import dbscan, optics, hdbscan
    df, dbscanColumnName = dbscan(df, continuousFeatures, eps= [0.1, 0.5, 1, 1.5, 2, 2.5, 3], min_samples=[1, 20])#eps= [0.1, 3], min_samples=[1, 20])
    #df, opticsColumnName = optics(df, continuousFeatures, min_samples=[1, 20], xi=[0.01, 0.05, 0.1], min_cluster_size=[0.01, 0.05, 0.1])
    df, hdbscanColumnName = hdbscan(df, continuousFeatures,min_cluster_size=[50, 200])
    saveDfToFile (df)


    #this is a dictionary filled with all the names of the columns related 
    #to clustering that we putted in the dataset
    #IMPORTANT: some values in the dataset might me simple strings but some others might be list of strings
    #since, for example, we don't have to do only 1 number of clusters for kmeans clustering
    #so the value for the key kMeans would be something like: 'kMeans': ['kMeans2', 'kMeans3', 'kMeans4', 'kMeans5', ...]
    keys = ["hierarchicalCentroidLinkage", "hierarchicalCompleteLinkage", "hierarchicalSingleLinkage", "hierarchicalGroupAverage",
                    "kMeans", "bisectingKmeans", "kModes", #"xMeans", 
                    "mixtureGuassian",
                    "dbscan", "hdbscan",]#"optics",]
    values = [hierarchicalColumnName, hierarchicalCompleteLinkageColumnName, hierarchicalSingleLinkColumnName, hierarchicalGroupAverageColumnName,
                    kMeansColumnName, bisectingKmeansColumnName, kModesColumnName, #xMeansColumnName, 
                    mixtureGuassianColumnName,
                    dbscanColumnName, hdbscanColumnName,]#opticsColumnName, ]
    clusteringColumnsNames = {method: column for method, column in zip(keys, values)}

    return df, clusteringColumnsNames

#all the methods imported in the following method should get:
#as INPUT:
#the dataframe
#as OUTPUT:
#the plots and metrics generated saved in the relative folder
########################################################################
#Side note: let's say for example that you want to generate the 
#visualization of Hierarchical Clustering using Dendrograms
#the path of the folder where you put the image in would be:
#src/clustering/visualization/hierarchicalClusteringDendogram.png
#or
#src/clustering/visualization/hierarchicalCompleteLinkageClusteringDendogram.png
#and for metrics would instead be:
#src/clustering/measures/hierarchicalClusteringScore.txt (or whatever else instead of "score" at the end, also the file can be a txt or a csv or any other kind of file you feel like is better)
#or
#src/clustering/visualization/hierarchicalCompleteLinkageClusteringScore.txt (or whatever else instead of "score" at the end, also the file can be a txt or a csv or any other kind of file you feel like is better)
#(obviously the dendograms are visual and not measuerments, it's 
#just that i don't want to find another thing that might mess the
#idea up)
def measuresAndVisualizationsForDeterminingClustersQuality (df:pd.DataFrame, clusteringColumnsNames:dict):
    allClusteringColumns = [
        clusteringColumnsNames.get("hierarchicalCentroidLinkage"), 
        clusteringColumnsNames.get("hierarchicalCompleteLinkage"), 
        clusteringColumnsNames.get("hierarchicalSingleLinkage"), 
        clusteringColumnsNames.get("hierarchicalGroupAverage"),
        clusteringColumnsNames.get("kMeans"),
        clusteringColumnsNames.get("bisectingKmeans"),
        #clusteringColumnsNames.get("xMeans"),
        clusteringColumnsNames.get("kModes"),
        clusteringColumnsNames.get("mixtureGuassian"),
        clusteringColumnsNames.get("dbscan"),
        #clusteringColumnsNames.get("optics"),
        clusteringColumnsNames.get("hdbscan"),
        ]

    from measuresAndVisualizations.visualizationAndPlot import dendogram
    dendogram (df, [
        clusteringColumnsNames.get("hierarchicalCentroidLinkage"), 
        clusteringColumnsNames.get("hierarchicalCompleteLinkage"), 
        clusteringColumnsNames.get("hierarchicalSingleLinkage"), 
        clusteringColumnsNames.get("hierarchicalGroupAverage")
        ])
    
    '''from measuresAndVisualizations.visualizationAndPlot import correlationMatrix
    correlationMatrix (df, [
        clusteringColumnsNames.get("hierarchicalCentroidLinkage"), 
        clusteringColumnsNames.get("hierarchicalCompleteLinkage"), 
        clusteringColumnsNames.get("hierarchicalSingleLinkage"), 
        clusteringColumnsNames.get("hierarchicalGroupAverage"),
        #clusteringColumnsNames.get("mixtureGuassian"),
        ]
    )'''

    from measuresAndVisualizations.visualizationAndPlot import clusterBarChart
    clusterBarChart (df, allClusteringColumns, categoricalFeaturesForPlotting)

    '''from measuresAndVisualizations.visualizationAndPlot import similarityMatrix
    similarityMatrix (df, [
        clusteringColumnsNames.get("hierarchicalCentroidLinkage"), 
        clusteringColumnsNames.get("hierarchicalCompleteLinkage"), 
        clusteringColumnsNames.get("hierarchicalSingleLinkage"), 
        clusteringColumnsNames.get("hierarchicalGroupAverage"),
        #clusteringColumnsNames.get("dbscan"),
        #clusteringColumnsNames.get("optics"),
        #clusteringColumnsNames.get("hdbscan"),
        ])'''

    ################################################################
    
    from measuresAndVisualizations.metrics import entropyMetric
    """
    from chatgpt:
    what's entropy in dataset clustering?
    Entropy in dataset clustering is a measure used to assess the purity of clusters created during 
    a clustering algorithm's execution. 
    It's commonly used in the context of hierarchicalCentroidLinkage clustering, particularly when performing 
    agglomerative clustering. The goal of clustering is to group similar data points together into 
    clusters, and entropy helps quantify the homogeneity of these clusters.
    """
    entropyMetric (df, allClusteringColumns)
    
    from measuresAndVisualizations.metrics import sse
    """sse (df, [
        clusteringColumnsNames.get("kMeans"),
        clusteringColumnsNames.get("bisectingKmeans"),
        #clusteringColumnsNames.get("xMeans"),
        #clusteringColumnsNames.get("mixtureGuassian"),
    ])"""

    from measuresAndVisualizations.metrics import clustersCohesionAndSeparation
    clustersCohesionAndSeparation (df, allClusteringColumns)
    
    from measuresAndVisualizations.metrics import silhouette  
    silhouette (df, [
        clusteringColumnsNames.get("hierarchicalCentroidLinkage"), 
        clusteringColumnsNames.get("hierarchicalCompleteLinkage"), 
        clusteringColumnsNames.get("hierarchicalSingleLinkage"), 
        clusteringColumnsNames.get("hierarchicalGroupAverage")
        clusteringColumnsNames.get("kMeans"),
        clusteringColumnsNames.get("bisectingKmeans"),
        #clusteringColumnsNames.get("xMeans"),
        clusteringColumnsNames.get("kModes"),
        clusteringColumnsNames.get("mixtureGuassian"),
        clusteringColumnsNames.get("dbscan"),
        #clusteringColumnsNames.get("optics"),
        clusteringColumnsNames.get("hdbscan"),
    ])

finalTrainDataset = pd.read_csv (datasetPath)
finalTrainDatasetWithClustering, clusteringColumnsNames = clusterings (finalTrainDataset)
measuresAndVisualizationsForDeterminingClustersQuality (finalTrainDatasetWithClustering, clusteringColumnsNames)
