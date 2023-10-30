import pandas as pd
import os.path as path

#all the methods we're gonna use (unless they're not in there should be from those two libraries (cause the teahers were using)):
import sklearn #usually you import the modules from this library and not the whole thing
import scipy as sp #TODO check if the abbreviation ("sp") is correct

#TODO: name the final file for the train dataset "trainFinalVersion.csv"
datasetPath = path.join(path.abspath(path.dirname(__file__)), "../../dataset (missing + split)/trainFinalVersion.csv") 

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

#all the method imported in the following methods should get 
#as INPUT:
#the dataset
#the originalDatasetColumnsToUse 
#as OUTPUT:
#the same dataset with an added column named after the method of clustering we're using and the relative values in it
#the name of the column used
def clusterings (df):
    from clusteringMethods.hierarchical import hierarchical, hierarchicalCompleteLinkage, hierarchicalSingleLink, hierarchicalGroupAverage
    df, hierarchicalColumnName = hierarchical(df, originalDatasetColumnsToUse)
    df, hierarchicalCompleteLinkageColumnName = hierarchicalCompleteLinkage(df, originalDatasetColumnsToUse)
    df, hierarchicalSingleLinkColumnName = hierarchicalSingleLink(df, originalDatasetColumnsToUse)
    df, hierarchicalGroupAverageColumnName = hierarchicalGroupAverage(df, originalDatasetColumnsToUse)

    from clusteringMethods.kMeans import kMeans, bisectingKmeans, xMeans, kModes
    df, kMeansColumnName = kMeans(df, originalDatasetColumnsToUse)
    df, bisectingKmeansColumnName = bisectingKmeans(df, originalDatasetColumnsToUse)
    df, xMeansColumnName = xMeans(df, originalDatasetColumnsToUse)
    df, kModesColumnName = kModes(df, originalDatasetColumnsToUse)

    from clusteringMethods.mixtureGuassianModel import mixtureGuassian
    df, mixtureGuassianColumnName = mixtureGuassian(df, originalDatasetColumnsToUse)

    from clusteringMethods.dbscan import dbscan, optics, hdbscan
    df, dbscanColumnName = dbscan(df, originalDatasetColumnsToUse)
    df, opticsColumnName = optics(df, originalDatasetColumnsToUse)
    df, hdbscanColumnName = hdbscan(df, originalDatasetColumnsToUse)


    #this is a dictionary filled with all the names of the columns related 
    #to clustering that we putted in the dataset
    keys = ["hierarchical", "hierarchicalCompleteLinkage", "hierarchicalSingleLink", "hierarchicalGroupAverage",
                    "kMeans", "bisectingKmeans", "xMeans", "kModes",
                    "mixtureGuassian",
                    "dbscan", "optics", "hdbscan"]
    values = [hierarchicalColumnName, hierarchicalCompleteLinkageColumnName, hierarchicalSingleLinkColumnName, hierarchicalGroupAverageColumnName,
                    kMeansColumnName, bisectingKmeansColumnName, xMeansColumnName, kModesColumnName,
                    mixtureGuassianColumnName,
                    dbscanColumnName, opticsColumnName, hdbscanColumnName]
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
        clusteringColumnsNames.get("hierarchical"), 
        clusteringColumnsNames.get("hierarchicalCompleteLinkage"), 
        clusteringColumnsNames.get("hierarchicalSingleLink"), 
        clusteringColumnsNames.get("hierarchicalGroupAverage"),
        clusteringColumnsNames.get("kMeans"),
        clusteringColumnsNames.get("bisectingKmeans"),
        clusteringColumnsNames.get("xMeans"),
        clusteringColumnsNames.get("kModes"),
        clusteringColumnsNames.get("mixtureGuassian"),
        clusteringColumnsNames.get("dbscan"),
        clusteringColumnsNames.get("optics"),
        clusteringColumnsNames.get("hdbscan"),
        ]

    from measuresAndVisualizations.visualizationAndPlot import dendogram
    dendogram (df, [
        clusteringColumnsNames.get("hierarchical"), 
        clusteringColumnsNames.get("hierarchicalCompleteLinkage"), 
        clusteringColumnsNames.get("hierarchicalSingleLink"), 
        clusteringColumnsNames.get("hierarchicalGroupAverage")
        ])
    
    from measuresAndVisualizations.visualizationAndPlot import correlationMatrix
    correlationMatrix (df, [
        clusteringColumnsNames.get("hierarchical"), 
        clusteringColumnsNames.get("hierarchicalCompleteLinkage"), 
        clusteringColumnsNames.get("hierarchicalSingleLink"), 
        clusteringColumnsNames.get("hierarchicalGroupAverage"),
        clusteringColumnsNames.get("mixtureGuassian"),
        ]
    )

    from measuresAndVisualizations.visualizationAndPlot import countPlotBetweenEachFeatureAndEachCluster
    countPlotBetweenEachFeatureAndEachCluster (df, allClusteringColumns)

    from measuresAndVisualizations.visualizationAndPlot import similarityMatrix
    similarityMatrix (df, [
        clusteringColumnsNames.get("hierarchical"), 
        clusteringColumnsNames.get("hierarchicalCompleteLinkage"), 
        clusteringColumnsNames.get("hierarchicalSingleLink"), 
        clusteringColumnsNames.get("hierarchicalGroupAverage"),
        clusteringColumnsNames.get("dbscan"),
        clusteringColumnsNames.get("optics"),
        clusteringColumnsNames.get("hdbscan"),
        ])

    ################################################################
    
    from measuresAndVisualizations.metrics import entropy
    """
    from chatgpt:
    what's entropy in dataset clustering?
    Entropy in dataset clustering is a measure used to assess the purity of clusters created during 
    a clustering algorithm's execution. 
    It's commonly used in the context of hierarchical clustering, particularly when performing 
    agglomerative clustering. The goal of clustering is to group similar data points together into 
    clusters, and entropy helps quantify the homogeneity of these clusters.
    """
    entropy (df, allClusteringColumns)
    
    from measuresAndVisualizations.metrics import sseZeroToFiftyClusters
    sseZeroToFiftyClusters (df, [
        clusteringColumnsNames.get("kMeans"),
        clusteringColumnsNames.get("bisectingKmeans"),
        clusteringColumnsNames.get("xMeans"),
        clusteringColumnsNames.get("mixtureGuassian"),
    ])

    from measuresAndVisualizations.metrics import clustersCohesionAndSeparation
    clustersCohesionAndSeparation (df, allClusteringColumns)
    
    from measuresAndVisualizations.metrics import silhouette  
    silhouette (df, [
        clusteringColumnsNames.get("kMeans"),
        clusteringColumnsNames.get("bisectingKmeans"),
        clusteringColumnsNames.get("xMeans"),
        clusteringColumnsNames.get("kModes"),
        clusteringColumnsNames.get("mixtureGuassian"),
        clusteringColumnsNames.get("dbscan"),
        clusteringColumnsNames.get("optics"),
        clusteringColumnsNames.get("hdbscan"),
    ])

    from measuresAndVisualizations.metrics import kthNeighborDistance
    kthNeighborDistance (df, [
        clusteringColumnsNames.get("dbscan"),
        clusteringColumnsNames.get("optics"),
        clusteringColumnsNames.get("hdbscan"),
    ])


finalTrainDataset = pd.read_csv (datasetPath)
finalTrainDatasetWithClustering, clusteringColumnsNames = clusterings (finalTrainDataset)
measuresAndVisualizationsForDeterminingClustersQuality (finalTrainDatasetWithClustering, clusteringColumnsNames)