import matplotlib.pyplot as plt
import os.path as path
from scipy.cluster.hierarchy import dendrogram, cophenet
from scipy.spatial.distance import pdist
import numpy as np
from measuresAndVisualizations.measuresAndVisualizationsUtils import saveDictToFile, getMetricsFolderPath, getPlotsFolderPath, checkSubFoldersExists, visualizazionAlreadyExists
from measuresAndVisualizations.clusteringUtility import getMidRunObject
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


def dendogram(df, listOfClusteringColumns):
    #@SaraHoxha
    results = {}
    filePath = path.join (getMetricsFolderPath (), "cophenetCoefficients.csv")
    for clusteringType in listOfClusteringColumns:
        for clusteringColumn in clusteringType:
            if "hierarchical" in clusteringColumn:
                    hierarchicalType = clusteringColumn.split (" ")[0] 
                    threshold = int(''.join(filter(str.isdigit, clusteringColumn.split (" ")[1]))) 
                    imageName = hierarchicalType + " " + clusteringColumn + " dendogram.png"
                    imgPath = path.join(getPlotsFolderPath(), "dendogram", imageName)
                    if not visualizazionAlreadyExists (imgPath):
                        plt.figure(figsize=(21, 9))
                        plt.title(f"Dendrogram for Clustering Type: {hierarchicalType} {clusteringColumn}")
                        
                        linkage_matrix = getMidRunObject ("(just object) " + hierarchicalType + "LinkageMatrix")
                        dendrogram(linkage_matrix, orientation="top", truncate_mode='lastp', p=threshold)

                        c, coph_dists = cophenet(linkage_matrix, pdist(df[continuousFeatures].values))
                        results[clusteringColumn] = c
                        
                        checkSubFoldersExists(imgPath)
                        plt.savefig(imgPath)
                        plt.close()
            checkSubFoldersExists (filePath)
            saveDictToFile(results, filePath, custom_headers=["clustering type", "cophenet"])

def correlationMatrix(df, listOfClusteringColumns):
    #@RafaelUrbina
    """for clusteringType in listOfClusteringColumns:#[["", ""],["", ""]]
        for clusteringColumn in clusteringType:#["", ""]
            if "hierarchical" in clusteringColumn: #clusteringColumn example: "hierarchicalGroupAverage threshold3 criterion=blabla"
                hierarchicalType = clusteringColumn.split (" ")[0] #get just the clustering type since the formatting of the strings for hierarchical is hierarchicalType + ' ' + 'threshold' + n + ' ' + 'criterion=' + n
                imageName = hierarchicalType + " correlationMatrix.png"
                imgPath = path.join(getPlotsFolderPath(), "correlationMatrix", imageName)

                if not visualizazionAlreadyExists (imgPath):
                    linkage_matrix = getMidRunObject ("(just object) " + hierarchicalType + "LinkageMatrix")
                    
                    distanceMatrix = distance_matrix (linkage_matrix)
                    
                    #Reorder in terms of the labels
                    dist_ordered_matrix = reorderDistanceMatrixhierarchical(distanceMatrix, linkage_matrix)

                    #Correlation coef.
                    correlation_coef = np.corrcoef(dist_ordered_matrix)

                    #set plt features
                    plt.figure(figsize=(21, 9))
                    plt.title(imageName)
                    plt.suptitle("Correlation coeficient: "+str(correlation_coef))

                    # Plot the correlation matrix
                    sns.heatmap(dist_ordered_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
                
                    checkSubFoldersExists(imgPath)
                    plt.savefig(imgPath, dpi=100)
                    plt.close()

            elif "mixtureGaussian" in clusteringColumn:
                tempDfScal = copyAndScaleDataset (df, continuousFeatures)
                gmmtype = clusteringColumn
                imageName = gmmtype + " correlationMatrix.png"
                imgPath = path.join(getPlotsFolderPath(), "correlationMatrix", imageName)
                if not visualizazionAlreadyExists (imgPath):
                    # correlation matrix for mixture guassian clustering
                    labels = df[gmmtype]
                    component_means = getMidRunObject("(just object) gmmComponent_means "+ gmmtype)
                    distanceMatrix = pairwise_distances(tempDfScal, component_means, metric='euclidean')
                    dist_ordered_matrix = reorderDistanceMatrixgmm(distanceMatrix,labels,component_means)

                    #set plt features
                    plt.figure(figsize=(21, 9))
                    plt.title(imageName)

                    # Plot the correlation matrix
                    sns.heatmap(dist_ordered_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
                
                    checkSubFoldersExists(imgPath)
                    plt.savefig(imgPath, dpi=100)
                    plt.close()
            else:
                pass"""
def clusterBarChart(df, listOfClusteringTypes, featuresToPlotOn):
    return
    #@AlessandroCarella
    for clusteringType in listOfClusteringTypes:
        for clusteringColumn in clusteringType:
            if clusteringColumn in df.columns:
                for categoricalColumn in featuresToPlotOn:
                    imgPath = path.join(getPlotsFolderPath(), "clusterBarChart", f'{categoricalColumn} for {clusteringColumn}.png')
                    if not visualizazionAlreadyExists (imgPath):
                        # Group the data by the categorical column and the cluster column
                        grouped = df.groupby([categoricalColumn, clusteringColumn]).size().unstack().fillna(0)

                        # Create the plot
                        fig, ax = plt.subplots()

                        # Plot the data
                        for cluster in grouped.columns:
                            ax.bar(grouped.index, grouped[cluster], label=f'Cluster {cluster}')

                        # Set plot labels and legend
                        ax.set_xlabel(categoricalColumn)
                        ax.set_ylabel('Count')
                        ax.set_title(f'{categoricalColumn} for {clusteringColumn}')
                        ax.legend()
                        
                        # Set the figure size to 21:9 (3440x1440 pixels)
                        fig.set_size_inches(21, 9)

                        checkSubFoldersExists(imgPath)
                        plt.savefig(imgPath, dpi=100)
                        plt.close()

def similarityMatrix(df, listOfClusteringColumns):
    #@RafaelUrbina
    """for clusteringType in listOfClusteringColumns:#[["", ""],["", ""]]
        for clusteringColumn in clusteringType:#["", ""]
            if "hierarchical" in clusteringColumn: #clusteringColumn example: "hierarchicalGroupAverage threshold3 criterion=blabla"
                hierarchicalType = clusteringColumn.split (" ")[0] #get just the clustering type since the formatting of the strings for hierarchical is hierarchicalType + ' ' + 'threshold' + n + ' ' + 'criterion=' + n
                imageName = hierarchicalType + " similarityMatrix.png"
                imgPath = path.join(getPlotsFolderPath(), "similarityMatrix", imageName)

                if not visualizazionAlreadyExists (imgPath):
                    linkage_matrix = getMidRunObject ("(just object) " + hierarchicalType + "LinkageMatrix")
                    
                    distanceMatrix = distance_matrix (linkage_matrix)
                    
                    #Reorder in terms of the labels
                    dist_ordered_matrix = reorderDistanceMatrixhierarchical(distanceMatrix, linkage_matrix)

                    # Convert distance matrix to similarity matrix (smaller distances should mean higher similarity)
                    similarity_matrix = 1 / (1 + dist_ordered_matrix)

                    #set plt features
                    plt.figure(figsize=(21, 9))
                    plt.title(imageName)

                    # Plot the correlation matrix
                    sns.heatmap(similarity_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
                
                    checkSubFoldersExists(imgPath)
                    plt.savefig(imgPath, dpi=100)
                    plt.close()
            elif "dbscan" in clusteringColumn:
                tempDfScal = copyAndScaleDataset (df, continuousFeatures)
                dbscantype = clusteringColumn
                imageName = dbscantype + " similarityMatrix.png"
                imgPath = path.join(getPlotsFolderPath(), "similarityMatrix", imageName)
                if not visualizazionAlreadyExists (imgPath):
                    distanceMatrix = pairwise_distances(tempDfScal)
                    labels = df[dbscantype]
                    dist_ordered_matrix = reorderDistanceMatrixdensity(distanceMatrix,labels)
                    similarity_matrix = 1 / (1 + dist_ordered_matrix)

                    #set plt features
                    plt.figure(figsize=(21, 9))
                    plt.title(imageName)

                    # Plot the correlation matrix
                    sns.heatmap(similarity_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
                
                    checkSubFoldersExists(imgPath)
                    plt.savefig(imgPath, dpi=100)
                    plt.close()
            elif "optics" in clusteringColumn:
                tempDfScal = copyAndScaleDataset (df, continuousFeatures)
                opticstype = clusteringColumn
                imageName = opticstype + " similarityMatrix.png"
                imgPath = path.join(getPlotsFolderPath(), "similarityMatrix", imageName)
                if not visualizazionAlreadyExists (imgPath):
                    distanceMatrix = pairwise_distances(tempDfScal)
                    labels = df[opticstype]
                    dist_ordered_matrix = reorderDistanceMatrixdensity(distanceMatrix,labels)
                    similarity_matrix = 1 / (1 + dist_ordered_matrix)

                    #set plt features
                    plt.figure(figsize=(21, 9))
                    plt.title(imageName)

                    # Plot the correlation matrix
                    sns.heatmap(similarity_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
                
                    checkSubFoldersExists(imgPath)
                    plt.savefig(imgPath, dpi=100)
                    plt.close()
            elif "hdbscan" in clusteringColumn:
                tempDfScal = copyAndScaleDataset (df, continuousFeatures)
                hdbscantype = clusteringColumn
                imageName = hdbscantype + " similarityMatrix.png"
                imgPath = path.join(getPlotsFolderPath(), "similarityMatrix", imageName)
                if not visualizazionAlreadyExists (imgPath):
                    distanceMatrix = pairwise_distances(tempDfScal)
                    labels = df[hdbscantype]
                    dist_ordered_matrix = reorderDistanceMatrixdensity(distanceMatrix,labels)
                    similarity_matrix = 1 / (1 + dist_ordered_matrix)

                    #set plt features
                    plt.figure(figsize=(21, 9))
                    plt.title(imageName)

                    # Plot the correlation matrix
                    sns.heatmap(similarity_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
                
                    checkSubFoldersExists(imgPath)
                    plt.savefig(imgPath, dpi=100)
                    plt.close()
                    
            else:
                pass"""