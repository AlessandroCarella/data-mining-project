import pandas as pd
from sklearn.preprocessing import StandardScaler
import os.path as path
import os
import pickle

def saveDfToFile (df:pd.DataFrame):
    df.to_csv (path.join(path.abspath(path.dirname(__file__)), "../../dataset (missing + split)/trainFinalWithClustering.csv"), index=False)

def columnAlreadyInDf (columnName, df):
    return columnName in df.columns

def copyAndScaleDataset (df, columnsToUse):
    #create a copy of the dataset to select only certain features
    tempDf = df.copy()
    tempDf = tempDf [columnsToUse]

    #scale the temp dataset to use the clustering algorithm
    scaler = StandardScaler()
    scaler.fit(tempDf)
    tempDfScal = scaler.transform(tempDf)

    return tempDfScal

"""
this method is needed since some of the object created during the clustering algorithms
are needed for some of the representations or metrics we want to calculate

for example, during the k means clustering algorithm (semplified for example porpuses):
def kMeans(df):
    for k in range (Krange[0], Krange[1] + 1):
        # Initialize the K-means model
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        # Fit the model to your data
        kmeans.fit(tempDfScal)

        # Get the cluster assignments for each data point
        labels = kmeans.labels_

        ***###HERE###*** 
        #other than the labels also the centroids are 
        centers = kmeans.cluster_centers_
        ***###HERE###*** 

        # Add the cluster assignment as a new column in the DataFrame
        df[newColumnName] = labels

    return df

the centers are needed for one of the visualizations.

So we need to save those objects and to do so we can use this method.

The idea is to save a dictionary that will be strutured in a similar way to the
dictionary we use to save the names of the columns with the clustering values
in the DataFrame

so for example, since we have ["kMeans=2", "kMeans=3", "kMeans=4", ...]
the dictionary will be like

centroidsKmeans = 
{
    "kMeans=2" = listOfCentroidsForKmeansWithValue2, 
    "kMeans=3" = listOfCentroidsForKmeansWithValue3, 
    "kMeans=4" = listOfCentroidsForKmeansWithValue4, 
    ...
}

see the actual implementations in the various methods to understand better how it works
"""
def saveMidRunObjectToFile (obj, filePath):
    #please use the src/clustering/additionalObjects folder to save the file in
    #also please, when the object saved is not a dictionary (since only one object is needed
    #specify it in the name of the file saying "(just object) nameOfTheObject") at the 
    #beginning of the name of the file
    if not path.exists(path.dirname(filePath)):
        os.makedirs(path.dirname(filePath))

    if not filePath.endswith(".pickle"):
        filePath += ".pickle"

    with open (filePath, "wb") as file:
        pickle.dump (obj, file)

def getMidRunObject (filePath):
    if path.isfile (filePath):
        with open (filePath, "rb") as file:
            return pickle.load (file)
    
    #passing just the file name (without extension)
    with open (path.join(path.dirname (__file__), "../additionalObjects", filePath + ".pickle"), "rb") as file:
        return pickle.load (file)

def getMidRunObjectFolderPath ():
    return path.join(path.dirname (__file__), "../additionalObjects")