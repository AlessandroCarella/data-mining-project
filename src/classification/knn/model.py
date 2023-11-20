import os.path as path
import os
import pickle
import math
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from classificationUtils import getModelPath, getTrainDatasetPath, splitDataset, copyAndScaleDataset, continuousFeatures, getModelFromPickleFile, saveModelToPickleFile, getSampleSizeList, downsampleDataset

def getRangeForK (numberOfSamplesInTheDataset:int, numberOfKs = 100) -> list[int]:
    #k should be sqrt of the number of samples in the training dataset, so maybe we can implement a range around that number
    middleK = int(math.sqrt(numberOfSamplesInTheDataset))
    lower_limit = max(3, middleK - 50)
    upper_limit = lower_limit + (numberOfKs * 2) - 1

    if lower_limit % 2 == 0:
        lower_limit += 1

    # Generate a list of odd numbers
    return list(range(lower_limit, upper_limit + 1, 2))

def getKnnModel ():
    if not path.exists(getModelPath ("knn")):
        dataset = pd.read_csv (getTrainDatasetPath())
        
        knnDict = {}
        splittedDatasets = {}
        for datasetDimensions in getSampleSizeList (dataset.shape[0]):#for the learning curve plot
            datasetSplitted = splitDataset (downsampleDataset(copyAndScaleDataset (dataset, continuousFeatures)))
            splittedDatasets[datasetDimensions] = datasetSplitted #to save the dataset split on file
            for k in getRangeForK (dataset.shape[0]): #for different values of k
                model = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
                model.fit ()
                #TODO: add cross validation (maybe instead of dataset split)
                metrics = knnMetrics (model, k, datasetSplitted)
                knnDict[k] = {"model": model, "metrics": metrics, "k": k, "datasetDimensions":datasetDimensions}
                saveModelToPickleFile (knnDict, "knn")
    else:
        return getModelFromPickleFile ("knn")
