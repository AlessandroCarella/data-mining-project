import os.path as path
import math
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

from classificationUtils import getModelPath, getTrainDatasetPath, getModelFromPickleFile, saveModelToPickleFile, getSampleSizeList, downsampleDataset, splitDataset, copyAndScaleDataset, continuousFeatures, saveOtherInfoModelDict
from metrics import knnMetrics, saveMetricsToFile

def getRangeForK (numberOfSamplesInTheDataset:int, numberOfKs = 10) -> list[int]:
    #k should be sqrt of the number of samples in the training dataset, so maybe we can implement a range around that number
    middleK = int(math.sqrt(numberOfSamplesInTheDataset))
    lower_limit = max(3, middleK - 10)
    upper_limit = lower_limit + (numberOfKs * 2) - 1

    if lower_limit % 2 == 0:
        lower_limit += 1

    # Generate a list of odd numbers
    return list(range(lower_limit, upper_limit + 1, 2))

def getDownsampledDataset (dataset:pd.DataFrame) -> dict:
    dowsampledDatasets = {}
    for datasetDimension in getSampleSizeList (dataset.shape[0]):#for the learning curve plot
        dowsampledDatasets[datasetDimension] = downsampleDataset(dataset, datasetDimension)
    return dowsampledDatasets

def makeknnRegDictValue (k:int, model, weight:str, metrics:dict, datasetDimension:int, splitNumber:int, trainIndex:np.ndarray, testIndex:np.ndarray, targetVariable:str, predictions:np.ndarray, y_test:pd.Series):
    return {
        "k":k, 
        "model": model, 
        "weight": weight,
        "metrics": metrics,
        "datasetDimension":datasetDimension,
        "splitNumber":splitNumber, 
        "kFoldTrainIndexAndTestIndex":{"trainIndex":trainIndex.tolist(), "testIndex":testIndex.tolist()}, 
        "targetVariable":targetVariable,
        "predictions":predictions,#saving those for 
        "groundTruth":y_test,#roc curve comupation
    }

def getKnnRegressorModel (targetVariable = "genre"):
    if not path.exists(getModelPath ("knnReg")):
        import time
        startTime = time.time()

        dataset = pd.read_csv (getTrainDatasetPath())
        
        """
        SEE COMMENT IN model.py in the knn folder
        """

        dowsampledDatasets = getDownsampledDataset (dataset=dataset)

        kf = KFold(n_splits=33, shuffle=True, random_state=42)

        weights = ['uniform', 'distance']

        knnRegDict = {}
        for weight in weights:
            for datasetDimension, subDataset in dowsampledDatasets.items ():
                X = copyAndScaleDataset (df=subDataset, columnsToUse=continuousFeatures)
                y = subDataset[targetVariable]
                
                splitNumber = 1
                for trainIndex, testIndex in kf.split (X):
                    for k in getRangeForK (datasetDimension): #for different values of k
                        X_train, X_test = X.iloc[trainIndex], X.iloc[testIndex]
                        y_train, y_test = y.iloc[trainIndex], y.iloc[testIndex]
                        
                        model = KNeighborsRegressor(n_neighbors=k, n_jobs=-1)#ONLY LINE CHANGED FROM KNN FOLDER
                        model.fit (X_train, y_train)

                        predictions=model.predict(X_test)
                        metrics = knnMetrics (predictions=model.predict(X_test), groundTruth=y_test)


                        knnRegDictKey = f"k:{k}, splitNumber:{splitNumber}, datasetDimension:{datasetDimension}, weights:{weight}"
                        knnRegDict[knnRegDictKey] = makeknnRegDictValue (k, model, weight, metrics, datasetDimension, splitNumber, trainIndex, testIndex, targetVariable, predictions, y_test)

                        #To check on the advancement
                        print (time.time() - startTime)
                        print (knnRegDictKey)
                    splitNumber += 1
        saveModelToPickleFile (knnRegDict)
        saveMetricsToFile (knnRegDict)
        #saveOtherInfoModelDict (knnRegDict)
        return knnRegDict
    else:
        return getModelFromPickleFile ("knnReg")
    
getKnnRegressorModel()