import os.path as path
import math
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
from sklearn.tree import DecisionTreeRegressor

from classificationUtils import getModelPath, getTrainDatasetPath, getModelFromPickleFile, saveModelToPickleFile, getSampleSizeList, downsampleDataset, splitDataset, copyAndScaleDataset, continuousFeatures, saveOtherInfoModelDict
from metrics import decisionTreeRegMetrics, saveMetricsToFile

def fun ():
    pass

def getDownsampledDataset (dataset:pd.DataFrame) -> dict:
    dowsampledDatasets = {}
    for datasetDimension in getSampleSizeList (dataset.shape[0]):#for the learning curve plot
        dowsampledDatasets[datasetDimension] = downsampleDataset(dataset, datasetDimension)
    return dowsampledDatasets

def makeDecisionTreeRegDictValue (criterion:str, maxDepth:int, minSampleLeaf:int, minSampleSplit:int, splitNumber:int, model, metrics:dict, trainIndex:np.ndarray, testIndex:np.ndarray, targetVariable:str, predictions:np.ndarray, y_test:pd.Series):
    return {
        "criterion":criterion, 
        "maxDepth":maxDepth, 
        "minSampleLeaf":minSampleLeaf, 
        "minSampleSplit":minSampleSplit, 
        "splitNumber":splitNumber,
        "model": model, 
        "metrics": metrics,
        "kFoldTrainIndexAndTestIndex":{"trainIndex":trainIndex.tolist(), "testIndex":testIndex.tolist()}, 
        "targetVariable":targetVariable,
        "predictions":predictions,#saving those for 
        "groundTruth":y_test,#roc curve comupation
    }

def getDecisionTreeRegressorModel (targetVariable = "popularity"):
    columnsToUse = continuousFeatures
    if targetVariable in columnsToUse:
        columnsToUse.remove (targetVariable)

    if not path.exists(getModelPath ("DecisionTreeReg")):
        import time
        startTime = time.time()

        dataset = pd.read_csv (getTrainDatasetPath())
        
        """
        SEE COMMENT IN model.py in the knn folder
        """

        #no need for decision tree + it was useless in knn
        #dowsampledDatasets = getDownsampledDataset (dataset=dataset)

        kf = KFold(n_splits=33, shuffle=True, random_state=42)

        X = copyAndScaleDataset (df=dataset, columnsToUse=columnsToUse)
        y = dataset[targetVariable]

        criterions = ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
        maxDepths = [None] + list(np.arange(2, 20))
        minSamplesLeaf = [1, 5, 10, 20, 30, 50, 100]
        minSamplesSplit = [2, 5, 10, 20, 30, 50, 100]

        decisionTreeRegDict = {}
        for criterion in criterions:
            for maxDepth in maxDepths:
                for minSampleLeaf in minSamplesLeaf:
                    for minSampleSplit in minSamplesSplit:
                        splitNumber = 1
                        for trainIndex, testIndex in kf.split (X):
                            X_train, X_test = X.iloc[trainIndex], X.iloc[testIndex]
                            y_train, y_test = y.iloc[trainIndex], y.iloc[testIndex]
                            
                            model = DecisionTreeRegressor(
                                max_depth=maxDepth, 
                                criterion=criterion, 
                                min_samples_leaf=minSampleLeaf, 
                                min_samples_split=minSampleSplit, 
                                #n_jobs=-1
                            )
                            model.fit (X_train, y_train)

                            predictions=model.predict(X_test)
                            metrics = decisionTreeRegMetrics (predictions=predictions, groundTruth=y_test)

                            decisionTreeRegDictKey = f"criterion:{criterion}, maxDepth:{maxDepth}, minSampleLeaf:{minSampleLeaf}, minSampleSplit:{minSampleSplit} splitNumber:{splitNumber}"
                            decisionTreeRegDict[decisionTreeRegDictKey] = makeDecisionTreeRegDictValue (criterion, maxDepth, minSampleLeaf, minSampleSplit, splitNumber, model, metrics, trainIndex, testIndex, targetVariable, predictions, y_test)

                            #To check on the advancement
                            if splitNumber % 10 == 0:
                                print(time.time() - startTime)
                                print(decisionTreeRegDictKey)
                            splitNumber += 1
        saveModelToPickleFile (decisionTreeRegDict)
        saveMetricsToFile (decisionTreeRegDict)
        #saveOtherInfoModelDict (decisionTreeRegDict)
        return decisionTreeRegDict
    else:
        return getModelFromPickleFile ("DecisionTreeReg")
    
getDecisionTreeRegressorModel()