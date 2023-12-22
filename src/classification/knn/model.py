import os.path as path
import os
import pickle
import math
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
import numpy as np

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

def makeKnnDictValue (k:int, model, weight:str, metrics:dict, datasetDimension:int, splitNumber:int, trainIndex:np.ndarray, testIndex:np.ndarray, targetVariable:str, predictions:np.ndarray, y_test:pd.Series):
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

def getKnnModel (targetVariable = "genre"):
    columnsToUse = continuousFeatures
    if targetVariable in columnsToUse:
        columnsToUse.remove (targetVariable)

    if not path.exists(getModelPath ("knnGroupByGenre")):
    #if True:
        import time
        startTime = time.time()

        dataset = pd.read_csv (getTrainDatasetPath())
        
        """
        You
        does it make sense to use both downsampled dataset iterations and k fold cross validation on a dataset? 
        iterating through the downsampled datasets (10%,20%,..., 100%) of the datasets as the outer loop 
        and 
        k fold cross validation as an inner loop?

        ChatGPT
        Yes, it can make sense to use both downsampled dataset iterations and k-fold cross-validation in a nested manner. This approach is often referred to as nested cross-validation and is commonly used in machine learning for model evaluation, especially when dealing with imbalanced datasets.

        Here's a brief explanation of each component:

        Downsampled Dataset Iterations:

        Purpose: Dealing with imbalanced datasets where one class is significantly more prevalent than the others.
        Procedure: Create multiple versions of the dataset by downsampling the majority class to different proportions (e.g., 10%, 20%, ..., 100% of the original size).
        Reasoning: This helps the model to train on various degrees of class imbalance, providing a more robust understanding of how well the model generalizes to different imbalance levels.
        k-Fold Cross-Validation:

        Purpose: Assessing the model's performance by partitioning the dataset into k subsets (folds) and training the model k times, each time using a different fold as the test set.
        Procedure: The dataset is divided into k subsets, and the model is trained and validated k times, with a different fold used as the test set in each iteration.
        Reasoning: This helps to obtain a more reliable estimate of the model's performance by reducing the impact of the data partitioning on the evaluation results.
        When combined in a nested manner, you are essentially performing model evaluation at two levels. The outer loop (downsampled dataset iterations) helps you understand how well your model generalizes across different degrees of class imbalance, while the inner loop (k-fold cross-validation) provides a robust estimate of the model's performance for a given downsampling level.

        Here's a possible nested structure:

        Outer Loop:

        Iterate through different downsampled datasets (e.g., 10%, 20%, ..., 100%).
        Inner Loop:

        For each downsampled dataset, perform k-fold cross-validation.
        This nested approach provides a more comprehensive evaluation of your model's performance, considering both class imbalance and generalization across different subsets of the data.
        """
        
        dowsampledDatasets = getDownsampledDataset (dataset=dataset)

        kf = KFold(n_splits=33, shuffle=True, random_state=42)

        weights = ['uniform', 'distance']

        knnDict = {}
        for weight in weights:
            for datasetDimension, subDataset in dowsampledDatasets.items ():
                X = copyAndScaleDataset (df=subDataset, columnsToUse=columnsToUse)
                y = subDataset[targetVariable]
                
                splitNumber = 1
                for trainIndex, testIndex in kf.split (X):
                    for k in getRangeForK (datasetDimension): #for different values of k
                        X_train, X_test = X.iloc[trainIndex], X.iloc[testIndex]
                        y_train, y_test = y.iloc[trainIndex], y.iloc[testIndex]
                        
                        model = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
                        model.fit (X_train, y_train)

                        predictions=model.predict(X_test)
                        metrics = knnMetrics (predictions=predictions, groundTruth=y_test)


                        knnDictKey = f"k:{k}, splitNumber:{splitNumber}, datasetDimension:{datasetDimension}, weights:{weight}"
                        knnDict[knnDictKey] = makeKnnDictValue (k, model, weight, metrics, datasetDimension, splitNumber, trainIndex, testIndex, targetVariable, predictions, y_test)

                        #To check on the advancement
                        print (time.time() - startTime)
                        print (knnDictKey)
                    splitNumber += 1
        saveModelToPickleFile (knnDict)
        saveMetricsToFile (knnDict)
        #saveOtherInfoModelDict (knnDict)
        return knnDict
    else:
        return getModelFromPickleFile ("knnGroupByGenre")

getKnnModel()