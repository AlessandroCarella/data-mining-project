import os.path as path
import os
import pickle
import math
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold

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

def getKnnModel ():
    if not path.exists(getModelPath ("knn")):
        #TODO: change the target variable to the actual variable
        targetVariable = "genre"

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
        
        dowsampledDatasets = {}
        for datasetDimension in getSampleSizeList (dataset.shape[0]):#for the learning curve plot
            dowsampledDatasets[datasetDimension] = downsampleDataset(dataset, datasetDimension)

        kf = KFold(n_splits=33, shuffle=True, random_state=42)

        knnDict = {}
        for datasetDimension, subDataset in dowsampledDatasets.items ():
            X = copyAndScaleDataset (df=subDataset, columnsToUse=continuousFeatures)
            y = subDataset[targetVariable]
            splitNumber = 1
            for trainIndex, testIndex in kf.split (X):
                for k in getRangeForK (dataset.shape[0]): #for different values of k
                    X_train, X_test = X.iloc[trainIndex], X.iloc[testIndex]
                    y_train, y_test = y.iloc[trainIndex], y.iloc[testIndex]
                    
                    model = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
                    model.fit (X_train, y_train)

                    predictions=model.predict(X_test)
                    metrics = knnMetrics (predictions=predictions, groundTruth=y_test)
                    knnDict[f"k:{k}, splitNumber:{splitNumber}, datasetDimension:{datasetDimension}"] = {
                        "k":k, 
                        "model": model, 
                        "metrics": metrics,
                        "datasetDimension":datasetDimension,
                        "splitNumber":splitNumber, 
                        "kFoldTrainIndexAndTestIndex":{"trainIndex":trainIndex.tolist(), "testIndex":testIndex.tolist()}, 
                        "targetVariable":targetVariable,
                        "predictions":predictions,#saving those for 
                        "groundTruth":y_test,#roc curve comupation
                    }
                    saveModelToPickleFile (knnDict)
                    saveMetricsToFile (knnDict)
                    saveOtherInfoModelDict (knnDict)
                    print (f"k:{k}, splitNumber:{splitNumber}, datasetDimension:{datasetDimension}")
                splitNumber += 1
        
        return knnDict
    else:
        return getModelFromPickleFile ("knn")

getKnnModel()