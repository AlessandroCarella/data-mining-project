import os.path as path
import math
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from utils import getTrainDatasetPath, continousAndCategorialFeaturesForClassification, saveModelParams

def makeDecisionTreeClassifierDictValue (criterion:str, maxDepth:int, minSampleLeaf:int, minSampleSplit:int, splitNumber:int, ccp_alpha, metrics):
    return {
        "criterion":criterion, 
        "max_depth":maxDepth, 
        "min_samples_leaf":minSampleLeaf, 
        "min_samples_split":minSampleSplit, 
        "splitNumber":splitNumber,
        "ccp_alpha":ccp_alpha,
        "metrics": metrics
    }

def getDecisionTreeModel (targetVariable = "genre"):
    columnsToUse = continousAndCategorialFeaturesForClassification  
    
    import time
    startTime = time.time()

    dataset = pd.read_csv (getTrainDatasetPath())[columnsToUse]
    X = dataset.copy().drop(targetVariable, axis=1)
    dataset[targetVariable]= LabelEncoder().fit_transform(dataset[targetVariable])     
    Y =  dataset[targetVariable]
    kf = KFold(n_splits=33, shuffle=True, random_state=42)

    criterions =  ['gini', 'entropy']
    maxDepths = [None] + list(np.arange(2, 20))
    minSamplesLeaf=list(np.arange(2, 20))
    minSamplesSplit=list(np.arange(2, 20))
    ccp_alphas= list(np.arange(0.0, 0.1))

    decisionTreeClassifierDict = {}
    for criterion in criterions:
        for maxDepth in maxDepths:
            for minSampleLeaf in minSamplesLeaf:
                for minSampleSplit in minSamplesSplit:
                    for ccp_alpha in ccp_alphas:
                        splitNumber = 1
                        for trainIndex, testIndex in kf.split (X):
                                X_train, X_test = X.iloc[trainIndex], X.iloc[testIndex]
                                y_train, y_test = Y.iloc[trainIndex], Y.iloc[testIndex]
                                model = DecisionTreeClassifier(
                                    max_depth=maxDepth, 
                                    criterion=criterion, 
                                    min_samples_leaf=minSampleLeaf, 
                                    min_samples_split=minSampleSplit, 
                                    ccp_alpha=ccp_alpha
                                )
                                model.fit (X_train, y_train)

                                predictions=model.predict(X_test)
                                accuracyScore = accuracy_score(predictions, y_test)
                                print('accuracyScore' + str(accuracyScore))
                                if accuracyScore < 0.7:
                                    continue
                                
                                f1Score = f1_score(predictions, y_test, average="weighted")
                                
                                metrics = {'accuracyScore':accuracyScore, "f1Score":f1Score}

                                decisionTreeRegDictKey = f"criterion:, maxDepth:, minSampleLeaf:, minSampleSplit:, splitNumber:, ccp_alpha:, metrics:"
                                decisionTreeClassifierDict[decisionTreeRegDictKey] = makeDecisionTreeClassifierDictValue (criterion, maxDepth, minSampleLeaf, minSampleSplit, splitNumber,ccp_alpha,  metrics)

                                #To check on the advancement
                                if splitNumber % 10 == 0:

                                    print(time.time() - startTime)
                                    print(decisionTreeRegDictKey)
                                splitNumber += 1
                                saveModelParams (decisionTreeClassifierDict)
        return decisionTreeClassifierDict
    
getDecisionTreeModel()