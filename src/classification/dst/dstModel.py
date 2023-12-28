import os.path as path
import math
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from utils import getTrainDatasetPath, continousAndCategorialFeaturesForClassification, saveModelParams

def makeDecisionTreeClassifierDictValue (criterion:str, maxDepth:int, minSampleLeaf:int, minSampleSplit:int, splitNumber:int, ccp_alpha, metrics,model):
    return {
        "criterion":criterion, 
        "max_depth":maxDepth, 
        "min_samples_leaf":minSampleLeaf, 
        "min_samples_split":minSampleSplit, 
        "splitNumber":splitNumber,
        "ccp_alpha":ccp_alpha,
        "metrics": metrics,
        "model": model
    }

def getDecisionTreeModel (targetVariable):
    columnsToUse = continousAndCategorialFeaturesForClassification

    dataset = pd.read_csv (getTrainDatasetPath())[columnsToUse]

    if(targetVariable == "mode"):
        dataset["genre"]= LabelEncoder().fit_transform(dataset["genre"])
    
    X = dataset.copy().drop(targetVariable, axis=1)  
    Y =  dataset[targetVariable]
    kf = KFold(n_splits=20, shuffle=True, random_state=42)

    criterions =  ['entropy', 'gini']
    maxDepths = list(np.arange(2, 3)) + [None]
    minSamplesLeaf= [ 0.1, 0.2, 1, 2 ,3 ,4 ,5 , 6]
    minSamplesSplit=[ 0.05, 0.1, 0.2]
    ccp_alphas= list(np.arange(0.01, 0.1))

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
                                
                                f1Score = f1_score(predictions, y_test, average="weighted")
                                recallScore = recall_score(predictions, y_test, average="weighted", zero_division=1)
                                precisionScore = precision_score(predictions, y_test, average="weighted", zero_division=1)
                                
                                metrics = {'accuracyScore':accuracyScore, "f1Score":f1Score, "precisionScore": precisionScore, "recallScore":recallScore}

                                decisionTreeRegKey = f"DST with criterion:{criterion}, maxDepth:{maxDepth}, minSampleLeaf:{minSampleLeaf}, minSampleSplit:{minSampleSplit}, splitNumber:{splitNumber}, ccp_alpha:{ccp_alpha}, metrics:{metrics}"
                                decisionTreeClassifierDict[decisionTreeRegKey] = makeDecisionTreeClassifierDictValue (criterion, maxDepth, minSampleLeaf, minSampleSplit, splitNumber,ccp_alpha,  metrics, model)

                                print(decisionTreeRegKey)
                                splitNumber += 1
                                #saveModelParams (decisionTreeClassifierDict, targetVariable)
    return decisionTreeClassifierDict