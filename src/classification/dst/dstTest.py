import os.path as path
import math
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from utils import getTestDatasetPath, continousAndCategorialFeaturesForClassification, saveModelParams

def testModel (targetVariable, tree):

    testDataset = pd.read_csv (getTestDatasetPath())[continousAndCategorialFeaturesForClassification]
    testDataset[targetVariable] = LabelEncoder().fit_transform(testDataset[targetVariable])

    X_test = testDataset.drop(targetVariable, axis=1) 
    Y_test = testDataset[targetVariable]
    
    predictions = tree.predict(X_test)
    
    print('Test Accuracy %s' % accuracy_score(Y_test, predictions))
    print('Test F1-score %s' % f1_score(Y_test, predictions, average='weighted'))
    print('Test Precision Score %s' % precision_score(Y_test, predictions, average='weighted'))
    print('Test Recall Score %s' % recall_score(Y_test, predictions, average='weighted'))
    