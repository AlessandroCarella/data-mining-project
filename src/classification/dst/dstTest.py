import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from utils import getTestDatasetPath, getTrainDatasetPath
from plotTrees import plotDecisionTree, plotConfusionMatrix
from sklearn.preprocessing import KBinsDiscretizer

def testModel (targetVariable, tree, desc, columnsToUse):
    dataset = pd.read_csv (getTrainDatasetPath())[columnsToUse]
    testDataset = pd.read_csv (getTestDatasetPath())[columnsToUse]
    
    if(targetVariable == "mode"):
        mode_value = testDataset[targetVariable].mode().iloc[0]  
        testDataset[targetVariable].fillna(mode_value, inplace=True)
        dataset["genre"]= LabelEncoder().fit_transform(dataset["genre"]) 
        testDataset["genre"]= LabelEncoder().fit_transform(testDataset["genre"])
    
    
        X_train = dataset.copy().drop(targetVariable, axis=1)
        Y_train = dataset[targetVariable]


    X_test = testDataset.drop(targetVariable, axis=1) 
    Y_test = testDataset[targetVariable]
    
    print(desc + "\n")
    y_train_pred = tree.predict(X_train)
    print('Train Accuracy %s' % accuracy_score(Y_train, y_train_pred))
    print('Train F1-score %s' % f1_score(Y_train, y_train_pred, average='weighted'))
    print('Train Precision Score %s' % precision_score(Y_train, y_train_pred, average='weighted',zero_division=1))
    print('Train Recall Score %s' % recall_score(Y_train, y_train_pred, average='weighted', zero_division=1))

    predictions = tree.predict(X_test)
    print('Test Accuracy %s' % accuracy_score(Y_test, predictions))
    print('Test F1-score %s' % f1_score(Y_test, predictions, average='weighted'))
    print('Test Precision Score %s' % precision_score(Y_test, predictions, average='weighted',zero_division=1))
    print('Test Recall Score %s' % recall_score(Y_test, predictions, average='weighted',zero_division=1))
    
    plotDecisionTree(tree)
    plotConfusionMatrix(Y_test, predictions)