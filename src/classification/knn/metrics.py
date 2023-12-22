from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import json
import os.path as path

from classificationUtils import getMetricsPath

def knnMetrics (predictions, groundTruth):
    # Accuracy
    accuracy = accuracy_score(groundTruth, predictions)

    # Precision
    precision = precision_score(groundTruth, predictions, average='weighted', zero_division=0)

    # Recall
    recall = recall_score(groundTruth, predictions, average='macro', zero_division=0)

    # F1 Score
    f1 = f1_score(groundTruth, predictions, average='weighted')
    
    # Compute ROC curve
    #fpr, tpr, _ = roc_curve(groundTruth, predictions)
    # Compute area under the ROC curve
    #roc_auc = auc(fpr, tpr)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1Score": f1,
        #"rocCurve": {
        #    "fpr": fpr,
        #    "tpr": tpr,
        #    "rocAuc": roc_auc,
        #}
        
    }

def rocCurve ():
    #TODO: ask andrea how to do this, also check the jupyter from the class
    pass

def saveMetricsToFile (knnDict:dict):
    def addSingleInstanceToFile (filename, kValueAndOtherIdInfo, dictToSave):
        with open(filename, 'a') as file:
            file.write(f"{kValueAndOtherIdInfo}\n")
            json.dump(dictToSave, file, indent=4)  # Write the dictionary with indentation
            file.write("\n---------------------\n")

    #create or reset the metrics file    
    with open (getMetricsPath("knnGroupByGenre"), "w") as file:
        file.write ("knnGroupByGenre classificator metrics:\n")
        file.write("\n---------------------\n")

    for key, values in knnDict.items():
        addSingleInstanceToFile (getMetricsPath ("knnGroupByGenre"), key, values.get ("metrics"))
