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
    recall = recall_score(groundTruth, predictions, average='weighted', zero_division=0)

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

def plotROCcurve (fpr, tpr, rocAuc):
    #all the inputs are found in the output of the knnMetrics method.get("rocCurve").get("fpr" or "tpr" or "rocAuc")
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {rocAuc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='black', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - KNN Model')
    plt.legend()
    plt.show()

def plotROCcurvesByKRange (kRange:list[int], metrics:dict):
    #krange shound be a list of Ks such as [3,5,7,9,11,...]
    #metrics should be the dictionary returned by the method knnMetrics
    for k in kRange:
        plt.figure(figsize=(8, 6))
        
        values = metrics.get(k)
        plt.plot(values.get("fpr"), values.get("tpr"), label=f'K={k} (AUC = {values.get("roc_auc"):.2f})')

    plt.plot([0, 1], [0, 1], linestyle='--', color='black', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Different Values of K')
    plt.legend()
    plt.show()

def saveMetricsToFile (knnDict:dict):
    def addSingleInstanceToFile (filename, kValueAndOtherIdInfo, dictToSave):
        with open(filename, 'a') as file:
            file.write(f"k:{kValueAndOtherIdInfo}\n")
            json.dump(dictToSave, file, indent=4)  # Write the dictionary with indentation
            file.write("\n---------------------\n")

    #create or reset the metrics file    
    with open (getMetricsPath("knn"), "w") as file:
        file.write ("knn classificator metrics:\n")
        file.write("\n---------------------\n")

    for key, values in knnDict.items():
        addSingleInstanceToFile (getMetricsPath ("knn"), key, values.get ("metrics"))
