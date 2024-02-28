from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import json
import os.path as path

from classificationUtils import getMetricsPath

def decisionTreeRegMetrics(predictions, groundTruth):
    # R2 Score
    r2 = r2_score(groundTruth, predictions)

    # Mean Squared Error (MSE)
    mse = mean_squared_error(groundTruth, predictions)

    # Mean Absolute Deviation (MAD)
    mad = mean_absolute_error(groundTruth, predictions)

    return {
        "r2Score": r2,
        "meanSquaredError": mse,
        "meanAbsoluteDeviation": mad,
    }

def rocCurve ():
    #TODO: ask andrea how to do this, also check the jupyter from the class
    pass

def saveMetricsToFile (decisionTreeRegressorDict:dict):
    def addSingleInstanceToFile (filename, kValueAndOtherIdInfo, dictToSave):
        with open(filename, 'a') as file:
            file.write(f"{kValueAndOtherIdInfo}\n")
            json.dump(dictToSave, file, indent=4)  # Write the dictionary with indentation
            file.write("\n---------------------\n")

    #create or reset the metrics file    
    with open (getMetricsPath("decision tree regressor"), "w") as file:
        file.write ("knn classificator metrics:\n")
        file.write("\n---------------------\n")

    for key, values in decisionTreeRegressorDict.items():
        addSingleInstanceToFile (getMetricsPath ("decision tree regressor"), key, values.get ("metrics"))
