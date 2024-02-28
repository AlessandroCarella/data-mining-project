import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import pickle
import os.path as path
from sklearn.tree import DecisionTreeRegressor
import time

from classificationUtils import downsampleDataset, getTrainDatasetPath, copyAndScaleDataset, continuousFeatures, modelNameToModelObject

def compareMetrics (metrics1:dict, metrics2:dict):
    #return True if metrics1 are better than metrics2, False otherwise
    # Check if MSE and MAD are lower in the first model, and if R2 is higher
    mse_condition = metrics1.get("meanSquaredError") < metrics2.get("meanSquaredError")
    mad_condition = metrics1.get("meanAbsoluteDeviation") < metrics2.get("meanAbsoluteDeviation")
    r2_condition = metrics1.get("r2Score") > metrics2.get("r2Score")

    # Return True if all conditions are true, False otherwise
    return mse_condition and mad_condition and r2_condition

"""
def getComapreModelResults (data):
    result_dict = {
        #key=datasetDimension
        #value={k, metrics}
    }

    metrics_to_compare = ['r2Score', 'meanSquaredError', 'meanAbsoluteDeviation']

    for item in data:
        # Create a key for grouping
        key = item['datasetDimension']

        k_value = item['k']
        metrics = item['metrics']
        weight = item["weight"]
    
        # If key is not present in result_dict or metrics are better, update the result_dict
        if key not in result_dict:
            result_dict[key] = {
                'k': k_value,
                'metrics': metrics,
                'weight': weight
            }
        else:
            better = compareMetrics (metrics1=metrics, metrics2=result_dict[key]["metrics"], metrics_to_compare=metrics_to_compare)
            if better:
                #print (metrics)
                #print ()
                #print (result_dict[key]["metrics"])
                result_dict[key] = {
                    'k': k_value,
                    'metrics': metrics,
                    'weight':weight
                }

    return result_dict
        
def compareModels ():
    #models = getModelFromPickleFile ("knn")
    decisionTreeRegModels = modelNameToModelObject ("decisionTreeReg")
    values = []
    for key, value in decisionTreeRegModels.items():
        values.append(value)

    results = getComapreModelResults (values)

    return results

def compareBestModels ():
    bestModels = compareModels ()

    for key, value in bestModels.items():
        print (key)
        print (value)
        print()

    bestModel = {'r2Score': float('-inf'), 'meanSquaredError': float('inf'), 'meanAbsoluteDeviation': float('inf')}
    for key, value in bestModels.items():
        if compareMetrics (value.get("metrics"), bestModel):
            bestModel = value.get("metrics")
            bestSize = key
            bestK = value.get("k")
            bestWeight = value.get("weight")
    
    print ("The best model performances are achieved")
    print ("With a dataset of size " + str(bestSize))
    print ("This k:", bestK)
    print ("Using this weight:", bestWeight)
    print ("And those metrics:")
    for key, value in bestModel.items ():
        print (key, ":", value)

def learningCurveForDifferentDatasetSize ():
    data = compareModels ()
    dataset_sizes = sorted(data.keys())

    plt.figure(figsize=(10, 6))

    r2score_values = [data[size]['metrics']['r2Score'] for size in dataset_sizes]
    plt.plot(dataset_sizes, r2score_values, marker='o', label='r2Score')

    # Plotting the learning curves
    meanSquaredError_values = [data[size]['metrics']['meanSquaredError'] for size in dataset_sizes]
    plt.plot(dataset_sizes, meanSquaredError_values, marker='o', label='meanSquaredError')

    meanAbsoluteDeviation_values = [data[size]['metrics']['meanAbsoluteDeviation'] for size in dataset_sizes]
    plt.plot(dataset_sizes, meanAbsoluteDeviation_values, marker='o', label='meanAbsoluteDeviation')
    
    # Adding labels and title
    plt.xlabel('Dataset Size')
    plt.ylabel('Metrics Value')
    plt.title('Learning Curve for Different Dataset Sizes')
    plt.legend()
    plt.xticks(dataset_sizes)  # Set x-axis ticks to dataset sizes
    plt.grid(True)
    plt.show()
"""
     
def getBestDecisionTreeRegressorModel (maxDepth:int=0, criterion:str="squared_error", minSampleLeaf:int=1, minSampleSplit:int=1)->DecisionTreeRegressor:
    return DecisionTreeRegressor(
        max_depth=maxDepth, 
        criterion=criterion, 
        min_samples_leaf=minSampleLeaf, 
        min_samples_split=minSampleSplit, 
        random_state=69
        #n_jobs=-1
    )

import pickle
def getTempModel ():
    with open ("decisionTreeRegTemp.pickle", "rb") as f:
        return pickle.load(f)

def getBestDecsionTreeReg ():
    decisionTreeRegModels = modelNameToModelObject ("decisionTreeReg duration_ms")
    #decisionTreeRegModels = getTempModel()

    for key, elem in decisionTreeRegModels.items ():
        print (elem["metrics"])

    bestModel = {"metrics":{"meanSquaredError":float ('inf'), "meanAbsoluteDeviation":float ('inf'), "r2Score":float ('-inf')}}
    for key, tempModel in decisionTreeRegModels.items():
        print (key)
        if compareMetrics (tempModel["metrics"], bestModel["metrics"]):
            bestModel = tempModel

    return bestModel

#compareBestModels ()
#learningCurveForDifferentDatasetSize ()

"""model = getBestDecsionTreeReg ()
print("Criterion:", model["criterion"])
print("Max Depth:", model["maxDepth"])
print("Min Sample Leaf:", model["minSampleLeaf"])
print("Min Sample Split:", model["minSampleSplit"])
for metric, score in model["metrics"].items():
    print(metric, score)"""

#getBestDecisionTreeRegressorModel (maxDepth=top10Models[0]["maxDepth"], criterion=top10Models[0]["criterion"], minSampleLeaf=top10Models[0]["minSampleLeaf"], minSampleSplit=top10Models[0]["minSampleSplit"])