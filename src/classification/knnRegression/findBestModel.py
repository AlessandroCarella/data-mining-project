import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import pickle
import os.path as path

from classificationUtils import downsampleDataset, getTrainDatasetPath, copyAndScaleDataset, continuousFeatures, modelNameToModelObject

def compareMetrics (metrics1:dict, metrics2:dict, metrics_to_compare:list=['r2Score', 'meanSquaredError', 'meanAbsoluteDeviation']):
    #return True if metrics1 are better than metrics2, False otherwise
    # Check if MSE and MAD are lower in the first model, and if R2 is higher
    mse_condition = metrics1.get("meanSquaredError", float('inf')) < metrics2.get("meanSquaredError", float('inf'))
    mad_condition = metrics1.get("meanAbsoluteDeviation", float('inf')) < metrics2.get("meanAbsoluteDeviation", float('inf'))
    r2_condition = metrics1.get("r2Score", float('-inf')) > metrics2.get("r2Score", float('-inf'))

    # Return True if all conditions are true, False otherwise
    return mse_condition and mad_condition and r2_condition

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
                """print (metrics)
                print ()
                print (result_dict[key]["metrics"])"""
                result_dict[key] = {
                    'k': k_value,
                    'metrics': metrics,
                    'weight':weight
                }

    return result_dict
        
def compareModels ():
    #models = getModelFromPickleFile ("knn")
    knnModels = modelNameToModelObject ("knn")
    values = []
    for key, value in knnModels.items():
        values.append(value)

    results = getComapreModelResults (values)
    sorted_keys = sorted(results.keys())
    for key in sorted_keys:
        #print (key)
        #print (results.get(key))
        #print ()
        pass

    return results

def compareBestModels ():
    bestModels = compareModels ()

    for key, value in bestModels.items():
        print (key)
        print (value)
        print()

    bestModel = {'r2Score': float('-inf'), 'meanSquaredError': float('-inf'), 'meanAbsoluteDeviation': float('-inf')}
    bestSize = 0
    bestK = 0
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

def getBestKnnModel (k=127, datasetSize=3000, targetVariable="genre"):
    #this method is needed just to get the best model found by the metrics
    #this method should train the classifier on the whole training set, not the train/validation split
    modelFilePath = path.join(path.dirname(__file__), "..", "results", "knnBestModel.pickle")
    #if not path.exists(modelFilePath):
    dataset = pd.read_csv (getTrainDatasetPath())
    
    if dataset.shape[0] > datasetSize:
        dowsampledDataset = downsampleDataset (dataset, datasetSize)
    else:
        dowsampledDataset = dataset

    X = copyAndScaleDataset (df=dowsampledDataset, columnsToUse=continuousFeatures)
    y = dowsampledDataset [targetVariable]
    
    model = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    model.fit (X, y)

    with open (modelFilePath, "wb") as file:
        pickle.dump (model, file)
    
    return model
    """else:
        with open (modelFilePath, "rb") as file:
            return pickle.load (file) """


"""compareBestModels ()
learningCurveForDifferentDatasetSize ()
getBestKnnModel ()"""