import pandas as pd
from sklearn.preprocessing import StandardScaler

from classificationUtils import continuousFeatures, downsampleDataset, getTrainDatasetPath, getTestDatasetPath, copyAndScaleDataset
from findBestModel import getBestKnnModel
from metrics import knnMetrics

def getDatasetScaler (originalDataset:pd.DataFrame, datasetDownscaleSize=-1, columnsToUse=continuousFeatures) -> StandardScaler:
    if datasetDownscaleSize != -1:
        downsampledDataset = downsampleDataset (dataset=originalDataset, n=datasetDownscaleSize)
    else:
        downsampledDataset = originalDataset

    tempDf = downsampledDataset.copy()
    tempDf = tempDf [columnsToUse]

    scaler = StandardScaler()
    scaler.fit(tempDf)

    return scaler

def modelTest (targetVariable="genre"):
    originalDataset = pd.read_csv (getTrainDatasetPath())
    scaler = getDatasetScaler (originalDataset=originalDataset, datasetDownscaleSize=3000)

    testDataset = pd.read_csv (getTestDatasetPath())
    testDatasetScaled = copyAndScaleDataset (df=testDataset, columnsToUse=continuousFeatures, scaler=scaler)
    groundTruth = testDataset[targetVariable]

    #The model is already trained on the targetVariable="genre",
    #remeber to change the model if you want to predict for other target variable
    #knnModel = getBestKnnModel ()

    #Since the results with the best model found are 
    #{'accuracy': 0.4112, 'precision': 0.4348223885204935, 'recall': 0.4112, 'f1Score': 0.38187370215787847}
    #I decided to try with the full dataset to check if the results are better
    #those are the best results for the full dataset btw:
    #15000: {'k': 113, 'metrics': {'accuracy': 0.512087912087912, 'precision': 0.5302678139424123, 'recall': 0.512087912087912, 'f1Score': 0.4994938879433043}}
    knnModel = getBestKnnModel (k=113, datasetSize=15000)
    #and i got those results
    #{'accuracy': 0.4112, 'precision': 0.4348223885204935, 'recall': 0.4112, 'f1Score': 0.38187370215787847}
    #TODO: ask the teacher if it's normale that i got the same results for both models

    predictions = knnModel.predict (testDatasetScaled)

    return knnMetrics (predictions=predictions, groundTruth=groundTruth)

for key, value in modelTest().items():
    print (key,":")
    print (value)

print (modelTest())