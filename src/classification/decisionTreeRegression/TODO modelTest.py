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

    #TODO change this for the new results
    #Since the results with the best model found are 
    #{'accuracy': 0.4164, 'precision': 0.43231061115016034, 'recall': 0.41640000000000005, 'f1Score': 0.39274632191616515}
    #I decided to try with the full dataset to check if the results are better if i use the same k for the smaller dataset
    #those are the best results for the full dataset btw:
    #{'accuracy': 0.48, 'precision': 0.48525256281376805, 'recall': 0.48, 'f1Score': 0.4635663793692188}
    #but when using k=35 as used for the 15000 dataset the results are slightly better
    #{'accuracy': 0.5074, 'precision': 0.5067468815233543, 'recall': 0.5074, 'f1Score': 0.49382314980438785}
    knnModel = getBestKnnModel (k=25, datasetSize=15000)
    #and i got those results
    #

    predictions = knnModel.predict (testDatasetScaled)

    return knnMetrics (predictions=predictions, groundTruth=groundTruth)

for key, value in modelTest().items():
    print (key,":")
    print (value)

print (modelTest())