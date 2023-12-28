import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

from classificationUtils import continuousFeatures, downsampleDataset, getTrainDatasetPath, getTestDatasetPath, copyAndScaleDataset
from metrics import knnRegMetrics

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

def modelTest (targetVariable="popularity", k=33, sampleSize=1500):
    columnsToUse = continuousFeatures
    if targetVariable in columnsToUse:
        columnsToUse.remove (targetVariable)
        
    originalDataset = pd.read_csv (getTrainDatasetPath())
    scaler = getDatasetScaler (originalDataset=originalDataset, datasetDownscaleSize=sampleSize)

    testDataset = pd.read_csv (getTestDatasetPath())
    testDatasetScaled = copyAndScaleDataset (df=testDataset, columnsToUse=continuousFeatures, scaler=scaler)
    groundTruth = testDataset[targetVariable]

    dowsampledDataset = downsampleDataset (originalDataset, n=sampleSize)
    X = copyAndScaleDataset (df=dowsampledDataset, columnsToUse=continuousFeatures, scaler=scaler)
    y = dowsampledDataset [targetVariable]
    
    model = KNeighborsRegressor(n_neighbors=k, weights="uniform", n_jobs=-1)
    model.fit (X, y)

    predictions = model.predict (testDatasetScaled)

    return knnRegMetrics (predictions=predictions, groundTruth=groundTruth)

for key, value in modelTest("danceability", 47).items():
    print (key,":")
    print (value)

print ()
print (modelTest("danceability", 47))
print ()