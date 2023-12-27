import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

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
    scaler = getDatasetScaler (originalDataset=originalDataset, datasetDownscaleSize=1500)

    testDataset = pd.read_csv (getTestDatasetPath())
    testDatasetScaled = copyAndScaleDataset (df=testDataset, columnsToUse=continuousFeatures, scaler=scaler)
    groundTruth = testDataset[targetVariable]

    dowsampledDataset = downsampleDataset (originalDataset, n=1500)
    X = copyAndScaleDataset (df=dowsampledDataset, columnsToUse=continuousFeatures, scaler=scaler)
    y = dowsampledDataset [targetVariable]
    
    model = KNeighborsClassifier(n_neighbors=35, weights="uniform", n_jobs=-1)
    model.fit (X, y)

    predictions = model.predict (testDatasetScaled)

    return knnMetrics (predictions=predictions, groundTruth=groundTruth)

for key, value in modelTest().items():
    print (key,":")
    print (value)

print ()
print (modelTest())
print ()