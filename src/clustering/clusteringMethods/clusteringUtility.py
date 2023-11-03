import pandas as pd
from sklearn.preprocessing import StandardScaler
import os.path as path

def saveDfToFile (df:pd.DataFrame):
    df.to_csv (path.join(path.abspath(path.dirname(__file__)), "../../../dataset (missing + split)/trainFinalWithClustering.csv"), index=False)

def columnAlreadyInDf (columnName, df):
    return columnName in df.columns

def copyAndScaleDataset (df, columnsToUse):
    #create a copy of the dataset to select only certain features
    tempDf = df.copy()
    tempDf = tempDf [columnsToUse]

    #scale the temp dataset to use the clustering algorithm
    scaler = StandardScaler()
    scaler.fit(tempDf)
    tempDfScal = scaler.transform(tempDf)

    return tempDfScal