import pandas as pd
import os.path as path

def saveDfToFile (df:pd.DataFrame):
    df.to_csv (path.join(path.abspath(path.dirname(__file__)), "../../dataset (missing + split)/trainFinalWithClustering.csv"), index=False)

def columnAlreadyInDf (columnName, df):
    return columnName in df.columns