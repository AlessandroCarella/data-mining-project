import os.path as path
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import math
from sklearn.utils import resample
import json

originalDatasetColumnsToUse = [
    "name",
    "duration_ms",
    "explicit",
    "popularity",
    "artists",
    "album_name",
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature",
    "n_beats",
    "genre"
]

categoricalFeatures = [
    "name",
    "explicit",
    "artists",
    "album_name",
    "key",
    "genre",
    "time_signature"
]

categoricalFeaturesForPlotting = [
    #"name",
    #"explicit",
    #"artists",
    #"album_name",
    "key",
    "genre",
    "time_signature"
]

continuousFeatures = [
    "duration_ms",
    "popularity",
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "n_beats"
]

def getModelsPickleFolder ():
    return path.join(path.abspath(path.dirname(__file__)), "..", "modelPickle")

def getModelPath (modelName:str):
    for file in os.listdir(getModelsPickleFolder()):
        if modelName.lower() in file.lower():
            return path.join(getModelsPickleFolder(), file)
    return path.join(getModelsPickleFolder(), "Knn Model.pickle")

def createModelPath (modelName:str):
    return path.join(getModelsPickleFolder(), modelName + ".pickle")

def getMetricsPath (modelName:str):
    return path.join(path.abspath(path.dirname(__file__)), f"{modelName} metrics.txt")

def saveModelToPickleFile (model):
    with open (createModelPath ("knn"), "wb") as f:
        pickle.dump (model, f)

def getModelFromPickleFile(modelName:str):
    if path.exists (getModelPath(modelName)):
        with open (getModelPath (modelName), "rb") as f:
            return pickle.load ()
    else:
        raise Exception ("Could not find model")
    
def getTrainDatasetPath ():
    return path.join(path.abspath(path.dirname(__file__)), "..", "..", "..", "dataset (missing + split)", "trainFilledWithoutUselessFeatures.csv")

def saveScalerToPickleObject (scaler, modelName):
    if not path.exists (path.join (getModelsPickleFolder (), f"scaler {modelName}.pickle")):
        with open (path.join (getModelsPickleFolder (), f"scaler {modelName}.pickle"), "wb") as f:
            pickle.dump (scaler, f)

def getScalerPickleObject ():
    with open (path.join (getModelsPickleFolder (), "scaler.pickle"), "rb") as f:
        return pickle.load (f)

def getSampleSizeList (number):
    percentages = []

    for i in range(10, 101, 10):
        percentage_value = number * i / 100
        rounded_value = math.ceil(percentage_value)  # Round up to the next integer
        percentages.append (rounded_value)

    return percentages

def downsampleDataset (dataset:pd.DataFrame, n:int, randomState=69):
    resampled_array = resample(dataset, n_samples=n, replace=False, random_state=randomState)
    return pd.DataFrame(resampled_array, columns=dataset.columns)

def copyAndScaleDataset (df:pd.DataFrame, columnsToUse:list[str])->pd.DataFrame:
    #create a copy of the dataset to select only certain features
    tempDf = df.copy()
    tempDf = tempDf [columnsToUse]

    #scale the temp dataset to use the clustering algorithm
    scaler = StandardScaler()
    scaler.fit(tempDf)

    #saveScalerToPickleObject (scaler, modelName)

    tempDfScal = scaler.transform(tempDf)

    return pd.DataFrame(tempDfScal, columns=columnsToUse)

def wrongSplitDataset(dataset: pd.DataFrame, targetVariable:str, datasetTrainPercentage: int, datasetTrainForValidationPercentage: int, randomState:int=69) -> dict:
    """
    the results will always be the same if the random state parameter of the method is not changed
    the result will vary for different target variables tho

    splits the dataset in 
                  --------datasetTrainForValidation
                 |
        ----datasetTrain
        |        |
        |         --------datasetValidation
    dataset
        |
        ----datasetTest
    and relatives yTrain, yTest, yTrainForValidation, yValidation

    Splits the dataset into train, validation, and test sets along with their respective labels.

    Parameters:
    - dataset: pd.DataFrame, the input dataset.
    - datasetTrainPercentage: int, percentage of the dataset used for training.
    - datasetTrainForValidationPercentage: int, percentage of the dataset used for training the validation set.

    Returns:
    - dict with keys 'X_train', 'y_train', 'X_validation', 'y_validation', 'X_test', 'y_test'.
    """
    # Splitting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(dataset.drop(targetVariable, axis=1), dataset[targetVariable],
                                                        test_size=(100 - datasetTrainPercentage) / 100,
                                                        random_state=randomState)

    # Splitting the training set for validation
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train,
                                                                    test_size=(100 - datasetTrainForValidationPercentage) / 100,
                                                                    random_state=randomState)

    # Creating the result dictionary
    result_dict = {
        'X_train': X_train,
        'y_train': y_train,
        'X_validation': X_validation,
        'y_validation': y_validation,
        'X_test': X_test,
        'y_test': y_test
    }

    return result_dict

def splitDataset(dataset: pd.DataFrame, targetVariable:str, datasetTrainPercentage: int, datasetTrainForValidationPercentage: int, randomState:int=69) -> dict:
    """
    THIS IS AN ALMOST COPY OF THE METHOD BEFORE, THE DIFFERENCE IS THAT WE DON'T NEED TO SPLIT BETWEEN TRAIN AND TEST (AND THAN TRAIN INTO VALIDATIONS)
    SINCE WE ALREADY HAVE A TEST DATASET THAT WE CAN USE IN ANOTHER FILE
    SO WE JUST NEED ONE OF THE SPLITS (THE FIRST ONE)

    the results will always be the same if the random state parameter of the method is not changed
    the result will vary for different target variables tho

    splits the dataset in 
                  --------datasetTrainForValidation
                 |
        ----datasetTrain
        |        |
        |         --------datasetValidation
    dataset
        |
        ----datasetTest
    and relatives yTrain, yTest, yTrainForValidation, yValidation

    Splits the dataset into train, validation, and test sets along with their respective labels.

    Parameters:
    - dataset: pd.DataFrame, the input dataset.
    - datasetTrainPercentage: int, percentage of the dataset used for training.
    - datasetTrainForValidationPercentage: int, percentage of the dataset used for training the validation set.

    Returns:
    - dict with keys 'X_train', 'y_train', 'X_validation', 'y_validation', 'X_test', 'y_test'.
    """
    # Splitting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(dataset.drop(targetVariable, axis=1), dataset[targetVariable],
                                                        test_size=(100 - datasetTrainForValidationPercentage) / 100, #datasetTrainPercentage) / 100,
                                                        random_state=randomState)

    # Splitting the training set for validation
    #X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train,
    #                                                                test_size=(100 - datasetTrainForValidationPercentage) / 100,
    #                                                                random_state=randomState)

    # Creating the result dictionary
    result_dict = {
        'X_train': X_train,
        'y_train': y_train,
        'X_validation': X_test, #X_validation,
        'y_validation': y_test, #y_validation,
        #'X_test': X_test,
        #'y_test': y_test
    }

    return result_dict

def getInfoPath (modelName):
    return path.join(path.abspath(os.path.dirname(__file__)), f"{modelName} infos.txt")

def saveOtherInfoModelDict (modelDict:dict):
    def addSingleInstanceToFile (filename, kValueAndOtherIdInfo, dictToSave):
        with open(filename, 'a') as file:
            file.write(f"k:{kValueAndOtherIdInfo}\n")
            json.dump(dictToSave, file, indent=4)  # Write the dictionary with indentation
            file.write("\n---------------------\n")

    #create or reset the info file    
    with open (getInfoPath("knn"), "w") as file:
        file.write ("knn dict infos:\n")
        file.write("\n---------------------\n")

    for key, values in modelDict.items():
        addSingleInstanceToFile (getInfoPath ("knn"), key, {
                "k":values.get("k"),
                "datasetDimension":values.get ("datasetDimension"), 
                "splitNumber":values.get("splitNumber"),
                "kFoldTrainIndexAndTestIndex":values.get ("kFoldTrainIndexAndTestIndex"),
                "targetVariable":values.get ("targetVariable")
                }
            )
