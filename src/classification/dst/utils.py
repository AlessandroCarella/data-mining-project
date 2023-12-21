import os.path as path
import os
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

continousAndCategorialFeaturesForClassification = [
    "mode",
    "key",
    #"genre",
    #"time_signature",
    #"duration_ms",
    #"popularity",
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

def getTrainDatasetPath ():
    return path.join(path.abspath(path.dirname(__file__)), "..", "..", "..", "dataset (missing + split)", "trainFilledWithoutUselessFeatures + Encoding.csv")

def getTestDatasetPath ():
    #TODO check if the test dataset is ok as it is right now, might need to do the various operations we did on the train one 
    #(so the path might also change when we create the test dataset with different values, fills etc)
    return path.join(path.abspath(path.dirname(__file__)), "..", "..", "..", "dataset (missing + split)", "test.csv")
    
directoryName = "decisionTreeClassifierModels"
fileName= "decisionTreeClassifierModelsParams"
def create_directory(base_directory='src/classification/dst'):
        model_directory = os.path.join(base_directory, directoryName)

        if not os.path.exists(model_directory):
            os.makedirs(model_directory)

def saveModelParams(tree_values, targetVariable, base_directory='src/classification/dst'):
        create_directory(base_directory)
        file_path = os.path.join(base_directory, directoryName, fileName + " FOR " + targetVariable + " .txt")

        with open(file_path, 'a') as file:
                for key,value in tree_values.items():
                        file.write(f"{key}\n")
                        file.write(f"{value}\n")
                        file.write('-' * 30 + '\n')
