import pandas as pd
import os.path as path
import time 
import tqdm as tqdm  
import seaborn as sns
import matplotlib.pyplot as plt
import os
import os.path as path

def removeUselessFeatures ():
    trainDfFilled = pd.read_csv (path.join(path.abspath(path.dirname(__file__)), "../../dataset (missing + split)/trainFilled.csv"))
    originalFeatures = list(trainDfFilled.keys())
    """
    ['name', 'duration_ms', 'explicit', 'popularity', 'artists',
        'album_name', 'danceability', 'energy', 'key', 'loudness', 'mode',      
        'speechiness', 'acousticness', 'instrumentalness', 'liveness',
        'valence', 'tempo', 'features_duration_ms', 'time_signature', 'n_beats',
        'n_bars', 'popularity_confidence', 'processing', 'genre']
    """

    #copy of duration_ms, correlation 1
    originalFeatures.remove('features_duration_ms')
    #copy of n_beats, correlation 0.98
    originalFeatures.remove ('n_bars')
    #very few values in it
    originalFeatures.remove ('popularity_confidence')
    #key and processing don't have a direct correlation but they have 
    #matching values everytime so i think that keeping only key 
    #(which is in the spotify api) is the best option
    originalFeatures.remove ('processing')

    filteredFeatures = originalFeatures
    trainDfFilledWithoutUselessFeatures = trainDfFilled[filteredFeatures]

    #print (trainDfFilled.shape)# (15000, 24)
    #print (trainDfFilledWithoutUselessFeatures.shape) #(15000, 20)

    #missing_values = trainDfFilledWithoutUselessFeatures.isna().sum()
    #print("Missing values in each column:")
    #print(missing_values)
    """
    Missing values in each column:
    name                0
    duration_ms         0
    explicit            0
    popularity          0
    artists             0
    album_name          0
    danceability        0
    energy              0
    key                 0
    loudness            0
    mode                0
    speechiness         0
    acousticness        0
    instrumentalness    0
    liveness            0
    valence             0
    tempo               0
    time_signature      0
    n_beats             0
    genre               0
    """
    
    return trainDfFilledWithoutUselessFeatures

def generateClusterMap (data):
    corr_matrix = data.select_dtypes(include=['int', 'float']).corr()

    # Create a larger figure with extra height for better readability at the bottom
    plt.figure(figsize=(21, 9))

    # Use hierarchical clustering to reorder the correlation matrix and display values
    clustermap = sns.clustermap(corr_matrix, cmap='coolwarm', annot=True, figsize=(21, 12))

    # Create a directory to save files
    output_directory = path.join(path.dirname(__file__), '../dataUnderstaing','analysis')

    # Save the clustermap image with higher resolution
    clustermap.savefig(os.path.join(output_directory, "visualizations", "clustermapWithoutUselessFeatures.png"), dpi=600)

    # Close the previous figure
    plt.close()

noUselessFeaturesDf = removeUselessFeatures()
generateClusterMap(noUselessFeaturesDf)

noUselessFeaturesDf.to_csv (path.join(path.abspath(path.dirname(__file__)), "../../dataset (missing + split)/trainFilled WithoutUselessFeatures.csv"), index=False)
    