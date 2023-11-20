from dbscan import dbscan
import pandas as pd

from clusteringUtility import copyAndScaleDataset

df = pd.read_csv ("originalDataset.csv")
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

df, dbscanColumnName = dbscan(df, continuousFeatures, eps= [4, 20], min_samples=[3, 7])#eps= [0.1, 3], min_samples=[1, 20])

df.to_csv("original and dbscan.csv")