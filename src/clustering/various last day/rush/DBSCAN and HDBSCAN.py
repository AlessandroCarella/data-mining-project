from dbscan import dbscan, optics, hdbscan
import pandas as pd

def createDatasetOnlyDbscanHdbscan ():
    df = pd.read_csv("trainFinalWithClustering.csv")

    datasetColumns = [
        "name", "duration_ms", "explicit", "popularity", "artists", "album_name",
        "danceability", "energy", "key", "loudness", "mode", "speechiness", 
        "acousticness", "instrumentalness", "liveness", "valence", "tempo", 
        "time_signature", "n_beats", "genre"
    ]
    dbscan_and_hdbscan_columns = [col for col in df.columns if "dbscan" in col]

    selected_df = df[datasetColumns + dbscan_and_hdbscan_columns]

    selected_df.to_csv ("trainFinalWithDbscanHdbscanClustering.csv")
    return selected_df

#createDatasetOnlyDbscanHdbscan ()

df = pd.read_csv ("original and dbscan.csv")

from metrics import sse, silhouette, continuousFeatures

"""# Find columns with "hdbscan"
hdbscan_columns = [col for col in df.columns if 'hdbscan' in col.lower()]
#sse (df, hdbscan_columns, "hdbscan")
#silhouette (df, hdbscan_columns, "hdbscan")
"""
# Find columns with "dbscan"
dbscan_columns = [col for col in df.columns if 'dbscan' in col.lower()]
#sse (df, dbscan_columns, "dbscan")
silhouette (df, dbscan_columns, "dbscan2")