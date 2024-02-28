import pandas as pd

def createDatasetOnlyDbscanHdbscan ():
    df = pd.read_csv("trainFinalWithClustering 16 11 2023.csv")

    datasetColumns = [
        "name", "duration_ms", "explicit", "popularity", "artists", "album_name",
        "danceability", "energy", "key", "loudness", "mode", "speechiness", 
        "acousticness", "instrumentalness", "liveness", "valence", "tempo", 
        "time_signature", "n_beats", "genre"
    ]
    dbscan_and_hdbscan_columns = [col for col in df.columns if "dbscan" in col]

    selected_df = df[datasetColumns + dbscan_and_hdbscan_columns]

    selected_df.to_csv ("x.csv")
    return selected_df

#createDatasetOnlyDbscanHdbscan ()

df = pd.read_csv ("x.csv")
dbscan_and_hdbscan_columns = [col for col in df.columns if "dbscan" in col]
uniqueValues = {}
for col in dbscan_and_hdbscan_columns:
    uniqueValues[col] = (df[col].unique())

# Assuming you have opened a file named 'output.txt' in write mode
with open('output.txt', 'w') as file:
    for key, value in uniqueValues.items():
        file.write(str(key) + '\n')
        file.write(str(value) + '\n')
        file.write('\n')
        file.write("-------------------------------------\n")
        file.write('\n')
