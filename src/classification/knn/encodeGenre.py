from classificationUtils import getTrainDatasetPath
import pandas as pd
import os.path as path

df = pd.read_csv (getTrainDatasetPath())

mapping =  {'j-idol': 14, 'afrobeat': 0, 'spanish': 17, 'study': 18, 'breakbeat': 4, 'forro': 7, 'idm': 9, 'mandopop': 15, 'chicago-house': 5, 'bluegrass': 2, 'black-metal': 1, 'brazil': 3, 'iranian': 12, 'happy': 8, 'indian': 10, 'j-dance': 13, 'industrial': 11, 'techno': 19, 'disney': 6, 'sleep': 16}

df["genreEncoded"] = df["genre"].map (mapping)

group1 = [0,2,3,6,7,10,13,15,17]
group2 = [16,18]
group3 = [1,4,5,9,12,19]
group4 = [8,11,14]

# Function to map groups
def map_groups(val):
    if val in group1:
        return 0
    elif val in group2:
        return 1
    elif val in group3:
        return 2
    elif val in group4:
        return 3
    else:
        return None  # Handle values not in any group

# Create a new column with grouped categories
df['knnGroupByGenre'] = df['genreEncoded'].apply(map_groups)

df.to_csv(path.join(path.abspath(path.dirname(__file__)), "..", "..", "..", "dataset (missing + split)", "trainFilledWithoutUselessFeatures + Encoding.csv"))
