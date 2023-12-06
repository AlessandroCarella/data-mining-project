import numpy as np
import pandas as pd
import math
import os
from sklearn.preprocessing import LabelEncoder
os.chdir("../clustering/Methods")


#Bring the df for all necesary transformations
df = pd.read_csv("../dataset (missing + split)/PRUEBA_Dataset_nomissing_nofeatur_noutlier.csv", skip_blank_lines=True)

categoricalFeatures = [
    #"name",
    "explicit",
    #"artists",
    #"album_name",
    "key",
    "genre",
    "time_signature"
]

continuousFeatures = [
    "duration_ms",
    "popularity",
    #"danceability",
    #"energy",
    "loudness",
    "speechiness",
    #"acousticness",
    "instrumentalness",
    "liveness",
    #"valence",
    "tempo",
    "n_beats"
]


#z-score normalization
def normalize_df(df, continuousFeatures):
    df = df[continuousFeatures]
    for col in continuousFeatures:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df

def select_categorical_features(df, categoricalFeatures):
    df = df[categoricalFeatures]
    selected_df = df[categoricalFeatures]
    return selected_df

def label_encode_strings(df):
    # Make a copy of the DataFrame to avoid modifying the original
    df_encoded = df.copy()

    # Select columns with object (string) datatype
    string_columns = df.select_dtypes(include=['object']).columns.tolist()

    # Apply label encoding to string columns and store mapping in a dictionary
    label_encoders = {}
    encoded_values_mapping = {}
    
    for col in string_columns:
        label_encoders[col] = LabelEncoder()
        encoded_values = label_encoders[col].fit_transform(df[col])
        df_encoded[col] = encoded_values

        # Create a dictionary to map original string values to their encoded counterparts
        encoded_values_mapping[col] = dict(zip(df[col], encoded_values))

    return df_encoded, label_encoders, encoded_values_mapping

normlz_df = normalize_df(df,continuousFeatures)
normlz_df.to_csv("../dataset (missing + split)/R_Norm_Prueba_Dataset_nomissing_nofeatur_noutlier.csv", index=False)

categ_df = select_categorical_features(df,categoricalFeatures)
categ_df, categories, mapping = label_encode_strings(categ_df)
categ_df.to_csv("../dataset (missing + split)/R_Cat_Prueba_Dataset_nomissing_nofeatur_noutlier.csv", index=False)


