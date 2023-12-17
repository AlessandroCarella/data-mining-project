import numpy as np
import pandas as pd
import math
import os
from sklearn.preprocessing import LabelEncoder
os.chdir("../cleaning")


#Bring the df for all necesary transformations
df = pd.read_csv("../dataset (missing + split)/PRUEBA_Dataset_nomissing_nofeatur_noutlier.csv", skip_blank_lines=True)

categoricalFeatures = [
    #"name",
    "explicit",
    "artists",
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

#One-hot encoding
def label_encode_strings(df):


    # Make a copy of the DataFrame to avoid modifying the original
    df_encoded = df.copy()

    # since the condition does not detect true false let's do it manuallly in the method
    df_encoded["explicit"] = df_encoded["explicit"].map({True: 1, False: 0})
    
    #Time_signature should be an int
    df['time_signature'] = pd.to_numeric(df['time_signature'], errors='coerce')

    df['time_signature'] = df['time_signature'].astype(int)


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


#Log transformation adding a small constant so that zeros and negatives don't disappear.
def log_transform_df(df, continuous_features, constant=1e-6):
    transformed_df = df[continuousFeatures]
    for col in continuous_features:
        if (df[col] <= 0).any():
            # Check for values less than or equal to zero
            transformed_df[col] = np.log(df[col] + constant)
            # Adding a small constant value to handle zero or negative values
            print(f"Column '{col}' contains values less than or equal to zero. Added a constant before log transformation.")
        else:
            transformed_df[col] = np.log(df[col] + constant)
    return transformed_df


categ_df = select_categorical_features(df,categoricalFeatures)
categ_df, categories, mapping = label_encode_strings(categ_df)
categ_df.to_csv("../dataset (missing + split)/R_Cat_Prueba_Dataset_nomissing_nofeatur_noutlier.csv", index=False)

normlz_df = normalize_df(df,continuousFeatures)
normlz_df['genre'] = categ_df['genre']
#For classification models we still need to have the target pasted in the continuous df so...
normlz_df.to_csv("../dataset (missing + split)/R_Norm_Prueba_Dataset_nomissing_nofeatur_noutlier.csv", index=False)

#Creating the df for logistic reg
normlz_df['explicit'] = categ_df['explicit']
#We need to take into account only the songs that contain lyrics, since the explicit label is for songs that contain strong verbal language
normlz_df = normlz_df[normlz_df['speechiness'] >= 0.6]
#THIS IS THE LOGISTIC REG DF
normlz_df.to_csv("../dataset (missing + split)/R_LogisticReg_Norm_Prueba_Dataset_nomissing_nofeatur_noutlier.csv", index=False)

#Applying the log transformation into the already normalized data
log_normlz_df = log_transform_df(normlz_df,continuousFeatures,constant=3)
log_normlz_df.to_csv("../dataset (missing + split)/R_Log_Norm_Prueba_Dataset_nomissing_nofeatur_noutlier.csv", index=False)

#Creating a log tranform data without the previous normalization
#Constant = 10000 (Fine results)
#Constant = 19 (Minimum possible without falling into NaN's) 
log_df = log_transform_df(df,continuousFeatures,constant=10000)
log_df['genre'] = categ_df['genre']
log_df.to_csv("../dataset (missing + split)/R_Log_Prueba_Dataset_nomissing_nofeatur_noutlier.csv", index=False)


