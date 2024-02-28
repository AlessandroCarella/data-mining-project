import pandas as pd

import os.path as path

def getTrainDataset ():
    return path.join(path.dirname(__file__), "../../../dataset (missing + split)/trainFilledWithoutUselessFeatures.csv")

def getDatasetWithPatterMiningFeatures()->pd.DataFrame:
    df = pd.read_csv(getTrainDataset ())
    selected_cols=['duration_ms', 'explicit', 'popularity', 'mode', 'genre', 'danceability', 'energy','speechiness', 'acousticness', 'liveness']
    
    df = df[selected_cols]

    #duration_ms transformation
    df['duration_category'] = pd.cut(df['duration_ms'],
                                    bins=[0, 90000, 120000, 180000, 240000, float('inf')],
                                    labels=['veryShortDuration', 'shortDuration', 'normalDuration', 'longDuration', 'veryLongDuration'], include_lowest=True)
    
    #danceability transformation
    df['danceability_category'] = pd.cut(df['danceability'], bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=['veryLowDanceability', 'lowDanceability', 'moderateDanceability', 'highDanceability', 'veryHighDanceability'], include_lowest=True, right=True)

    #energy transformation
    df['energy_category'] = pd.cut(df['energy'], bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=['veryLowEnergy', 'lowEnergy', 'moderateEnergy', 'highEnergy', 'veryHighEnergy'], include_lowest=True, right=True)
    
    #speechiness transformation
    df['speechiness_category'] = pd.cut(df['speechiness'], bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=['veryLowSpeechiness', 'lowSpeechiness', 'moderateSpeechiness', 'highSpeechiness', 'veryHighSpeechiness'], include_lowest=True, right=True)
    
    
    #acousticness transformation
    df['acousticness_category'] = pd.cut(df['acousticness'], bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=['veryLowAcousticness', 'lowAcousticness', 'moderateAcousticness', 'highAcousticness', 'veryHighAcousticness'], include_lowest=True, right=True)
    
    #liveness transformation
    df['liveness_category'] = pd.cut(df['liveness'], bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=['veryLowLiveness', 'lowLiveness', 'moderateLiveness', 'highLiveness', 'veryHighLiveness'], include_lowest=True, right=True)
    
    #explicit transformation
    df['explicit_category'] = df['explicit'].apply(lambda x: 'Explicit' if x else 'notExplicit')

    #popularity transformation
    df['popularity_category'] = pd.cut(df['popularity'], bins=[0, 25, 50, 75, 100],
                                    labels=['veryUnpopular', 'unpopular', 'popular', 'veryPopular'], include_lowest=True, right=True)

    #mode transformation
    df['mode_category'] = df['mode'].apply(lambda x: 'Major' if x == 1 else 'Minor')

    #genre
    df['genre_category'] = df['genre']


    return df[['popularity_category','acousticness_category', 'liveness_category', 'danceability_category','energy_category' ,'speechiness_category', 'duration_category', 'explicit_category', 'mode_category', 'genre_category']]