import pandas as pd

import os.path as path

def getTrainDataset ():
    return path.join(path.dirname(__file__), "../../../dataset (missing + split)/trainFilledWithoutUselessFeatures.csv")

def getDatasetWithPatterMiningFeatures()->pd.DataFrame:
    df = pd.read_csv(getTrainDataset ())
    selected_cols=['duration_ms', 'explicit', 'popularity', 'mode', 'genre', 'danceability', 'energy','speechiness', 'instrumentalness', 'acousticness', 'liveness', 'key']
    
    df = df[selected_cols]

    #duration_ms transformation
    df['duration_category'] = pd.cut(df['duration_ms'],
                                    bins=[0, 90000, 120000, 180000, 240000, float('inf')],
                                    labels=['veryShortDuration', 'shortDuration', 'normalDuration', 'longDuration', 'veryLongDuration'])
    
    #danceability transformation
    df['danceability_category'] = pd.cut(df['danceability'], bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=['veryLowDanceability', 'lowDanceability', 'moderateDanceability', 'highDanceability', 'veryHighDanceability'])

    #energy transformation
    df['energy_category'] = pd.cut(df['energy'], bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=['veryLowEnergy', 'lowEnergy', 'moderateEnergy', 'highEnergy', 'veryHighEnergy'])
    
    #speechiness transformation
    df['speechiness_category'] = pd.cut(df['speechiness'], bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=['veryLowSpeechiness', 'lowSpeechiness', 'moderateSpeechiness', 'highSpeechiness', 'veryHighSpeechiness'])
    
    #instrumentalness transformation
    df['instrumentalness_category'] = pd.cut(df['instrumentalness'], bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=['veryLowInstrumentalness', 'lowInstrumentalness', 'moderateInstrumentalness', 'highInstrumentalness', 'veryHighInstrumentalness'])
    
    #acousticness transformation
    df['acousticness_category'] = pd.cut(df['acousticness'], bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=['veryLowAcousticness', 'lowAcousticness', 'moderateAcousticness', 'highAcousticness', 'veryHighAcousticness'])
    
    #liveness transformation
    df['liveness_category'] = pd.cut(df['liveness'], bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=['veryLowLiveness', 'lowLiveness', 'moderateLiveness', 'highLiveness', 'veryHighLiveness'])
    
    #explicit transformation
    df['explicit_category'] = df['explicit'].apply(lambda x: 'Explicit' if x else 'notExplicit')

    #popularity transformation
    df['popularity_category'] = pd.cut(df['popularity'], bins=[0, 25, 50, 75, 100],
                                    labels=['veryUnpopular', 'unpopular', 'popular', 'veryPopular'])

    #mode transformation
    df['mode_category'] = df['mode'].apply(lambda x: 'Major' if x == 1 else 'Minor')

    #genre
    df['genre_category'] = df['genre']
    
    #key transformation
    key_mapping = {-1: 'C', 0: 'C#', 1: 'D', 2: 'D#', 3: 'E', 4: 'F', 5: 'F#', 6: 'G', 7: 'G#', 8: 'A', 9: 'A#', 10: 'B', 11: 'C'}
    df['key_category'] = df['key'].map(key_mapping)
    

    return df[['acousticness_category', 'liveness_category', 'danceability_category','energy_category' ,'speechiness_category', 'duration_category', 'instrumentalness_category', 'explicit_category', 'popularity_category', 'mode_category', 'genre_category', 'key_category']]