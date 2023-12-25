import pandas as pd

from utils import getTrainDataset

def getDatasetWithPatterMiningFeatures()->pd.DataFrame:
    df = pd.read_csv(getTrainDataset ())

    #Get the usueful columns
    df = df[['duration_ms', 'explicit', 'popularity', 'artists', 'genre']]

    #duration_ms transformation
    df['duration_category'] = pd.cut(df['duration_ms'],
                                    bins=[0, 90_000, 120_000, 180_000, 240_000, float('inf')],
                                    labels=['veryShortDuration', 'shortDuration', 'normalDuration', 'longDuration', 'veryLongDuration'])

    #explicit transformation
    df['explicit_category'] = df['explicit'].apply(lambda x: 'explicit' if x else 'notExplicit')

    #popularity transformation
    df['popularity_category'] = pd.cut(df['popularity'], bins=[-float('inf'), 25, 50, 75, float('inf')],
                                    labels=['veryUnpopular', 'unpopular', 'popular', 'veryPopular'],
                                    right=False)

    df['artists_category'] = df['artists']

    df['genre_category'] = df['genre']

    return df[['duration_category', 'explicit_category', 'popularity_category', 'artists_category', 'genre_category']]