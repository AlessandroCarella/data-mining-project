import os.path as path
import pandas as pd

from createCsvWithNameArtistsAlbum_nameExplicitPopularityTrack_id import createCsvWithNameArtistsAlbum_nameExplicitPopularityTrack_id 

datasetPath = path.join(path.abspath(path.dirname(__file__)), "../../dataset (missing + split)/train.csv")
dataset = pd.read_csv(datasetPath)
names = list(dataset["name"])# [:30]
albums = list(dataset["album_name"])# [:30]
artists = list(dataset["artists"])# [:30]

createCsvWithNameArtistsAlbum_nameExplicitPopularityTrack_id (names, artists, albums)