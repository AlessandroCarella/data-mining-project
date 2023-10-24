import pandas as pd
import os.path as path

datasetSpotifySongsStats = pd.read_csv(path.join(path.abspath(path.dirname(__file__)), "spotify_song_stats.csv"))
datasetAudioFeatures = pd.read_csv(path.join(path.abspath(path.dirname(__file__)), "spotify_song_audio_features.csv"))

# Merge the datasets on the 'track_id' column
merged_dataset = datasetSpotifySongsStats.merge(datasetAudioFeatures, on='track_id')

# Define the desired column order
desired_columns = [
    'name', 'duration_ms', 'explicit', 'popularity', 'artists', 'album_name',
    'danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence',
    'tempo', 'time_signature', #'features_duration_ms', 'n_beats', 'n_bars',
    #'popularity_confidence', 'processing', 'genre'
]

# Reorder the columns
merged_dataset = merged_dataset[desired_columns]

# Save the merged dataset to a new CSV file
merged_file_path = path.join(path.join(path.abspath(path.dirname(__file__)), "merged_dataset.csv"))
merged_dataset.to_csv(merged_file_path, index=False)

