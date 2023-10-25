import pandas as pd
import os
from fuzzywuzzy import fuzz

# Read the CSV file
df = pd.read_csv('dataset (missing + split)/train.csv')

# Define the output directory
output_dir_unique = os.path.join(os.path.dirname(os.path.abspath(__file__ or '.')), "Semantic Inconsistencies in Dataset")
if not os.path.exists(output_dir_unique):
    os.makedirs(output_dir_unique, exist_ok=True)

# Create file
file_different_artists_name = os.path.join(output_dir_unique, 'albumsWithDiffArtists.txt')

df['artists'] = df['artists'].str.split(';')
grouped = df.groupby('album_name')

def find_similar_artists(artist_names, threshold=90):
    similar_artists = set()
    artist_names = [name for name in artist_names if name is not None]
    for i in range(len(artist_names)):
        for j in range(i + 1, len(artist_names)):
            similarity = fuzz.ratio(artist_names[i], artist_names[j])
            if similarity >= threshold:
                similar_artists.add(artist_names[i])
                similar_artists.add(artist_names[j])
    return similar_artists

similar_artists = []

for name, group in grouped:
    album = name
    artist_names = group['artists'].str.join(',').tolist()
    similar_artists_list = find_similar_artists(artist_names)
    
    if similar_artists_list and len(similar_artists_list) > 1:
        similar_artists.extend([(album, "; ".join(similar_artists_list))])

similar_df = pd.DataFrame(similar_artists, columns=['Album', 'Similar_Artists'])
similar_df.to_csv(file_different_artists_name, header=True)