import pandas as pd
import os
from fuzzywuzzy import fuzz

df = pd.read_csv('dataset (missing + split)/train.csv')

# Define the output directory
output_dir_unique = os.path.join(os.path.dirname(os.path.abspath(__file__ or '.')), "Semantic Inconsistencies in Dataset")
os.makedirs(output_dir_unique, exist_ok=True)

# Create files
file_similar_albums_name = os.path.join(output_dir_unique, 'artistsWithSimilarAlbums.txt')

#Check for artists with similar album names
grouped = df.groupby('artists')

def find_similar_albums(album_name, album_list, threshold=80):
    similar_albums = set()
    for other_album in album_list:
        if album_name != other_album:
            similarity = fuzz.ratio(album_name, other_album)
            if similarity >= threshold:
                similar_albums.add(other_album)
    return similar_albums

similar_albums = []
processed_albums = set()

for name, group in grouped:
    for _, row in group.iterrows():
        artist = row['artists']
        album = row['album_name']
        # Check if the album has already been processed
        if album not in processed_albums:
            similar_albums_list = find_similar_albums(album, group['album_name'])
            if similar_albums_list:
                similar_albums.extend([(artist, album, ", ".join(similar_albums_list))])
                processed_albums.add(album)

similar_df = pd.DataFrame(similar_albums, columns=['Artist', 'Album', 'Similar_Albums'])
similar_df.to_csv(file_similar_albums_name, header=True)