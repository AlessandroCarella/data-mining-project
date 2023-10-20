import pandas as pd
import os.path as path
import time 

from spotipyManagement import getSpotifyObj

# Define a function to get Spotify song features
def get_spotify_features(sp, song_name, artist_name, album_name):
    try:
        # Search for the track
        search_results = sp.search(q=f"track:{song_name} artist:{artist_name} album:{album_name}", type='track', limit=1)
        if not search_results['tracks']['items']:
            print(f"No results found for {song_name} by {artist_name} from {album_name}, trying with track artist query")
            search_results = sp.search(q=f"track:{song_name} artist:{artist_name}", type='track', limit=1)
            if not search_results['tracks']['items']:
                if ";" in artist_name:
                    print(f"No results found for {song_name} by {artist_name}, trying with only the first artist")
                    search_results = sp.search(q=f"track:{song_name} artist:{(artist_name).split(';')[0]}", type='track', limit=1)
                    if not search_results['tracks']['items']:
                        print(f"No results found for {song_name} by {artist_name}")
                        return None, None, None
                else:
                    return None, None, None
        
        track_info = search_results['tracks']['items'][0]

        # Get track information
        track_id = track_info['id']
        explicit = track_info['explicit']
        popularity = track_info['popularity']

        return track_id, explicit, popularity
    except Exception as e:
        print(f"Error for {song_name} by {artist_name} from {album_name}: {str(e)}")
        print(e)
        return None, None, None

def createCsvWithNameArtistsAlbum_nameExplicitPopularityTrack_id(songs, artists, albums):
    sp = getSpotifyObj ()

    # Create an empty list to store the results
    results_list = []

    numberOfRequests = 0

    # Loop through the songs, artists, and albums
    for song, artist, album in zip(songs, artists, albums):
        track_id, explicit, popularity = get_spotify_features(sp, song, artist, album)
        result = {
            "name": song,
            "artists": artist,
            "album_name": album,
            "explicit": explicit,
            "popularity": popularity,
            "track_id": track_id,
        }
        results_list.append(result)
        
        numberOfRequests += 1
        # Sleep for a moment to respect Spotify's API rate limits
        if numberOfRequests % 10 == 0:
            time.sleep(1.5)  # You can adjust this as needed.

    # Convert the list of results to a DataFrame
    results_df = pd.DataFrame(results_list)

    # Save the results to a CSV file
    results_df.to_csv(path.join(path.abspath(path.dirname(__file__)), "spotify_song_stats.csv"), index=False)

