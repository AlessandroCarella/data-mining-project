import pandas as pd
import os.path as path
import time 
import tqdm as tqdm

from spotipyManagement import getSpotifyObj

def removeNans(ids):
    return [item for item in ids if not pd.isna(item)]

def split_list_into_chunks(input_list, chunk_size):
    for i in range(0, len(input_list), chunk_size):
        yield input_list[i:i + chunk_size]

def createCsvWithAudioFeatures ():
    idDataset = pd.read_csv (path.join(path.abspath(path.dirname(__file__)), "spotify_song_stats.csv"))

    ids = list(idDataset["track_id"])

    idsWithoutNan = removeNans (ids)

    idsChunked = split_list_into_chunks (idsWithoutNan, 100)

    results_list = []
    sp = getSpotifyObj()
    for idsList in tqdm(idsChunked):
        audio_features = sp.audio_features(idsList)
        for singleId, idRelatedAudioFeatures in tqdm(zip(idsList, audio_features), total=len(len(idsList))):
            if idRelatedAudioFeatures:
                result = {
                    "track_id": singleId,
                    "acousticness": idRelatedAudioFeatures["acousticness"],
                    "danceability": idRelatedAudioFeatures["danceability"],
                    "duration_ms": idRelatedAudioFeatures["duration_ms"],
                    "energy": idRelatedAudioFeatures["energy"],
                    "instrumentalness": idRelatedAudioFeatures["instrumentalness"],
                    "key": idRelatedAudioFeatures["key"],
                    "liveness": idRelatedAudioFeatures["liveness"],
                    "loudness": idRelatedAudioFeatures["loudness"],
                    "mode": idRelatedAudioFeatures["mode"],
                    "speechiness": idRelatedAudioFeatures["speechiness"],
                    "tempo": idRelatedAudioFeatures["tempo"],
                    "time_signature": idRelatedAudioFeatures["time_signature"],
                    "valence": idRelatedAudioFeatures["valence"],
                }
                results_list.append(result)
            else:#handling when the spotify api doesnt have any value for the id passed
                results_list.append(
                    {
                        "track_id": singleId,
                        "acousticness": "",
                        "danceability": "",
                        "duration_ms": "",
                        "energy": "",
                        "instrumentalness": "",
                        "key": "",
                        "liveness": "",
                        "loudness": "",
                        "mode": "",
                        "speechiness": "",
                        "tempo": "",
                        "time_signature": "",
                        "valence": ""
                    }
                )

        time.sleep(1.5)
    
    # Convert the list of results to a DataFrame
    results_df = pd.DataFrame(results_list)

    # Save the results to a CSV file
    results_df.to_csv(path.join(path.abspath(path.dirname(__file__)), "spotify_song_audio_features.csv"), index=False)

createCsvWithAudioFeatures()