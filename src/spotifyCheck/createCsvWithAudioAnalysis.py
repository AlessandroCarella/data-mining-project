import pandas as pd
import os.path as path
import time 
import tqdm as tqdm  

from spotipyManagement import getSpotifyObj

def removeNans(ids):
    return [item for item in ids if not pd.isna(item)]

def createCsvWithAudioAnalysis ():
    idDataset = pd.read_csv (path.join(path.abspath(path.dirname(__file__)), "spotify_song_stats.csv"))

    ids = list(idDataset["track_id"])

    idsWithoutNan = removeNans (ids)

    results_list = []
    sp = getSpotifyObj()
    numberOfRequests = 0
    for singleId in idsWithoutNan:
        audio_analysis = sp.audio_analysis(singleId)
        numberOfRequests += 1
        if audio_analysis:
            results_list.append(
                {
                    "track_id": singleId,
                    "n_beats": len(audio_analysis["bars"]),
                    "n_bars": len(audio_analysis["beats"]),
                }
            ) 
        else:#handling when the spotify api doesnt have any value for the id passed
            results_list.append(
                {
                    "track_id": singleId,
                    "n_beats": -1,#using -1 as missing value
                    "n_bars": -1,#using -1 as missing value
                }
            )

        if numberOfRequests % 10 == 0:
            time.sleep(1.5)
    
    # Convert the list of results to a DataFrame
    results_df = pd.DataFrame(results_list)

    # Save the results to a CSV file
    results_df.to_csv(path.join(path.abspath(path.dirname(__file__)), "spotify_song_audio_analysis.csv"), index=False)

createCsvWithAudioAnalysis()
