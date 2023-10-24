import pandas as pd
import os.path as path

def mergeDatasetOnTrackIdTheDumbWayCauseTheSmartOneDoesntWork(spotifySongStats, spotifySongAudioFeatures):
    # Create a dictionary to map track_id to (mode, time_signature)
    audio_features_dict = dict(zip(spotifySongAudioFeatures['track_id'], 
                                   spotifySongAudioFeatures[['mode', 'time_signature']].values))

    # Create new columns for mode and time_signature in spotifySongStats
    spotifySongStats['mode'] = 0
    spotifySongStats['time_signature'] = 0

    for index, row in spotifySongStats.iterrows():
        track_id = row['track_id']
        if track_id in audio_features_dict:
            mode, time_signature = audio_features_dict[track_id]
            spotifySongStats.at[index, 'mode'] = mode
            spotifySongStats.at[index, 'time_signature'] = time_signature

    return spotifySongStats

def mergeDatasetOnTrackIdTheDumbWayCauseTheSmartOneDoesntWorkFunnierWay(datasetTrain, audioFeaturesAndStatsMerged):
    # Create a dictionary to map track_id to (mode, time_signature)
    audio_features_dict = dict(zip(audioFeaturesAndStatsMerged['name'], 
                                   audioFeaturesAndStatsMerged[['mode', 'time_signature']].values))
    
    for index, row in datasetTrain.iterrows():
        name = row['name']
        if name in audio_features_dict:
            mode, time_signature = audio_features_dict[name]
            if pd.isna(datasetTrain.at[index, 'mode']):
                datasetTrain.at[index, 'mode'] = mode
            if pd.isna(datasetTrain.at[index, 'time_signature']):
                datasetTrain.at[index, 'time_signature'] = time_signature

    return datasetTrain

def createTrainDatasetWithModeAndTimeSignatureValues ():
    datasetTrain = pd.read_csv(path.join(path.abspath(path.dirname(__file__)), "../../dataset (missing + split)/train.csv"))
    print (datasetTrain["name"].is_unique)#name is a primary key IMPORTANT, SEE UNDENEATH

    spotifySongStats = pd.read_csv(path.join(path.abspath(path.dirname(__file__)), "spotify_song_stats.csv"))
    spotifySongAudioFeatures = pd.read_csv(path.join(path.abspath(path.dirname(__file__)), "spotify_song_audio_features.csv"))

    #filter on the values i need
    spotifySongStats = spotifySongStats[['name', 'artists', 'album_name', 'track_id']]
    print (spotifySongStats.shape)
    spotifySongAudioFeatures = spotifySongAudioFeatures[['track_id', 'mode','time_signature']]
    print (spotifySongAudioFeatures.shape)

    audioFeaturesAndStatsMerged = mergeDatasetOnTrackIdTheDumbWayCauseTheSmartOneDoesntWork (spotifySongStats, spotifySongAudioFeatures)
    print (audioFeaturesAndStatsMerged.shape)
    print (audioFeaturesAndStatsMerged.keys())#['name', 'artists', 'album_name', 'track_id', 'mode', 'time_signature']
    print (audioFeaturesAndStatsMerged["name"].is_unique)#name is a primary key IMPORTANT, SEE ABOVE


    print(datasetTrain.isnull().sum())#check how many values are missing from the original dataset
    fullyMerged = mergeDatasetOnTrackIdTheDumbWayCauseTheSmartOneDoesntWorkFunnierWay(datasetTrain, audioFeaturesAndStatsMerged)
    print (fullyMerged.shape)
    print(fullyMerged.isnull().sum())#check how many values are missing from the merged one

    fullyMerged.to_csv(path.join(path.abspath(path.dirname(__file__)), "train with almost all mode values.csv"), index=False)

def findMissingModeValue ():
    newTrainDf = pd.read_csv(path.join(path.abspath(path.dirname(__file__)), "train with almost all mode values.csv"))
    missing_mode_rows = newTrainDf[newTrainDf['mode'].isna()]

    output = ""
    for index, row in missing_mode_rows.iterrows():
        output += (f"Row {index}:\n")
        # Iterate through columns and print their values
        for column, value in row.items():
            output+= (f"{column}: {value}\n")
        output+=("\n")  # Add a blank line between rows for clarity
    #print (output)

    """
    the song with the missing mode value is 
    Row 7900:
    name: Pink Noise for Sleeping
    duration_ms: 240000
    explicit: False
    popularity: 35
    artists: Pink Noise
    album_name: Pink Noise
    danceability: 0.0
    energy: 7.95e-05
    key: 10
    loudness: -12.882
    mode: nan
    speechiness: 0.0
    acousticness: 0.118
    instrumentalness: 0.971
    liveness: 0.235
    valence: 0.0
    tempo: 0.0
    features_duration_ms: 240000
    time_signature: 0.0
    n_beats: 0.0
    n_bars: 0.0
    popularity_confidence: nan
    processing: 0.7573891625615764
    genre: sleep

    the spotify api is not able to find it by query and it makes sense cause there are like 10 artists with pink noise as a name

    so i searched manually in the spotify_song_stats.csv file and the track_id is 2ex3O9bv4muJcuVI3BQbtj
    """

    from spotipyManagement import getSpotifyObj

    sp = getSpotifyObj()
    #audio_features = sp.audio_features(["2ex3O9bv4muJcuVI3BQbtj"])[0]['mode']
    """
    this returns a None value
    BUT, if you search "2ex3O9bv4muJcuVI3BQbtj" on google the track appears
    and from there i can get the link to the song, which is "https://open.spotify.com/intl-it/track/2ex3O9bv4muJcuVI3BQbtj"
    so i can
    """

    # Get the track ID from the URI (correct format)
    track = sp.track("2ex3O9bv4muJcuVI3BQbtj")
    print (track["uri"])

    # Get the audio features of the track
    audio_features = sp.audio_analysis(track["uri"])
    """
    i have no idea why but audio_features call return None
    but the audio_analysis call has the mode value in it soooooo
    """
    # Get the audio features of the track
    audio_analysis = sp.audio_analysis(track["uri"])
    
    # Print the audio features
    return audio_analysis["track"]["mode"]


trainDfWithOneModeValueMissing = pd.read_csv(path.join(path.abspath(path.dirname(__file__)), "train with almost all mode values.csv"))
missing_mode_rows = trainDfWithOneModeValueMissing[trainDfWithOneModeValueMissing['mode'].isna()]
print (missing_mode_rows)

trainDfWithOneModeValueMissing['mode'].fillna(findMissingModeValue, inplace=True)
trainWithAllValues = trainDfWithOneModeValueMissing
# Count missing values in each column
missing_values = trainWithAllValues.isna().sum()
# Print the number of missing values for each column
print(missing_values)

trainWithAllValues.to_csv(path.join(path.abspath(path.dirname(__file__)), "train with all values.csv"), index=False)