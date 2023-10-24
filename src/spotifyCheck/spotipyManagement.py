import os.path as path
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

#returns a list with client id in the first position and secret in the second from a file outside of the project folder
def getClientIdAndSecret():
    with open (path.join(path.abspath(path.dirname(__file__)), "../../../spotikeys.txt")) as f:
        lines = f.readlines()
        return lines[0].strip(), lines[1].strip()
    
def getSpotifyObj ():
    # Spotify API credentials
    client_id = getClientIdAndSecret()[0]
    client_secret = getClientIdAndSecret()[1]
    client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    return spotipy.Spotify(client_credentials_manager=client_credentials_manager)