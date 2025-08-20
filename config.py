import os

CLIENT_ID = "76849ce2923a4b5daeb1b51a91e6a9b9"
CLIENT_SECRET = "c652faa0dae142ae905086d64261b1ee"
REDIRECT_URI = 'http://127.0.0.1:5000/callback'

SCOPE = 'playlist-read-private, playlist-read-collaborative, user-library-read'

SECRET_KEY = os.urandom(64)
CSV_OUTPUT = "spotify_tracks.csv"
