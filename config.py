import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

CLIENT_ID = os.environ.get('CLIENT_ID')
CLIENT_SECRET = os.environ.get('CLIENT_SECRET')
REDIRECT_URI = 'http://127.0.0.1:8501/callback'
SCOPE = 'playlist-read-private playlist-read-collaborative user-library-read user-top-read playlist-modify-public playlist-modify-private'

CSV_OUTPUT = "spotify_tracks.csv"
USER_TRACKS_CSV = "spotify_user_tracks.csv"
FEATURES_CACHE = "features_cache.parquet"

# Reccobeats API settings
RECCOBEATS_API_BASE = "https://api.reccobeats.com/v1"
RECCOBEATS_HEADERS = {'Accept': 'application/json'}

# Batch processing settings
BATCH_SIZE = 40
MAX_WORKERS = 8
