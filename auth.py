from flask import session, redirect
from spotipy import Spotify
from spotipy.oauth2 import SpotifyOAuth
from spotipy.cache_handler import FlaskSessionCacheHandler
import config

cache_handler = FlaskSessionCacheHandler(session)
sp_oauth = SpotifyOAuth(
    client_id=config.CLIENT_ID,
    client_secret=config.CLIENT_SECRET,
    redirect_uri=config.REDIRECT_URI,
    scope=config.SCOPE,
    cache_handler=cache_handler,
    show_dialog=True
)

def get_spotify_client():
    """Return a fresh Spotify client after validating token or redirect to auth."""
    token_info = cache_handler.get_cached_token()

    if not token_info or not sp_oauth.validate_token(token_info):
        # Token expired or not present
        return redirect(sp_oauth.get_authorize_url())

    # Refresh if necessary
    if sp_oauth.is_token_expired(token_info):
        token_info = sp_oauth.refresh_access_token(token_info['refresh_token'])

    sp = Spotify(auth=token_info['access_token'])
    return sp