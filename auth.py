import os
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import config
import pathlib

def get_spotify_client():
    """
    Initialize and return a Spotify client with proper authentication.
    Uses per-user cache files for Streamlit compatibility.
    """
    # Initialize temporary client to get username
    temp_sp = spotipy.Spotify(
        auth_manager=SpotifyOAuth(
            client_id=config.CLIENT_ID,
            client_secret=config.CLIENT_SECRET,
            redirect_uri=config.REDIRECT_URI,
            scope=config.SCOPE,
            show_dialog=True
        )
    )
    
    # Get username for per-user cache
    try:
        username = temp_sp.current_user()["id"]
    except Exception as e:
        st.error(f"Authentication failed: {e}")
        return None
    
    user_cache = f".cache-{username}"
    
    # Create auth manager with per-user cache
    auth_manager = SpotifyOAuth(
        client_id=config.CLIENT_ID,
        client_secret=config.CLIENT_SECRET,
        redirect_uri=config.REDIRECT_URI,
        scope=config.SCOPE,
        cache_path=user_cache,
        show_dialog=True
    )
    
    # Return authenticated Spotify client
    return spotipy.Spotify(auth_manager=auth_manager), user_cache

def logout(user_cache):
    """Remove user cache file to logout."""
    if os.path.exists(user_cache):
        os.remove(user_cache)
        st.success("You have been logged out. Please refresh to log in again.")
        st.stop()

def get_current_user_info(sp):
    """Get current user information."""
    try:
        user = sp.current_user()
        return user
    except Exception as e:
        st.error(f"Failed to get user info: {e}")
        return None