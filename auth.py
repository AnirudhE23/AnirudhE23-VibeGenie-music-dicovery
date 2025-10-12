import os
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.cache_handler import CacheFileHandler
import config
import pathlib
import json

def get_spotify_client():
    """
    Initialize and return a Spotify client with proper authentication.
    Uses per-user cache files for Streamlit compatibility.
    Handles OAuth flow through web redirects only (Docker-compatible).
    
    This implementation avoids using SpotifyOAuth's automatic flow methods
    to prevent local server startup or CLI prompts.
    """
    # Check if we have an authorization code from the callback
    auth_code = st.query_params.get('code')
    
    # If we have an auth code, exchange it for a token
    if auth_code:
        try:
            # Create auth manager just for token exchange
            auth_manager = SpotifyOAuth(
                client_id=config.CLIENT_ID,
                client_secret=config.CLIENT_SECRET,
                redirect_uri=config.REDIRECT_URI,
                scope=config.SCOPE,
                show_dialog=True,
            )
            
            # Manually exchange authorization code for access token
            # This method should work without triggering local server
            token_info = auth_manager.get_access_token(auth_code, as_dict=True, check_cache=False)
            
            if token_info:
                # Create temporary client to get username
                temp_sp = spotipy.Spotify(auth=token_info['access_token'])
                username = temp_sp.current_user()["id"]
                
                # Create per-user cache file
                user_cache = f".cache-{username}"
                
                # Save token manually to avoid any auto-flow triggers
                with open(user_cache, 'w') as f:
                    json.dump(token_info, f)
                
                # Create new auth manager with user-specific cache
                cache_handler = CacheFileHandler(cache_path=user_cache)
                user_auth_manager = SpotifyOAuth(
                    client_id=config.CLIENT_ID,
                    client_secret=config.CLIENT_SECRET,
                    redirect_uri=config.REDIRECT_URI,
                    scope=config.SCOPE,
                    cache_handler=cache_handler,
                    show_dialog=True,
                )
                
                # Create authenticated client
                sp = spotipy.Spotify(auth_manager=user_auth_manager)
                return sp, user_cache
        except Exception as e:
            st.error(f"Failed to exchange authorization code: {e}")
            import traceback
            st.error(f"Details: {traceback.format_exc()}")
            return None
    
    # Check for existing cached tokens
    cache_files = [f for f in os.listdir('.') if f.startswith('.cache-')]
    if cache_files:
        # Try to use the most recent cache file
        latest_cache = max(cache_files, key=lambda f: os.path.getmtime(f))
        
        try:
            # Read cached token manually
            with open(latest_cache, 'r') as f:
                token_info = json.load(f)
            
            # Create auth manager with cached credentials
            cache_handler = CacheFileHandler(cache_path=latest_cache)
            cached_auth_manager = SpotifyOAuth(
                client_id=config.CLIENT_ID,
                client_secret=config.CLIENT_SECRET,
                redirect_uri=config.REDIRECT_URI,
                scope=config.SCOPE,
                cache_handler=cache_handler,
                show_dialog=True,
            )
            
            # Validate cached token
            valid_token = cached_auth_manager.validate_token(token_info)
            if valid_token:
                sp = spotipy.Spotify(auth_manager=cached_auth_manager)
                # Verify the token works
                sp.current_user()
                return sp, latest_cache
        except Exception as e:
            # Token is invalid or expired, need to re-authenticate
            pass
    
    # No valid token - manually redirect to authorization URL
    # Create a temporary auth manager just to get the auth URL
    temp_auth_manager = SpotifyOAuth(
        client_id=config.CLIENT_ID,
        client_secret=config.CLIENT_SECRET,
        redirect_uri=config.REDIRECT_URI,
        scope=config.SCOPE,
        show_dialog=True,
    )
    
    auth_url = temp_auth_manager.get_authorize_url()
    
    # Use meta refresh for reliable redirect in Streamlit
    st.markdown(f'<meta http-equiv="refresh" content="0;url={auth_url}">', unsafe_allow_html=True)
    st.info("🔄 Redirecting to Spotify for authorization...")
    st.stop()
    
    return None

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