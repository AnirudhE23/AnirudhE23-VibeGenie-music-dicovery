import pandas as pd
import streamlit as st

def get_all_playlists(sp):
    """Get all user playlists."""
    playlists = []
    results = sp.current_user_playlists(limit=50)
    while results:
        playlists.extend(results['items'])
        if results['next']:
            results = sp.next(results)
        else:
            results = None
    return playlists

def get_playlist_tracks(sp, playlist_id):
    """Get all tracks from a specific playlist."""
    tracks = []
    results = sp.playlist_tracks(playlist_id, limit=100)
    while results:
        tracks.extend(results['items'])
        if results['next']:
            results = sp.next(results)
        else:
            results = None
    return tracks

def get_user_liked_songs(sp):
    """Get all user's liked songs."""
    tracks = []
    offset = 0
    while True:
        results = sp.current_user_saved_tracks(limit=50, offset=offset)
        if not results["items"]:
            break
        tracks.extend(results["items"])
        offset += 50
        if offset >= results["total"]:
            break
    return tracks

def get_user_top_tracks(sp, time_range="medium_term", limit=50):
    """Get user's top tracks for a given time range."""
    try:
        results = sp.current_user_top_tracks(limit=limit, time_range=time_range)
        return results["items"]
    except Exception as e:
        st.error(f"Failed to get top tracks: {e}")
        return []

def extract_track_info(track_item, playlist_name="N/A", owner_id=""):
    """Extract standardized track information from a track item."""
    track = track_item.get("track") if isinstance(track_item, dict) and "track" in track_item else track_item
    
    if not track or not track.get("id"):
        return None
    
    return {
        "Track Name": track["name"],
        "Album Name": track["album"]["name"],
        "Artist(s)": ", ".join([a["name"] for a in track["artists"]]),
        "Playlist Name": playlist_name,
        "Track ID": track["id"],
        "Popularity": track.get("popularity", 0),
        "Owner ID": owner_id,
        "ISRC": (track.get("external_ids") or {}).get("isrc")
    }

def get_all_user_tracks(sp):
    """
    Collect all user tracks from:
    1. Playlists
    2. Liked songs
    3. Top tracks (short, medium, long term)
    """
    all_tracks = []
    
    # 1. Playlists
    playlists = sp.current_user_playlists(limit=50)["items"]
    for playlist in playlists:
        playlist_name = playlist["name"]
        owner_id = playlist["owner"]["id"]
        playlist_id = playlist["id"]

        results = sp.playlist_tracks(playlist_id, limit=100)
        while results:
            for item in results["items"]:
                track_info = extract_track_info(item, playlist_name, owner_id)
                if track_info:
                    all_tracks.append(track_info)
            results = sp.next(results) if results.get("next") else None

    # 2. Liked songs
    liked_tracks = get_user_liked_songs(sp)
    for item in liked_tracks:
        track_info = extract_track_info(item, "Liked Songs", "")
        if track_info:
            all_tracks.append(track_info)

    # 3. Top tracks (short, medium, long term)
    for term in ["short_term", "medium_term", "long_term"]:
        top_tracks = get_user_top_tracks(sp, term)
        for track in top_tracks:
            track_info = extract_track_info(track, f"Top Tracks ({term})", "")
            if track_info:
                all_tracks.append(track_info)

    # Remove duplicates and return DataFrame
    tracks_df = pd.DataFrame(all_tracks).drop_duplicates(subset=["Track ID"])
    return tracks_df