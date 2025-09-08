import pandas as pd
import streamlit as st
import time

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

def create_playlist_from_recommendations(sp, recommendations, playlist_name="VibeGenie Playlist", is_public=False):
    """
    Create a Spotify playlist from AI recommendations using track IDs from the dataset.
    
    Args:
        sp: Authenticated Spotify client
        recommendations: List of recommendation dictionaries with track_id field
        playlist_name: Name for the new playlist
        is_public: Whether the playlist should be public (default: False)
    
    Returns:
        dict: Result with success status, playlist info, and statistics
    """
    try:
        # Get current user info
        user = sp.current_user()
        user_id = user['id']
        
        # Debug: Print user info
        print(f"Creating playlist for user: {user.get('display_name', 'Unknown')} (ID: {user_id})")
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Create empty playlist
        status_text.text("Creating playlist...")
        progress_bar.progress(0.1)
        
        print(f"Creating playlist: '{playlist_name}' (public: {is_public})")
        
        playlist = sp.user_playlist_create(
            user=user_id,
            name=playlist_name,
            public=is_public,
            description="AI-powered music recommendations from VibeGenie"
        )
        
        playlist_id = playlist['id']
        playlist_url = playlist['external_urls']['spotify']
        
        print(f"Playlist created successfully: {playlist_id}")
        print(f"Playlist URL: {playlist_url}")
        
        # Step 2: Extract track IDs from recommendations
        status_text.text("Preparing tracks...")
        progress_bar.progress(0.2)
        
        track_ids = []
        skipped_tracks = []
        
        print(f"Processing {len(recommendations)} recommendations...")
        
        for i, rec in enumerate(recommendations):
            if 'track_id' in rec and rec['track_id']:
                track_ids.append(rec['track_id'])
                print(f"Track {i+1}: {rec.get('track_name', 'Unknown')} - ID: {rec['track_id']}")
            else:
                skipped_tracks.append({
                    'track_name': rec.get('track_name', 'Unknown'),
                    'artists': rec.get('artists', 'Unknown'),
                    'reason': 'No track ID found'
                })
                print(f"Track {i+1}: {rec.get('track_name', 'Unknown')} - SKIPPED (no track ID)")
        
        print(f"Total tracks to add: {len(track_ids)}")
        print(f"Tracks skipped: {len(skipped_tracks)}")
        
        # Check if we have enough tracks
        if len(track_ids) < 3:
            return {
                'success': False,
                'error': f'Not enough valid tracks found ({len(track_ids)} tracks). Need at least 3 tracks to create a playlist.',
                'playlist_info': None,
                'stats': {
                    'total_recommendations': len(recommendations),
                    'tracks_added': len(track_ids),
                    'tracks_skipped': len(skipped_tracks)
                }
            }
        
        # Step 3: Add tracks to playlist in batches (Spotify allows max 100 tracks per request)
        status_text.text(f"Adding {len(track_ids)} tracks to playlist...")
        progress_bar.progress(0.3)
        
        added_tracks = 0
        failed_tracks = []
        
        # Process tracks in batches of 100
        batch_size = 100
        for i in range(0, len(track_ids), batch_size):
            batch = track_ids[i:i + batch_size]
            
            try:
                # Add batch to playlist
                result = sp.playlist_add_items(playlist_id, batch)
                
                # Check for any failed tracks in this batch
                if 'snapshot_id' in result:
                    added_tracks += len(batch)
                else:
                    # If the request failed, add all tracks in this batch to failed list
                    failed_tracks.extend(batch)
                
                # Update progress
                progress = 0.3 + (0.6 * (i + len(batch)) / len(track_ids))
                progress_bar.progress(min(progress, 0.9))
                
                # Small delay to respect rate limits
                time.sleep(0.1)
                
            except Exception as e:
                # If batch fails, try adding tracks individually
                for track_id in batch:
                    try:
                        sp.playlist_add_items(playlist_id, [track_id])
                        added_tracks += 1
                    except:
                        failed_tracks.append(track_id)
                
                # Update progress
                progress = 0.3 + (0.6 * (i + len(batch)) / len(track_ids))
                progress_bar.progress(min(progress, 0.9))
        
        # Step 4: Finalize
        status_text.text("Playlist created successfully!")
        progress_bar.progress(1.0)
        
        # Prepare result
        result = {
            'success': True,
            'playlist_info': {
                'id': playlist_id,
                'name': playlist_name,
                'url': playlist_url,
                'public': is_public,
                'total_tracks': added_tracks
            },
            'stats': {
                'total_recommendations': len(recommendations),
                'tracks_added': added_tracks,
                'tracks_skipped': len(skipped_tracks),
                'tracks_failed': len(failed_tracks)
            },
            'skipped_tracks': skipped_tracks,
            'failed_tracks': failed_tracks
        }
        
        # Clear progress indicators after a short delay
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        return result
        
    except Exception as e:
        # Clear progress indicators on error
        try:
            progress_bar.empty()
            status_text.empty()
        except:
            pass
            
        return {
            'success': False,
            'error': f'Failed to create playlist: {str(e)}',
            'playlist_info': None,
            'stats': {
                'total_recommendations': len(recommendations),
                'tracks_added': 0,
                'tracks_skipped': 0
            }
        }