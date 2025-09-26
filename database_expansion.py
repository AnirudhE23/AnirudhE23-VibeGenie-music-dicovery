import pandas as pd
import streamlit as st
import time
import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import config
import spotify_utils
import reccobeats_utils
import data_collection
from database import db

def add_user_tracks_to_database_tracks(user_tracks_df):
    """
    Add user's collected tracks to the tracks table in database.
    This ensures the tracks table includes the user's music taste.
    
    Args:
        user_tracks_df: DataFrame with user's tracks and audio features
    
    Returns:
        Number of new tracks added to tracks table
    """

    if user_tracks_df.empty:
        return 0
    
    # Debug: Print available columns
    print(f"Available columns in user_tracks_df: {list(user_tracks_df.columns)}")
    
    # filter user tracks that have COMPLETE audio features (no NaN values)
    feature_cols = ['acousticness', 'danceability', 'energy', 'instrumentalness', 
                   'key_value', 'liveness', 'loudness', 'mode_value', 'speechiness', 'tempo', 'valence']
    
    # Check which feature columns are actually available
    available_features = [col for col in feature_cols if col in user_tracks_df.columns]
    missing_features = [col for col in feature_cols if col not in user_tracks_df.columns]
    
    print(f"Available feature columns: {available_features}")
    print(f"Missing feature columns: {missing_features}")
    
    if missing_features:
        print(f"Warning: Missing feature columns: {missing_features}")
        return 0
    
    # Only include tracks that have ALL audio features (no missing values)
    user_tracks_with_features = user_tracks_df.dropna(subset=feature_cols)
    
    # Additional check: ensure no empty strings or invalid values
    for col in feature_cols:
        if col in user_tracks_with_features.columns:
            # Remove rows where the feature is empty string or NaN
            user_tracks_with_features = user_tracks_with_features[
                (user_tracks_with_features[col] != '') & 
                (user_tracks_with_features[col].notna())
            ]
    
    if user_tracks_with_features.empty:
        return 0
    
    # Get existing track ids from database
    existing_tracks = db.get_all_track_ids()
    existing_ids = set(existing_tracks) if existing_tracks else set()
    
    # Filter out tracks that already exist in the database
    new_tracks = user_tracks_with_features[~user_tracks_with_features["spotify_track_id"].isin(existing_ids)]

    if len(new_tracks) == 0:
        return 0

    
    # store new tracks in database
    stored_count = 0
    for idx, row in new_tracks.iterrows():
        try:
            # Validate that all required features are present and not NULL
            required_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 
                              'key_value', 'liveness', 'loudness', 'mode_value', 'speechiness', 'tempo', 'valence']
            
            # Check if all required features are present and not NULL
            missing_features = []
            for feature in required_features:
                if feature not in row or pd.isna(row[feature]) or row[feature] == '':
                    missing_features.append(feature)
            
            if missing_features:
                print(f"Skipping track {row.get('track_name', 'Unknown')} - missing features: {missing_features}")
                continue
            
            track_data = {
                'track_id': row['spotify_track_id'],
                'track_name': row['track_name'],
                'artists': row['artists'],
                'popularity': row.get('popularity', 0),
                'acousticness': row.get('acousticness'),
                'danceability': row.get('danceability'),
                'energy': row.get('energy'),
                'instrumentalness': row.get('instrumentalness'),
                'key_value': row.get('key_value'),
                'liveness': row.get('liveness'),
                'loudness': row.get('loudness'),
                'mode_value': row.get('mode_value'),
                'speechiness': row.get('speechiness'),
                'tempo': row.get('tempo'),
                'valence': row.get('valence')
            }

            # store in tracks table
            success = db.add_track_to_database(track_data)
            if success:
                stored_count += 1
        except Exception as e:
            st.warning(f"Failed to store track {row.get('track_name', 'Unknown')}: {e}")
    
    return stored_count

def clean_database_tracks():
    """
    Clean the tracks table by removing tracks without complete audio features.
    This ensures the database only contains tracks with valid audio features.
    
    Returns:
        Number of tracks removed
    """
    try:
        # Get current database stats
        stats = db.get_database_stats()
        original_count = stats.get('total_tracks', 0)
        
        # Get all tracks from database
        all_tracks = db.get_all_tracks()
        
        if not all_tracks:
            return 0
        
        # Convert to DataFrame for easier processing
        tracks_df = pd.DataFrame(all_tracks)
        
        # Define required audio features
        feature_cols = ['acousticness', 'danceability', 'energy', 'instrumentalness', 
                       'key_value', 'liveness', 'loudness', 'mode_value', 'speechiness', 'tempo', 'valence']
        
        # Remove tracks that are missing any audio features
        cleaned_df = tracks_df.dropna(subset=feature_cols)

        # Additional check: remove tracks with empty strings in audio features
        for col in feature_cols:
            if col in cleaned_df.columns:
                cleaned_df = cleaned_df[
                    (cleaned_df[col] != '') & 
                    (cleaned_df[col].notna())
                ]
        
        # Remove tracks that don't have complete features
        tracks_to_remove = tracks_df[~tracks_df['spotify_track_id'].isin(cleaned_df['spotify_track_id'])]
        
        # Remove incomplete tracks from database
        removed_count = 0
        for track_id in tracks_to_remove['spotify_track_id']:
            try:
                success = db.remove_track_from_database(track_id)
                if success:
                    removed_count += 1
            except Exception as e:
                st.warning(f"Failed to remove track {track_id}: {e}")
        
        return removed_count
        
    except Exception as e:
        st.error(f"Error cleaning database: {e}")
        return 0

def expand_database_tracks_background(sp, target_size=150000):
    """
    Automatically expand the tracks table in the background to improve recommendation quality.
    This happens behind the scenes and adds diverse tracks to the database.
    
    Args:
        sp: Spotify client
        target_size: Target total size for the tracks table
    
    Returns:
        Number of new tracks added
    """
    # Get current database size
    stats = db.get_database_stats()
    current_size = stats.get('total_tracks', 0)
    
    if current_size >= target_size:
        return 0
    
    needed_tracks = target_size - current_size
    
    # Load existing cache
    cache_df = data_collection.load_features_cache()
    existing_ids = db.get_all_track_ids()
    
    # Smart expansion strategies for database
    new_tracks = []
    
    # Strategy 1: Popular tracks from diverse genres and years
    new_tracks.extend(get_diverse_database_tracks(sp, needed_tracks // 3))

    # Strategy 2: Trending tracks from various time periods
    new_tracks.extend(get_trending_database_tracks(sp, needed_tracks // 3))
    
    # Strategy 3: Tracks from curated playlists (discovery-focused)
    new_tracks.extend(get_curated_database_tracks(sp, needed_tracks // 3))
    
    if not new_tracks:
        return 0
    
    # Convert to DataFrame and remove duplicates
    new_tracks_df = pd.DataFrame(new_tracks)
    new_tracks_df = new_tracks_df.drop_duplicates(subset=["track_id"])
    
    # Remove tracks already in database
    new_tracks_df = new_tracks_df[~new_tracks_df["track_id"].isin(existing_ids)]
    
    if len(new_tracks_df) == 0:
        return 0
    
    # Get audio features for new tracks
    new_tracks_with_features = get_features_for_database_tracks(new_tracks_df, cache_df)
    
    # Store in database
    stored_count = 0
    for idx, row in new_tracks_with_features.iterrows():
        try:
            # Validate that all required features are present and not NULL
            required_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 
                              'key_value', 'liveness', 'loudness', 'mode_value', 'speechiness', 'tempo', 'valence']
            
            # Check if all required features are present and not NULL
            missing_features = []
            for feature in required_features:
                if feature not in row or pd.isna(row[feature]) or row[feature] == '':
                    missing_features.append(feature)
            
            if missing_features:
                print(f"Skipping track {row.get('track_name', 'Unknown')} - missing features: {missing_features}")
                continue
            
            track_data = {
                'track_id': row['track_id'],
                'track_name': row['track_name'],
                'artists': row['artists'],
                'popularity': row.get('popularity', 0),
                'acousticness': row.get('acousticness'),
                'danceability': row.get('danceability'),
                'energy': row.get('energy'),
                'instrumentalness': row.get('instrumentalness'),
                'key_value': row.get('key_value'),
                'liveness': row.get('liveness'),
                'loudness': row.get('loudness'),
                'mode_value': row.get('mode_value'),
                'speechiness': row.get('speechiness'),
                'tempo': row.get('tempo'),
                'valence': row.get('valence')
            }
            
            # Store in tracks table
            success = db.add_track_to_database(track_data)
            if success:
                stored_count += 1
                
        except Exception as e:
            st.warning(f"Failed to store track {row.get('track_name', 'Unknown')}: {e}")
    
    return stored_count

def get_diverse_database_tracks(sp, max_tracks):
    """Get diverse tracks for database expansion."""
    new_tracks = []
    
    # Search for popular tracks in various genres and years
    genre_queries = [
        "year:2024", "year:2023", "year:2022", "year:2021", "year:2020",  # Recent years
        "genre:pop", "genre:rock", "genre:hip-hop", "genre:electronic",
        "genre:indie", "genre:alternative", "genre:r&b", "genre:country",
        "genre:jazz", "genre:classical", "genre:reggae", "genre:blues"
    ]
    
    for query in genre_queries:
        if len(new_tracks) >= max_tracks:
            break
            
        try:
            results = sp.search(q=query, type='track', limit=25)
            for track in results['tracks']['items']:
                if len(new_tracks) >= max_tracks:
                    break
                    
                # Only include tracks with good popularity
                if track['popularity'] >= 20:  # Lower threshold for more diversity
                    track_info = {
                        'track_id': track['id'],
                        'track_name': track['name'],
                        'artists': ', '.join([a['name'] for a in track['artists']]),
                        'popularity': track['popularity']
                    }
                    new_tracks.append(track_info)
                        
        except Exception as e:
            continue

    return new_tracks

def get_trending_database_tracks(sp, max_tracks):
    """Get trending tracks for database expansion."""
    new_tracks = []
    
    # Search for trending/viral tracks
    trending_queries = [
        "viral", "trending", "hits", "popular",
        "new releases", "fresh", "latest", "top hits"
    ]
    
    for query in trending_queries:
        if len(new_tracks) >= max_tracks:
            break
            
        try:
            results = sp.search(q=query, type='track', limit=20)
            for track in results['tracks']['items']:
                if len(new_tracks) >= max_tracks:
                    break
                    
                if track['popularity'] >= 30:
                    track_info = {
                        'track_id': track['id'],
                        'track_name': track['name'],
                        'artists': ', '.join([a['name'] for a in track['artists']]),
                        'popularity': track['popularity']
                    }
                    new_tracks.append(track_info)
                        
        except Exception as e:
            continue
    
    return new_tracks

def get_curated_database_tracks(sp, max_tracks):
    """Get tracks from Spotify's curated playlists for database expansion."""
    new_tracks = []
    
    # Search for Spotify's curated playlists
    curated_queries = [
        "Discover Weekly", "Release Radar", "New Music Friday",
        "Today's Top Hits", "Global Top 50", "Viral Hits",
        "RapCaviar", "Rock This", "All Out 2010s", "All Out 2000s"
    ]
    
    for query in curated_queries:
        if len(new_tracks) >= max_tracks:
            break
            
        try:
            results = sp.search(q=query, type='playlist', limit=3)
            for playlist in results['playlists']['items']:
                if len(new_tracks) >= max_tracks:
                    break
                    
                # Get tracks from this playlist
                try:
                    playlist_tracks = spotify_utils.get_playlist_tracks(sp, playlist['id'])
                    for track_item in playlist_tracks[:15]:  # More tracks per playlist
                        if len(new_tracks) >= max_tracks:
                            break
                        track = track_item.get('track', {})
                        if track and track.get('id'):
                            track_info = {
                                'track_id': track['id'],
                                'track_name': track['name'],
                                'artists': ', '.join([a['name'] for a in track['artists']]),
                                'popularity': track.get('popularity', 50)
                            }
                            new_tracks.append(track_info)
                            
                except Exception as e:
                    continue
                    
        except Exception as e:
            continue
    
    return new_tracks

def get_features_for_database_tracks(new_tracks_df, cache_df):
    """Get audio features for new database tracks using your existing pipeline."""
    if new_tracks_df.empty:
        return new_tracks_df
    
    # Get track IDs that need features
    need_ids = [tid for tid in new_tracks_df["track_id"].dropna().unique().tolist() 
                if tid not in cache_df["Track ID"].unique()]
    
    if not need_ids:
        return new_tracks_df
    
    # Use your existing batch processing approach
    for i in range(0, len(need_ids), config.BATCH_SIZE):
        batch = need_ids[i:i+config.BATCH_SIZE]
        features = reccobeats_utils.get_features_from_recco_batch(batch)
        
        if features is not None and not features.empty:
            cache_df = pd.concat([cache_df, features], ignore_index=True) if not cache_df.empty else features
            data_collection.save_features_cache(cache_df)
    
    # Fallback for tracks that didn't get features in batch
    features_df = cache_df.copy()
    have_ids = set(features_df["Track ID"].unique()) if not features_df.empty else set()
    remaining = new_tracks_df[~new_tracks_df["track_id"].isin(have_ids)].copy()
    
    if len(remaining) > 0:
        spotify_to_recco = {}
        fallback_results = []

        def run_fallback(row): 
            time.sleep(random.uniform(0.05, 0.15)) 
            result = reccobeats_utils.get_reccobeats_features( 
                track_id=row["track_id"], 
                track_name=row["track_name"], 
                artist_name=row["artists"], 
                spotify_to_recco=spotify_to_recco
            )
            return result

        with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as ex:
            futures = {ex.submit(run_fallback, row): idx for idx, row in remaining.iterrows()}
            for f in as_completed(futures):
                try:
                    res = f.result()
                    if res is not None and not res.empty:
                        fallback_results.append(res)
                        cache_df = pd.concat([cache_df, res], ignore_index=True) if not cache_df.empty else res
                        data_collection.save_features_cache(cache_df)
                except Exception:
                    pass

        if fallback_results:
            fb_df = pd.concat(fallback_results, ignore_index=True)
            features_df = pd.concat([features_df, fb_df], ignore_index=True).drop_duplicates(subset=["Track ID"])

    # Merge new tracks with their features and format for database
    final_df = pd.merge(new_tracks_df, features_df, left_on="track_id", right_on="Track ID", how="left")
    
    # Only keep tracks that have ALL audio features (no missing values)
    feature_cols = ['acousticness', 'danceability', 'energy', 'instrumentalness', 
                   'key_value', 'liveness', 'loudness', 'mode_value', 'speechiness', 'tempo', 'valence']
    
    # Filter out tracks without complete audio features
    tracks_with_complete_features = final_df.dropna(subset=feature_cols)
    
    # Additional check: ensure no empty strings or invalid values
    for col in feature_cols:
        if col in tracks_with_complete_features.columns:
            tracks_with_complete_features = tracks_with_complete_features[
                (tracks_with_complete_features[col] != '') & 
                (tracks_with_complete_features[col].notna())
            ]
    
    if tracks_with_complete_features.empty:
        return pd.DataFrame()
    
    # Format to match database structure
    database_format = {
        'track_id': tracks_with_complete_features['track_id'],
        'track_name': tracks_with_complete_features['track_name'],
        'artists': tracks_with_complete_features['artists'],
        'popularity': tracks_with_complete_features['popularity'],
        'acousticness': tracks_with_complete_features['acousticness'],
        'danceability': tracks_with_complete_features['danceability'],
        'energy': tracks_with_complete_features['energy'],
        'instrumentalness': tracks_with_complete_features['instrumentalness'],
        'key_value': tracks_with_complete_features['key_value'],
        'liveness': tracks_with_complete_features['liveness'],
        'loudness': tracks_with_complete_features['loudness'],
        'mode_value': tracks_with_complete_features['mode_value'],
        'speechiness': tracks_with_complete_features['speechiness'],
        'tempo': tracks_with_complete_features['tempo'],
        'valence': tracks_with_complete_features['valence']
    }
    
    return pd.DataFrame(database_format)
