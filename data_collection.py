import pandas as pd
import streamlit as st
import pathlib
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import config
import spotify_utils
import reccobeats_utils
from database import db
import database_expansion

CACHE_PATH = pathlib.Path(config.FEATURES_CACHE)

def load_features_cache():
    """Load cached audio features from parquet file."""
    if CACHE_PATH.exists():
        return pd.read_parquet(CACHE_PATH)
    return pd.DataFrame()

def save_features_cache(df):
    """Save audio features to cache, removing duplicates."""
    df = df.drop_duplicates(subset=["Track ID"])
    df.to_parquet(CACHE_PATH, index=False)

def collect_user_tracks(sp):
    """
    Fetches user tracks and stores directly in database with audio features.
    Uses caching to avoid unnecessary API calls.
    """
    # Get user info for database
    user = sp.current_user()
    spotify_user_id = user['id']
    display_name = user.get('display_name', 'Unknown')
    
    # Create user in database
    db.create_user(spotify_user_id, display_name)
    
    # Load features cache (keep your existing caching system)
    cache_df = load_features_cache()
    cached_ids = set(cache_df["Track ID"].unique()) if not cache_df.empty else set()
    
    # Get all user tracks using spotify_utils
    tracks_df = spotify_utils.get_all_user_tracks(sp)
    
    st.write(f"Total tracks collected: {len(tracks_df)}")
    st.write(f"Already cached: {len(cached_ids)}")
    
    # Which tracks still need features
    need_ids = [tid for tid in tracks_df["Track ID"].dropna().unique().tolist() if tid not in cached_ids]
    st.write(f"Need features: {len(need_ids)}")

    # Batch fetch features for new tracks (keep your existing logic)
    if need_ids:
        batch_progress = st.progress(0, text="Fetching audio features (batch)")
        status = st.empty()
        for i in range(0, len(need_ids), config.BATCH_SIZE):
            batch = need_ids[i:i+config.BATCH_SIZE]
            features = reccobeats_utils.get_features_from_recco_batch(batch)
            if features is not None and not features.empty:
                cache_df = pd.concat([cache_df, features], ignore_index=True) if not cache_df.empty else features
                save_features_cache(cache_df)
                st.write(f"Cached features for {len(features)} tracks (total cache: {len(cache_df)})")
            progress_value = min((i + config.BATCH_SIZE) / len(need_ids), 1.0)
            batch_progress.progress(progress_value)
            status.text(f"Batch processed {min(i+config.BATCH_SIZE, len(need_ids))}/{len(need_ids)} tracks")

    # Get features for all tracks (from cache + new fetches)
    features_df = cache_df.copy()
    have_ids = set(features_df["Track ID"].unique()) if not features_df.empty else set()
    remaining = tracks_df[~tracks_df["Track ID"].isin(have_ids)].copy()

    # Fallback pass for remaining tracks (keep your existing logic)
    if len(remaining) > 0:
        remaining = remaining.sort_values("Popularity", ascending=False)
        fb_progress = st.progress(0, text=f"Fallback lookups ({len(remaining)} tracks)")
        fb_status = st.empty()
        spotify_to_recco = {}
        fallback_results = []

        def run_fallback(row): 
            time.sleep(random.uniform(0.05, 0.15)) 
            result = reccobeats_utils.get_reccobeats_features( 
                track_id=row["Track ID"], 
                track_name=row["Track Name"], 
                artist_name=row["Artist(s)"], 
                spotify_to_recco=spotify_to_recco
            )
            return result

        with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as ex:
            futures = {ex.submit(run_fallback, row): idx for idx, row in remaining.iterrows()}
            done = 0
            for f in as_completed(futures):
                try:
                    res = f.result()
                    if res is not None and not res.empty:
                        fallback_results.append(res)
                        cache_df = pd.concat([cache_df, res], ignore_index=True) if not cache_df.empty else res
                        save_features_cache(cache_df)
                except Exception:
                    pass
                finally:
                    done += 1
                    fb_progress.progress(done / len(remaining))
                    fb_status.text(f"Fallback processed {done}/{len(remaining)}")

        if fallback_results:
            fb_df = pd.concat(fallback_results, ignore_index=True)
            features_df = pd.concat([features_df, fb_df], ignore_index=True).drop_duplicates(subset=["Track ID"])

    # NEW: Store user tracks directly in database (not in tracks table)
    st.info("Storing user tracks in database...")
    db_progress = st.progress(0, text="Storing user tracks in database")
    
    stored_count = 0
    for idx, row in tracks_df.iterrows():
        try:
            # Get features for this track
            track_features = features_df[features_df["Track ID"] == row["Track ID"]]
            
            if not track_features.empty:
                feature_row = track_features.iloc[0]

                # Prepare track data for database
                track_data = {
                    'track_id': row['Track ID'],
                    'track_name': row['Track Name'],
                    'artists': row['Artist(s)'],
                    'popularity': row.get('Popularity', 0),
                    'acousticness': feature_row.get('acousticness'),
                    'danceability': feature_row.get('danceability'),
                    'energy': feature_row.get('energy'),
                    'instrumentalness': feature_row.get('instrumentalness'),
                    'key': feature_row.get('key_value'),
                    'liveness': feature_row.get('liveness'),
                    'loudness': feature_row.get('loudness'),
                    'mode': feature_row.get('mode_value'),
                    'speechiness': feature_row.get('speechiness'),
                    'tempo': feature_row.get('tempo'),
                    'valence': feature_row.get('valence')
                }
                
                # Store in user_tracks table
                success = db.add_user_track(spotify_user_id, track_data, row.get('Playlist Name', 'Unknown'))
                if success:
                    stored_count += 1
                    
        except Exception as e:
            st.warning(f"Failed to store track {row.get('Track Name', 'Unknown')}: {e}")
        
        # Update progress
        progress_value = min((idx + 1) / len(tracks_df), 1.0)
        db_progress.progress(progress_value)
    
    st.success(f"✅ Stored {stored_count} user tracks in database!")
    
    # Database expansion (instead of CSV expansion)
    st.info("Expanding database with diverse tracks...")
    try:
        # Step 1: Clean existing tracks table (remove tracks without audio features)
        removed_tracks = database_expansion.clean_database_tracks()
        if removed_tracks > 0:
            st.info(f"Cleaned database: removed {removed_tracks} tracks without audio features")
        
        # Step 2: Add user tracks to tracks table (get from database with audio features)
        user_tracks_with_features = db.get_user_tracks_with_features(spotify_user_id)
        if not user_tracks_with_features.empty:
            user_tracks_added = database_expansion.add_user_tracks_to_database_tracks(user_tracks_with_features)
            if user_tracks_added > 0:
                st.success(f"Added {user_tracks_added} of your tracks to database!")
        else:
            st.info("No user tracks with audio features found for expansion.")
            user_tracks_added = 0
        
        # Step 3: Expand tracks table with diverse tracks
        new_tracks = database_expansion.expand_database_tracks_background(sp, target_size=150000)
        if new_tracks > 0:
            st.success(f"Database expansion added {new_tracks:,} new tracks!")
            st.info("💡 The recommendation model now has access to a larger, more diverse pool of songs!")
        else:
            st.info("Database is already at target size.")
            
        # Step 4: Show recommendation status
        if user_tracks_added > 0 or new_tracks > 0:
            st.success("✅ **Great!** Your tracks have been added to the database!")
            st.info("🎵 **You can now generate recommendations** - the system will use your tracks to find similar songs!")
                        
            # Show recommendation status
            st.markdown("""
            <div style="background: var(--bg-secondary); padding: 1.5rem; border-radius: 15px; border: 1px solid var(--border-color); margin: 1rem 0;">
                <h4 style="color: var(--primary-color); margin-top: 0;">🚀 How Recommendations Work Now:</h4>
                <ul style="color: var(--text-secondary); line-height: 1.8;">
                    <li><strong>Database-First:</strong> All data stored in PostgreSQL for fast access</li>
                    <li><strong>Immediate Use:</strong> Your tracks are used to find similar songs in the database</li>
                    <li><strong>No Waiting:</strong> No need to retrain the model - recommendations work right away!</li>
                    <li><strong>Scalable:</strong> Database handles multiple users efficiently</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No new tracks added, database remains unchanged")
            
        # Return user tracks count for compatibility
        return stored_count
            
    except Exception as e:
        st.warning(f"Database expansion failed: {e}")
        
        # Return user tracks count for compatibility
        return stored_count if stored_count is not None else 0

def collect_playlist_data(sp):
    """Legacy function for collecting only playlist data (for backward compatibility)."""
    all_data = []
    playlists = sp.current_user_playlists()['items']

    for playlist in playlists:
        playlist_name = playlist['name']
        playlist_id = playlist['id']
        owner_id = playlist['owner']['id']

        st.write(f"Fetching tracks from playlist: {playlist_name}")
        results = sp.playlist_tracks(playlist_id)
        tracks = results['items']

        for item in tracks:
            track = item['track']
            if not track:
                continue
            track_id = track['id']
            if not track_id:  # skip local/private tracks
                continue

            track_name = track['name']
            artists = ", ".join([artist['name'] for artist in track['artists']])
            album = track['album']['name']
            popularity = track['popularity']

            # Note: Using Reccobeats API for audio features instead of deprecated Spotify endpoint
            # This is handled in the main collect_user_tracks function
            all_data.append({
                'playlist_name': playlist_name,
                'playlist_id': playlist_id,
                'owner_id': owner_id,
                'track_id': track_id,
                'track_name': track_name,
                'artists': artists,
                'album': album,
                'popularity': popularity
            })
            time.sleep(0.05)

    df = pd.DataFrame(all_data)
    df.to_csv(config.CSV_OUTPUT, index=False)
    st.success(f"Saved {len(df)} tracks to {config.CSV_OUTPUT}")
    return df