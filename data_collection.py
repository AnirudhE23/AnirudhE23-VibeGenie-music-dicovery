import pandas as pd
import streamlit as st
import pathlib
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import config
import spotify_utils
import reccobeats_utils

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
    Fetches:
    1. Current user's playlists
    2. Current user's liked songs
    3. Current user's top tracks (short, medium, long term)
    Returns a single DataFrame with audio features (Spotify + Reccobeats)
    """
    cache_df = load_features_cache()
    cached_ids = set(cache_df["Track ID"].unique()) if not cache_df.empty else set()
    
    # Get all user tracks using spotify_utils
    tracks_df = spotify_utils.get_all_user_tracks(sp)
    
    st.write(f"Total tracks collected: {len(tracks_df)}")
    st.write(f"Already cached: {len(cached_ids)}")
    
    # Which tracks still need features
    need_ids = [tid for tid in tracks_df["Track ID"].dropna().unique().tolist() if tid not in cached_ids]
    st.write(f"Need features: {len(need_ids)}")

    # Batch fetch Spotify IDs
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
            batch_progress.progress(min((i + config.BATCH_SIZE) / len(need_ids), 1.0))
            status.text(f"Batch processed {min(i+config.BATCH_SIZE, len(need_ids))}/{len(need_ids)} tracks")

    features_df = cache_df.copy()
    have_ids = set(features_df["Track ID"].unique()) if not features_df.empty else set()
    remaining = tracks_df[~tracks_df["Track ID"].isin(have_ids)].copy()

    # Fallback pass (artist->track walk, enhanced)
    if len(remaining) > 0:
        # Sort by popularity to prioritize popular tracks first
        remaining = remaining.sort_values("Popularity", ascending=False)
        fb_progress = st.progress(0, text=f"Fallback lookups ({len(remaining)} tracks)")
        fb_status = st.empty()
        spotify_to_recco = {}
        fallback_results = []

        def run_fallback(row): 
            """Return a FEATURES DATAFRAME or None.""" 
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

    # Merge everything into final dataframe
    final_df = pd.merge(tracks_df.drop(columns=["ISRC"]), features_df, on="Track ID", how="left")
    final_df.to_csv(config.USER_TRACKS_CSV, index=True)

    feature_cols = [c for c in features_df.columns if c != "Track ID"] if not features_df.empty else []
    rows_with_features = final_df[feature_cols].notna().any(axis=1).sum() if feature_cols else 0
    cached_unique = features_df["Track ID"].nunique() if not features_df.empty else 0

    st.write(f"Final rows: {len(final_df)}. With features: {rows_with_features} (unique cached tracks: {cached_unique})")
    return final_df

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

            try:
                features = sp.audio_features(track_id)[0]
            except Exception as e:
                st.error(f"Skipping track {track_id} due to error: {e}")
                continue

            if not features:
                continue

            all_data.append({
                'playlist_name': playlist_name,
                'playlist_id': playlist_id,
                'owner_id': owner_id,
                'track_id': track_id,
                'track_name': track_name,
                'artists': artists,
                'album': album,
                'popularity': popularity,
                **{key: features[key] for key in features if key != 'type'}
            })
            time.sleep(0.05)

    df = pd.DataFrame(all_data)
    df.to_csv(config.CSV_OUTPUT, index=False)
    st.success(f"Saved {len(df)} tracks to {config.CSV_OUTPUT}")
    return df