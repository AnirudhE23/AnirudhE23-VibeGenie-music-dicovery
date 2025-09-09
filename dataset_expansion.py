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

def add_user_tracks_to_training_dataset(user_tracks_df):
    """
    Add user's collected tracks to the training dataset.
    This ensures the training dataset includes the user's music taste.
    
    Args:
        user_tracks_df: DataFrame with user's tracks and audio features
    
    Returns:
        Number of new tracks added to training dataset
    """
    if user_tracks_df.empty:
        return 0
    
    # Load existing training dataset
    try:
        training_df = pd.read_csv("Final_training_dataset.csv")
    except FileNotFoundError:
        st.error("Final_training_dataset.csv not found!")
        return 0
    
    # Filter user tracks that have COMPLETE audio features (no NaN values)
    feature_cols = ['acousticness', 'danceability', 'energy', 'instrumentalness', 
                   'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'valence']
    
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
    
    # Format user tracks to match training dataset structure
    user_training_format = {
        'track_id': user_tracks_with_features['Track ID'],
        'track_name': user_tracks_with_features['Track Name'],
        'artists': user_tracks_with_features['Artist(s)'],
        'popularity': user_tracks_with_features['Popularity'],
        'acousticness': user_tracks_with_features['acousticness'],
        'danceability': user_tracks_with_features['danceability'],
        'energy': user_tracks_with_features['energy'],
        'instrumentalness': user_tracks_with_features['instrumentalness'],
        'key': user_tracks_with_features['key'],
        'liveness': user_tracks_with_features['liveness'],
        'loudness': user_tracks_with_features['loudness'],
        'mode': user_tracks_with_features['mode'],
        'speechiness': user_tracks_with_features['speechiness'],
        'tempo': user_tracks_with_features['tempo'],
        'valence': user_tracks_with_features['valence']
    }
    
    user_training_df = pd.DataFrame(user_training_format)
    
    # Remove tracks already in training dataset
    existing_ids = set(training_df["track_id"].unique())
    new_user_tracks = user_training_df[~user_training_df["track_id"].isin(existing_ids)]
    
    if len(new_user_tracks) == 0:
        return 0
    
    # Add user tracks to training dataset
    expanded_training_df = pd.concat([training_df, new_user_tracks], ignore_index=True)
    expanded_training_df = expanded_training_df.drop_duplicates(subset=["track_id"])
    
    # Save updated training dataset
    expanded_training_df.to_csv("Final_training_dataset.csv", index=False)
    
    return len(new_user_tracks)

def clean_training_dataset():
    """
    Clean the training dataset by removing tracks without complete audio features.
    This ensures the model only trains on tracks with valid audio features.
    
    Returns:
        Number of tracks removed
    """
    try:
        training_df = pd.read_csv("Final_training_dataset.csv")
    except FileNotFoundError:
        return 0
    
    original_count = len(training_df)
    
    # Define required audio features
    feature_cols = ['acousticness', 'danceability', 'energy', 'instrumentalness', 
                   'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'valence']
    
    # Remove tracks that are missing any audio features
    cleaned_df = training_df.dropna(subset=feature_cols)
    
    # Additional check: remove tracks with empty strings in audio features
    for col in feature_cols:
        if col in cleaned_df.columns:
            cleaned_df = cleaned_df[
                (cleaned_df[col] != '') & 
                (cleaned_df[col].notna())
            ]
    
    # Save cleaned dataset
    cleaned_df.to_csv("Final_training_dataset.csv", index=False)
    
    removed_count = original_count - len(cleaned_df)
    return removed_count

def retrain_model_with_expanded_dataset():
    """
    Retrain the model with the expanded training dataset.
    This generates new embeddings and updates the model files.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Import training modules
        from models.train import train_model
        from models.utils import save_models_and_data
        
        st.info("🔄 Retraining model with expanded dataset...")
        
        # Train the model with the updated dataset
        training_results = train_model()
        
        if training_results:
            st.success("✅ Model retrained successfully with expanded dataset!")
            return True
        else:
            st.error("❌ Model retraining failed!")
            return False
            
    except Exception as e:
        st.error(f"❌ Error retraining model: {e}")
        return False

def expand_training_dataset_background(sp, target_size=150000):
    """
    Automatically expand the training dataset in the background to improve recommendation quality.
    This happens behind the scenes and adds diverse tracks to the training pool.
    
    Args:
        sp: Spotify client
        target_size: Target total size for the training dataset
    
    Returns:
        Number of new tracks added
    """
    # Load existing training dataset
    try:
        existing_df = pd.read_csv("Final_training_dataset.csv")
    except FileNotFoundError:
        return 0
    
    if len(existing_df) >= target_size:
        return 0
    
    needed_tracks = target_size - len(existing_df)
    
    # Load existing cache
    cache_df = data_collection.load_features_cache()
    existing_ids = set(existing_df["track_id"].unique())
    
    # Smart expansion strategies for training dataset
    new_tracks = []
    
    # Strategy 1: Popular tracks from diverse genres and years
    new_tracks.extend(get_diverse_training_tracks(sp, needed_tracks // 3))
    
    # Strategy 2: Trending tracks from various time periods
    new_tracks.extend(get_trending_training_tracks(sp, needed_tracks // 3))
    
    # Strategy 3: Tracks from curated playlists (discovery-focused)
    new_tracks.extend(get_curated_training_tracks(sp, needed_tracks // 3))
    
    if not new_tracks:
        return 0
    
    # Convert to DataFrame and remove duplicates
    new_tracks_df = pd.DataFrame(new_tracks)
    new_tracks_df = new_tracks_df.drop_duplicates(subset=["track_id"])
    
    # Remove tracks already in training dataset
    new_tracks_df = new_tracks_df[~new_tracks_df["track_id"].isin(existing_ids)]
    
    if len(new_tracks_df) == 0:
        return 0
    
    # Get audio features for new tracks
    new_tracks_with_features = get_features_for_training_tracks(new_tracks_df, cache_df)
    
    # Combine with existing training data
    expanded_df = pd.concat([existing_df, new_tracks_with_features], ignore_index=True)
    expanded_df = expanded_df.drop_duplicates(subset=["track_id"])
    
    # Save expanded training dataset
    expanded_df.to_csv("Final_training_dataset.csv", index=False)
    
    return len(new_tracks_with_features)

def get_diverse_training_tracks(sp, max_tracks):
    """Get diverse tracks for training dataset expansion."""
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

def get_trending_training_tracks(sp, max_tracks):
    """Get trending tracks for training dataset expansion."""
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

def get_curated_training_tracks(sp, max_tracks):
    """Get tracks from Spotify's curated playlists for training dataset expansion."""
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

def analyze_user_music_profile(existing_df):
    """Analyze user's music profile to understand their taste."""
    profile = {
        'avg_popularity': existing_df['Popularity'].mean() if 'Popularity' in existing_df.columns else 50,
        'popularity_std': existing_df['Popularity'].std() if 'Popularity' in existing_df.columns else 20,
        'top_artists': existing_df['Artist(s)'].value_counts().head(10).index.tolist() if 'Artist(s)' in existing_df.columns else [],
        'total_tracks': len(existing_df)
    }
    return profile

def get_related_tracks_from_artists(sp, existing_df, max_tracks):
    """Get related tracks from user's top artists."""
    related_tracks = []
    
    # Get top artists from existing data
    artist_counts = existing_df['Artist(s)'].value_counts()
    top_artists = artist_counts.head(10).index.tolist()
    
    st.write(f"Top artists: {', '.join(top_artists[:5])}...")
    
    for artist_name in top_artists:
        if len(related_tracks) >= max_tracks:
            break
            
        try:
            # Search for the artist
            results = sp.search(q=f'artist:"{artist_name}"', type='artist', limit=1)
            if not results['artists']['items']:
                continue
                
            artist = results['artists']['items'][0]
            artist_id = artist['id']
            
            # Get artist's top tracks
            top_tracks = sp.artist_top_tracks(artist_id, country='US')
            
            for track in top_tracks['tracks'][:5]:  # Get top 5 tracks per artist
                if len(related_tracks) >= max_tracks:
                    break
                    
                track_info = spotify_utils.extract_track_info(
                    track, 
                    playlist_name="Expansion - Related Artist", 
                    owner_id=""
                )
                if track_info:
                    related_tracks.append(track_info)
                    
        except Exception as e:
            st.warning(f"Error getting tracks for artist {artist_name}: {e}")
            continue
    
    return related_tracks

def get_tracks_from_similar_playlists(sp, existing_df, max_tracks):
    """Get tracks from playlists that might be similar to user's taste."""
    related_tracks = []
    
    # Get genres from existing data (if available)
    # This is a simplified approach - in reality, you'd need genre data
    try:
        # Get some popular playlists that might match user's taste
        # This is a basic implementation - you could enhance this with genre matching
        
        # Search for popular playlists with keywords from user's music
        keywords = get_music_keywords(existing_df)
        
        for keyword in keywords[:3]:  # Try top 3 keywords
            if len(related_tracks) >= max_tracks:
                break
                
            try:
                results = sp.search(q=keyword, type='playlist', limit=5)
                for playlist in results['playlists']['items']:
                    if len(related_tracks) >= max_tracks:
                        break
                        
                    # Get tracks from this playlist
                    playlist_tracks = spotify_utils.get_playlist_tracks(sp, playlist['id'])
                    
                    for track_item in playlist_tracks[:10]:  # Limit per playlist
                        if len(related_tracks) >= max_tracks:
                            break
                            
                        track_info = spotify_utils.extract_track_info(
                            track_item,
                            playlist_name=f"Expansion - {playlist['name']}",
                            owner_id=playlist['owner']['id']
                        )
                        if track_info:
                            related_tracks.append(track_info)
                            
            except Exception as e:
                st.warning(f"Error searching for keyword {keyword}: {e}")
                continue
                
    except Exception as e:
        st.warning(f"Error in playlist expansion: {e}")
    
    return related_tracks

def get_popular_tracks_from_genres(sp, existing_df, max_tracks):
    """Get popular tracks from genres present in user's library."""
    related_tracks = []
    
    # This is a simplified approach since we don't have genre data
    # In a real implementation, you'd analyze the existing tracks for genres
    
    # For now, we'll get some popular tracks that might match the user's taste
    # based on the popularity distribution of their existing tracks
    
    try:
        # Get average popularity of user's tracks
        avg_popularity = existing_df['Popularity'].mean()
        
        # Search for popular tracks that might match
        search_terms = ["popular", "trending", "hits"]
        
        for term in search_terms:
            if len(related_tracks) >= max_tracks:
                break
                
            try:
                results = sp.search(q=term, type='track', limit=20)
                for track in results['tracks']['items']:
                    if len(related_tracks) >= max_tracks:
                        break
                        
                    # Only include tracks with similar popularity
                    if abs(track['popularity'] - avg_popularity) <= 20:
                        track_info = spotify_utils.extract_track_info(
                            track,
                            playlist_name="Expansion - Popular Tracks",
                            owner_id=""
                        )
                        if track_info:
                            related_tracks.append(track_info)
                            
            except Exception as e:
                st.warning(f"Error searching for {term}: {e}")
                continue
                
    except Exception as e:
        st.warning(f"Error in genre expansion: {e}")
    
    return related_tracks

def get_music_keywords(existing_df):
    """Extract keywords from user's music for playlist search."""
    keywords = []
    
    # Get common words from track names and artists
    all_text = " ".join(existing_df['Track Name'].fillna("").astype(str))
    all_text += " " + " ".join(existing_df['Artist(s)'].fillna("").astype(str))
    
    # Simple keyword extraction (you could make this more sophisticated)
    words = all_text.lower().split()
    word_counts = pd.Series(words).value_counts()
    
    # Filter out common words and get meaningful keywords
    common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    keywords = word_counts[~word_counts.index.isin(common_words)].head(10).index.tolist()
    
    return keywords

def get_features_for_training_tracks(new_tracks_df, cache_df):
    """Get audio features for new training tracks using your existing pipeline."""
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

    # Merge new tracks with their features and format for training dataset
    final_df = pd.merge(new_tracks_df, features_df, left_on="track_id", right_on="Track ID", how="left")
    
    # Only keep tracks that have ALL audio features (no missing values)
    feature_cols = ['acousticness', 'danceability', 'energy', 'instrumentalness', 
                   'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'valence']
    
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
    
    # Format to match training dataset structure
    training_format = {
        'track_id': tracks_with_complete_features['track_id'],
        'track_name': tracks_with_complete_features['track_name'],
        'artists': tracks_with_complete_features['artists'],
        'popularity': tracks_with_complete_features['popularity'],
        'acousticness': tracks_with_complete_features['acousticness'],
        'danceability': tracks_with_complete_features['danceability'],
        'energy': tracks_with_complete_features['energy'],
        'instrumentalness': tracks_with_complete_features['instrumentalness'],
        'key': tracks_with_complete_features['key'],
        'liveness': tracks_with_complete_features['liveness'],
        'loudness': tracks_with_complete_features['loudness'],
        'mode': tracks_with_complete_features['mode'],
        'speechiness': tracks_with_complete_features['speechiness'],
        'tempo': tracks_with_complete_features['tempo'],
        'valence': tracks_with_complete_features['valence']
    }
    
    return pd.DataFrame(training_format)

def get_smart_recommendations(sp, user_tracks_df, n_recommendations=20):
    """
    Get smart recommendations that include both dataset tracks and external discoveries.
    This is the main function for the recommendation system.
    """
    if user_tracks_df.empty:
        return [], "No user data available"
    
    # Analyze user's music profile
    user_profile = analyze_user_music_profile(user_tracks_df)
    
    # Get recommendations from multiple sources
    all_recommendations = []
    
    # 1. Get recommendations from existing dataset (using your existing system)
    try:
        from recommendation_system import recommendation_engine
        dataset_recs, error = recommendation_engine.get_recommendations(user_tracks_df, n_recommendations // 2)
        if dataset_recs and not error:
            all_recommendations.extend(dataset_recs)
    except Exception as e:
        st.warning(f"Could not get dataset recommendations: {e}")
    
    # 2. Get external recommendations based on user's taste
    external_recs = get_external_recommendations(sp, user_profile, n_recommendations // 2)
    all_recommendations.extend(external_recs)
    
    # 3. Mix and rank all recommendations
    final_recommendations = rank_and_mix_recommendations(all_recommendations, user_profile)
    
    return final_recommendations[:n_recommendations], None

def get_external_recommendations(sp, user_profile, n_recommendations):
    """Get recommendations from external sources based on user's taste profile."""
    external_recs = []
    
    # Strategy 1: Search for tracks similar to user's top artists
    for artist in user_profile['top_artists'][:3]:  # Top 3 artists
        if len(external_recs) >= n_recommendations:
            break
            
        try:
            # Search for tracks by similar artists
            results = sp.search(q=f'artist:"{artist}"', type='track', limit=10)
            for track in results['tracks']['items']:
                if len(external_recs) >= n_recommendations:
                    break
                    
                # Only include tracks with similar popularity to user's taste
                if abs(track['popularity'] - user_profile['avg_popularity']) <= user_profile['popularity_std']:
                    rec = {
                        'track_name': track['name'],
                        'artists': ', '.join([a['name'] for a in track['artists']]),
                        'popularity': track['popularity'],
                        'similarity_score': 0.8,  # High similarity for same artist
                        'source': 'External - Same Artist',
                        'track_id': track['id']
                    }
                    external_recs.append(rec)
                    
        except Exception as e:
            continue
    
    # Strategy 2: Search for tracks in user's preferred popularity range
    if len(external_recs) < n_recommendations:
        try:
            # Search for popular tracks in user's range
            results = sp.search(q='year:2024', type='track', limit=20)
            for track in results['tracks']['items']:
                if len(external_recs) >= n_recommendations:
                    break
                    
                if (user_profile['avg_popularity'] - user_profile['popularity_std'] <= 
                    track['popularity'] <= user_profile['avg_popularity'] + user_profile['popularity_std']):
                    
                    rec = {
                        'track_name': track['name'],
                        'artists': ', '.join([a['name'] for a in track['artists']]),
                        'popularity': track['popularity'],
                        'similarity_score': 0.6,  # Medium similarity for popularity match
                        'source': 'External - Popularity Match',
                        'track_id': track['id']
                    }
                    external_recs.append(rec)
                    
        except Exception as e:
            pass
    
    # Strategy 3: Get tracks from curated playlists that match user's taste
    if len(external_recs) < n_recommendations:
        curated_playlists = ["New Music Friday", "Today's Top Hits", "Global Top 50"]
        
        for playlist_name in curated_playlists:
            if len(external_recs) >= n_recommendations:
                break
                
            try:
                results = sp.search(q=playlist_name, type='playlist', limit=1)
                if results['playlists']['items']:
                    playlist = results['playlists']['items'][0]
                    playlist_tracks = spotify_utils.get_playlist_tracks(sp, playlist['id'])
                    
                    for track_item in playlist_tracks[:5]:  # Limit per playlist
                        if len(external_recs) >= n_recommendations:
                            break
                            
                        track = track_item.get('track', {})
                        if track and track.get('id'):
                            rec = {
                                'track_name': track['name'],
                                'artists': ', '.join([a['name'] for a in track['artists']]),
                                'popularity': track.get('popularity', 50),
                                'similarity_score': 0.7,  # Good similarity for curated content
                                'source': f'External - {playlist_name}',
                                'track_id': track['id']
                            }
                            external_recs.append(rec)
                            
            except Exception as e:
                continue
    
    return external_recs

def rank_and_mix_recommendations(all_recommendations, user_profile):
    """Rank and mix recommendations from different sources."""
    if not all_recommendations:
        return []
    
    # Remove duplicates based on track name and artist
    seen = set()
    unique_recs = []
    for rec in all_recommendations:
        key = (rec['track_name'].lower(), rec['artists'].lower())
        if key not in seen:
            seen.add(key)
            unique_recs.append(rec)
    
    # Sort by similarity score and popularity
    unique_recs.sort(key=lambda x: (x['similarity_score'], x['popularity']), reverse=True)
    
    return unique_recs
