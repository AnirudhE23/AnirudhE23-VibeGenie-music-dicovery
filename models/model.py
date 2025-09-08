# Model architecture definition
# Define your model classes and functions here

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from .config import MODEL_CONFIG


def create_music_autoencoder(input_dim, embedding_dim=None):
    """
    Create autoencoder for learning music embeddings
    Based on research showing autoencoders work well for music recommendation
    
    Args:
        input_dim (int): Number of input features
        embedding_dim (int): Dimension of the embedding space
    
    Returns:
        tuple: (autoencoder_model, encoder_model)
    """
    if embedding_dim is None:
        embedding_dim = MODEL_CONFIG['embedding_dim']
    
    # Input layer
    input_layer = tf.keras.layers.Input(shape=(input_dim,), name='audio_features')
    
    # Encoder
    encoded = tf.keras.layers.Dense(256, activation='relu', name='encoder_1')(input_layer)
    encoded = tf.keras.layers.BatchNormalization()(encoded)
    encoded = tf.keras.layers.Dropout(0.3)(encoded)

    encoded = tf.keras.layers.Dense(128, activation='relu', name='encoder_2')(encoded)
    encoded = tf.keras.layers.BatchNormalization()(encoded)
    encoded = tf.keras.layers.Dropout(0.2)(encoded)

    # Bottleneck layer
    embedding = tf.keras.layers.Dense(embedding_dim, activation='relu', name='song_embedding')(encoded)

    # Decoder
    decoded = tf.keras.layers.Dense(128, activation='relu', name='decoder_1')(embedding)
    decoded = tf.keras.layers.BatchNormalization()(decoded)
    decoded = tf.keras.layers.Dropout(0.2)(decoded)

    decoded = tf.keras.layers.Dense(256, activation='relu', name='decoder_2')(decoded)
    decoded = tf.keras.layers.BatchNormalization()(decoded)
    decoded = tf.keras.layers.Dropout(0.3)(decoded)

    # Output layer
    output = tf.keras.layers.Dense(input_dim, activation='sigmoid', name='reconstructed')(decoded)
        
    # Create models
    autoencoder = tf.keras.models.Model(inputs=input_layer, outputs=output, name='music_autoencoder')
    encoder = tf.keras.Model(input_layer, embedding, name='music_encoder')
    
    return autoencoder, encoder


def create_two_tower_model(input_dim, embedding_dim=None):
    """
    Two-tower architecture for user-song matching
    Based on Spotify's research on contextual embeddings
    
    Args:
        input_dim (int): Number of input features
        embedding_dim (int): Dimension of the embedding space
    
    Returns:
        tuple: (model, song_encoder, user_encoder)
    """
    if embedding_dim is None:
        embedding_dim = MODEL_CONFIG['embedding_dim']
    
    # Song tower
    song_input = tf.keras.layers.Input(shape=(input_dim,), name='song_features')
    song_tower = tf.keras.layers.Dense(256, activation='relu')(song_input)
    song_tower = tf.keras.layers.Dropout(0.3)(song_tower)
    song_tower = tf.keras.layers.Dense(128, activation='relu')(song_tower)
    song_tower = tf.keras.layers.Dropout(0.2)(song_tower)
    song_embedding = tf.keras.layers.Dense(embedding_dim, activation='relu', name='song_emb')(song_tower)
    
    # User tower (will be used at inference time)
    user_input = tf.keras.layers.Input(shape=(input_dim,), name='user_profile')
    user_tower = tf.keras.layers.Dense(256, activation='relu')(user_input)
    user_tower = tf.keras.layers.Dropout(0.3)(user_tower)
    user_tower = tf.keras.layers.Dense(128, activation='relu')(user_tower)
    user_tower = tf.keras.layers.Dropout(0.2)(user_tower)
    user_embedding = tf.keras.layers.Dense(embedding_dim, activation='relu', name='user_emb')(user_tower)
    
    # Similarity computation
    dot_product = tf.keras.layers.Dot(axes=1, normalize=True)([user_embedding, song_embedding])
    
    model = tf.keras.Model([user_input, song_input], dot_product, name='two_tower')
    song_encoder = tf.keras.Model(song_input, song_embedding, name='song_encoder')
    user_encoder = tf.keras.Model(user_input, user_embedding, name='user_encoder')
    
    return model, song_encoder, user_encoder


class MusicRecommender:
    """
    Music recommendation system using learned embeddings
    """
    
    def __init__(self, song_embeddings, track_ids, df, encoder, scalers):
        """
        Initialize the recommender
        
        Args:
            song_embeddings (np.array): Pre-computed song embeddings
            track_ids (np.array): Array of track IDs corresponding to embeddings
            df (pd.DataFrame): Original dataset with track information
            encoder (tf.keras.Model): Trained encoder model
            scalers (dict): Dictionary containing fitted scalers
        """
        self.song_embeddings = song_embeddings
        self.track_ids = track_ids
        self.df = df
        self.encoder = encoder
        self.scaler_bounded = scalers['bounded']
        self.scaler_unbounded = scalers['unbounded']

        # Building the Nearest Neighbors index for fast similarity search
        self.nn_model = NearestNeighbors(n_neighbors=50, metric='cosine')
        self.nn_model.fit(song_embeddings)
        
        # OPTIMIZATION: Cache for user embeddings to avoid recomputation
        self.user_embedding_cache = {}
        self.cache_max_size = 100  # Limit cache size

    def preprocess_user_tracks(self, user_track_features):
        """
        Preprocess user tracks to match the model's input format
        
        Args:
            user_track_features (pd.DataFrame): User's track features
            
        Returns:
            np.array: Preprocessed features
        """
        user_features = user_track_features.copy()

        # CORRECTED: Handle both 'popularity' and 'popularity_norm' columns
        # The model expects 12 features: 11 audio + popularity_norm
        
        # Define the expected feature columns (what the model was trained on)
        expected_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 
                            'liveness', 'speechiness', 'valence', 'loudness', 'tempo', 'key', 'mode', 'popularity_norm']
        
        # Ensure all columns are numeric
        for col in expected_features:
            if col in user_features.columns:
                user_features[col] = pd.to_numeric(user_features[col], errors='coerce')
        
        # Handle popularity column - convert 'popularity' to 'popularity_norm' if needed
        if 'popularity' in user_features.columns and 'popularity_norm' not in user_features.columns:
            user_features['popularity_norm'] = user_features['popularity'] / 100.0
        elif 'popularity_norm' not in user_features.columns:
            # If neither exists, use default value
            user_features['popularity_norm'] = 0.5
            print("⚠️ No popularity column found. Using default value 0.5")

        # Apply same scaling as training
        bounded_cols = ['acousticness', 'danceability', 'energy', 'instrumentalness', 
                    'liveness', 'speechiness', 'valence']
        unbounded_cols = ['loudness', 'tempo']

        # Only process columns that exist and have valid data
        bounded_cols = [col for col in bounded_cols if col in user_features.columns]
        unbounded_cols = [col for col in unbounded_cols if col in user_features.columns]

        if bounded_cols:
            user_features[bounded_cols] = self.scaler_bounded.transform(user_features[bounded_cols])
        if unbounded_cols:
            user_features[unbounded_cols] = self.scaler_unbounded.transform(user_features[unbounded_cols])
        
        # Handle categorical features normalization
        if 'key' in user_features.columns:
            user_features['key'] = user_features['key'] / 11.0
        if 'mode' in user_features.columns:
            user_features['mode'] = user_features['mode']  # Already 0-1
        
        # Select only the expected features in the correct order
        final_features = []
        for feature in expected_features:
            if feature in user_features.columns:
                final_features.append(feature)
            else:
                print(f"⚠️ Missing feature: {feature}")
                # Add default value for missing features
                if feature == 'popularity_norm':
                    final_features.append(0.5)
                else:
                    final_features.append(0.0)
        
        return user_features[expected_features].values

    def create_user_profile(self, user_track_features):
        """
        Create a user taste profile from their listening history with weighted averaging
        
        Args:
            user_track_features (pd.DataFrame): User's track features
            
        Returns:
            np.array: User profile embedding
        """
        # Preprocess user tracks
        processed_features = self.preprocess_user_tracks(user_track_features)
        
        # Generate embeddings for user's tracks using existing encoder
        user_track_embeddings = self.encoder.predict(processed_features)
        
        # Create weighted user profile based on track characteristics
        user_profile = self._create_weighted_user_profile(user_track_features, user_track_embeddings)
    
        return user_profile
    
    def find_similar_songs_to_user_tracks(self, user_track_features, n_recommendations=20, exclude_track_ids=None):
        """
        Find songs similar to user's tracks using existing model embeddings
        OPTIMIZED VERSION for faster performance
        
        Args:
            user_track_features (pd.DataFrame): User's track features
            n_recommendations (int): Number of recommendations per user track
            exclude_track_ids (set): Track IDs to exclude
            
        Returns:
            list: List of recommendation dictionaries
        """
        import time
        start_time = time.time()
        # OPTIMIZATION: Check cache first
        cache_key = self._get_cache_key(user_track_features)
        if cache_key in self.user_embedding_cache:
            user_track_embeddings = self.user_embedding_cache[cache_key]
        else:
            # Preprocess user tracks
            processed_features = self.preprocess_user_tracks(user_track_features)
            
            # Generate embeddings for user's tracks
            user_track_embeddings = self.encoder.predict(processed_features)
            
            # Cache the embeddings
            self._cache_user_embeddings(cache_key, user_track_embeddings)
        
        # OPTIMIZATION 1: Create track_id to index mapping for fast lookups
        track_id_to_idx = {track_id: idx for idx, track_id in enumerate(self.track_ids)}
        
        # OPTIMIZATION 2: Pre-filter excluded tracks
        exclude_indices = set()
        if exclude_track_ids:
            for track_id in exclude_track_ids:
                if track_id in track_id_to_idx:
                    exclude_indices.add(track_id_to_idx[track_id])
        
        # OPTIMIZATION 3: Batch cosine similarity calculation
        # Calculate similarities for all user tracks at once
        all_similarities = cosine_similarity(user_track_embeddings, self.song_embeddings)
        
        all_recommendations = []
        seen_track_ids = set(exclude_track_ids) if exclude_track_ids else set()
        
        # OPTIMIZATION 4: Process user tracks in batches for better performance
        for i, (idx, row) in enumerate(user_track_features.iterrows()):
            similarities = all_similarities[i]
            
            # OPTIMIZATION 5: Use numpy operations for faster sorting
            # Get top similar songs (more than needed for diversity selection)
            top_k = min(len(similarities), n_recommendations * 5)  # Get 5x more for diversity
            similar_indices = np.argpartition(similarities, -top_k)[-top_k:]
            similar_indices = similar_indices[np.argsort(similarities[similar_indices])[::-1]]
            
            # Collect recommendations for this user track
            track_recommendations = []
            for sim_idx in similar_indices:
                # Skip excluded tracks
                if sim_idx in exclude_indices:
                    continue
                    
                track_id = self.track_ids[sim_idx]
                
                # Skip if already seen
                if track_id in seen_track_ids:
                    continue
                
                # OPTIMIZATION 6: Direct array access instead of DataFrame lookup
                track_info = {
                    'track_id': track_id,
                    'track_name': self.df.iloc[sim_idx]['track_name'],
                    'artists': self.df.iloc[sim_idx]['artists'],
                    'popularity': self.df.iloc[sim_idx]['popularity'],
                    'similarity_score': similarities[sim_idx],
                    'source_track': row.get('Track Name', 'Unknown'),
                    'source_artist': row.get('Artist(s)', 'Unknown')
                }
                
                track_recommendations.append(track_info)
                seen_track_ids.add(track_id)
                
                if len(track_recommendations) >= n_recommendations:
                    break
            
            all_recommendations.extend(track_recommendations)
        
        # Sort all recommendations by similarity score
        all_recommendations.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Return top recommendations with diversity
        final_recommendations = self._apply_diversity_to_recommendations(all_recommendations, n_recommendations)
        
        # Performance monitoring
        end_time = time.time()
        print(f"⚡ Recommendation generation took {end_time - start_time:.2f} seconds")
        
        return final_recommendations
    
    def _apply_diversity_to_recommendations(self, recommendations, n_recommendations, diversity_factor=0.3):
        """
        Apply diversity to recommendations to avoid repetitive results
        """
        if len(recommendations) <= n_recommendations:
            return recommendations
        
        # Select diverse recommendations
        diverse_recommendations = []
        used_artists = set()
        used_tracks = set()
        
        # First pass: select highest similarity recommendations
        for rec in recommendations:
            if len(diverse_recommendations) >= n_recommendations:
                break
            
            # Check for diversity
            artist_key = rec['artists'].lower()
            track_key = rec['track_name'].lower()
            
            # Allow some repetition but not too much
            artist_count = sum(1 for r in diverse_recommendations if r['artists'].lower() == artist_key)
            track_count = sum(1 for r in diverse_recommendations if r['track_name'].lower() == track_key)
            
            if artist_count < 2 and track_count == 0:  # Max 2 songs per artist, no duplicate tracks
                diverse_recommendations.append(rec)
        
        # If we need more recommendations, be less strict about diversity
        if len(diverse_recommendations) < n_recommendations:
            for rec in recommendations:
                if len(diverse_recommendations) >= n_recommendations:
                    break
                
                if rec not in diverse_recommendations:
                    diverse_recommendations.append(rec)
        
        return diverse_recommendations[:n_recommendations]
    
    def _get_cache_key(self, user_track_features):
        """
        Generate a cache key for user track features
        """
        # Create a hash of the track IDs and features for caching
        track_ids = sorted(user_track_features['track_id'].tolist())
        return hash(tuple(track_ids))
    
    def _cache_user_embeddings(self, cache_key, embeddings):
        """
        Cache user embeddings with size limit
        """
        # Limit cache size
        if len(self.user_embedding_cache) >= self.cache_max_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.user_embedding_cache))
            del self.user_embedding_cache[oldest_key]
        
        self.user_embedding_cache[cache_key] = embeddings
    
    def _create_weighted_user_profile(self, user_track_features, user_track_embeddings):
        """
        Create a weighted user profile that emphasizes the user's dominant music characteristics
        """
        # Calculate weights based on track characteristics
        weights = []
        
        for idx, row in user_track_features.iterrows():
            weight = 1.0  # Base weight
            
            # Weight by popularity (more popular tracks get higher weight)
            if 'popularity' in row and pd.notna(row['popularity']):
                popularity_weight = min(2.0, max(0.5, row['popularity'] / 50.0))  # Scale 0-100 to 0.5-2.0
                weight *= popularity_weight
            
            # Weight by energy level (if user has high-energy tracks, emphasize them)
            if 'energy' in row and pd.notna(row['energy']):
                # If user has high-energy tracks, give them more weight
                if row['energy'] > 0.7:  # High energy
                    weight *= 1.5
                elif row['energy'] < 0.3:  # Low energy
                    weight *= 0.8
            
            # Weight by danceability (if user has danceable tracks, emphasize them)
            if 'danceability' in row and pd.notna(row['danceability']):
                if row['danceability'] > 0.7:  # High danceability
                    weight *= 1.3
                elif row['danceability'] < 0.3:  # Low danceability
                    weight *= 0.9
            
            # Weight by valence (mood)
            if 'valence' in row and pd.notna(row['valence']):
                if row['valence'] > 0.7:  # Positive mood
                    weight *= 1.2
                elif row['valence'] < 0.3:  # Negative mood
                    weight *= 1.1  # Still give some weight to negative mood tracks
            
            weights.append(weight)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize so they sum to 1
        
        # Create weighted average of embeddings
        user_profile = np.average(user_track_embeddings, axis=0, weights=weights)
        
        return user_profile

    def recommend_songs(self, user_profile, n_recommendations=20, exclude_track_ids=None, diversity_factor=0.3):
        """
        Recommend songs based on user profile with diversity and randomization
        
        Args:
            user_profile (np.array): User's taste profile embedding
            n_recommendations (int): Number of recommendations to return
            exclude_track_ids (set): Set of track IDs to exclude from recommendations
            diversity_factor (float): Factor to control diversity (0.0 = no diversity, 1.0 = max diversity)
            
        Returns:
            list: List of recommendation dictionaries
        """
        # Find similar songs using cosine similarity
        similarities = cosine_similarity([user_profile], self.song_embeddings)[0]
        
        # Get top similar songs (more than needed for diversity selection)
        similar_indices = np.argsort(similarities)[::-1]
        
        # Filter out excluded tracks
        available_indices = []
        for idx in similar_indices:
            track_id = self.track_ids[idx]
            if not exclude_track_ids or track_id not in exclude_track_ids:
                available_indices.append(idx)
        
        if len(available_indices) == 0:
            return []
        
        # Apply diversity and randomization
        recommendations = self._select_diverse_recommendations(
            available_indices, similarities, n_recommendations, diversity_factor
        )
        
        return recommendations
    
    def _select_diverse_recommendations(self, indices, similarities, n_recommendations, diversity_factor):
        """
        Select diverse recommendations using a combination of similarity and diversity
        """
        import random
        
        recommendations = []
        used_indices = set()
        
        # Get top candidates (3x more than needed for diversity selection)
        top_candidates = indices[:min(len(indices), n_recommendations * 3)]
        
        # First, add the most similar song
        if top_candidates:
            best_idx = top_candidates[0]
            recommendations.append(self._create_recommendation(best_idx, similarities[best_idx]))
            used_indices.add(best_idx)
        
        # Then select diverse recommendations
        while len(recommendations) < n_recommendations and len(used_indices) < len(top_candidates):
            candidates = [idx for idx in top_candidates if idx not in used_indices]
            if not candidates:
                break
            
            # Calculate diversity scores
            diversity_scores = []
            for idx in candidates:
                # Combine similarity with diversity
                similarity_score = similarities[idx]
                
                # Calculate diversity (distance from already selected recommendations)
                diversity_score = 0
                if recommendations:
                    for rec in recommendations:
                        # Use embedding distance as diversity measure
                        embedding_distance = np.linalg.norm(
                            self.song_embeddings[idx] - self.song_embeddings[rec['original_index']]
                        )
                        diversity_score += embedding_distance
                    diversity_score /= len(recommendations)
                
                # Normalize diversity score (0-1 range)
                max_possible_distance = np.sqrt(self.song_embeddings.shape[1]) * 2  # Max distance in embedding space
                diversity_score = diversity_score / max_possible_distance
                
                # Combine similarity and diversity
                combined_score = (1 - diversity_factor) * similarity_score + diversity_factor * diversity_score
                diversity_scores.append((combined_score, idx))
            
            # Sort by combined score and add some randomness
            diversity_scores.sort(reverse=True)
            
            # Add randomness to selection (select from top 30% with some randomness)
            top_percent = max(1, int(len(diversity_scores) * 0.3))
            selected_candidates = diversity_scores[:top_percent]
            
            # Random selection from top candidates
            if len(selected_candidates) > 1:
                # Weighted random selection (higher scores more likely)
                scores = [score for score, _ in selected_candidates]
                weights = np.array(scores)
                weights = weights / weights.sum()  # Normalize weights
                
                selected_idx = np.random.choice(
                    len(selected_candidates), 
                    p=weights
                )
                chosen_idx = selected_candidates[selected_idx][1]
            else:
                chosen_idx = selected_candidates[0][1]
            
            recommendations.append(self._create_recommendation(chosen_idx, similarities[chosen_idx]))
            used_indices.add(chosen_idx)
        
        return recommendations
    
    def _create_recommendation(self, idx, similarity_score):
        """Create a recommendation dictionary"""
        track_id = self.track_ids[idx]
        track_info = self.df[self.df['track_id'] == track_id].iloc[0]
        
        return {
            'track_id': track_id,
            'track_name': track_info['track_name'],
            'artists': track_info['artists'],
            'popularity': track_info['popularity'],
            'similarity_score': similarity_score,
            'original_index': idx  # Store for diversity calculations
        }

            




