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

    def preprocess_user_tracks(self, user_track_features):
        """
        Preprocess user tracks to match the model's input format
        
        Args:
            user_track_features (pd.DataFrame): User's track features
            
        Returns:
            np.array: Preprocessed features
        """
        user_features = user_track_features.copy()

        # Ensure all columns are numeric and handle any data type issues
        # Include all 12 features that the model was trained on (11 audio + popularity)
        numeric_cols = ['acousticness', 'danceability', 'energy', 'instrumentalness', 
                       'liveness', 'speechiness', 'valence', 'loudness', 'tempo', 'key', 'mode', 'popularity']
        
        for col in numeric_cols:
            if col in user_features.columns:
                # Convert to numeric, coercing errors to NaN
                user_features[col] = pd.to_numeric(user_features[col], errors='coerce')

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
            user_features['mode'] = user_features['mode'] / 1.0  # mode is 0 or 1
        
        # Handle popularity normalization (0-100 scale)
        if 'popularity' in user_features.columns:
            user_features['popularity'] = user_features['popularity'] / 100.0
        
        # Select only the numeric columns for the model
        feature_cols = [col for col in numeric_cols if col in user_features.columns]
        return user_features[feature_cols].values

    def create_user_profile(self, user_track_features):
        """
        Create a user taste profile from their listening history
        
        Args:
            user_track_features (pd.DataFrame): User's track features
            
        Returns:
            np.array: User profile embedding
        """
        # Preprocess user tracks
        processed_features = self.preprocess_user_tracks(user_track_features)
        
        # Generate embeddings for user's tracks
        user_track_embeddings = self.encoder.predict(processed_features)
        
        # Create user profile as average of their track embeddings
        user_profile = np.mean(user_track_embeddings, axis=0)
    
        return user_profile

    def recommend_songs(self, user_profile, n_recommendations=20, exclude_track_ids=None):
        """
        Recommend songs based on user profile
        
        Args:
            user_profile (np.array): User's taste profile embedding
            n_recommendations (int): Number of recommendations to return
            exclude_track_ids (set): Set of track IDs to exclude from recommendations
            
        Returns:
            list: List of recommendation dictionaries
        """
        # Find similar songs using cosine similarity
        similarities = cosine_similarity([user_profile], self.song_embeddings)[0]
        
        # Get top similar songs
        similar_indices = np.argsort(similarities)[::-1]
        
        recommendations = []
        for idx in similar_indices:
            track_id = self.track_ids[idx]
            
            # Skip if track is in user's history
            if exclude_track_ids and track_id in exclude_track_ids:
                continue
                
            track_info = self.df[self.df['track_id'] == track_id].iloc[0]
            
            recommendations.append({
                'track_id': track_id,
                'track_name': track_info['track_name'],
                'artists': track_info['artists'],
                'popularity': track_info['popularity'],
                'similarity_score': similarities[idx]
            })
            
            if len(recommendations) >= n_recommendations:
                break
                
        return recommendations

            




