# Training utilities
# Helper functions, data processing utilities, and common operations

import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from .config import FILE_PATHS, FEATURE_CONFIG


def save_models_and_data(autoencoder, encoder, song_embeddings, track_ids, df, scalers):
    """
    Save all trained models and data for later use
    
    Args:
        autoencoder: Trained autoencoder model
        encoder: Trained encoder model
        song_embeddings (np.array): Generated song embeddings
        track_ids (np.array): Track IDs
        df (pd.DataFrame): Original dataset
        scalers (dict): Fitted scalers
    """
    print("Saving models and preprocessors...")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save models
    autoencoder.save(FILE_PATHS['autoencoder_model'])
    encoder.save(FILE_PATHS['encoder_model'])
    
    # Save preprocessors
    joblib.dump(scalers['bounded'], FILE_PATHS['scaler_bounded'])
    joblib.dump(scalers['unbounded'], FILE_PATHS['scaler_unbounded'])
    
    # Save embeddings and metadata
    np.save(FILE_PATHS['song_embeddings'], song_embeddings)
    np.save(FILE_PATHS['track_ids'], track_ids)
    
    # Save metadata for quick retrieval
    metadata_df = df[['track_id', 'track_name', 'artists', 'popularity']].copy()
    metadata_df.to_csv(FILE_PATHS['song_metadata'], index=False)
    
    print("All models and data saved successfully!")
    print("Ready for deployment with Streamlit!")


def load_models_and_data():
    """
    Load all saved models and data
    
    Returns:
        dict: Dictionary containing all loaded components
    """
    print("Loading saved models and data...")
    
    try:
        # Load models with custom_objects to handle compatibility issues
        custom_objects = {
            'mse': tf.keras.metrics.MeanSquaredError,
            'mae': tf.keras.metrics.MeanAbsoluteError,
            'accuracy': tf.keras.metrics.Accuracy,
            'binary_accuracy': tf.keras.metrics.BinaryAccuracy,
            'categorical_accuracy': tf.keras.metrics.CategoricalAccuracy
        }
        
        # Try loading with custom_objects first
        try:
            autoencoder = tf.keras.models.load_model(
                FILE_PATHS['autoencoder_model'], 
                custom_objects=custom_objects,
                compile=False  # Don't compile the model
            )
            encoder = tf.keras.models.load_model(
                FILE_PATHS['encoder_model'], 
                custom_objects=custom_objects,
                compile=False  # Don't compile the model
            )
        except Exception as e:
            print(f"Warning: Could not load with custom_objects: {e}")
            # Fallback: try loading without custom_objects
            try:
                autoencoder = tf.keras.models.load_model(
                    FILE_PATHS['autoencoder_model'], 
                    compile=False
                )
                encoder = tf.keras.models.load_model(
                    FILE_PATHS['encoder_model'], 
                    compile=False
                )
            except Exception as e2:
                print(f"Warning: Could not load models directly: {e2}")
                print("Attempting to recreate models from architecture...")
                
                # Recreate the model architecture and load weights
                from .model import create_music_autoencoder
                
                # Create models with the same architecture
                autoencoder, encoder = create_music_autoencoder(input_dim=12, embedding_dim=64)
                
                # Try to load weights
                try:
                    autoencoder.load_weights(FILE_PATHS['autoencoder_model'])
                    encoder.load_weights(FILE_PATHS['encoder_model'])
                    print("✅ Successfully loaded model weights!")
                except Exception as e3:
                    print(f"❌ Could not load weights either: {e3}")
                    raise e3
        
        # Load preprocessors
        scaler_bounded = joblib.load(FILE_PATHS['scaler_bounded'])
        scaler_unbounded = joblib.load(FILE_PATHS['scaler_unbounded'])
        
        # Load embeddings and metadata
        song_embeddings = np.load(FILE_PATHS['song_embeddings'])
        track_ids = np.load(FILE_PATHS['track_ids'], allow_pickle=True)
        
        # Load metadata
        metadata_df = pd.read_csv(FILE_PATHS['song_metadata'])
        
        print("All models and data loaded successfully!")
        
        return {
            'autoencoder': autoencoder,
            'encoder': encoder,
            'song_embeddings': song_embeddings,
            'track_ids': track_ids,
            'metadata_df': metadata_df,
            'scalers': {
                'bounded': scaler_bounded,
                'unbounded': scaler_unbounded
            }
        }
        
    except FileNotFoundError as e:
        print(f"Error loading models: {e}")
        print("Please train the model first using train.py")
        return None


def create_recommender_from_saved_data():
    """
    Create a recommender instance from saved data
    
    Returns:
        MusicRecommender: Initialized recommender
    """
    from .model import MusicRecommender
    
    # Load saved data
    saved_data = load_models_and_data()
    
    if saved_data is None:
        return None
    
    # Create recommender
    recommender = MusicRecommender(
        song_embeddings=saved_data['song_embeddings'],
        track_ids=saved_data['track_ids'],
        df=saved_data['metadata_df'],
        encoder=saved_data['encoder'],
        scalers=saved_data['scalers']
    )
    
    return recommender


def test_recommendation_system(recommender, df, X_test, track_ids_test, n_test_tracks=5, n_recommendations=10):
    """
    Test the recommendation system using UNSEEN test data
    
    Args:
        recommender: MusicRecommender instance
        df (pd.DataFrame): Full dataset
        X_test (np.array): Test features (unseen during training)
        track_ids_test (np.array): Test track IDs (unseen during training)
        n_test_tracks (int): Number of test tracks to use
        n_recommendations (int): Number of recommendations to generate
    """
    
    # Get the actual feature names from the test data
    # The test data should have the same number of columns as the training data
    n_features = X_test.shape[1]
    print(f"Test data has {n_features} features")
    
    # Create feature names based on the actual data structure
    if n_features == 12:  # 11 audio features + popularity_norm
        feature_names = ['acousticness', 'danceability', 'energy', 'instrumentalness', 
                        'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'valence', 'popularity_norm']
    elif n_features == 11:  # Just audio features
        feature_names = ['acousticness', 'danceability', 'energy', 'instrumentalness', 
                        'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'valence']
    else:
        # Fallback: create generic feature names
        feature_names = [f'feature_{i}' for i in range(n_features)]
        print(f"⚠️ Unexpected number of features ({n_features}). Using generic names.")
    print(f"Testing recommendation system with {n_test_tracks} UNSEEN test tracks...")
    
    # Randomly sample from test set (unseen during training)
    test_indices = np.random.choice(len(X_test), min(n_test_tracks, len(X_test)), replace=False)
    
    print("\n" + "="*60)
    print("RECOMMENDATION SYSTEM TEST - UNSEEN DATA")
    print("="*60)
    
    for i, test_idx in enumerate(test_indices, 1):
        # Get test track features and ID
        test_features = X_test[test_idx]
        test_track_id = track_ids_test[test_idx]
        
        # Get track info
        track_info = df[df['track_id'] == test_track_id].iloc[0]
        print(f"\n{i}. Testing with: {track_info['track_name']} by {track_info['artists']}")
        print("-" * 50)
        
        # Create user profile from this single track
        # This simulates a user who only has this one song
        user_tracks = pd.DataFrame([test_features], columns=feature_names)
        
        # Create user profile
        user_profile = recommender.create_user_profile(user_tracks)
        
        # Get recommendations (exclude the test track itself)
        exclude_tracks = {test_track_id}
        recommendations = recommender.recommend_songs(
            user_profile, 
            n_recommendations=n_recommendations,
            exclude_track_ids=exclude_tracks
        )
        
        print(f"Generated {len(recommendations)} recommendations:")
        for j, rec in enumerate(recommendations[:5], 1):  # Show top 5
            print(f"  {j}. {rec['track_name']} by {rec['artists']}")
            print(f"     Similarity: {rec['similarity_score']:.3f}")
            print(f"     Popularity: {rec['popularity']}")
    
    print("\n" + "="*60)
    print("TESTING COMPLETED!")
    print("="*60)


def get_model_info(model):
    """
    Get information about a trained model
    
    Args:
        model: Trained model
        
    Returns:
        dict: Model information
    """
    info = {
        'model_type': type(model).__name__,
        'input_shape': model.input_shape,
        'output_shape': model.output_shape,
        'total_params': model.count_params(),
        'trainable_params': sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]),
        'non_trainable_params': sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    }
    
    return info


def print_model_summary(model):
    """
    Print a formatted model summary
    
    Args:
        model: Trained model
    """
    info = get_model_info(model)
    
    print("\n" + "="*50)
    print("MODEL SUMMARY")
    print("="*50)
    print(f"Model Type: {info['model_type']}")
    print(f"Input Shape: {info['input_shape']}")
    print(f"Output Shape: {info['output_shape']}")
    print(f"Total Parameters: {info['total_params']:,}")
    print(f"Trainable Parameters: {info['trainable_params']:,}")
    print(f"Non-trainable Parameters: {info['non_trainable_params']:,}")
    print("="*50)


def check_data_quality(df, feature_columns):
    """
    Check the quality of the dataset
    
    Args:
        df (pd.DataFrame): Dataset
        feature_columns (list): List of feature column names
        
    Returns:
        dict: Data quality metrics
    """
    quality_metrics = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df[feature_columns].isnull().sum().to_dict(),
        'feature_ranges': {},
        'data_types': df[feature_columns].dtypes.to_dict()
    }
    
    # Calculate feature ranges
    for col in feature_columns:
        if col in df.columns:
            quality_metrics['feature_ranges'][col] = {
                'min': df[col].min(),
                'max': df[col].max(),
                'mean': df[col].mean(),
                'std': df[col].std()
            }
    
    return quality_metrics


def print_data_quality_report(quality_metrics):
    """
    Print a formatted data quality report
    
    Args:
        quality_metrics (dict): Data quality metrics
    """
    print("\n" + "="*50)
    print("DATA QUALITY REPORT")
    print("="*50)
    print(f"Total Rows: {quality_metrics['total_rows']:,}")
    print(f"Total Columns: {quality_metrics['total_columns']}")
    
    print("\nMissing Values:")
    for col, missing in quality_metrics['missing_values'].items():
        if missing > 0:
            print(f"  {col}: {missing} ({missing/quality_metrics['total_rows']*100:.2f}%)")
    
    print("\nFeature Ranges:")
    for col, ranges in quality_metrics['feature_ranges'].items():
        print(f"  {col}: [{ranges['min']:.3f}, {ranges['max']:.3f}] (mean: {ranges['mean']:.3f})")
    
    print("="*50)
