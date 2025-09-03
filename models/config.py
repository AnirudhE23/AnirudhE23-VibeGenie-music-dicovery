# Training hyperparameters and configuration
# Define your model parameters, training settings, and constants

# Model Configuration
MODEL_CONFIG = {
    'embedding_dim': 64,
    'learning_rate': 0.001,
    'batch_size': 128,
    'epochs': 100,
    'test_size': 0.2,
    'random_state': 42
}

# Feature Configuration
FEATURE_CONFIG = {
    'audio_features': [
        'acousticness', 'danceability', 'energy', 'instrumentalness', 
        'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'valence'
    ],
    'bounded_features': [
        'acousticness', 'danceability', 'energy', 'instrumentalness', 
        'liveness', 'speechiness', 'valence'
    ],
    'unbounded_features': ['loudness', 'tempo'],
    'categorical_features': ['key', 'mode']
}

# Training Callbacks Configuration
CALLBACKS_CONFIG = {
    'early_stopping': {
        'patience': 15,
        'restore_best_weights': True,
        'monitor': 'val_loss'
    },
    'reduce_lr': {
        'factor': 0.5,
        'patience': 8,
        'min_lr': 1e-7,
        'monitor': 'val_loss'
    },
    'model_checkpoint': {
        'filepath': 'best_music_autoencoder.h5',
        'save_best_only': True,
        'monitor': 'val_loss'
    }
}

# File Paths
FILE_PATHS = {
    'dataset': 'Final_training_dataset.csv',
    'autoencoder_model': 'best_music_autoencoder.h5',
    'encoder_model': 'music_encoder.h5',
    'scaler_bounded': 'scaler_bounded.pkl',
    'scaler_unbounded': 'scaler_unbounded.pkl',
    'song_embeddings': 'song_embeddings.npy',
    'track_ids': 'track_ids.npy',
    'song_metadata': 'song_metadata.csv'
}
