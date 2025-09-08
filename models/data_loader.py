# Data preprocessing and loading
# Handle data loading, cleaning, and preparation for training

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from .config import FEATURE_CONFIG, MODEL_CONFIG, FILE_PATHS


class SpotifyDataLoader:
    def __init__(self, dataset_path=None):
        """
        Initialize the data loader
        
        Args:
            dataset_path (str): Path to the dataset CSV file
        """
        self.dataset_path = dataset_path or FILE_PATHS['dataset']
        self.df = None
        self.X_features = None
        self.scaler_bounded = None
        self.scaler_unbounded = None
        self.track_ids = None
        
    def load_data(self):
        """Load the dataset"""
        print(f"Loading dataset from {self.dataset_path}")
        self.df = pd.read_csv(self.dataset_path)
        print(f"Dataset shape: {self.df.shape}")
        return self.df
    
    def preprocess_features(self):
        """Preprocess and normalize audio features"""
        print("Preprocessing audio features...")
        
        # Get audio features (excluding popularity_norm for now)
        base_audio_features = [col for col in FEATURE_CONFIG['audio_features'] if col != 'popularity_norm']
        bounded_features = FEATURE_CONFIG['bounded_features']
        unbounded_features = FEATURE_CONFIG['unbounded_features']
        
        print(f"Base audio features: {base_audio_features}")
        
        # Create feature matrix with base audio features
        self.X_features = self.df[base_audio_features].copy()
        
        # Scale bounded features (ensure 0-1 range)
        self.scaler_bounded = MinMaxScaler()
        self.X_features[bounded_features] = self.scaler_bounded.fit_transform(
            self.X_features[bounded_features]
        )
        
        # Standardize unbounded features
        self.scaler_unbounded = StandardScaler()
        self.X_features[unbounded_features] = self.scaler_unbounded.fit_transform(
            self.X_features[unbounded_features]
        )
        
        # Normalize categorical features
        self.X_features['key'] = self.X_features['key'] / 11.0  # 0-11 → 0-1
        self.X_features['mode'] = self.X_features['mode']  # Already 0-1
        
        # Add normalized popularity (if available)
        if 'popularity' in self.df.columns:
            self.X_features['popularity_norm'] = self.df['popularity'] / 100.0
        else:
            # If no popularity column, add a default value
            self.X_features['popularity_norm'] = 0.5  # Default to middle popularity
            print("⚠️ No 'popularity' column found in dataset. Using default value 0.5")
        
        print(f"Normalized features shape: {self.X_features.shape}")
        print("Feature ranges:")
        print(self.X_features.describe())
        
        return self.X_features
    
    def prepare_training_data(self):
        """Prepare data for training, validation, and recommendation testing"""
        # Get track IDs
        self.track_ids = self.df['track_id'].values
        
        # Convert to numpy arrays
        X = self.X_features.values
        
        # First split: 80% for training+validation, 20% for final testing
        X_train_val, X_test = train_test_split(
            X, 
            test_size=0.2, 
            random_state=42
        )
        
        # Second split: Split the 80% into 64% training, 16% validation
        X_train, X_val = train_test_split(
            X_train_val,
            test_size=0.2,  # 20% of 80% = 16% of total
            random_state=42
        )
        
        # Also split the track IDs to keep track of which songs are in which set
        track_ids_train_val, track_ids_test = train_test_split(
            self.track_ids,
            test_size=0.2,
            random_state=42
        )
        
        track_ids_train, track_ids_val = train_test_split(
            track_ids_train_val,
            test_size=0.2,
            random_state=42
        )
        
        print(f"Training set: {X_train.shape} (64% of total)")
        print(f"Validation set: {X_val.shape} (16% of total)")
        print(f"Test set: {X_test.shape} (20% of total)")
        
        return X_train, X_val, X_test, X, self.track_ids, track_ids_train, track_ids_val, track_ids_test
    
    def get_scalers(self):
        """Get the fitted scalers for later use"""
        return {
            'bounded': self.scaler_bounded,
            'unbounded': self.scaler_unbounded
        }
    
    def get_feature_names(self):
        """Get the feature column names"""
        return self.X_features.columns.tolist()
    
    def get_complete_feature_names(self):
        """Get the complete feature names including popularity_norm"""
        return self.X_features.columns.tolist()
    
    def get_dataset_info(self):
        """Get information about the dataset"""
        return {
            'total_tracks': len(self.df),
            'feature_dim': len(self.X_features.columns),
            'audio_features': FEATURE_CONFIG['audio_features'],
            'dataset_shape': self.df.shape
        }


