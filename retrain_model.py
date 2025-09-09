#!/usr/bin/env python3
"""
Model Retraining Script

This script retrains the recommendation model with the updated training dataset
that includes user tracks. Run this after collecting user data to ensure
the model uses the user's music taste for recommendations.

Usage:
    python retrain_model.py
"""

import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
from models.train import train_model
from models.utils import save_models_and_data, load_models_and_data
import dataset_expansion

def check_training_dataset():
    """Check if the training dataset exists and has sufficient data"""
    try:
        df = pd.read_csv("Final_training_dataset.csv")
        print(f"Training dataset found with {len(df)} tracks")
        
        # Check for required columns
        required_cols = ['track_id', 'track_name', 'artists', 'popularity', 
                        'acousticness', 'danceability', 'energy', 'instrumentalness',
                        'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'valence']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return False
        
        # Check for tracks with complete audio features
        feature_cols = ['acousticness', 'danceability', 'energy', 'instrumentalness',
                       'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'valence']
        
        complete_tracks = df.dropna(subset=feature_cols)
        print(f"Tracks with complete audio features: {len(complete_tracks)}")
        
        if len(complete_tracks) < 1000:
            print("❌ Insufficient tracks with complete audio features (need at least 1000)")
            return False
        
        return True
        
    except FileNotFoundError:
        print("❌ Training dataset not found: Final_training_dataset.csv")
        return False
    except Exception as e:
        print(f"❌ Error checking training dataset: {e}")
        return False

def backup_existing_model():
    """Backup existing model files before retraining"""
    model_files = [
        'best_music_autoencoder.h5',
        'music_encoder.h5', 
        'song_embeddings.npy',
        'track_ids.npy',
        'scaler_bounded.pkl',
        'scaler_unbounded.pkl',
        'song_metadata.csv'
    ]
    
    backup_dir = 'model_backup'
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    
    backed_up = []
    for file in model_files:
        if os.path.exists(file):
            import shutil
            backup_path = os.path.join(backup_dir, file)
            shutil.copy2(file, backup_path)
            backed_up.append(file)
    
    if backed_up:
        print(f"✅ Backed up existing model files: {backed_up}")
    else:
        print("ℹ️ No existing model files to backup")
    
    return len(backed_up) > 0

def retrain_model():
    """Retrain the model with the updated training dataset"""
    print("🔄 Starting model retraining...")
    
    try:
        # Train the model
        training_results = train_model()
        
        if training_results:
            print("✅ Model retrained successfully!")
            
            # Verify the new model works
            try:
                models_data = load_models_and_data()
                print(f"✅ Model verification successful:")
                print(f"   - Song embeddings shape: {models_data['song_embeddings'].shape}")
                print(f"   - Track IDs count: {len(models_data['track_ids'])}")
                print(f"   - Dataset size: {len(models_data['df'])}")
                
                return True
            except Exception as e:
                print(f"❌ Model verification failed: {e}")
                return False
        else:
            print("❌ Model training failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error during model training: {e}")
        return False

def main():
    """Main retraining function"""
    print("🎵 Music Recommendation Model Retraining")
    print("=" * 50)
    
    # Step 1: Check training dataset
    print("\n1. Checking training dataset...")
    if not check_training_dataset():
        print("❌ Training dataset check failed. Please collect user data first.")
        return False
    
    # Step 2: Backup existing model
    print("\n2. Backing up existing model...")
    backup_existing_model()
    
    # Step 3: Retrain model
    print("\n3. Retraining model...")
    success = retrain_model()
    
    if success:
        print("\n🎉 Model retraining completed successfully!")
        print("💡 Your recommendation system now uses the updated dataset with user tracks.")
        print("🔄 You can now generate recommendations that better match your music taste.")
    else:
        print("\n❌ Model retraining failed!")
        print("💡 Check the error messages above and try again.")
        print("🔄 You can restore the backup model if needed.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)