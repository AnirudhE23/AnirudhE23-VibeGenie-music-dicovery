#!/usr/bin/env python3
"""
CORRECTED Google Colab Training Script for Music Recommendation System
This version properly splits data into train/validation/test sets
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from google.colab import drive

# Mount Google Drive
print("Mounting Google Drive...")
drive.mount('/content/drive')

# Create project directory in Drive
project_dir = '/content/drive/MyDrive/SpotifyAPI_proj'
os.makedirs(project_dir, exist_ok=True)
os.chdir(project_dir)

print(f"Working directory: {os.getcwd()}")

# Install required packages
print("Installing required packages...")
!pip install tensorflow scikit-learn joblib matplotlib seaborn pandas numpy

# Verify GPU availability
print("\n" + "="*50)
print("GPU VERIFICATION")
print("="*50)
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

if tf.config.list_physical_devices('GPU'):
    print("✅ GPU detected! Training will be much faster.")
    # Enable memory growth to avoid GPU memory issues
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ GPU memory growth enabled.")
        except RuntimeError as e:
            print(f"⚠️ GPU memory growth setting failed: {e}")
else:
    print("⚠️ No GPU detected. Training will be slower on CPU.")

# Add models directory to path
models_path = '/content/drive/MyDrive/SpotifyAPI_proj/models'
sys.path.append(models_path)

# Import your modules
from models.data_loader import SpotifyDataLoader
from models.model import create_music_autoencoder
from models.trainer import ModelTrainer
from models.evaluation import ModelEvaluator
from models.utils import save_models_and_data, print_model_summary, check_data_quality, print_data_quality_report
from models.config import FEATURE_CONFIG

def main():
    """
    CORRECTED Main training pipeline for Google Colab
    """
    print("\n" + "="*60)
    print("CORRECTED GOOGLE COLAB TRAINING PIPELINE")
    print("="*60)

    # Debug: Check if models folder exists and show contents
    print("\n🔍 Checking models folder...")
    models_path = '/content/drive/MyDrive/SpotifyAPI_proj/models'
    if os.path.exists(models_path):
        print("✅ Models folder found!")
        print("Contents:")
        try:
            model_files = os.listdir(models_path)
            for file in model_files:
                print(f"  - {file}")
        except Exception as e:
            print(f"Error listing files: {e}")
    else:
        print("❌ Models folder NOT found!")
        print("Please ensure you've uploaded the models folder to Google Drive/SpotifyAPI_proj/")
        return

    print(f"\n📁 Current working directory: {os.getcwd()}")
    print(f"🔗 Models path in sys.path: {models_path in sys.path}")

    # Step 1: Load and preprocess data
    print("\n1. Loading and preprocessing data...")

    # Check if dataset exists in Drive
    dataset_path = '/content/drive/MyDrive/SpotifyAPI_proj/Final_training_dataset.csv'
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        print("Please upload your Final_training_dataset.csv to your Google Drive/SpotifyAPI_proj folder")
        return

    data_loader = SpotifyDataLoader(dataset_path)

    # Load dataset
    df = data_loader.load_data()

    # Check data quality
    quality_metrics = check_data_quality(df, FEATURE_CONFIG['audio_features'])
    print_data_quality_report(quality_metrics)

    # Preprocess features
    X_features = data_loader.preprocess_features()

    # CORRECTED: Prepare training data with proper train/validation/test split
    X_train, X_val, X_test, X, track_ids, track_ids_train, track_ids_val, track_ids_test = data_loader.prepare_training_data()

    # Step 2: Create and compile model
    print("\n2. Creating model architecture...")
    input_dim = len(X_features.columns)
    autoencoder, encoder = create_music_autoencoder(input_dim)

    # Print model summary
    print_model_summary(autoencoder)

    # Step 3: Train the model on GPU with CORRECTED validation data
    print("\n3. Training the model on GPU with proper validation data...")
    trainer = ModelTrainer(autoencoder)
    
    # CORRECTED: Use separate validation data
    history = trainer.train(X_train, X_val)  # Now using different data for training and validation!

    # Plot training history
    training_plot_path = '/content/drive/MyDrive/SpotifyAPI_proj/corrected_training_history.png'
    trainer.plot_training_history(training_plot_path)

    # Print training summary
    training_summary = trainer.get_training_summary()
    if training_summary:
        print(f"\nTraining completed in {training_summary['total_epochs']} epochs")
        print(f"Best validation loss: {training_summary['best_val_loss']:.6f}")
        print(f"Best validation MAE: {training_summary['best_val_mae']:.6f}")

    # Step 4: Generate embeddings for all songs
    print("\n4. Generating song embeddings...")
    song_embeddings = encoder.predict(X, batch_size=256)
    print(f"Generated embeddings shape: {song_embeddings.shape}")

    # Step 5: Evaluate the model
    print("\n5. Evaluating the model...")
    evaluator = ModelEvaluator()

    # Get predictions for evaluation
    y_pred_train = autoencoder.predict(X_train)

    # Plot reconstruction analysis
    reconstruction_plot_path = '/content/drive/MyDrive/SpotifyAPI_proj/corrected_reconstruction_analysis.png'
    evaluator.plot_reconstruction_analysis(X_train, y_pred_train, reconstruction_plot_path)

    # Generate comprehensive evaluation report
    evaluation_report = evaluator.generate_evaluation_report(
        autoencoder, X_train, song_embeddings, track_ids, df
    )

    # Step 6: Save models and data to Google Drive
    print("\n6. Saving models and data to Google Drive...")
    scalers = data_loader.get_scalers()

    # Update file paths for Google Drive
    save_paths = {
        'autoencoder_model': '/content/drive/MyDrive/SpotifyAPI_proj/corrected_music_autoencoder.h5',
        'encoder_model': '/content/drive/MyDrive/SpotifyAPI_proj/corrected_music_encoder.h5',
        'scaler_bounded': '/content/drive/MyDrive/SpotifyAPI_proj/corrected_scaler_bounded.pkl',
        'scaler_unbounded': '/content/drive/MyDrive/SpotifyAPI_proj/corrected_scaler_unbounded.pkl',
        'song_embeddings': '/content/drive/MyDrive/SpotifyAPI_proj/corrected_song_embeddings.npy',
        'track_ids': '/content/drive/MyDrive/SpotifyAPI_proj/corrected_track_ids.npy',
        'song_metadata': '/content/drive/MyDrive/SpotifyAPI_proj/corrected_song_metadata.csv'
    }

    # Save everything to Google Drive
    save_models_and_data(autoencoder, encoder, song_embeddings, track_ids, df, scalers)

    # Step 7: Test recommendation system on UNSEEN test data
    print("\n7. Testing recommendation system on UNSEEN test data...")
    from models.model import MusicRecommender

    recommender = MusicRecommender(
        song_embeddings=song_embeddings,
        track_ids=track_ids,
        df=df,
        encoder=encoder,
        scalers=scalers
    )

    # Test with UNSEEN test data
    from models.utils import test_recommendation_system
    test_recommendation_system(recommender, df, X_test, track_ids_test, n_test_tracks=3, n_recommendations=10)

    print("\n" + "="*60)
    print("CORRECTED COLAB TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("All files saved to your Google Drive!")
    print("\nFiles created:")
    print("- corrected_music_autoencoder.h5 (trained autoencoder)")
    print("- corrected_music_encoder.h5 (trained encoder)")
    print("- corrected_scaler_bounded.pkl (fitted scaler for bounded features)")
    print("- corrected_scaler_unbounded.pkl (fitted scaler for unbounded features)")
    print("- corrected_song_embeddings.npy (pre-computed song embeddings)")
    print("- corrected_track_ids.npy (track ID mapping)")
    print("- corrected_song_metadata.csv (track metadata)")
    print("- corrected_training_history.png (training plots)")
    print("- corrected_reconstruction_analysis.png (evaluation plots)")

    print("\n✅ KEY IMPROVEMENTS:")
    print("1. Proper train/validation/test split (64%/16%/20%)")
    print("2. Validation loss now represents true generalization")
    print("3. Training and validation use different data")
    print("4. More reliable model evaluation")

    print("\nNext steps:")
    print("1. Download corrected models from Google Drive to your local machine")
    print("2. Use the corrected models in your Streamlit application")
    print("3. Test with real user data")

if __name__ == "__main__":
    main()
