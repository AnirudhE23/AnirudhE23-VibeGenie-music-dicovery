# Main training script
# This is where you'll put your main training logic

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to the path so we can import from models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.data_loader import SpotifyDataLoader
from models.model import create_music_autoencoder
from models.trainer import ModelTrainer
from models.evaluation import ModelEvaluator
from models.utils import save_models_and_data, print_model_summary, check_data_quality, print_data_quality_report
from models.config import FEATURE_CONFIG


def main():
    """
    Main training pipeline for the music recommendation system
    """
    print("="*60)
    print("MUSIC RECOMMENDATION SYSTEM - TRAINING PIPELINE")
    print("="*60)
    
    # Step 1: Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    data_loader = SpotifyDataLoader()
    
    # Load dataset
    df = data_loader.load_data()
    
    # Check data quality
    quality_metrics = check_data_quality(df, FEATURE_CONFIG['audio_features'])
    print_data_quality_report(quality_metrics)
    
    # Preprocess features
    X_features = data_loader.preprocess_features()
    
    # Prepare training data
    X_train, X_test, X, track_ids, track_ids_train, track_ids_test = data_loader.prepare_training_data()
    
    # Step 2: Create and compile model
    print("\n2. Creating model architecture...")
    input_dim = len(X_features.columns)
    autoencoder, encoder = create_music_autoencoder(input_dim)
    
    # Print model summary
    print_model_summary(autoencoder)
    
    # Step 3: Train the model on TRAINING DATA ONLY
    print("\n3. Training the model on training data only...")
    trainer = ModelTrainer(autoencoder)
    history = trainer.train(X_train, X_train)  # Train on training data only
    
    # Plot training history
    trainer.plot_training_history('training_history.png')
    
    # Print training summary
    training_summary = trainer.get_training_summary()
    if training_summary:
        print(f"\nTraining completed in {training_summary['total_epochs']} epochs")
        print(f"Best validation loss: {training_summary['best_val_loss']:.6f}")
        print(f"Best validation MAE: {training_summary['best_val_mae']:.6f}")
    
    # Step 4: Generate embeddings for ALL songs (for recommendations)
    print("\n4. Generating song embeddings for all tracks...")
    song_embeddings = encoder.predict(X, batch_size=256)
    print(f"Generated embeddings shape: {song_embeddings.shape}")
    
    # Step 5: Evaluate the model on training data reconstruction
    print("\n5. Evaluating the model on training data...")
    evaluator = ModelEvaluator()
    
    # Get predictions for evaluation (use training data for reconstruction evaluation)
    y_pred_train = autoencoder.predict(X_train)
    
    # Plot reconstruction analysis
    evaluator.plot_reconstruction_analysis(X_train, y_pred_train, 'reconstruction_analysis.png')
    
    # Generate comprehensive evaluation report
    evaluation_report = evaluator.generate_evaluation_report(
        autoencoder, X_train, song_embeddings, track_ids, df
    )
    
    # Step 6: Save models and data
    print("\n6. Saving models and data...")
    scalers = data_loader.get_scalers()
    save_models_and_data(autoencoder, encoder, song_embeddings, track_ids, df, scalers)
    
    # Step 7: Test recommendation system on UNSEEN data
    print("\n7. Testing recommendation system on unseen data...")
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
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nNext steps:")
    print("1. Use the saved models for recommendations")
    print("2. Integrate with your Streamlit application")
    print("3. Test with real user data")
    print("\nFiles created:")
    print("- music_autoencoder.h5 (trained autoencoder)")
    print("- music_encoder.h5 (trained encoder)")
    print("- scaler_bounded.pkl (fitted scaler for bounded features)")
    print("- scaler_unbounded.pkl (fitted scaler for unbounded features)")
    print("- song_embeddings.npy (pre-computed song embeddings)")
    print("- track_ids.npy (track ID mapping)")
    print("- song_metadata.csv (track metadata)")
    print("- training_history.png (training plots)")
    print("- reconstruction_analysis.png (evaluation plots)")


if __name__ == "__main__":
    main()
