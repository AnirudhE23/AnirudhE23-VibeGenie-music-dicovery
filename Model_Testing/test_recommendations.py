#!/usr/bin/env python3
"""
Test script for the recommendation system
Run this after training to verify everything works
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.utils import load_models_and_data, test_recommendation_system
from models.config import FEATURE_CONFIG

def test_saved_recommendations():
    """Test the recommendation system with saved models"""
    print("Testing saved recommendation system...")
    
    # Load saved models and data
    saved_data = load_models_and_data()
    
    if saved_data is None:
        print("❌ No saved models found. Please train the model first.")
        return
    
    print("✅ Models loaded successfully!")
    
    # Create recommender
    from models.model import MusicRecommender
    recommender = MusicRecommender(
        song_embeddings=saved_data['song_embeddings'],
        track_ids=saved_data['track_ids'],
        df=saved_data['metadata_df'],
        encoder=saved_data['encoder'],
        scalers=saved_data['scalers']
    )
    
    print("✅ Recommender created successfully!")
    
    # Test with a few sample tracks
    print("\nTesting recommendation system...")
    
    # Get a few sample tracks from the metadata
    sample_tracks = saved_data['metadata_df'].sample(3)
    
    for i, (_, track) in enumerate(sample_tracks.iterrows(), 1):
        print(f"\n{i}. Testing with: {track['track_name']} by {track['artists']}")
        
        # Create a simple user profile (you'd normally get this from Spotify API)
        # For testing, we'll use dummy features with proper column names
        import numpy as np
        import pandas as pd
        
        # Create dummy features with proper column names
        # The model expects 12 features: 11 audio features + popularity_norm
        feature_names = FEATURE_CONFIG['audio_features'] + ['popularity_norm']
        dummy_features = np.random.random(len(feature_names))
        user_tracks = pd.DataFrame([dummy_features], columns=feature_names)
        
        # Create user profile
        user_profile = recommender.create_user_profile(user_tracks)
        
        # Get recommendations
        recommendations = recommender.recommend_songs(
            user_profile, 
            n_recommendations=5,
            exclude_track_ids=set()
        )
        
        print(f"   Generated {len(recommendations)} recommendations:")
        for j, rec in enumerate(recommendations[:3], 1):
            print(f"     {j}. {rec['track_name']} by {rec['artists']} (similarity: {rec['similarity_score']:.3f})")
    
    print("\n✅ Recommendation testing completed!")

if __name__ == "__main__":
    test_saved_recommendations()
