import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
from models.model import MusicRecommender
from models.utils import load_models_and_data

class Recommendation_Engine:
    """Main recommendation engine that integrates with the trained model"""

    def __init__(self):
        self.recommender = None
        self.is_loaded = False
        self.load_error = None

    def load_model(self):
        """Load the trained model and recommender system"""
        try:
            print("Loading recommendation system...")
            
            # Load models and data
            models_data = load_models_and_data()
            
            # Create recommender instance
            self.recommender = MusicRecommender(
                song_embeddings=models_data['song_embeddings'],
                track_ids=models_data['track_ids'],
                df=models_data['df'],
                encoder=models_data['encoder'],
                scalers=models_data['scalers']
            )
            
            self.is_loaded = True
            self.load_error = None
            print("Recommendation system loaded successfully!")
            
        except Exception as e:
            self.load_error = str(e)
            print(f"Error loading recommendation system: {e}")
            self.is_loaded = False

    def get_recommendations(self, user_tracks_df, n_recommendations=20, quick_mode=False, user_preferences=None):
        """
        Get music recommendations for a user based on their tracks and preferences
        
        Args:
            user_tracks_df (pd.DataFrame): User's track features
            n_recommendations (int): Number of recommendations to return
            quick_mode (bool): Use faster recommendation method with less diversity
            user_preferences (dict): User's mood and diversity preferences
            
        Returns:
            list: List of recommendation dictionaries
        """
        if not self.is_loaded:
            return None, "Model not loaded. Please try again."
        
        try:
            # Create a copy to avoid modifying the original
            df_copy = user_tracks_df.copy()
            
            # Map column names to match the expected format
            column_mapping = {
                'Track ID': 'track_id',
                'Track Name': 'track_name', 
                'Artist(s)': 'artists',
                'Popularity': 'popularity'
            }
            
            # Rename columns if they exist
            for old_name, new_name in column_mapping.items():
                if old_name in df_copy.columns:
                    df_copy = df_copy.rename(columns={old_name: new_name})
            
            # Check if user has tracks with audio features
            required_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 
                               'liveness', 'speechiness', 'valence', 'loudness', 'tempo', 'key', 'mode', 'popularity']
            
            available_features = [col for col in required_features if col in df_copy.columns]
            tracks_with_features = df_copy.dropna(subset=available_features)
            
            print(f"DEBUG: User tracks with features: {len(tracks_with_features)}")
            print(f"DEBUG: Available features: {available_features}")
            
            if len(tracks_with_features) == 0:
                return None, "No tracks with audio features found. Please collect more data."
            
            if len(tracks_with_features) < 3:
                return None, "Need at least 3 tracks with audio features for good recommendations."
            
            # NEW: Use mood-based recommendations if preferences are provided
            if user_preferences and user_preferences.get('selected_moods'):
                print("DEBUG: Using mood-based recommendation system")
                recommendations = self._get_mood_based_recommendations(
                    tracks_with_features, 
                    n_recommendations, 
                    user_preferences,
                    quick_mode
                )
            else:
                print("DEBUG: Using original recommendation system")
                # Original logic for backward compatibility
                exclude_tracks = set(tracks_with_features['track_id'].tolist())
                
                if quick_mode:
                    user_profile = self.recommender.create_user_profile(tracks_with_features)
                    recommendations = self.recommender.recommend_songs(
                        user_profile, 
                        n_recommendations=n_recommendations,
                        exclude_track_ids=exclude_tracks,
                        diversity_factor=0.1
                    )
                else:
                    try:
                        recommendations = self.recommender.find_similar_songs_to_user_tracks(
                            tracks_with_features,
                            n_recommendations=n_recommendations,
                            exclude_track_ids=exclude_tracks
                        )
                        
                        if not recommendations or len(recommendations) == 0:
                            user_profile = self.recommender.create_user_profile(tracks_with_features)
                            recommendations = self.recommender.recommend_songs(
                                user_profile, 
                                n_recommendations=n_recommendations,
                                exclude_track_ids=exclude_tracks,
                                diversity_factor=0.3
                            )
                    except Exception as e:
                        print(f"DEBUG: Error in find_similar_songs_to_user_tracks: {e}")
                        user_profile = self.recommender.create_user_profile(tracks_with_features)
                        recommendations = self.recommender.recommend_songs(
                            user_profile, 
                            n_recommendations=n_recommendations,
                            exclude_track_ids=exclude_tracks,
                            diversity_factor=0.3
                        )
            
            return recommendations, None
            
        except Exception as e:
            return None, f"Error generating recommendations: {str(e)}"
    
    def get_model_status(self):
        """Get the current status of the recommendation system"""
        return {
            'is_loaded': self.is_loaded,
            'load_error': self.load_error,
            'model_info': self._get_model_info() if self.is_loaded else None
        }
    
    def _get_model_info(self):
        """Get information about the loaded model"""
        if not self.is_loaded:
            return None
            
        return {
            'total_songs': len(self.recommender.track_ids),
            'embedding_dim': self.recommender.song_embeddings.shape[1],
            'model_type': 'Music Autoencoder with Embeddings'
        }
    
    def _get_mood_based_recommendations(self, tracks_with_features, n_recommendations, user_preferences, quick_mode):
        """
        Generate recommendations using mood-based profiles
        
        Args:
            tracks_with_features (pd.DataFrame): User's tracks with audio features
            n_recommendations (int): Number of recommendations to return
            user_preferences (dict): User's mood and diversity preferences
            quick_mode (bool): Use faster recommendation method
            
        Returns:
            list: List of recommendation dictionaries
        """
        selected_moods = user_preferences.get('selected_moods', ['High Energy', 'Chill', 'Medium Energy'])
        diversity_level = user_preferences.get('diversity_level', 0.3)
        energy_preference = user_preferences.get('energy_preference', 'Match My Taste')
        taste_analysis = user_preferences.get('taste_analysis')
        
        print(f"DEBUG: Selected moods: {selected_moods}")
        print(f"DEBUG: Diversity level: {diversity_level}")
        print(f"DEBUG: Energy preference: {energy_preference}")
        
        # Get recommendations for each selected mood
        mood_recommendations = {}
        exclude_tracks = set(tracks_with_features['track_id'].tolist())
        
        for mood in selected_moods:
            print(f"DEBUG: Generating recommendations for {mood}")
            
            # Get tracks that match this mood
            mood_tracks = self._filter_tracks_by_mood(tracks_with_features, mood)
            
            if len(mood_tracks) > 0:
                # Generate recommendations for this mood
                if quick_mode:
                    user_profile = self.recommender.create_user_profile(mood_tracks)
                    mood_recs = self.recommender.recommend_songs(
                        user_profile,
                        n_recommendations=n_recommendations // len(selected_moods) + 5,  # Extra for diversity
                        exclude_track_ids=exclude_tracks,
                        diversity_factor=diversity_level
                    )
                else:
                    mood_recs = self.recommender.find_similar_songs_to_user_tracks(
                        mood_tracks,
                        n_recommendations=n_recommendations // len(selected_moods) + 5,
                        exclude_track_ids=exclude_tracks
                    )
                
                mood_recommendations[mood] = mood_recs if mood_recs else []
            else:
                print(f"DEBUG: No tracks found for mood {mood}")
                mood_recommendations[mood] = []
        
        # Combine and diversify recommendations
        final_recommendations = self._combine_mood_recommendations(
            mood_recommendations, 
            n_recommendations, 
            diversity_level,
            energy_preference,
            taste_analysis
        )
        
        return final_recommendations
    
    def _filter_tracks_by_mood(self, tracks_with_features, target_mood):
        """
        Filter tracks to only include those that match the target mood
        """
        mood_tracks = []
        
        for idx, row in tracks_with_features.iterrows():
            track_mood = self.recommender._determine_track_mood(row)
            if track_mood == target_mood:
                mood_tracks.append(row)
        
        return pd.DataFrame(mood_tracks) if mood_tracks else pd.DataFrame()
    
    def _combine_mood_recommendations(self, mood_recommendations, n_recommendations, diversity_level, energy_preference, taste_analysis):
        """
        Combine recommendations from different moods with proper weighting and diversity
        """
        all_recommendations = []
        
        # Collect all recommendations
        for mood, recs in mood_recommendations.items():
            for rec in recs:
                rec['source_mood'] = mood
                all_recommendations.append(rec)
        
        if not all_recommendations:
            return []
        
        # Remove duplicates while preserving mood information
        seen_tracks = set()
        unique_recommendations = []
        
        for rec in all_recommendations:
            track_id = rec.get('track_id')
            if track_id not in seen_tracks:
                seen_tracks.add(track_id)
                unique_recommendations.append(rec)
        
        # Sort by similarity score
        unique_recommendations.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        
        # Apply energy preference filtering if specified
        if energy_preference != 'Match My Taste':
            unique_recommendations = self._apply_energy_preference(unique_recommendations, energy_preference)
        
        # Apply diversity selection
        final_recommendations = self._select_diverse_mood_recommendations(
            unique_recommendations, 
            n_recommendations, 
            diversity_level,
            taste_analysis
        )
        
        return final_recommendations
    
    def _apply_energy_preference(self, recommendations, energy_preference):
        """
        Filter recommendations based on energy preference
        """
        # This would require access to the track features in the dataset
        # For now, we'll return the recommendations as-is
        # TODO: Implement energy-based filtering
        return recommendations
    
    def _select_diverse_mood_recommendations(self, recommendations, n_recommendations, diversity_level, taste_analysis):
        """
        Select diverse recommendations ensuring good representation from different moods
        """
        if len(recommendations) <= n_recommendations:
            return recommendations
        
        # Group recommendations by mood
        mood_groups = {}
        for rec in recommendations:
            mood = rec.get('source_mood', 'Unknown')
            if mood not in mood_groups:
                mood_groups[mood] = []
            mood_groups[mood].append(rec)
        
        # Calculate how many recommendations to take from each mood
        selected_recommendations = []
        moods = list(mood_groups.keys())
        
        if taste_analysis and taste_analysis.get('mood_statistics'):
            # Weight by user's actual mood distribution
            mood_weights = {}
            for mood, stats in taste_analysis['mood_statistics'].items():
                if mood in moods:
                    mood_weights[mood] = stats['percentage'] / 100.0
            
            # Normalize weights
            total_weight = sum(mood_weights.values())
            if total_weight > 0:
                for mood in mood_weights:
                    mood_weights[mood] = mood_weights[mood] / total_weight
        else:
            # Equal weighting
            mood_weights = {mood: 1.0 / len(moods) for mood in moods}
        
        # Select recommendations from each mood
        for mood in moods:
            mood_recs = mood_groups[mood]
            n_from_mood = max(1, int(n_recommendations * mood_weights.get(mood, 1.0 / len(moods))))
            n_from_mood = min(n_from_mood, len(mood_recs))
            
            # Take top recommendations from this mood
            selected_recommendations.extend(mood_recs[:n_from_mood])
        
        # If we need more recommendations, fill from remaining
        if len(selected_recommendations) < n_recommendations:
            remaining = [rec for rec in recommendations if rec not in selected_recommendations]
            selected_recommendations.extend(remaining[:n_recommendations - len(selected_recommendations)])
        
        # Sort final recommendations by similarity score
        selected_recommendations.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        
        return selected_recommendations[:n_recommendations]

# Global instance
recommendation_engine = Recommendation_Engine()