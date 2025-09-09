"""
Simple test script to validate recommendation system
"""

import pandas as pd
import numpy as np
from test_sample_data_generator import SampleDataGenerator
from recommendation_system import recommendation_engine

def main():
    print("="*50)
    print("SIMPLE RECOMMENDATION SYSTEM TEST")
    print("="*50)
    
    # Step 1: Load recommendation engine
    print("\n1. Loading recommendation engine...")
    recommendation_engine.load_model()
    
    if not recommendation_engine.is_loaded:
        print("❌ Failed to load recommendation engine!")
        print(f"Error: {recommendation_engine.load_error}")
        return False
    
    print("✅ Recommendation engine loaded successfully!")
    
    # Step 2: Generate test data
    print("\n2. Generating test data...")
    generator = SampleDataGenerator()
    
    # Create users with different preferences
    rock_user = generator.generate_user_profile('rock', 10)
    pop_user = generator.generate_user_profile('pop', 10)
    classical_user = generator.generate_user_profile('classical', 10)
    
    print(f"✅ Generated test users:")
    print(f"   - Rock user: {len(rock_user)} tracks")
    print(f"   - Pop user: {len(pop_user)} tracks")
    print(f"   - Classical user: {len(classical_user)} tracks")
    
    # Step 3: Test recommendations
    print("\n3. Testing recommendations...")
    
    users = {
        'Rock': rock_user,
        'Pop': pop_user,
        'Classical': classical_user
    }
    
    all_recommendations = {}
    
    for genre, user_df in users.items():
        print(f"\n   Testing {genre} user...")
        recommendations, error = recommendation_engine.get_recommendations(user_df, 10)
        
        if error:
            print(f"   ❌ Error: {error}")
        elif recommendations:
            print(f"   ✅ Generated {len(recommendations)} recommendations")
            all_recommendations[genre] = [rec['track_id'] for rec in recommendations]
            
            # Show first few recommendations
            print(f"   Top 3 recommendations:")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"     {i}. {rec['track_name']} by {rec['artists']} (similarity: {rec['similarity_score']:.3f})")
        else:
            print(f"   ❌ No recommendations returned")
    
    # Step 4: Check diversity
    print("\n4. Checking recommendation diversity...")
    
    if len(all_recommendations) >= 2:
        genres = list(all_recommendations.keys())
        for i in range(len(genres)):
            for j in range(i+1, len(genres)):
                genre1, genre2 = genres[i], genres[j]
                recs1 = set(all_recommendations[genre1])
                recs2 = set(all_recommendations[genre2])
                
                overlap = len(recs1.intersection(recs2))
                total_unique = len(recs1.union(recs2))
                overlap_ratio = overlap / total_unique if total_unique > 0 else 0
                
                print(f"   {genre1} vs {genre2}: {overlap_ratio:.3f} overlap ({overlap}/{total_unique} tracks)")
                
                if overlap_ratio < 0.5:  # Less than 50% overlap is good
                    print(f"   ✅ Good diversity between {genre1} and {genre2}")
                else:
                    print(f"   ⚠️  High overlap between {genre1} and {genre2}")
    
    # Step 5: Test with different numbers of recommendations
    print("\n5. Testing different recommendation counts...")
    
    test_user = rock_user
    for n_recs in [5, 15, 25]:
        recs, error = recommendation_engine.get_recommendations(test_user, n_recs)
        if error:
            print(f"   ❌ Failed to get {n_recs} recommendations: {error}")
        else:
            print(f"   ✅ Successfully got {len(recs)} recommendations (requested {n_recs})")
    
    print("\n" + "="*50)
    print("TEST COMPLETED!")
    print("="*50)
    
    return True

if __name__ == "__main__":
    main()
