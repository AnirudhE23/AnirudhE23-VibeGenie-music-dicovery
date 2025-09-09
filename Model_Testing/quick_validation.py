"""
Quick Validation Script for Recommendation System
Run this anytime to quickly validate that your system is working correctly
"""

import pandas as pd
import numpy as np
from test_sample_data_generator import SampleDataGenerator
from recommendation_system import recommendation_engine

def quick_validation():
    """Quick validation of the recommendation system"""
    print("🚀 QUICK RECOMMENDATION SYSTEM VALIDATION")
    print("=" * 50)
    
    # Load system
    print("Loading recommendation engine...")
    recommendation_engine.load_model()
    
    if not recommendation_engine.is_loaded:
        print("❌ System failed to load!")
        return False
    
    print("✅ System loaded successfully!")
    
    # Generate test users
    generator = SampleDataGenerator()
    
    # Test with 3 different genres
    test_users = {
        'Rock': generator.generate_user_profile('rock', 8),
        'Pop': generator.generate_user_profile('pop', 8),
        'Classical': generator.generate_user_profile('classical', 8)
    }
    
    print(f"✅ Generated test users for {len(test_users)} genres")
    
    # Get recommendations
    recommendations = {}
    
    for genre, user_df in test_users.items():
        recs, error = recommendation_engine.get_recommendations(user_df, 10)
        if error:
            print(f"❌ Error for {genre}: {error}")
            return False
        
        recommendations[genre] = [rec['track_id'] for rec in recs]
        print(f"✅ {genre}: {len(recs)} recommendations (avg similarity: {np.mean([r['similarity_score'] for r in recs]):.3f})")
    
    # Check diversity
    print("\n🔍 Checking diversity...")
    
    genres = list(recommendations.keys())
    diversity_ok = True
    
    for i in range(len(genres)):
        for j in range(i+1, len(genres)):
            genre1, genre2 = genres[i], genres[j]
            recs1 = set(recommendations[genre1])
            recs2 = set(recommendations[genre2])
            
            overlap = len(recs1.intersection(recs2))
            total_unique = len(recs1.union(recs2))
            overlap_ratio = overlap / total_unique if total_unique > 0 else 0
            
            if overlap_ratio < 0.3:
                print(f"✅ {genre1} vs {genre2}: {overlap_ratio:.3f} overlap (Good diversity)")
            else:
                print(f"⚠️  {genre1} vs {genre2}: {overlap_ratio:.3f} overlap (High overlap)")
                diversity_ok = False
    
    # Final assessment
    print("\n" + "=" * 50)
    if diversity_ok:
        print("🎉 VALIDATION PASSED!")
        print("✅ System is working correctly")
        print("✅ Recommendations are diverse")
        print("✅ Ready for users!")
    else:
        print("⚠️  VALIDATION WARNING!")
        print("⚠️  Some recommendations show high overlap")
        print("⚠️  Consider investigating further")
    
    return diversity_ok

if __name__ == "__main__":
    quick_validation()
