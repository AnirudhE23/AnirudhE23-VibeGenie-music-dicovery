"""
Comprehensive test with more user profiles and detailed analysis
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from test_sample_data_generator import SampleDataGenerator
from recommendation_system import recommendation_engine

def run_comprehensive_test():
    print("="*60)
    print("COMPREHENSIVE RECOMMENDATION SYSTEM TEST")
    print("="*60)
    
    # Load recommendation engine
    print("\n1. Loading recommendation engine...")
    recommendation_engine.load_model()
    
    if not recommendation_engine.is_loaded:
        print("❌ Failed to load recommendation engine!")
        return False
    
    print("✅ Recommendation engine loaded successfully!")
    
    # Generate diverse test users
    print("\n2. Generating diverse test users...")
    generator = SampleDataGenerator()
    
    # Create users for all genres
    test_users = {}
    genres = ['rock', 'pop', 'electronic', 'classical', 'jazz', 'hip_hop', 'country', 'blues']
    
    for genre in genres:
        test_users[genre] = generator.generate_user_profile(genre, 12)
        print(f"   ✅ Generated {genre} user with 12 tracks")
    
    # Create mixed genre users
    mixed_users = {
        'rock_pop': generator.generate_mixed_profile(['rock', 'pop'], 6),
        'electronic_hip_hop': generator.generate_mixed_profile(['electronic', 'hip_hop'], 6),
        'jazz_blues': generator.generate_mixed_profile(['jazz', 'blues'], 6),
        'classical_jazz': generator.generate_mixed_profile(['classical', 'jazz'], 6)
    }
    
    for name, user_df in mixed_users.items():
        test_users[name] = user_df
        print(f"   ✅ Generated {name} user with {len(user_df)} tracks")
    
    # Test recommendations for all users
    print("\n3. Testing recommendations for all users...")
    
    results = {}
    all_recommendations = {}
    
    for user_name, user_df in test_users.items():
        print(f"\n   Testing {user_name} user...")
        recommendations, error = recommendation_engine.get_recommendations(user_df, 15)
        
        if error:
            print(f"   ❌ Error: {error}")
            results[user_name] = {'error': error}
        else:
            print(f"   ✅ Generated {len(recommendations)} recommendations")
            all_recommendations[user_name] = [rec['track_id'] for rec in recommendations]
            
            # Calculate metrics
            similarity_scores = [rec['similarity_score'] for rec in recommendations]
            popularity_scores = [rec['popularity'] for rec in recommendations]
            
            results[user_name] = {
                'num_recommendations': len(recommendations),
                'avg_similarity': np.mean(similarity_scores),
                'min_similarity': np.min(similarity_scores),
                'max_similarity': np.max(similarity_scores),
                'avg_popularity': np.mean(popularity_scores),
                'similarity_std': np.std(similarity_scores),
                'top_3_recommendations': [
                    {
                        'track_name': rec['track_name'],
                        'artists': rec['artists'],
                        'similarity': rec['similarity_score'],
                        'popularity': rec['popularity']
                    }
                    for rec in recommendations[:3]
                ]
            }
            
            print(f"   Similarity range: {np.min(similarity_scores):.3f} - {np.max(similarity_scores):.3f}")
            print(f"   Average popularity: {np.mean(popularity_scores):.1f}")
    
    # Analyze diversity
    print("\n4. Analyzing recommendation diversity...")
    
    diversity_analysis = {}
    
    # Compare single-genre users
    single_genre_users = [name for name in test_users.keys() if name in genres]
    
    for i, genre1 in enumerate(single_genre_users):
        for genre2 in single_genre_users[i+1:]:
            if genre1 in all_recommendations and genre2 in all_recommendations:
                recs1 = set(all_recommendations[genre1])
                recs2 = set(all_recommendations[genre2])
                
                overlap = len(recs1.intersection(recs2))
                total_unique = len(recs1.union(recs2))
                overlap_ratio = overlap / total_unique if total_unique > 0 else 0
                
                diversity_analysis[f"{genre1}_vs_{genre2}"] = {
                    'overlap_ratio': overlap_ratio,
                    'overlap_count': overlap,
                    'total_unique': total_unique
                }
                
                status = "✅ Good" if overlap_ratio < 0.3 else "⚠️ High" if overlap_ratio < 0.6 else "❌ Very High"
                print(f"   {genre1} vs {genre2}: {overlap_ratio:.3f} overlap {status}")
    
    # Test edge cases
    print("\n5. Testing edge cases...")
    
    edge_case_results = {}
    
    # Test with minimal tracks
    minimal_user = generator.generate_user_profile('pop', 3)
    recs, error = recommendation_engine.get_recommendations(minimal_user, 10)
    edge_case_results['minimal_tracks'] = {
        'success': not error and recs is not None,
        'error': error if error else None,
        'num_recommendations': len(recs) if recs else 0
    }
    print(f"   Minimal tracks (3): {'✅ Success' if not error else '❌ Failed'}")
    
    # Test with many recommendations
    normal_user = generator.generate_user_profile('rock', 10)
    recs, error = recommendation_engine.get_recommendations(normal_user, 50)
    edge_case_results['many_recommendations'] = {
        'success': not error and recs is not None,
        'error': error if error else None,
        'num_recommendations': len(recs) if recs else 0
    }
    print(f"   Many recommendations (50): {'✅ Success' if not error else '❌ Failed'}")
    
    # Test with mixed genre user
    mixed_user = generator.generate_mixed_profile(['rock', 'pop', 'electronic'], 4)
    recs, error = recommendation_engine.get_recommendations(mixed_user, 15)
    edge_case_results['mixed_genre'] = {
        'success': not error and recs is not None,
        'error': error if error else None,
        'num_recommendations': len(recs) if recs else 0
    }
    print(f"   Mixed genre user: {'✅ Success' if not error else '❌ Failed'}")
    
    # Generate comprehensive report
    print("\n6. Generating comprehensive report...")
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'test_summary': {
            'total_users_tested': len(test_users),
            'successful_tests': len([r for r in results.values() if 'error' not in r]),
            'failed_tests': len([r for r in results.values() if 'error' in r]),
            'genres_tested': genres,
            'mixed_users_tested': list(mixed_users.keys())
        },
        'user_results': results,
        'diversity_analysis': diversity_analysis,
        'edge_case_results': edge_case_results,
        'recommendations': {
            'system_ready': len([r for r in results.values() if 'error' not in r]) >= len(test_users) * 0.8,
            'diversity_good': len([d for d in diversity_analysis.values() if d['overlap_ratio'] < 0.3]) >= len(diversity_analysis) * 0.7,
            'edge_cases_handled': all(ec['success'] for ec in edge_case_results.values())
        }
    }
    
    # Save report
    with open('comprehensive_test_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "="*60)
    print("COMPREHENSIVE TEST SUMMARY")
    print("="*60)
    
    successful_tests = len([r for r in results.values() if 'error' not in r])
    total_tests = len(results)
    success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
    
    print(f"Total users tested: {total_tests}")
    print(f"Successful tests: {successful_tests}")
    print(f"Success rate: {success_rate:.1f}%")
    
    # Diversity summary
    good_diversity = len([d for d in diversity_analysis.values() if d['overlap_ratio'] < 0.3])
    total_comparisons = len(diversity_analysis)
    diversity_rate = (good_diversity / total_comparisons * 100) if total_comparisons > 0 else 0
    
    print(f"\nDiversity analysis:")
    print(f"Good diversity pairs: {good_diversity}/{total_comparisons} ({diversity_rate:.1f}%)")
    
    # Edge cases summary
    edge_success = sum(1 for ec in edge_case_results.values() if ec['success'])
    edge_total = len(edge_case_results)
    edge_rate = (edge_success / edge_total * 100) if edge_total > 0 else 0
    
    print(f"\nEdge cases:")
    print(f"Successful edge cases: {edge_success}/{edge_total} ({edge_rate:.1f}%)")
    
    # Overall assessment
    overall_score = (success_rate + diversity_rate + edge_rate) / 3
    
    print(f"\nOverall system score: {overall_score:.1f}%")
    
    if overall_score >= 85:
        print("\n🎉 EXCELLENT! Your recommendation system is ready for production!")
        print("   The system shows excellent diversity, consistency, and handles edge cases well.")
    elif overall_score >= 70:
        print("\n✅ GOOD! Your recommendation system is working well.")
        print("   The system shows good performance with minor areas for improvement.")
    elif overall_score >= 50:
        print("\n⚠️  FAIR. Your recommendation system needs some improvements.")
        print("   The system works but has some issues that should be addressed.")
    else:
        print("\n❌ POOR. Your recommendation system needs significant work.")
        print("   Major issues detected. Review and fix before deployment.")
    
    print(f"\n📊 Detailed report saved to: comprehensive_test_report.json")
    
    return overall_score >= 70

if __name__ == "__main__":
    run_comprehensive_test()
