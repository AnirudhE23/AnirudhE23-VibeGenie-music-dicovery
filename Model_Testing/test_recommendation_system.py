"""
Comprehensive Test Script for Recommendation System
Tests recommendation diversity, quality, and consistency across different user profiles
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from test_sample_data_generator import SampleDataGenerator
from recommendation_system import recommendation_engine

class RecommendationSystemTester:
    """Comprehensive testing suite for the recommendation system"""
    
    def __init__(self):
        self.test_results = {}
        self.recommendation_engine = recommendation_engine
        self.data_generator = SampleDataGenerator()
        
    def run_all_tests(self):
        """Run all test suites"""
        print("="*60)
        print("RECOMMENDATION SYSTEM COMPREHENSIVE TESTING")
        print("="*60)
        
        # Load the recommendation engine
        print("\n1. Loading recommendation engine...")
        self.recommendation_engine.load_model()
        
        if not self.recommendation_engine.is_loaded:
            print("❌ Failed to load recommendation engine!")
            print(f"Error: {self.recommendation_engine.load_error}")
            return False
        
        print("✅ Recommendation engine loaded successfully!")
        
        # Generate test data
        print("\n2. Generating test user data...")
        test_users = self.data_generator.generate_test_users(num_users_per_genre=2)
        self.data_generator.save_test_data(test_users, 'test_data')
        
        # Run test suites
        print("\n3. Running test suites...")
        
        # Test 1: Basic functionality
        self.test_basic_functionality(test_users)
        
        # Test 2: Recommendation diversity
        self.test_recommendation_diversity(test_users)
        
        # Test 3: Genre consistency
        self.test_genre_consistency(test_users)
        
        # Test 4: User profile sensitivity
        self.test_user_profile_sensitivity(test_users)
        
        # Test 5: Recommendation quality metrics
        self.test_recommendation_quality(test_users)
        
        # Test 6: Edge cases
        self.test_edge_cases()
        
        # Generate comprehensive report
        self.generate_test_report()
        
        return True
    
    def test_basic_functionality(self, test_users):
        """Test basic recommendation functionality"""
        print("\n" + "="*40)
        print("TEST 1: BASIC FUNCTIONALITY")
        print("="*40)
        
        results = {
            'test_name': 'Basic Functionality',
            'passed': 0,
            'failed': 0,
            'errors': []
        }
        
        try:
            # Test with a rock user
            rock_user = test_users['rock'][0]
            recommendations, error = self.recommendation_engine.get_recommendations(rock_user, 10)
            
            if error:
                results['failed'] += 1
                results['errors'].append(f"Error getting recommendations: {error}")
            elif recommendations and len(recommendations) > 0:
                results['passed'] += 1
                print(f"✅ Successfully generated {len(recommendations)} recommendations")
            else:
                results['failed'] += 1
                results['errors'].append("No recommendations returned")
            
            # Test with different numbers of recommendations
            for n_recs in [5, 15, 25]:
                recs, error = self.recommendation_engine.get_recommendations(rock_user, n_recs)
                if error or len(recs) != n_recs:
                    results['failed'] += 1
                    results['errors'].append(f"Failed to get {n_recs} recommendations")
                else:
                    results['passed'] += 1
                    print(f"✅ Successfully generated {n_recs} recommendations")
            
        except Exception as e:
            results['failed'] += 1
            results['errors'].append(f"Exception in basic functionality test: {str(e)}")
        
        self.test_results['basic_functionality'] = results
        self._print_test_summary(results)
    
    def test_recommendation_diversity(self, test_users):
        """Test that different users get different recommendations"""
        print("\n" + "="*40)
        print("TEST 2: RECOMMENDATION DIVERSITY")
        print("="*40)
        
        results = {
            'test_name': 'Recommendation Diversity',
            'passed': 0,
            'failed': 0,
            'errors': [],
            'diversity_metrics': {}
        }
        
        try:
            all_recommendations = {}
            user_profiles = {}
            
            # Get recommendations for each genre
            for genre, users in test_users.items():
                if genre == 'mixed':  # Skip mixed for now, test separately
                    continue
                    
                genre_recs = []
                for i, user in enumerate(users):
                    recs, error = self.recommendation_engine.get_recommendations(user, 20)
                    if not error and recs:
                        genre_recs.extend([rec['track_id'] for rec in recs])
                        all_recommendations[f"{genre}_user_{i}"] = [rec['track_id'] for rec in recs]
                        user_profiles[f"{genre}_user_{i}"] = {
                            'genre': genre,
                            'avg_energy': user['energy'].mean(),
                            'avg_danceability': user['danceability'].mean(),
                            'avg_valence': user['valence'].mean()
                        }
                
                # Calculate diversity within genre
                unique_tracks = len(set(genre_recs))
                total_tracks = len(genre_recs)
                diversity_ratio = unique_tracks / total_tracks if total_tracks > 0 else 0
                results['diversity_metrics'][f"{genre}_diversity"] = diversity_ratio
                
                print(f"{genre.title()} diversity: {diversity_ratio:.3f} ({unique_tracks}/{total_tracks} unique)")
            
            # Test cross-genre diversity
            genre_pairs = [('rock', 'classical'), ('pop', 'jazz'), ('electronic', 'blues')]
            for genre1, genre2 in genre_pairs:
                if f"{genre1}_user_0" in all_recommendations and f"{genre2}_user_0" in all_recommendations:
                    recs1 = set(all_recommendations[f"{genre1}_user_0"])
                    recs2 = set(all_recommendations[f"{genre2}_user_0"])
                    overlap = len(recs1.intersection(recs2))
                    total_unique = len(recs1.union(recs2))
                    overlap_ratio = overlap / total_unique if total_unique > 0 else 0
                    
                    results['diversity_metrics'][f"{genre1}_vs_{genre2}_overlap"] = overlap_ratio
                    print(f"{genre1} vs {genre2} overlap: {overlap_ratio:.3f} ({overlap}/{total_unique})")
                    
                    # Good diversity means low overlap between very different genres
                    if overlap_ratio < 0.3:  # Less than 30% overlap
                        results['passed'] += 1
                    else:
                        results['failed'] += 1
                        results['errors'].append(f"High overlap between {genre1} and {genre2}: {overlap_ratio:.3f}")
            
            # Test that users get different recommendations
            user_pairs = list(all_recommendations.keys())
            for i in range(len(user_pairs)):
                for j in range(i+1, len(user_pairs)):
                    user1, user2 = user_pairs[i], user_pairs[j]
                    recs1 = set(all_recommendations[user1])
                    recs2 = set(all_recommendations[user2])
                    overlap = len(recs1.intersection(recs2))
                    total_unique = len(recs1.union(recs2))
                    overlap_ratio = overlap / total_unique if total_unique > 0 else 0
                    
                    if overlap_ratio < 0.8:  # Less than 80% overlap
                        results['passed'] += 1
                    else:
                        results['failed'] += 1
                        results['errors'].append(f"Very high overlap between {user1} and {user2}: {overlap_ratio:.3f}")
            
        except Exception as e:
            results['failed'] += 1
            results['errors'].append(f"Exception in diversity test: {str(e)}")
        
        self.test_results['recommendation_diversity'] = results
        self._print_test_summary(results)
    
    def test_genre_consistency(self, test_users):
        """Test that recommendations are consistent with user's genre preferences"""
        print("\n" + "="*40)
        print("TEST 3: GENRE CONSISTENCY")
        print("="*40)
        
        results = {
            'test_name': 'Genre Consistency',
            'passed': 0,
            'failed': 0,
            'errors': [],
            'consistency_metrics': {}
        }
        
        try:
            # Define expected feature ranges for each genre
            genre_expectations = {
                'rock': {'energy': (0.6, 1.0), 'danceability': (0.4, 0.8)},
                'pop': {'energy': (0.5, 0.9), 'danceability': (0.6, 0.9)},
                'electronic': {'energy': (0.7, 1.0), 'danceability': (0.7, 1.0)},
                'classical': {'energy': (0.1, 0.6), 'danceability': (0.1, 0.5)},
                'jazz': {'energy': (0.3, 0.7), 'danceability': (0.3, 0.7)},
                'hip_hop': {'energy': (0.5, 0.9), 'danceability': (0.6, 0.9)},
                'country': {'energy': (0.3, 0.7), 'danceability': (0.4, 0.8)},
                'blues': {'energy': (0.3, 0.7), 'danceability': (0.3, 0.6)}
            }
            
            for genre, users in test_users.items():
                if genre == 'mixed' or genre not in genre_expectations:
                    continue
                
                for i, user in enumerate(users):
                    recs, error = self.recommendation_engine.get_recommendations(user, 15)
                    if error or not recs:
                        continue
                    
                    # Get track IDs from recommendations
                    track_ids = [rec['track_id'] for rec in recs]
                    
                    # Load the full dataset to get features for recommended tracks
                    try:
                        # Try to load the metadata file
                        metadata_df = pd.read_csv('song_metadata.csv')
                        recommended_tracks = metadata_df[metadata_df['track_id'].isin(track_ids)]
                        
                        if len(recommended_tracks) > 0:
                            # Check if recommended tracks match genre expectations
                            expectations = genre_expectations[genre]
                            consistency_score = 0
                            total_checks = 0
                            
                            for feature, (min_val, max_val) in expectations.items():
                                if feature in recommended_tracks.columns:
                                    feature_values = recommended_tracks[feature]
                                    # Count how many tracks fall within expected range
                                    in_range = ((feature_values >= min_val) & (feature_values <= max_val)).sum()
                                    consistency_score += in_range / len(feature_values)
                                    total_checks += 1
                            
                            if total_checks > 0:
                                avg_consistency = consistency_score / total_checks
                                results['consistency_metrics'][f"{genre}_user_{i}"] = avg_consistency
                                
                                print(f"{genre} user {i} consistency: {avg_consistency:.3f}")
                                
                                if avg_consistency > 0.6:  # 60% of recommendations should match genre
                                    results['passed'] += 1
                                else:
                                    results['failed'] += 1
                                    results['errors'].append(f"Low consistency for {genre} user {i}: {avg_consistency:.3f}")
                        
                    except FileNotFoundError:
                        print(f"Warning: Could not load song_metadata.csv for consistency check")
                        results['failed'] += 1
                        results['errors'].append("Could not load metadata for consistency check")
        
        except Exception as e:
            results['failed'] += 1
            results['errors'].append(f"Exception in genre consistency test: {str(e)}")
        
        self.test_results['genre_consistency'] = results
        self._print_test_summary(results)
    
    def test_user_profile_sensitivity(self, test_users):
        """Test that the system responds to different user profiles"""
        print("\n" + "="*40)
        print("TEST 4: USER PROFILE SENSITIVITY")
        print("="*40)
        
        results = {
            'test_name': 'User Profile Sensitivity',
            'passed': 0,
            'failed': 0,
            'errors': [],
            'sensitivity_metrics': {}
        }
        
        try:
            # Test with users who have very different profiles
            test_pairs = [
                ('rock', 'classical'),
                ('electronic', 'jazz'),
                ('pop', 'blues')
            ]
            
            for genre1, genre2 in test_pairs:
                if genre1 in test_users and genre2 in test_users:
                    user1 = test_users[genre1][0]
                    user2 = test_users[genre2][0]
                    
                    recs1, error1 = self.recommendation_engine.get_recommendations(user1, 20)
                    recs2, error2 = self.recommendation_engine.get_recommendations(user2, 20)
                    
                    if error1 or error2 or not recs1 or not recs2:
                        continue
                    
                    # Calculate profile differences
                    profile_diff = self._calculate_profile_difference(user1, user2)
                    
                    # Calculate recommendation differences
                    rec_ids1 = set([rec['track_id'] for rec in recs1])
                    rec_ids2 = set([rec['track_id'] for rec in recs2])
                    rec_diff = 1 - (len(rec_ids1.intersection(rec_ids2)) / len(rec_ids1.union(rec_ids2)))
                    
                    results['sensitivity_metrics'][f"{genre1}_vs_{genre2}"] = {
                        'profile_difference': profile_diff,
                        'recommendation_difference': rec_diff
                    }
                    
                    print(f"{genre1} vs {genre2}:")
                    print(f"  Profile difference: {profile_diff:.3f}")
                    print(f"  Recommendation difference: {rec_diff:.3f}")
                    
                    # Good sensitivity: different profiles should lead to different recommendations
                    if rec_diff > 0.3:  # At least 30% different recommendations
                        results['passed'] += 1
                    else:
                        results['failed'] += 1
                        results['errors'].append(f"Low sensitivity for {genre1} vs {genre2}: {rec_diff:.3f}")
        
        except Exception as e:
            results['failed'] += 1
            results['errors'].append(f"Exception in sensitivity test: {str(e)}")
        
        self.test_results['user_profile_sensitivity'] = results
        self._print_test_summary(results)
    
    def test_recommendation_quality(self, test_users):
        """Test recommendation quality metrics"""
        print("\n" + "="*40)
        print("TEST 5: RECOMMENDATION QUALITY")
        print("="*40)
        
        results = {
            'test_name': 'Recommendation Quality',
            'passed': 0,
            'failed': 0,
            'errors': [],
            'quality_metrics': {}
        }
        
        try:
            all_quality_metrics = []
            
            for genre, users in test_users.items():
                if genre == 'mixed':
                    continue
                    
                for i, user in enumerate(users):
                    recs, error = self.recommendation_engine.get_recommendations(user, 20)
                    if error or not recs:
                        continue
                    
                    # Calculate quality metrics
                    similarity_scores = [rec['similarity_score'] for rec in recs]
                    popularity_scores = [rec['popularity'] for rec in recs]
                    
                    quality_metrics = {
                        'genre': genre,
                        'user_id': f"{genre}_user_{i}",
                        'avg_similarity': np.mean(similarity_scores),
                        'min_similarity': np.min(similarity_scores),
                        'max_similarity': np.max(similarity_scores),
                        'similarity_std': np.std(similarity_scores),
                        'avg_popularity': np.mean(popularity_scores),
                        'popularity_diversity': len(set(popularity_scores)) / len(popularity_scores)
                    }
                    
                    all_quality_metrics.append(quality_metrics)
                    
                    print(f"{genre} user {i}:")
                    print(f"  Avg similarity: {quality_metrics['avg_similarity']:.3f}")
                    print(f"  Similarity range: {quality_metrics['min_similarity']:.3f} - {quality_metrics['max_similarity']:.3f}")
                    print(f"  Avg popularity: {quality_metrics['avg_popularity']:.1f}")
                    
                    # Quality checks
                    if quality_metrics['avg_similarity'] > 0.5:  # Good similarity scores
                        results['passed'] += 1
                    else:
                        results['failed'] += 1
                        results['errors'].append(f"Low average similarity for {genre} user {i}: {quality_metrics['avg_similarity']:.3f}")
                    
                    if quality_metrics['similarity_std'] > 0.05:  # Good diversity in similarity scores
                        results['passed'] += 1
                    else:
                        results['failed'] += 1
                        results['errors'].append(f"Low similarity diversity for {genre} user {i}: {quality_metrics['similarity_std']:.3f}")
            
            results['quality_metrics'] = all_quality_metrics
        
        except Exception as e:
            results['failed'] += 1
            results['errors'].append(f"Exception in quality test: {str(e)}")
        
        self.test_results['recommendation_quality'] = results
        self._print_test_summary(results)
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        print("\n" + "="*40)
        print("TEST 6: EDGE CASES")
        print("="*40)
        
        results = {
            'test_name': 'Edge Cases',
            'passed': 0,
            'failed': 0,
            'errors': []
        }
        
        try:
            # Test with empty dataframe
            empty_df = pd.DataFrame()
            recs, error = self.recommendation_engine.get_recommendations(empty_df, 10)
            if error and "No tracks" in error:
                results['passed'] += 1
                print("✅ Correctly handles empty dataframe")
            else:
                results['failed'] += 1
                results['errors'].append("Failed to handle empty dataframe")
            
            # Test with insufficient tracks
            minimal_df = self.data_generator.generate_user_profile('rock', 2)
            recs, error = self.recommendation_engine.get_recommendations(minimal_df, 10)
            if error and "at least 3 tracks" in error:
                results['passed'] += 1
                print("✅ Correctly handles insufficient tracks")
            else:
                results['failed'] += 1
                results['errors'].append("Failed to handle insufficient tracks")
            
            # Test with missing features
            incomplete_df = self.data_generator.generate_user_profile('rock', 5)
            # Remove some feature columns
            incomplete_df = incomplete_df.drop(columns=['acousticness', 'danceability'])
            recs, error = self.recommendation_engine.get_recommendations(incomplete_df, 10)
            if error and "audio features" in error:
                results['passed'] += 1
                print("✅ Correctly handles missing features")
            else:
                results['failed'] += 1
                results['errors'].append("Failed to handle missing features")
            
            # Test with very large number of recommendations
            normal_df = self.data_generator.generate_user_profile('pop', 10)
            recs, error = self.recommendation_engine.get_recommendations(normal_df, 1000)
            if not error and recs and len(recs) > 0:
                results['passed'] += 1
                print(f"✅ Handles large recommendation requests: {len(recs)} recommendations")
            else:
                results['failed'] += 1
                results['errors'].append("Failed to handle large recommendation requests")
        
        except Exception as e:
            results['failed'] += 1
            results['errors'].append(f"Exception in edge cases test: {str(e)}")
        
        self.test_results['edge_cases'] = results
        self._print_test_summary(results)
    
    def _calculate_profile_difference(self, user1_df, user2_df):
        """Calculate the difference between two user profiles"""
        features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 
                   'liveness', 'speechiness', 'valence', 'loudness', 'tempo', 'key', 'mode']
        
        diff_sum = 0
        for feature in features:
            if feature in user1_df.columns and feature in user2_df.columns:
                mean1 = user1_df[feature].mean()
                mean2 = user2_df[feature].mean()
                diff_sum += abs(mean1 - mean2)
        
        return diff_sum / len(features)
    
    def _print_test_summary(self, results):
        """Print a summary of test results"""
        total_tests = results['passed'] + results['failed']
        pass_rate = (results['passed'] / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\n{results['test_name']} Results:")
        print(f"  Passed: {results['passed']}")
        print(f"  Failed: {results['failed']}")
        print(f"  Pass Rate: {pass_rate:.1f}%")
        
        if results['errors']:
            print("  Errors:")
            for error in results['errors'][:3]:  # Show first 3 errors
                print(f"    - {error}")
            if len(results['errors']) > 3:
                print(f"    ... and {len(results['errors']) - 3} more errors")
    
    def generate_test_report(self):
        """Generate a comprehensive test report"""
        print("\n" + "="*60)
        print("COMPREHENSIVE TEST REPORT")
        print("="*60)
        
        # Calculate overall statistics
        total_passed = sum(results['passed'] for results in self.test_results.values())
        total_failed = sum(results['failed'] for results in self.test_results.values())
        total_tests = total_passed + total_failed
        overall_pass_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\nOVERALL RESULTS:")
        print(f"  Total Tests: {total_tests}")
        print(f"  Passed: {total_passed}")
        print(f"  Failed: {total_failed}")
        print(f"  Overall Pass Rate: {overall_pass_rate:.1f}%")
        
        print(f"\nTEST SUITE BREAKDOWN:")
        for test_name, results in self.test_results.items():
            test_total = results['passed'] + results['failed']
            test_pass_rate = (results['passed'] / test_total * 100) if test_total > 0 else 0
            status = "✅ PASS" if test_pass_rate >= 80 else "❌ FAIL"
            print(f"  {test_name.replace('_', ' ').title()}: {status} ({test_pass_rate:.1f}%)")
        
        # Save detailed report
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'overall_results': {
                'total_tests': total_tests,
                'passed': total_passed,
                'failed': total_failed,
                'pass_rate': overall_pass_rate
            },
            'test_results': self.test_results
        }
        
        with open('test_report.json', 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"\n📊 Detailed report saved to: test_report.json")
        
        # Generate recommendations
        if overall_pass_rate >= 80:
            print("\n🎉 RECOMMENDATION SYSTEM IS READY FOR PRODUCTION!")
            print("   The system shows good diversity, consistency, and quality.")
        elif overall_pass_rate >= 60:
            print("\n⚠️  RECOMMENDATION SYSTEM NEEDS IMPROVEMENT")
            print("   The system works but has some issues that should be addressed.")
        else:
            print("\n❌ RECOMMENDATION SYSTEM NEEDS MAJOR WORK")
            print("   Significant issues detected. Review and fix before deployment.")
        
        return overall_pass_rate >= 80

def main():
    """Run the comprehensive test suite"""
    tester = RecommendationSystemTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n✅ All tests completed successfully!")
    else:
        print("\n❌ Some tests failed. Check the report for details.")
    
    return success

if __name__ == "__main__":
    main()
