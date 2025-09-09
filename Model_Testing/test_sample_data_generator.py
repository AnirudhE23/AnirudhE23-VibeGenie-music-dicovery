"""
Sample Data Generator for Testing Recommendation System
Creates diverse user profiles with different music preferences to test recommendation variety
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime
import os

class SampleDataGenerator:
    """Generate diverse sample user data for testing recommendation system"""
    
    def __init__(self):
        self.genres = {
            'rock': {
                'acousticness': (0.01, 0.3),
                'danceability': (0.4, 0.7),
                'energy': (0.7, 0.95),
                'instrumentalness': (0.0, 0.1),
                'liveness': (0.1, 0.4),
                'speechiness': (0.02, 0.1),
                'valence': (0.4, 0.8),
                'loudness': (-12, -3),
                'tempo': (100, 160),
                'key': (0, 11),
                'mode': (0, 1),
                'popularity': (60, 90)
            },
            'pop': {
                'acousticness': (0.1, 0.4),
                'danceability': (0.6, 0.9),
                'energy': (0.6, 0.9),
                'instrumentalness': (0.0, 0.05),
                'liveness': (0.1, 0.3),
                'speechiness': (0.03, 0.15),
                'valence': (0.5, 0.9),
                'loudness': (-8, -2),
                'tempo': (110, 140),
                'key': (0, 11),
                'mode': (0, 1),
                'popularity': (70, 95)
            },
            'electronic': {
                'acousticness': (0.0, 0.2),
                'danceability': (0.7, 0.95),
                'energy': (0.8, 0.98),
                'instrumentalness': (0.1, 0.8),
                'liveness': (0.05, 0.2),
                'speechiness': (0.02, 0.1),
                'valence': (0.4, 0.8),
                'loudness': (-6, -1),
                'tempo': (120, 180),
                'key': (0, 11),
                'mode': (0, 1),
                'popularity': (50, 85)
            },
            'classical': {
                'acousticness': (0.8, 0.99),
                'danceability': (0.1, 0.4),
                'energy': (0.1, 0.5),
                'instrumentalness': (0.7, 0.99),
                'liveness': (0.1, 0.3),
                'speechiness': (0.0, 0.05),
                'valence': (0.1, 0.6),
                'loudness': (-20, -8),
                'tempo': (60, 120),
                'key': (0, 11),
                'mode': (0, 1),
                'popularity': (20, 60)
            },
            'jazz': {
                'acousticness': (0.3, 0.8),
                'danceability': (0.3, 0.7),
                'energy': (0.3, 0.7),
                'instrumentalness': (0.2, 0.9),
                'liveness': (0.1, 0.4),
                'speechiness': (0.02, 0.1),
                'valence': (0.2, 0.7),
                'loudness': (-15, -5),
                'tempo': (80, 140),
                'key': (0, 11),
                'mode': (0, 1),
                'popularity': (15, 50)
            },
            'hip_hop': {
                'acousticness': (0.0, 0.3),
                'danceability': (0.6, 0.9),
                'energy': (0.5, 0.9),
                'instrumentalness': (0.0, 0.1),
                'liveness': (0.1, 0.3),
                'speechiness': (0.1, 0.4),
                'valence': (0.3, 0.8),
                'loudness': (-10, -3),
                'tempo': (80, 120),
                'key': (0, 11),
                'mode': (0, 1),
                'popularity': (60, 90)
            },
            'country': {
                'acousticness': (0.4, 0.9),
                'danceability': (0.4, 0.8),
                'energy': (0.3, 0.7),
                'instrumentalness': (0.0, 0.2),
                'liveness': (0.1, 0.4),
                'speechiness': (0.03, 0.15),
                'valence': (0.4, 0.8),
                'loudness': (-12, -5),
                'tempo': (90, 130),
                'key': (0, 11),
                'mode': (0, 1),
                'popularity': (40, 80)
            },
            'blues': {
                'acousticness': (0.2, 0.7),
                'danceability': (0.3, 0.6),
                'energy': (0.3, 0.7),
                'instrumentalness': (0.0, 0.3),
                'liveness': (0.1, 0.4),
                'speechiness': (0.02, 0.1),
                'valence': (0.1, 0.5),
                'loudness': (-15, -6),
                'tempo': (70, 120),
                'key': (0, 11),
                'mode': (0, 1),
                'popularity': (25, 65)
            }
        }
        
        # Sample track names and artists for each genre
        self.sample_tracks = {
            'rock': [
                ('Bohemian Rhapsody', 'Queen'),
                ('Stairway to Heaven', 'Led Zeppelin'),
                ('Hotel California', 'Eagles'),
                ('Sweet Child O Mine', 'Guns N Roses'),
                ('Smells Like Teen Spirit', 'Nirvana'),
                ('Thunderstruck', 'AC/DC'),
                ('Enter Sandman', 'Metallica'),
                ('Born to Run', 'Bruce Springsteen'),
                ('Livin on a Prayer', 'Bon Jovi'),
                ('Black', 'Pearl Jam')
            ],
            'pop': [
                ('Shape of You', 'Ed Sheeran'),
                ('Blinding Lights', 'The Weeknd'),
                ('Levitating', 'Dua Lipa'),
                ('Watermelon Sugar', 'Harry Styles'),
                ('Good 4 U', 'Olivia Rodrigo'),
                ('Stay', 'The Kid LAROI & Justin Bieber'),
                ('Industry Baby', 'Lil Nas X'),
                ('Peaches', 'Justin Bieber'),
                ('Kiss Me More', 'Doja Cat'),
                ('Montero', 'Lil Nas X')
            ],
            'electronic': [
                ('One More Time', 'Daft Punk'),
                ('Strobe', 'Deadmau5'),
                ('Levels', 'Avicii'),
                ('Titanium', 'David Guetta'),
                ('Clarity', 'Zedd'),
                ('Faded', 'Alan Walker'),
                ('Animals', 'Martin Garrix'),
                ('Wake Me Up', 'Avicii'),
                ('Summertime Sadness', 'Lana Del Rey'),
                ('Midnight City', 'M83')
            ],
            'classical': [
                ('Canon in D', 'Johann Pachelbel'),
                ('Moonlight Sonata', 'Ludwig van Beethoven'),
                ('The Four Seasons', 'Antonio Vivaldi'),
                ('Clair de Lune', 'Claude Debussy'),
                ('Symphony No. 9', 'Ludwig van Beethoven'),
                ('The Nutcracker Suite', 'Pyotr Ilyich Tchaikovsky'),
                ('Eine kleine Nachtmusik', 'Wolfgang Amadeus Mozart'),
                ('Boléro', 'Maurice Ravel'),
                ('Adagio for Strings', 'Samuel Barber'),
                ('The Blue Danube', 'Johann Strauss II')
            ],
            'jazz': [
                ('Take Five', 'Dave Brubeck'),
                ('Blue in Green', 'Miles Davis'),
                ('Autumn Leaves', 'Chet Baker'),
                ('All of Me', 'Billie Holiday'),
                ('Summertime', 'Ella Fitzgerald'),
                ('My Funny Valentine', 'Chet Baker'),
                ('Round Midnight', 'Thelonious Monk'),
                ('So What', 'Miles Davis'),
                ('A Love Supreme', 'John Coltrane'),
                ('Kind of Blue', 'Miles Davis')
            ],
            'hip_hop': [
                ('Lose Yourself', 'Eminem'),
                ('Juicy', 'The Notorious B.I.G.'),
                ('Nuthin but a G Thang', 'Dr. Dre'),
                ('California Love', '2Pac'),
                ('C.R.E.A.M.', 'Wu-Tang Clan'),
                ('The Message', 'Grandmaster Flash'),
                ('Rapper\'s Delight', 'Sugarhill Gang'),
                ('Fight the Power', 'Public Enemy'),
                ('Straight Outta Compton', 'N.W.A'),
                ('It Was a Good Day', 'Ice Cube')
            ],
            'country': [
                ('Friends in Low Places', 'Garth Brooks'),
                ('The Gambler', 'Kenny Rogers'),
                ('Ring of Fire', 'Johnny Cash'),
                ('Jolene', 'Dolly Parton'),
                ('Take Me Home, Country Roads', 'John Denver'),
                ('Amarillo by Morning', 'George Strait'),
                ('Mama Tried', 'Merle Haggard'),
                ('He Stopped Loving Her Today', 'George Jones'),
                ('Coal Miner\'s Daughter', 'Loretta Lynn'),
                ('I Walk the Line', 'Johnny Cash')
            ],
            'blues': [
                ('The Thrill is Gone', 'B.B. King'),
                ('Cross Road Blues', 'Robert Johnson'),
                ('Sweet Home Chicago', 'Robert Johnson'),
                ('Hoochie Coochie Man', 'Muddy Waters'),
                ('Born Under a Bad Sign', 'Albert King'),
                ('Stormy Monday', 'T-Bone Walker'),
                ('Red House', 'Jimi Hendrix'),
                ('Pride and Joy', 'Stevie Ray Vaughan'),
                ('I\'m a Man', 'Bo Diddley'),
                ('Mannish Boy', 'Muddy Waters')
            ]
        }

    def generate_user_profile(self, genre, num_tracks=15, user_id=None):
        """
        Generate a user profile with tracks from a specific genre
        
        Args:
            genre (str): Genre to generate tracks for
            num_tracks (int): Number of tracks to generate
            user_id (str): Optional user ID
            
        Returns:
            pd.DataFrame: User profile with track data
        """
        if genre not in self.genres:
            raise ValueError(f"Genre '{genre}' not supported. Available: {list(self.genres.keys())}")
        
        if user_id is None:
            user_id = f"user_{genre}_{random.randint(1000, 9999)}"
        
        genre_config = self.genres[genre]
        tracks = self.sample_tracks[genre]
        
        user_tracks = []
        
        for i in range(num_tracks):
            # Select a track (with replacement if we need more tracks than available)
            track_name, artist = random.choice(tracks)
            
            # Generate features within genre ranges
            features = {}
            for feature, (min_val, max_val) in genre_config.items():
                if feature in ['key', 'mode']:
                    # Discrete values
                    features[feature] = random.randint(int(min_val), int(max_val))
                else:
                    # Continuous values
                    features[feature] = random.uniform(min_val, max_val)
            
            # Create track ID
            track_id = f"sample_{genre}_{i}_{random.randint(10000, 99999)}"
            
            user_tracks.append({
                'Track Name': track_name,
                'Album Name': f'{track_name} - Album',
                'Artist(s)': artist,
                'Playlist Name': f'{genre.title()} Favorites',
                'Track ID': track_id,
                'Popularity': int(features['popularity']),
                'Owner ID': user_id,
                'acousticness': features['acousticness'],
                'danceability': features['danceability'],
                'energy': features['energy'],
                'instrumentalness': features['instrumentalness'],
                'key': features['key'],
                'liveness': features['liveness'],
                'loudness': features['loudness'],
                'mode': features['mode'],
                'speechiness': features['speechiness'],
                'tempo': features['tempo'],
                'valence': features['valence'],
                'href': f'https://open.spotify.com/track/{track_id}'
            })
        
        return pd.DataFrame(user_tracks)

    def generate_mixed_profile(self, genres, num_tracks_per_genre=5, user_id=None):
        """
        Generate a user profile with tracks from multiple genres
        
        Args:
            genres (list): List of genres to include
            num_tracks_per_genre (int): Number of tracks per genre
            user_id (str): Optional user ID
            
        Returns:
            pd.DataFrame: Mixed genre user profile
        """
        if user_id is None:
            user_id = f"user_mixed_{random.randint(1000, 9999)}"
        
        all_tracks = []
        
        for genre in genres:
            if genre not in self.genres:
                print(f"Warning: Genre '{genre}' not supported, skipping...")
                continue
            
            genre_tracks = self.generate_user_profile(genre, num_tracks_per_genre, user_id)
            all_tracks.append(genre_tracks)
        
        if all_tracks:
            return pd.concat(all_tracks, ignore_index=True)
        else:
            raise ValueError("No valid genres provided")

    def generate_test_users(self, num_users_per_genre=2):
        """
        Generate multiple test users for comprehensive testing
        
        Args:
            num_users_per_genre (int): Number of users to generate per genre
            
        Returns:
            dict: Dictionary with genre as key and list of user profiles as values
        """
        test_users = {}
        
        for genre in self.genres.keys():
            test_users[genre] = []
            for i in range(num_users_per_genre):
                user_profile = self.generate_user_profile(genre, num_tracks=12)
                test_users[genre].append(user_profile)
        
        # Add some mixed genre users
        mixed_combinations = [
            ['rock', 'pop'],
            ['electronic', 'hip_hop'],
            ['jazz', 'blues'],
            ['classical', 'jazz'],
            ['rock', 'electronic', 'pop'],
            ['country', 'blues', 'rock']
        ]
        
        test_users['mixed'] = []
        for combo in mixed_combinations:
            user_profile = self.generate_mixed_profile(combo, num_tracks_per_genre=4)
            test_users['mixed'].append(user_profile)
        
        return test_users

    def save_test_data(self, test_users, output_dir='test_data'):
        """
        Save test user data to CSV files
        
        Args:
            test_users (dict): Test users dictionary
            output_dir (str): Output directory for test data
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for genre, users in test_users.items():
            for i, user_df in enumerate(users):
                filename = f"{output_dir}/{genre}_user_{i+1}.csv"
                user_df.to_csv(filename, index=False)
                print(f"Saved {filename} with {len(user_df)} tracks")
        
        # Create a summary file
        summary_data = []
        for genre, users in test_users.items():
            for i, user_df in enumerate(users):
                summary_data.append({
                    'genre': genre,
                    'user_id': f"{genre}_user_{i+1}",
                    'num_tracks': len(user_df),
                    'avg_popularity': user_df['Popularity'].mean(),
                    'avg_energy': user_df['energy'].mean(),
                    'avg_danceability': user_df['danceability'].mean(),
                    'avg_valence': user_df['valence'].mean()
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f"{output_dir}/test_users_summary.csv", index=False)
        print(f"Saved test users summary to {output_dir}/test_users_summary.csv")

def main():
    """Generate and save test data"""
    generator = SampleDataGenerator()
    
    print("Generating test user data...")
    test_users = generator.generate_test_users(num_users_per_genre=2)
    
    print("Saving test data...")
    generator.save_test_data(test_users)
    
    print("\nTest data generation complete!")
    print("Generated users for genres:", list(test_users.keys()))
    
    # Print summary
    total_users = sum(len(users) for users in test_users.values())
    total_tracks = sum(sum(len(user_df) for user_df in users) for users in test_users.values())
    print(f"Total users: {total_users}")
    print(f"Total tracks: {total_tracks}")

if __name__ == "__main__":
    main()
