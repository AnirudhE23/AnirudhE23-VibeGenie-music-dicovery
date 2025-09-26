import psycopg2
import psycopg2.extras
import pandas as pd
from typing import List, Dict, Optional, Tuple
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DatabaseManager:
    """Handles all database operations for the music recommendation system"""
    
    def __init__(self):
        self.connection_params = {
            'host': os.getenv('DB_HOST'),
            'port': os.getenv('DB_PORT'),
            'database': os.getenv('DB_NAME'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD')
        }
    
    def get_connection(self):
        """Get a database connection"""
        return psycopg2.connect(**self.connection_params)
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    return True
        except Exception as e:
            print(f"Database connection failed: {e}")
            return False
    
    # User operations
    def create_user(self, spotify_user_id: str, display_name: str = None) -> bool:
        """Create a new user in the database"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "INSERT INTO users (spotify_user_id, display_name) VALUES (%s, %s) ON CONFLICT (spotify_user_id) DO NOTHING",
                        (spotify_user_id, display_name)
                    )
                    conn.commit()
                    return True
        except Exception as e:
            print(f"Error creating user: {e}")
            return False
    
    def get_user(self, spotify_user_id: str) -> Optional[Dict]:
        """Get user information"""
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                    cursor.execute(
                        "SELECT * FROM users WHERE spotify_user_id = %s",
                        (spotify_user_id,)
                    )
                    result = cursor.fetchone()
                    return dict(result) if result else None
        except Exception as e:
            print(f"Error getting user: {e}")
            return None
    
    # Track operations
    def get_track_by_id(self, spotify_track_id: str) -> Optional[Dict]:
        """Get track information by Spotify track ID"""
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                    cursor.execute(
                        "SELECT * FROM tracks WHERE spotify_track_id = %s",
                        (spotify_track_id,)
                    )
                    result = cursor.fetchone()
                    return dict(result) if result else None
        except Exception as e:
            print(f"Error getting track: {e}")
            return None
    
    def search_tracks(self, query: str, limit: int = 10) -> List[Dict]:
        """Search tracks by name or artist"""
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                    cursor.execute(
                        """
                        SELECT * FROM tracks 
                        WHERE track_name ILIKE %s OR artists ILIKE %s 
                        ORDER BY popularity DESC 
                        LIMIT %s
                        """,
                        (f'%{query}%', f'%{query}%', limit)
                    )
                    results = cursor.fetchall()
                    return [dict(row) for row in results]
        except Exception as e:
            print(f"Error searching tracks: {e}")
            return []
    
    def get_tracks_by_features(self, min_energy: float = None, max_energy: float = None,min_danceability: float = None, max_danceability: float = None,
                              limit: int = 100) -> List[Dict]:
        """Get tracks filtered by audio features"""
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                    query = "SELECT * FROM tracks WHERE 1=1"
                    params = []

                    if min_energy is not None:
                        query += " AND energy >= %s"
                        params.append(min_energy)
                    if max_energy is not None:
                        query += " AND energy <= %s"
                        params.append(max_energy)
                    if min_danceability is not None:
                        query += " AND danceability >= %s"
                        params.append(min_danceability)
                    if max_danceability is not None:
                        query += " AND danceability <= %s"
                        params.append(max_danceability)

                    query += " ORDER BY popularity DESC LIMIT %s"
                    params.append(limit)

                    cursor.execute(query, params)
                    results = cursor.fetchall()
                    return [dict(row) for row in results]
        except Exception as e:
            print(f"Error getting tracks by features: {e}")
            return []
    
    # User tracks operations
    def add_user_track(self, spotify_user_id: str, track_data: Dict, playlist_name: str = 'Unknown') -> bool:
        """Add a track to user's collection"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO user_tracks (
                            spotify_user_id, spotify_track_id, track_name, artists,
                            acousticness, danceability, energy, instrumentalness,
                            key_value, liveness, loudness, mode_value, speechiness, tempo, valence,
                            playlist_name
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (spotify_user_id, spotify_track_id) DO NOTHING
                        """,
                        (
                            spotify_user_id,
                            track_data.get('track_id'),
                            track_data.get('track_name'),
                            track_data.get('artists'),
                            float(track_data.get('acousticness')) if track_data.get('acousticness') is not None else None,
                            float(track_data.get('danceability')) if track_data.get('danceability') is not None else None,
                            float(track_data.get('energy')) if track_data.get('energy') is not None else None,
                            float(track_data.get('instrumentalness')) if track_data.get('instrumentalness') is not None else None,
                            float(track_data.get('key')) if track_data.get('key') is not None else None,
                            float(track_data.get('liveness')) if track_data.get('liveness') is not None else None,
                            float(track_data.get('loudness')) if track_data.get('loudness') is not None else None,
                            float(track_data.get('mode')) if track_data.get('mode') is not None else None,
                            float(track_data.get('speechiness')) if track_data.get('speechiness') is not None else None,
                            float(track_data.get('tempo')) if track_data.get('tempo') is not None else None,
                            float(track_data.get('valence')) if track_data.get('valence') is not None else None,
                            playlist_name
                        )
                    )
                    conn.commit()
                    return True
        except Exception as e:
            print(f"Error adding user track: {e}")
            return False
    
    def get_user_tracks(self, spotify_user_id: str) -> pd.DataFrame:
        """Get all tracks for a user as a DataFrame"""
        try:
            with self.get_connection() as conn:
                query = """
                SELECT * FROM user_tracks 
                WHERE spotify_user_id = %s
                ORDER BY added_at DESC
                """
                return pd.read_sql_query(query, conn, params=[spotify_user_id])
        except Exception as e:
            print(f"Error getting user tracks: {e}")
            return pd.DataFrame()

    def get_user_tracks_with_features(self, spotify_user_id: str) -> pd.DataFrame:
        """Get all tracks for a user with audio features as a DataFrame"""
        try:
            with self.get_connection() as conn:
                query = """
                SELECT * FROM user_tracks 
                WHERE spotify_user_id = %s
                AND acousticness IS NOT NULL
                AND danceability IS NOT NULL
                AND energy IS NOT NULL
                AND key_value IS NOT NULL
                AND mode_value IS NOT NULL
                ORDER BY added_at DESC
                """
                return pd.read_sql_query(query, conn, params=[spotify_user_id])
        except Exception as e:
            print(f"Error getting user tracks with features: {e}")
            return pd.DataFrame()

    # statistics and analytics 
    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    stats = {}
                    
                    # Total tracks
                    cursor.execute("SELECT COUNT(*) FROM tracks")
                    stats['total_tracks'] = cursor.fetchone()[0]
                    
                    # Total users
                    cursor.execute("SELECT COUNT(*) FROM users")
                    stats['total_users'] = cursor.fetchone()[0]
                    
                    # Total user tracks
                    cursor.execute("SELECT COUNT(*) FROM user_tracks")
                    stats['total_user_tracks'] = cursor.fetchone()[0]
                    
                    return stats
        except Exception as e:
            print(f"Error getting database stats: {e}")
            return {}

    def get_all_track_ids(self) -> List[str]:
        """Get all track IDs from the tracks table"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT spotify_track_id FROM tracks")
                    results = cursor.fetchall()
                    return [row[0] for row in results]
        except Exception as e:
            print(f"Error getting track IDs: {e}")
            return []

    def get_all_tracks(self) -> List[Dict]:
        """Get all tracks from the tracks table"""
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                    cursor.execute("SELECT * FROM tracks")
                    results = cursor.fetchall()
                    return [dict(row) for row in results]
        except Exception as e:
            print(f"Error getting all tracks: {e}")
            return []

    def add_track_to_database(self, track_data: Dict) -> bool:
        """Add a track to the tracks table (for database expansion)"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO tracks (
                            spotify_track_id, track_name, artists, popularity,
                            acousticness, danceability, energy, instrumentalness,
                            key_value, liveness, loudness, mode_value, speechiness, tempo, valence
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (spotify_track_id) DO NOTHING
                        """,
                        (
                            track_data.get('track_id'),
                            track_data.get('track_name'),
                            track_data.get('artists'),
                            track_data.get('popularity'),
                            track_data.get('acousticness'),
                            track_data.get('danceability'),
                            track_data.get('energy'),
                            track_data.get('instrumentalness'),
                            track_data.get('key_value'),
                            track_data.get('liveness'),
                            track_data.get('loudness'),
                            track_data.get('mode_value'),
                            track_data.get('speechiness'),
                            track_data.get('tempo'),
                            track_data.get('valence')
                        )
                    )
                    conn.commit()
                    return True
        except Exception as e:
            print(f"Error adding track to database: {e}")
            return False
        
    def remove_track_from_database(self, track_id: str) -> bool:
        """Remove a track from the tracks table"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("DELETE FROM tracks WHERE spotify_track_id = %s", (track_id,))
                    conn.commit()
                    return True
        except Exception as e:
            print(f"Error removing track from database: {e}")
            return False

db = DatabaseManager()