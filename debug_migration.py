import pandas as pd
import psycopg2
import numpy as np

DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'music_recommender',
    'user': 'postgres',
    'password': 'Krishna$9$9'
}

def debug_key_mode():
    """Debug specifically key and mode values"""
    
    # Read CSV
    df = pd.read_csv('Final_training_dataset.csv')
    print("🔍 CSV Data:")
    print(f"Key column: {df['key'].dtype}")
    print(f"Mode column: {df['mode'].dtype}")
    print(f"First key value: {df.iloc[0]['key']} (type: {type(df.iloc[0]['key'])})")
    print(f"First mode value: {df.iloc[0]['mode']} (type: {type(df.iloc[0]['mode'])})")
    
    # Convert to Python types
    key_converted = float(df.iloc[0]['key']) if pd.notna(df.iloc[0]['key']) else None
    mode_converted = float(df.iloc[0]['mode']) if pd.notna(df.iloc[0]['mode']) else None
    
    print(f"\n🔍 After conversion:")
    print(f"Key converted: {key_converted} (type: {type(key_converted)})")
    print(f"Mode converted: {mode_converted} (type: {type(mode_converted)})")
    
    # Connect to database
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    try:
        # Clear existing data
        cursor.execute("DELETE FROM tracks")
        conn.commit()
        
        # Test insert with explicit values
        print(f"\n🔍 Testing direct insert...")
        test_query = """
        INSERT INTO tracks (
            spotify_track_id, track_name, artists, popularity,
            acousticness, danceability, energy, instrumentalness,
            key_value, liveness, loudness, mode_value, speechiness, tempo, valence
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        # Use the first row data with explicit conversion
        first_row = df.iloc[0]
        test_data = (
            str(first_row['track_id']),
            str(first_row['track_name']),
            str(first_row['artists']),
            int(first_row['popularity']),
            float(first_row['acousticness']),
            float(first_row['danceability']),
            float(first_row['energy']),
            float(first_row['instrumentalness']),
            float(first_row['key']),  # This should be 4.0
            float(first_row['liveness']),
            float(first_row['loudness']),
            float(first_row['mode']),  # This should be 0.0
            float(first_row['speechiness']),
            float(first_row['tempo']),
            float(first_row['valence'])
        )
        
        print(f"Test data: {test_data}")
        print(f"Key value in test data: {test_data[8]} (type: {type(test_data[8])})")
        print(f"Mode value in test data: {test_data[11]} (type: {type(test_data[11])})")
        
        cursor.execute(test_query, test_data)
        conn.commit()
        
        # Check what was actually inserted
        cursor.execute("SELECT spotify_track_id, key_value, mode_value FROM tracks")
        result = cursor.fetchone()
        print(f"\n🔍 What was actually inserted:")
        print(f"Track ID: {result[0]}")
        print(f"Key value: {result[1]} (type: {type(result[1])})")
        print(f"Mode value: {result[2]} (type: {type(result[2])})")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    debug_key_mode()