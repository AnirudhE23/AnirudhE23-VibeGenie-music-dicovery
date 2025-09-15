import pandas as pd
import psycopg2
from psycopg2 import sql
import os
import numpy as np

DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'music_recommender',
    'user': 'postgres',
    'password': 'Krishna$9$9'
}

def migrate_training_data():
    """Migrate data from Final_training_dataset.csv to tracks table"""

    # Read the training dataset
    print("✅ Reading training dataset...")
    df = pd.read_csv("Final_training_dataset.csv")
    print(f"✅ Loaded {len(df)} tracks from training dataset")

    # Connect to the database
    print("✅ Connecting to the database...")
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    try:
        # Clear existing data first
        print("🧹 Clearing existing tracks data...")
        cursor.execute("DELETE FROM tracks")
        conn.commit()
        print("✅ Existing data cleared")

        # Prepare insert statement
        insert_query = """
        INSERT INTO tracks (
            spotify_track_id, track_name, artists, popularity,
            acousticness, danceability, energy, instrumentalness,
            key_value, liveness, loudness, mode_value, speechiness, tempo, valence
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        # insert data in batches
        batch_size = 1000
        total_rows = len(df)

        print(f"✅ Inserting {total_rows} tracks in batches of {batch_size}")

        for i in range(0, total_rows, batch_size):
            batch = df.iloc[i:i+batch_size]
            # Prepare batch data
            batch_data = []
            for _, row in batch.iterrows():
                batch_data.append((
                    str(row['track_id']),  # Convert to string
                    str(row['track_name']),  # Convert to string
                    str(row['artists']),  # Convert to string
                    int(row['popularity']) if pd.notna(row['popularity']) else None,  # Convert to int
                    float(row['acousticness']) if pd.notna(row['acousticness']) else None,
                    float(row['danceability']) if pd.notna(row['danceability']) else None,
                    float(row['energy']) if pd.notna(row['energy']) else None,
                    float(row['instrumentalness']) if pd.notna(row['instrumentalness']) else None,
                    float(row['key']) if pd.notna(row['key']) else None,  # Convert numpy to float
                    float(row['liveness']) if pd.notna(row['liveness']) else None,
                    float(row['loudness']) if pd.notna(row['loudness']) else None,
                    float(row['mode']) if pd.notna(row['mode']) else None,  # Convert numpy to float
                    float(row['speechiness']) if pd.notna(row['speechiness']) else None,
                    float(row['tempo']) if pd.notna(row['tempo']) else None,
                    float(row['valence']) if pd.notna(row['valence']) else None
                ))
            # execute batch insert
            cursor.executemany(insert_query, batch_data)
            conn.commit()

            print(f"✅ Inserted batch {i//batch_size + 1} of {(total_rows-1)//batch_size + 1}")

        print(f"�� Successfully migrated {total_rows} tracks to database!")
    except psycopg2.Error as e:
        print(f"❌ Error during migration: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()
        print("✅ Database connection closed")

if __name__ == "__main__":
    migrate_training_data()