import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

def migrate_tracks_to_neon():
    """
    Migrate tracks from local database to Neon PostgreSQL
    """
    # Local database connection
    local_conn = psycopg2.connect(
        host='localhost',
        port=5432,
        database='music_recommender',
        user='postgres',
        password=os.getenv('DB_PASSWORD')
    )
    
    # Neon database connection using DATABASE_URL
    neon_conn = psycopg2.connect(os.getenv('DATABASE_URL'))
    
    print("Connected to both databases!")
    
    # Get tracks from local
    with local_conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM tracks")
        total = cur.fetchone()[0]
        print(f"Found {total:,} tracks in local database")
    
    # Check Neon database
    with neon_conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM tracks")
        neon_count = cur.fetchone()[0]
        print(f"Neon database currently has {neon_count:,} tracks")
    
    # Transfer tracks in batches
    batch_size = 1000
    transferred = 0
    
    with local_conn.cursor() as src_cur, neon_conn.cursor() as dst_cur:
        src_cur.execute("""
            SELECT spotify_track_id, track_name, artists, popularity,
                   acousticness, danceability, energy, instrumentalness,
                   key_value, liveness, loudness, mode_value,
                   speechiness, tempo, valence
            FROM tracks
        """)
        
        print("Starting migration...")
        
        while True:
            rows = src_cur.fetchmany(batch_size)
            if not rows:
                break
            
            # Insert batch
            dst_cur.executemany("""
                INSERT INTO tracks (
                    spotify_track_id, track_name, artists, popularity,
                    acousticness, danceability, energy, instrumentalness,
                    key_value, liveness, loudness, mode_value,
                    speechiness, tempo, valence
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (spotify_track_id) DO NOTHING
            """, rows)
            
            neon_conn.commit()
            transferred += len(rows)
            print(f"Transferred {transferred:,} / {total:,} tracks ({transferred/total*100:.1f}%)")
    
    print(f"✅ Migration complete! {transferred:,} tracks transferred.")
    
    # Verify final count
    with neon_conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM tracks")
        final_count = cur.fetchone()[0]
        print(f"Neon database now has {final_count:,} tracks")
    
    local_conn.close()
    neon_conn.close()

if __name__ == "__main__":
    migrate_tracks_to_neon()