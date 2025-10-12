-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    spotify_user_id VARCHAR(255) UNIQUE NOT NULL,
    display_name VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create tracks table  
CREATE TABLE IF NOT EXISTS tracks (
    id SERIAL PRIMARY KEY,
    spotify_track_id VARCHAR(255) UNIQUE NOT NULL,
    track_name VARCHAR(500),
    artists TEXT,
    popularity INTEGER,
    acousticness FLOAT,
    danceability FLOAT,
    energy FLOAT,
    instrumentalness FLOAT,
    key_value FLOAT,
    liveness FLOAT,
    loudness FLOAT,
    mode_value FLOAT,
    speechiness FLOAT,
    tempo FLOAT,
    valence FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create user_tracks table
CREATE TABLE IF NOT EXISTS user_tracks (
    id SERIAL PRIMARY KEY,
    spotify_user_id VARCHAR(255) NOT NULL,
    spotify_track_id VARCHAR(255) NOT NULL,
    track_name VARCHAR(500),
    artists TEXT,
    acousticness FLOAT,
    danceability FLOAT,
    energy FLOAT,
    instrumentalness FLOAT,
    key_value FLOAT,
    liveness FLOAT,
    loudness FLOAT,
    mode_value FLOAT,
    speechiness FLOAT,
    tempo FLOAT,
    valence FLOAT,
    playlist_name VARCHAR(500),
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(spotify_user_id, spotify_track_id)
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_users_spotify_id ON users(spotify_user_id);
CREATE INDEX IF NOT EXISTS idx_tracks_spotify_id ON tracks(spotify_track_id);
CREATE INDEX IF NOT EXISTS idx_user_tracks_user_id ON user_tracks(spotify_user_id);
CREATE INDEX IF NOT EXISTS idx_user_tracks_track_id ON user_tracks(spotify_track_id);