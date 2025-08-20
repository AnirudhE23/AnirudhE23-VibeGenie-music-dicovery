import time
import pandas as pd
from spotify_utils import get_all_playlists, get_playlist_tracks
import config
import spotipy

# BATCH_SIZE = 50  # max 100 for Spotify API

# def collect_data(sp):
#     """Collect tracks + audio features and save to CSV."""
#     all_data = []
#     playlists = get_all_playlists(sp)

#     for playlist in playlists:
#         playlist_name = playlist['name']
#         playlist_id = playlist['id']
#         owner_id = playlist['owner']['id']

#         print(f"Fetching tracks from playlist: {playlist_name}")
#         tracks = get_playlist_tracks(sp, playlist_id)

#         track_ids_batch = []
#         track_info_batch = []

#         for item in tracks:
#             track = item['track']
#             if not track or track.get('is_local'):
#                 continue  # skip local tracks

#             track_id = track['id']
#             track_name = track['name']
#             artists = ", ".join([artist['name'] for artist in track['artists']])
#             album = track['album']['name']
#             popularity = track['popularity']

#             if track_id:
#                 track_ids_batch.append(track_id)
#                 track_info_batch.append({
#                     'playlist_name': playlist_name,
#                     'playlist_id': playlist_id,
#                     'owner_id': owner_id,
#                     'track_id': track_id,
#                     'track_name': track_name,
#                     'artists': artists,
#                     'album': album,
#                     'popularity': popularity
#                 })

#             # Process batch when it reaches BATCH_SIZE
#             if len(track_ids_batch) >= BATCH_SIZE:
#                 _process_batch(sp, track_ids_batch, track_info_batch, all_data)
#                 track_ids_batch, track_info_batch = [], []

#         # Process remaining tracks in batch
#         if track_ids_batch:
#             _process_batch(sp, track_ids_batch, track_info_batch, all_data)

#     df = pd.DataFrame(all_data)
#     df.to_csv(config.CSV_OUTPUT, index=False)
#     print(f"Saved {len(df)} tracks to {config.CSV_OUTPUT}")


# def _process_batch(sp, track_ids, track_info_batch, all_data):
#     """Fetch audio features for a batch of tracks and append to all_data."""
#     try:
#         features_list = sp.audio_features(track_ids)
#         for info, features in zip(track_info_batch, features_list):
#             if features:
#                 info.update({k: features[k] for k in features if k != 'type'})
#                 all_data.append(info)
#     except spotipy.exceptions.SpotifyException as e:
#         print(f"Skipping batch due to Spotify API error: {e}")

#     time.sleep(0.1)  # slight delay to avoid rate limiting

def collect_data(sp):
    all_data = []
    playlists = sp.current_user_playlists()['items']

    for playlist in playlists:
        playlist_name = playlist['name']
        playlist_id = playlist['id']
        owner_id = playlist['owner']['id']

        print(f"Fetching tracks from playlist: {playlist_name}")
        results = sp.playlist_tracks(playlist_id)
        tracks = results['items']

        for item in tracks:
            track = item['track']
            if not track:
                continue
            track_id = track['id']
            if not track_id:  # skip local/private tracks
                continue

            track_name = track['name']
            artists = ", ".join([artist['name'] for artist in track['artists']])
            album = track['album']['name']
            popularity = track['popularity']

            try:
                features = sp.audio_features(track_id)[0]
            except Exception as e:
                print(f"Skipping track {track_id} due to error: {e}")
                continue

            if not features:
                continue

            all_data.append({
                'playlist_name': playlist_name,
                'playlist_id': playlist_id,
                'owner_id': owner_id,
                'track_id': track_id,
                'track_name': track_name,
                'artists': artists,
                'album': album,
                'popularity': popularity,
                **{key: features[key] for key in features if key != 'type'}
            })
            time.sleep(0.05)

    df = pd.DataFrame(all_data)
    df.to_csv("spotify_tracks.csv", index=False)
    print(f"Saved {len(df)} tracks to spotify_tracks.csv")