def get_all_playlists(sp):

    playlists = []
    results = sp.current_user_playlists(limit=50)
    while results:
        playlists.extend(results['items'])
        if results['next']:
            results = sp.next(results)
        else:
            results = None
    return playlists

def get_playlist_tracks(sp, playlist_id):
    tracks = []
    results = sp.playlist_tracks(playlist_id, limit=100)
    while results:
        tracks.extend(results['items'])
        if results['next']:
            results = sp.next(results)
        else:
            results = None
    return tracks