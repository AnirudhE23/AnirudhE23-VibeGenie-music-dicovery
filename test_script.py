import os
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
import pandas as pd
import requests
import time
from rapidfuzz import fuzz
import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import re
import unicodedata

# Load environment variables
load_dotenv()

CLIENT_ID = os.environ.get('CLIENT_ID')
CLIENT_SECRET = os.environ.get('CLIENT_SECRET')
REDIRECT_URI = 'http://127.0.0.1:8501/callback'
SCOPE = 'playlist-read-private playlist-read-collaborative user-library-read user-top-read'

# --- Spotify Auth with per-user cache ---
temp_sp = spotipy.Spotify(
    auth_manager=SpotifyOAuth(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope=SCOPE,
        show_dialog=True
    )
)
username = temp_sp.current_user()["id"]
user_cache = f".cache-{username}"

auth_manager = SpotifyOAuth(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    redirect_uri=REDIRECT_URI,
    scope=SCOPE,
    cache_path=user_cache,
    show_dialog=True
)

sp = spotipy.Spotify(auth_manager=auth_manager)

CACHE_PATH = pathlib.Path("features_cache.parquet")

def load_features_cache():
    if CACHE_PATH.exists():
        return pd.read_parquet(CACHE_PATH)
    return pd.DataFrame()

def save_features_cache(df):
    df = df.drop_duplicates(subset=["Track ID"])
    df.to_parquet(CACHE_PATH, index=False)


def normalize_title(s: str, strip_parentheses: bool = False) -> str:
    if not s:
        return ""
    s = s.lower()

    # remove soundtrack/OST suffixes
    s = re.sub(r'\s*-\s*from\s+["\'].*?["\'].*$', '', s)
    s = re.sub(r'\s*-\s*original\s+soundtrack.*$', '', s)

    if strip_parentheses:
        s = re.sub(r'\s*[\(\[\{].*?[\)\]\}]', '', s)

    s = re.sub(r'\s*-\s*(remaster(ed)?(\s*\d{2,4})?|live|radio\s*edit|single\s*version|album\s*version|mono|stereo|deluxe|bonus\s*track).*$', '', s)
    s = re.sub(r'\s*(feat\.|ft\.|with)\s+.*$', '', s)
    s = re.sub(r'[^a-z0-9\s]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def normalize_artist(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")  # remove accents
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", "", s)  # drop symbols like $, ¥
    return re.sub(r"\s+", " ", s).strip()

# fallback alias map
ARTIST_ALIASES = {
    "kanye west": ["ye"],
    "ty dolla sign": ["ty dolla $ign"],
    "ty dolla $ign": ["ty dolla sign"],
    "¥$": ["ye"],
    "Â¥$": ["ye"],
}

# --- Reccobeats Helper Functions ---
API_BASE = "https://api.reccobeats.com/v1"
HEADERS = {'Accept': 'application/json'}

def get_features_from_spotify_id(track_id: str):
    """Fast path: try audio-features with Spotify ID directly."""
    url = f"{API_BASE}/audio-features?ids={track_id}"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    if resp.status_code == 200:
        js = resp.json()
        if js.get("content"):
            df = pd.DataFrame(js["content"])
            df["Track ID"] = track_id
            return df.drop(columns=["id"])
    return None

def get_features_from_reccoid(recco_id: str, spotify_track_id_for_join: str):
    """Fetch audio-features using a Reccobeats track id, attach back to the Spotify Track ID for merging."""
    url = f"{API_BASE}/audio-features?ids={recco_id}"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    if resp.status_code == 200:
        js = resp.json()
        if js.get("content"):
            df = pd.DataFrame(js["content"])
            df["Track ID"] = spotify_track_id_for_join
            return df.drop(columns=["id"])
    return None

def find_recco_id_from_artist(
    spotify_id: str,
    artist_name: str,
    track_name: str,
    album_name: str = None,
    isrc: str = None,
    threshold: int = 80,
    soft_threshold: int = 60,
    max_pages: int = 25,
    album_first: bool = True
):
    """
    Locate a Reccobeats track ID.
    Priority:
      1. ISRC match (exact, fastest & most reliable)
      2. Album search (if album_name given)
      3. Artist catalog scan with exact & fuzzy title matching
    """

    if not artist_name or not track_name:
        return None

    wanted = normalize_title(track_name)
    wanted_no_parens = normalize_title(track_name, strip_parentheses=True)

    for candidate_artist in artist_name.split(", "):
        variants = {candidate_artist, normalize_artist(candidate_artist)}

        for var in variants:
            try:
                # Artist search
                search_resp = requests.get(
                    f"{API_BASE}/artist/search",
                    params={"searchText": var},
                    headers=HEADERS,
                    timeout=15
                )
                if search_resp.status_code != 200:
                    continue
                artists_json = search_resp.json()
                if not artists_json.get("content"):
                    continue

                for art in artists_json["content"]:
                    art_id = art.get("id")
                    if not art_id:
                        continue

                    # 1. ISRC first (artist tracks)
                    if isrc:
                        for page in range(min(3, max_pages)):
                            page_resp = requests.get(
                                f"{API_BASE}/artist/{art_id}/track",
                                params={"page": page, "size": 50},
                                headers=HEADERS,
                                timeout=15
                            )
                            if page_resp.status_code != 200:
                                break
                            for t in page_resp.json().get("content", []):
                                if t.get("isrc") == isrc:
                                    return t.get("id")

                    # 2. Album search
                    if album_first and album_name:
                        alb_resp = requests.get(
                            f"{API_BASE}/album/search",
                            params={"searchText": album_name},
                            headers=HEADERS,
                            timeout=20
                        )
                        if alb_resp.status_code == 200:
                            for alb in alb_resp.json().get("content", []):
                                alb_id = alb.get("id")
                                if not alb_id:
                                    continue
                                # fetch tracks
                                tr_resp = requests.get(
                                    f"{API_BASE}/album/{alb_id}/track",
                                    headers=HEADERS,
                                    timeout=20
                                )
                                if tr_resp.status_code != 200:
                                    continue
                                for t in tr_resp.json().get("content", []):
                                    title, tid = t.get("trackTitle"), t.get("id")
                                    if not title or not tid:
                                        continue

                                    norm_title = normalize_title(title)
                                    norm_title_no_parens = normalize_title(title, strip_parentheses=True)
                                    score = fuzz.ratio(wanted, norm_title)

                                    # Exact match first
                                    if norm_title == wanted or (soft_threshold <= score and score >= threshold):
                                        return tid
                                    # ISRC match
                                    if isrc and t.get("isrc") == isrc:
                                        return tid
                                    # Soft fuzzy with artist overlap
                                    if score >= soft_threshold:
                                        track_artists = [normalize_artist(a.get("name", "")) for a in t.get("artists", [])]
                                        if any(fuzz.ratio(normalize_artist(candidate_artist), a) > 70 for a in track_artists):
                                            return tid
                                    # Backup: match ignoring parentheses
                                    score_no_parens = fuzz.ratio(wanted_no_parens, norm_title_no_parens)
                                    if score_no_parens >= soft_threshold:
                                        track_artists = [normalize_artist(a.get("name", "")) for a in t.get("artists", [])]
                                        if any(fuzz.ratio(normalize_artist(candidate_artist), a) > 70 for a in track_artists):
                                            return tid

                    # 3. Artist page fallback
                    for page in range(max_pages):
                        page_resp = requests.get(
                            f"{API_BASE}/artist/{art_id}/track",
                            params={"page": page, "size": 50},
                            headers=HEADERS,
                            timeout=20
                        )
                        if page_resp.status_code != 200:
                            break
                        page_items = page_resp.json().get("content") or []
                        if not page_items:
                            break

                        for t in page_items:
                            title, tid = t.get("trackTitle"), t.get("id")
                            if not title or not tid:
                                continue

                            norm_title = normalize_title(title)
                            norm_title_no_parens = normalize_title(title, strip_parentheses=True)
                            score = fuzz.ratio(wanted, norm_title)

                            if norm_title == wanted or (score >= threshold):
                                return tid
                            # Soft fuzzy with artist overlap
                            if score >= soft_threshold:
                                track_artists = [normalize_artist(a.get("name", "")) for a in t.get("artists", [])]
                                if any(fuzz.ratio(normalize_artist(candidate_artist), a) > 70 for a in track_artists):
                                    return tid
                            # Backup: ignore parentheses
                            score_no_parens = fuzz.ratio(wanted_no_parens, norm_title_no_parens)
                            if score_no_parens >= soft_threshold:
                                track_artists = [normalize_artist(a.get("name", "")) for a in t.get("artists", [])]
                                if any(fuzz.ratio(normalize_artist(candidate_artist), a) > 70 for a in track_artists):
                                    return tid

            except Exception:
                continue

    return None


def get_reccobeats_features(track_id: str, track_name: str, artist_name: str, spotify_to_recco: dict):
    """
    Full chain:
    1) Try Spotify ID directly against /audio-features
    2) If miss, try artist->tracks to find a recco-id, then fetch via recco-id.
    """
    # 1) Try direct Spotify ID
    features = get_features_from_spotify_id(track_id)
    if features is not None:
        return features

    # 2) If we already mapped this Spotify ID to a Reccobeats ID, reuse it
    recco_id = spotify_to_recco.get(track_id)
    if recco_id:
        features = get_features_from_reccoid(recco_id, track_id)
        if features is not None:
            return features

    # 3) Artist->Track lookup to get a Reccobeats ID
    recco_id = find_recco_id_from_artist(track_id, artist_name, track_name)
    if recco_id:
        spotify_to_recco[track_id] = recco_id
        return get_features_from_reccoid(recco_id, track_id)

    return None


def get_features_from_recco_batch(track_ids):
    """Batch attempt with Spotify IDs (some will resolve, some won't)."""
    if not track_ids:
        return None
    url = f"{API_BASE}/audio-features?ids={','.join(track_ids)}"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    if resp.status_code == 200:
        js = resp.json()
        if js.get("content"):
            df = pd.DataFrame(js["content"])
            df["Track ID"] = df["id"]
            return df.drop(columns=["id"])
    elif resp.status_code == 429:
        retry_after = int(resp.headers.get("Retry-After", 5))
        st.warning(f"Rate limit hit. Retrying in {retry_after}s...")
        time.sleep(retry_after)
        return get_features_from_recco_batch(track_ids)
    return None

def collect_user_tracks():
    """
    Fetches:
    1. Current user's playlists
    2. Current user's liked songs
    3. Current user's top tracks (short, medium, long term)
    Returns a single DataFrame with audio features (Spotify + Reccobeats)
    """
    cache_df = load_features_cache()
    cached_ids = set(cache_df["Track ID"].unique()) if not cache_df.empty else set()
    all_tracks = []

    # --- 1. Playlists ---
    playlists = sp.current_user_playlists(limit=50)["items"]
    for playlist in playlists:
        playlist_name = playlist["name"]
        owner_id = playlist["owner"]["id"]
        playlist_id = playlist["id"]

        results = sp.playlist_tracks(playlist_id, limit=100)
        while results:
            for item in results["items"]:
                track = item["track"]
                if not track or not track.get("id"):
                    continue
                all_tracks.append({
                    "Track Name": track["name"],
                    "Album Name": track["album"]["name"],
                    "Artist(s)": ", ".join([a["name"] for a in track["artists"]]),
                    "Playlist Name": playlist_name if playlist_name else "N/A",
                    "Track ID": track["id"],
                    "Popularity": track.get("popularity", 0),
                    "Owner ID": owner_id,
                    "ISRC": (track.get("external_ids") or {}).get("isrc")
                })
            results = sp.next(results) if results.get("next") else None

    # --- 2. Liked songs ---
    offset = 0
    while True:
        results = sp.current_user_saved_tracks(limit=50, offset=offset)
        if not results["items"]:
            break
        for item in results["items"]:
            track = item["track"]
            if not track or not track.get("id"):
                continue
            all_tracks.append({
                "Track Name": track["name"],
                "Album Name": track["album"]["name"],
                "Artist(s)": ", ".join([a["name"] for a in track["artists"]]),
                "Playlist Name": "Liked Songs",
                "Track ID": track["id"],
                "Popularity": track.get("popularity", 0),
                "Owner ID": track.get("album", {}).get("id", ""),
                "ISRC": (track.get("external_ids") or {}).get("isrc")
            })
        offset += 50
        if offset >= results["total"]:
            break

    # --- 3. Top tracks (short, medium, long term) ---
    for term in ["short_term", "medium_term", "long_term"]:
        results = sp.current_user_top_tracks(limit=50, time_range=term)
        for track in results["items"]:
            if not track or not track.get("id"):
                continue
            all_tracks.append({
                "Track Name": track["name"],
                "Album Name": track["album"]["name"],
                "Artist(s)": ", ".join([a["name"] for a in track["artists"]]),
                "Playlist Name": f"Top Tracks ({term})",
                "Track ID": track["id"],
                "Popularity": track.get("popularity", 0),
                "Owner ID": track.get("album", {}).get("id", ""),
                "ISRC": (track.get("external_ids") or {}).get("isrc")
            })

    tracks_df = pd.DataFrame(all_tracks).drop_duplicates(subset=["Track ID"])

    # Add this debugging code after line 405
    st.write("=== DEBUG: Checking for Danger Zone ===")
    danger_zone_tracks = tracks_df[tracks_df["Track Name"].str.contains("Danger Zone", case=False, na=False)]
    if not danger_zone_tracks.empty:
        st.write(f"Found {len(danger_zone_tracks)} Danger Zone tracks in collection:")
        for _, track in danger_zone_tracks.iterrows():
            st.write(f"  - '{track['Track Name']}' by '{track['Artist(s)']}' (ID: {track['Track ID']})")
            st.write(f"    In cache: {track['Track ID'] in cached_ids}")
    else:
        st.write("❌ No Danger Zone tracks found in collection!")

    # --- Which tracks still need features ---
    need_ids = [tid for tid in tracks_df["Track ID"].dropna().unique().tolist() if tid not in cached_ids]
    
    st.write(f"Total tracks: {len(tracks_df)}")
    st.write(f"Already cached: {len(cached_ids)}")
    st.write(f"Need features: {len(need_ids)}")

    # ISRC dedupe - REMOVED: This was too aggressive and filtering out valid tracks
    # The original logic was removing tracks that share ISRCs, but we want to try all tracks
    # and let the Reccobeats API handle duplicates

    # --- Batch fetch Spotify IDs ---
    batch_size = 40
    if need_ids:
        batch_progress = st.progress(0, text="Fetching audio features (batch)")
        status = st.empty()
        for i in range(0, len(need_ids), batch_size):
            batch = need_ids[i:i+batch_size]
            features = get_features_from_recco_batch(batch)
            if features is not None and not features.empty:
                cache_df = pd.concat([cache_df, features], ignore_index=True) if not cache_df.empty else features
                save_features_cache(cache_df)
                st.write(f"Cached features for {len(features)} tracks (total cache: {len(cache_df)})")
            batch_progress.progress(min((i + batch_size) / len(need_ids), 1.0))
            status.text(f"Batch processed {min(i+batch_size, len(need_ids))}/{len(need_ids)} tracks")

    features_df = cache_df.copy()
    have_ids = set(features_df["Track ID"].unique()) if not features_df.empty else set()
    remaining = tracks_df[~tracks_df["Track ID"].isin(have_ids)].copy()

    # Add this debugging code after line 430
    st.write("=== DEBUG: Checking remaining tracks ===")
    danger_zone_remaining = remaining[remaining["Track Name"].str.contains("Danger Zone", case=False, na=False)]
    if not danger_zone_remaining.empty:
        st.write(f"✅ Danger Zone is in remaining tracks ({len(danger_zone_remaining)} versions)")
        for _, track in danger_zone_remaining.iterrows():
            st.write(f"  - '{track['Track Name']}' by '{track['Artist(s)']}' (ID: {track['Track ID']})")
    else:
        st.write("❌ Danger Zone not in remaining tracks")

    # --- Fallback pass (artist->track walk, enhanced) ---
    # Process ALL remaining tracks, not just a subset
    if len(remaining) > 0:
        # Sort by popularity to prioritize popular tracks first
        remaining = remaining.sort_values("Popularity", ascending=False)
        fb_progress = st.progress(0, text=f"Fallback lookups ({len(remaining)} tracks)")
        fb_status = st.empty()
        spotify_to_recco = {}
        fallback_results = []

        def run_fallback(row): 
            """Return a FEATURES DATAFRAME or None.""" 
            time.sleep(random.uniform(0.05, 0.15)) 
            
            # Log what we're trying to find
            st.write(f"Trying to find: '{row['Track Name']}' by '{row['Artist(s)']}'")
            
            result = get_reccobeats_features( 
                track_id=row["Track ID"], 
                track_name=row["Track Name"], 
                artist_name=row["Artist(s)"], 
                spotify_to_recco=spotify_to_recco
            )
            
            if result is not None:
                st.write(f"✓ Found features for: '{row['Track Name']}'")
            else:
                st.write(f"✗ No features found for: '{row['Track Name']}'")
                
            return result

        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = {ex.submit(run_fallback, row): idx for idx, row in remaining.iterrows()}
            done = 0
            for f in as_completed(futures):
                try:
                    res = f.result()
                    if res is not None and not res.empty:
                        fallback_results.append(res)
                        cache_df = pd.concat([cache_df, res], ignore_index=True) if not cache_df.empty else res
                        save_features_cache(cache_df)
                except Exception:
                    pass
                finally:
                    done += 1
                    fb_progress.progress(done / len(remaining))
                    fb_status.text(f"Fallback processed {done}/{len(remaining)}")

        if fallback_results:
            fb_df = pd.concat(fallback_results, ignore_index=True)
            features_df = pd.concat([features_df, fb_df], ignore_index=True).drop_duplicates(subset=["Track ID"])

    # --- Merge everything into final dataframe ---
    final_df = pd.merge(tracks_df.drop(columns=["ISRC"]), features_df, on="Track ID", how="left")
    final_df.to_csv("spotify_tracks.csv", index=True)

    feature_cols = [c for c in features_df.columns if c != "Track ID"] if not features_df.empty else []
    rows_with_features = final_df[feature_cols].notna().any(axis=1).sum() if feature_cols else 0
    cached_unique = features_df["Track ID"].nunique() if not features_df.empty else 0

    st.write(f"Final rows: {len(final_df)}. With features: {rows_with_features} (unique cached tracks: {cached_unique})")
    return final_df


# --- Streamlit UI ---
st.set_page_config(page_title="Spotify Data Extractor", page_icon="🎵")
st.title("Spotify Data Extractor")

# Add debugging section
st.header("Debug Individual Song")
if st.button("Test Danger Zone"):
    # Test the find_recco_id_from_artist function directly
    recco_id = find_recco_id_from_artist(
        spotify_id="test",
        artist_name="Kenny Loggins", 
        track_name="Danger Zone"
    )
    if recco_id:
        st.success(f"Found Reccobeats ID: {recco_id}")
        # Get features
        features = get_features_from_reccoid(recco_id, "test")
        if features is not None:
            st.write("Audio Features:", features)
        else:
            st.error("Could not fetch features")
    else:
        st.error("Could not find Reccobeats ID for Danger Zone")

if st.button("Fetch Playlists and Save CSV"):
    df = collect_user_tracks()
    st.success(f"Extracted {len(df)} tracks. Saved to spotify_user_tracks.csv")
    st.dataframe(df.head())

if st.button("Logout"):
    if os.path.exists(user_cache):
        os.remove(user_cache)
        st.success("You have been logged out. Please refresh to log in again.")
    st.stop()
