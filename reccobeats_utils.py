import requests
import pandas as pd
import streamlit as st
import time
import random
import re
import unicodedata
from rapidfuzz import fuzz
from concurrent.futures import ThreadPoolExecutor, as_completed
import config

def normalize_title(s: str, strip_parentheses: bool = False) -> str:
    """Normalize track title for matching."""
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
    """Normalize artist name for matching."""
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

def get_features_from_spotify_id(track_id: str):
    """Fast path: try audio-features with Spotify ID directly."""
    url = f"{config.RECCOBEATS_API_BASE}/audio-features?ids={track_id}"
    resp = requests.get(url, headers=config.RECCOBEATS_HEADERS, timeout=15)
    if resp.status_code == 200:
        js = resp.json()
        if js.get("content"):
            df = pd.DataFrame(js["content"])
            df["Track ID"] = track_id
            # Rename columns to match database schema
            if 'key' in df.columns:
                df = df.rename(columns={'key': 'key_value'})
            if 'mode' in df.columns:
                df = df.rename(columns={'mode': 'mode_value'})
            return df.drop(columns=["id"])
    return None

def get_features_from_reccoid(recco_id: str, spotify_track_id_for_join: str):
    """Fetch audio-features using a Reccobeats track id, attach back to the Spotify Track ID for merging."""
    url = f"{config.RECCOBEATS_API_BASE}/audio-features?ids={recco_id}"
    resp = requests.get(url, headers=config.RECCOBEATS_HEADERS, timeout=15)
    if resp.status_code == 200:
        js = resp.json()
        if js.get("content"):
            df = pd.DataFrame(js["content"])
            df["Track ID"] = spotify_track_id_for_join
            # Rename columns to match database schema
            if 'key' in df.columns:
                df = df.rename(columns={'key': 'key_value'})
            if 'mode' in df.columns:
                df = df.rename(columns={'mode': 'mode_value'})
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
                    f"{config.RECCOBEATS_API_BASE}/artist/search",
                    params={"searchText": var},
                    headers=config.RECCOBEATS_HEADERS,
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
                                f"{config.RECCOBEATS_API_BASE}/artist/{art_id}/track",
                                params={"page": page, "size": 50},
                                headers=config.RECCOBEATS_HEADERS,
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
                            f"{config.RECCOBEATS_API_BASE}/album/search",
                            params={"searchText": album_name},
                            headers=config.RECCOBEATS_HEADERS,
                            timeout=20
                        )
                        if alb_resp.status_code == 200:
                            for alb in alb_resp.json().get("content", []):
                                alb_id = alb.get("id")
                                if not alb_id:
                                    continue
                                # fetch tracks
                                tr_resp = requests.get(
                                    f"{config.RECCOBEATS_API_BASE}/album/{alb_id}/track",
                                    headers=config.RECCOBEATS_HEADERS,
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
                            f"{config.RECCOBEATS_API_BASE}/artist/{art_id}/track",
                            params={"page": page, "size": 50},
                            headers=config.RECCOBEATS_HEADERS,
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
    url = f"{config.RECCOBEATS_API_BASE}/audio-features?ids={','.join(track_ids)}"
    resp = requests.get(url, headers=config.RECCOBEATS_HEADERS, timeout=30)
    if resp.status_code == 200:
        js = resp.json()
        if js.get("content"):
            df = pd.DataFrame(js["content"])
            df["Track ID"] = df["id"]
            # Rename columns to match database schema
            if 'key' in df.columns:
                df = df.rename(columns={'key': 'key_value'})
            if 'mode' in df.columns:
                df = df.rename(columns={'mode': 'mode_value'})
            return df.drop(columns=["id"])
    elif resp.status_code == 429:
        retry_after = int(resp.headers.get("Retry-After", 5))
        st.warning(f"Rate limit hit. Retrying in {retry_after}s...")
        time.sleep(retry_after)
        return get_features_from_recco_batch(track_ids)
    return None
