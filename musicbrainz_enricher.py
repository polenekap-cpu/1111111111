#!/usr/bin/env python3
"""MusicBrainz metadata enrichment for the music catalog.

Fetches instrument credits and recording type (live vs studio) via the
MusicBrainz JSON API.  Results are cached locally in mb_cache.json.

MusicBrainz rate limit: 1 request/second — enforced below.
No API key required.  User-Agent must identify your application per
https://wiki.musicbrainz.org/MusicBrainz_API/Rate_Limiting

Cache key: "{artist_lower}|{title_lower}"
Cache TTL:  90 days (instrument data changes rarely)

Typical usage:
    # During background enrichment (slow — 1 req/s):
    from musicbrainz_enricher import enrich_catalog_mb
    cache = enrich_catalog_mb(catalog_index, data_dir, progress_cb=cb)

    # At query time (instant, read-only):
    from musicbrainz_enricher import load_mb_cache, get_tracks_with_instrument
    cache = load_mb_cache(data_dir)
    flute_indices = get_tracks_with_instrument("flute", catalog_index, cache)
"""

import json
import os
import re
import time

import requests

MB_API_URL     = "https://musicbrainz.org/ws/2/"
CACHE_FILENAME = "mb_cache.json"
CACHE_TTL_DAYS = 90
_REQ_INTERVAL  = 1.1       # seconds between requests (API limit: 1/s)
_SAVE_EVERY    = 20        # persist cache every N new fetches
_USER_AGENT    = (
    "AIPlaylistGenerator/1.0 "
    "(https://github.com/polenekap-cpu/1111111111)"
)

# ---------------------------------------------------------------------------
# Instrument tag normalisation map
# Covers English MB tags, transliterated variants and common synonyms.
# ---------------------------------------------------------------------------
_INSTR_NORM: dict[str, str] = {
    # Flute family
    "flute": "flute", "transverse flute": "flute", "piccolo": "flute",
    "alto flute": "flute", "bass flute": "flute", "pan flute": "flute",
    "recorder": "flute",
    # Violin / strings
    "violin": "violin", "fiddle": "violin",
    "viola": "violin",
    "cello": "cello", "violoncello": "cello",
    "double bass": "double bass", "upright bass": "double bass",
    "strings": "strings", "string section": "strings",
    # Piano / keyboard
    "piano": "piano", "grand piano": "piano", "upright piano": "piano",
    "electric piano": "piano", "keyboard": "piano",
    "harpsichord": "piano", "organ": "organ", "hammond organ": "organ",
    "accordion": "accordion",
    # Guitar family
    "guitar": "guitar", "acoustic guitar": "guitar",
    "electric guitar": "guitar", "classical guitar": "guitar",
    "bass guitar": "bass guitar", "electric bass": "bass guitar",
    # Brass
    "trumpet": "trumpet", "cornet": "trumpet", "flugelhorn": "trumpet",
    "trombone": "trombone", "horn": "horn", "french horn": "horn",
    "tuba": "tuba",
    # Woodwind
    "saxophone": "saxophone", "alto saxophone": "saxophone",
    "tenor saxophone": "saxophone", "soprano saxophone": "saxophone",
    "baritone saxophone": "saxophone",
    "clarinet": "clarinet", "bass clarinet": "clarinet",
    "oboe": "oboe", "bassoon": "bassoon",
    # Percussion / drums
    "drums": "drums", "drum kit": "drums", "percussion": "percussion",
    "snare drum": "drums", "bass drum": "drums",
    # Other
    "harp": "harp", "banjo": "banjo", "mandolin": "mandolin",
    "ukulele": "ukulele", "synthesizer": "synthesizer",
    "theremin": "theremin", "bagpipes": "bagpipes",
}


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_path(data_dir: str) -> str:
    return os.path.join(data_dir, CACHE_FILENAME)


def load_mb_cache(data_dir: str) -> dict:
    """Load mb_cache.json.  Returns {} if missing or corrupt."""
    path = _cache_path(data_dir)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_mb_cache(data_dir: str, cache: dict) -> None:
    os.makedirs(data_dir, exist_ok=True)
    tmp = _cache_path(data_dir) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    os.replace(tmp, _cache_path(data_dir))


def _is_fresh(entry: dict) -> bool:
    fetched = entry.get("fetched", "")
    if not fetched:
        return False
    try:
        from datetime import date
        return (date.today() - date.fromisoformat(fetched)).days < CACHE_TTL_DAYS
    except (ValueError, ImportError):
        return False


def cache_key(artist: str, title: str) -> str:
    """Normalised cache key for artist + title."""
    return f"{artist.strip().lower()}|{title.strip().lower()}"


# ---------------------------------------------------------------------------
# MusicBrainz API
# ---------------------------------------------------------------------------

def _escape_lucene(s: str) -> str:
    """Escape Lucene special characters for MB search queries."""
    return re.sub(r'([+\-&|!(){}\[\]^"~*?:\\/])', r'\\\1', s)


def _search_recording(artist: str, title: str) -> dict | None:
    """Query MusicBrainz recording search.

    Returns a normalised dict:
        {instruments: list[str], tags: list[str], is_live: bool, mb_id: str}
    or None on network/parse error.
    """
    query = (
        f'artist:"{_escape_lucene(artist)}" AND '
        f'recording:"{_escape_lucene(title)}"'
    )
    params = {"query": query, "limit": 5, "fmt": "json"}
    headers = {"User-Agent": _USER_AGENT}

    try:
        resp = requests.get(
            MB_API_URL + "recording/",
            params=params, headers=headers, timeout=15,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
    except Exception:
        return None

    recordings = data.get("recordings", [])
    if not recordings:
        return None

    rec = recordings[0]   # highest-score result

    # --- Instrument extraction from tags ---
    raw_tags   = [t["name"].lower() for t in rec.get("tags", [])]
    instruments: list[str] = []
    seen: set[str] = set()
    for tag in raw_tags:
        norm = _INSTR_NORM.get(tag)
        if norm and norm not in seen:
            instruments.append(norm)
            seen.add(norm)

    # --- Live vs studio ---
    releases = rec.get("releases", [])
    is_live  = False
    for rel in releases[:5]:
        rg_type = (rel.get("release-group") or {}).get("primary-type", "")
        if "live" in rg_type.lower():
            is_live = True
            break
        if "live" in rel.get("title", "").lower():
            is_live = True
            break

    return {
        "instruments": instruments,
        "tags":        raw_tags[:10],
        "is_live":     is_live,
        "mb_id":       rec.get("id", ""),
    }


# ---------------------------------------------------------------------------
# Main enrichment entry point
# ---------------------------------------------------------------------------

def enrich_catalog_mb(
    catalog_index: dict,
    data_dir: str,
    progress_cb=None,
    max_tracks: int = 500,
) -> dict:
    """Incrementally enrich catalog with MusicBrainz instrument data.

    Processes up to *max_tracks* new entries per call so the operation
    finishes in a reasonable time (≈ 8–9 minutes for 500 tracks).
    Call again to continue; already-cached entries are skipped.

    Args:
        catalog_index: {index: {artist, title, path, ...}}
        data_dir:      directory for mb_cache.json
        progress_cb:   optional callback(current, total, message)
        max_tracks:    maximum number of *new* tracks to fetch per call

    Returns:
        Full cache dict (including pre-existing entries).
    """
    cache     = load_mb_cache(data_dir)
    tracks    = [
        (idx, t) for idx, t in catalog_index.items()
        if t.get("artist") and t.get("title")
    ]
    total     = len(tracks)
    fetched   = 0
    today     = time.strftime("%Y-%m-%d")

    for i, (_, track) in enumerate(tracks):
        key = cache_key(track["artist"], track["title"])

        if _is_fresh(cache.get(key, {})):
            continue

        result = _search_recording(track["artist"], track["title"])

        if result is not None:
            cache[key] = {**result, "fetched": today}
        else:
            # Record attempt (empty) so we don't hammer the API on every run
            cache[key] = {
                "instruments": [], "tags": [],
                "is_live": False, "mb_id": "",
                "fetched": today,
            }

        fetched += 1

        if fetched % _SAVE_EVERY == 0:
            _save_mb_cache(data_dir, cache)

        if progress_cb:
            progress_cb(
                i + 1, total,
                f"MusicBrainz: {i+1}/{total} — "
                f"{track['artist']} – {track['title']}",
            )

        if fetched >= max_tracks:
            if progress_cb:
                progress_cb(
                    i + 1, total,
                    f"MusicBrainz: пауза ({max_tracks} треков обработано). "
                    "Запустите снова для продолжения.",
                )
            break

        time.sleep(_REQ_INTERVAL)

    _save_mb_cache(data_dir, cache)

    if progress_cb:
        progress_cb(
            total, total,
            f"MusicBrainz: готово. Записей в кэше: {len(cache):,}",
        )

    return cache


# ---------------------------------------------------------------------------
# Query-time helpers
# ---------------------------------------------------------------------------

def get_tracks_with_instrument(
    instrument: str,
    catalog_index: dict,
    mb_cache: dict,
) -> list[int]:
    """Return catalog track indices confirmed to feature *instrument*.

    Args:
        instrument:    normalised instrument name (e.g. "flute", "violin")
        catalog_index: {index: {artist, title, ...}}
        mb_cache:      from load_mb_cache()

    Returns:
        List of integer track indices.  Empty list if cache is empty or
        no matches found.
    """
    if not mb_cache:
        return []

    result = []
    for idx, track in catalog_index.items():
        key   = cache_key(track.get("artist", ""), track.get("title", ""))
        entry = mb_cache.get(key, {})
        if instrument in entry.get("instruments", []):
            result.append(idx)

    return result


def get_studio_track_indices(
    catalog_index: dict,
    mb_cache: dict,
) -> set[int]:
    """Return indices of tracks confirmed as studio recordings (not live).

    Useful for de-prioritising live versions when the query doesn't ask
    for them explicitly.
    """
    studio = set()
    for idx, track in catalog_index.items():
        key   = cache_key(track.get("artist", ""), track.get("title", ""))
        entry = mb_cache.get(key, {})
        if entry and not entry.get("is_live", False):
            studio.add(idx)
    return studio
