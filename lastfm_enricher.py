#!/usr/bin/env python3
"""Last.fm API enrichment for the music catalog.

Fetches genre tags and listener/play counts for all unique artists.
Results are cached locally in lastfm_cache.json (TTL = 30 days) to
minimise API calls on repeated scans.

Last.fm free API limits: 5 requests/second, no daily cap.
API key: https://www.last.fm/api/account/create  (free, instant)

Usage (from catalog_builder or query_engine):
    from lastfm_enricher import enrich_catalog_artists, load_cache

    # enrich after scan:
    cache = enrich_catalog_artists(api_key, artist_map, data_dir, cb)

    # at query time (read-only, instant):
    cache = load_cache(data_dir)
    info  = cache.get("the beatles", {})
    tags  = info.get("tags", [])         # ["classic rock", "pop", "60s"]
    listeners = info.get("listeners", 0) # 7 800 000
"""

import json
import os
import time
import requests
from datetime import date, timedelta

LASTFM_API_URL  = "https://ws.audioscrobbler.com/2.0/"
CACHE_FILENAME  = "lastfm_cache.json"
CACHE_TTL_DAYS  = 30
_REQ_INTERVAL   = 0.22   # ~4.5 req/s — safely under the 5 req/s limit
_MAX_TAGS       = 8      # keep only the most relevant tags per artist
_SAVE_EVERY     = 25     # persist cache every N new fetches


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

def _cache_path(data_dir: str) -> str:
    return os.path.join(data_dir, CACHE_FILENAME)


def load_cache(data_dir: str) -> dict:
    """Load the local Last.fm cache.  Returns {} if file is missing or corrupt."""
    path = _cache_path(data_dir)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache(data_dir: str, cache: dict) -> None:
    os.makedirs(data_dir, exist_ok=True)
    tmp = _cache_path(data_dir) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    os.replace(tmp, _cache_path(data_dir))


def _is_fresh(entry: dict) -> bool:
    """True if the cache entry was fetched within CACHE_TTL_DAYS."""
    fetched = entry.get("fetched", "")
    if not fetched:
        return False
    try:
        return (date.today() - date.fromisoformat(fetched)).days < CACHE_TTL_DAYS
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Last.fm API calls
# ---------------------------------------------------------------------------

def _fetch_artist(api_key: str, artist_name: str) -> dict | None:
    """Call artist.getinfo.  Returns enriched dict or None on error."""
    params = {
        "method":      "artist.getinfo",
        "artist":      artist_name,
        "api_key":     api_key,
        "format":      "json",
        "autocorrect": "1",
    }
    try:
        resp = requests.get(LASTFM_API_URL, params=params, timeout=10)
        data = resp.json()
    except Exception:
        return None

    if "error" in data or resp.status_code != 200:
        return None

    artist = data.get("artist", {})
    tags = [
        t["name"].lower()
        for t in artist.get("tags", {}).get("tag", [])
    ][:_MAX_TAGS]
    stats = artist.get("stats", {})
    return {
        "tags":      tags,
        "listeners": int(stats.get("listeners", 0)),
        "playcount": int(stats.get("playcount", 0)),
    }


# ---------------------------------------------------------------------------
# Main enrichment entry point
# ---------------------------------------------------------------------------

def enrich_catalog_artists(
    api_key: str,
    artist_map: dict,
    data_dir: str,
    progress_cb=None,
) -> dict:
    """Fetch / refresh Last.fm data for every artist in artist_map.

    Uses local cache (lastfm_cache.json).  Only fetches artists whose
    cache entry is absent or older than CACHE_TTL_DAYS.

    Args:
        api_key:     Last.fm API key (from settings).
        artist_map:  {artist_name_lower: {name, indices}} — output of
                     query_engine.build_artist_index().
        data_dir:    Directory for lastfm_cache.json (same as catalog).
        progress_cb: Optional callback(current, total, message).

    Returns:
        Full cache dict (including pre-existing entries).
    """
    cache = load_cache(data_dir)
    artists = list(artist_map.items())
    total   = len(artists)
    fetched = 0

    for i, (key, info) in enumerate(artists):
        if _is_fresh(cache.get(key, {})):
            if progress_cb and (i + 1) % 50 == 0:
                progress_cb(i + 1, total,
                            f"Last.fm: {i+1}/{total} (кэш актуален)")
            continue

        result = _fetch_artist(api_key, info["name"])

        today = str(date.today())
        if result is not None:
            cache[key] = {**result, "name": info["name"], "fetched": today}
        else:
            # Record attempt so we don't hammer a bad artist name repeatedly
            entry = dict(cache.get(key, {}))
            entry.setdefault("tags", [])
            entry.setdefault("listeners", 0)
            entry.setdefault("playcount", 0)
            entry["fetched"] = today
            cache[key] = entry

        fetched += 1
        if fetched % _SAVE_EVERY == 0:
            _save_cache(data_dir, cache)

        if progress_cb:
            progress_cb(i + 1, total,
                        f"Last.fm: {i+1}/{total} — {info['name']}")

        time.sleep(_REQ_INTERVAL)

    _save_cache(data_dir, cache)
    return cache


# ---------------------------------------------------------------------------
# Helpers for query_engine
# ---------------------------------------------------------------------------

def format_listeners(n: int) -> str:
    """Human-readable listener count: 7800000 → '7.8M', 450000 → '450K'."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M слуш."
    if n >= 1_000:
        return f"{n // 1_000}K слуш."
    if n > 0:
        return f"{n} слуш."
    return ""
