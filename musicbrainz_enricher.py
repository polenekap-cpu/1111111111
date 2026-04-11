#!/usr/bin/env python3
"""MusicBrainz metadata enrichment for the music catalog.

Fetches instrument credits and recording type (live vs studio) via the
MusicBrainz JSON API.  Results are cached locally in mb_cache.json.

Rate limit: 1 request/second (enforced globally via _RateLimiter).
No API key required.

KEY DESIGN: Per-artist batch strategy
--------------------------------------
Naively fetching one recording at a time would cost:
    40 000 tracks × 1.1 s/req  ≈  12 hours

Instead we query once per unique ARTIST and get up to 100 recordings
with tags in a single response:
    2 000 artists × 1.1 s/req  ≈  36 minutes  (20× speedup)

Additionally, get_instrument_tracks_global() does a single "tag:flute"
search to find well-known instrument tracks instantly (no pre-enrichment
needed) — useful at query time when the cache is still being built.

Cache key: "{artist_lower}|{title_lower}"
Cache TTL:  90 days
"""

import json
import os
import re
import threading
import time

import requests

MB_API_URL     = "https://musicbrainz.org/ws/2/"
CACHE_FILENAME = "mb_cache.json"
CACHE_TTL_DAYS = 90
_REQ_INTERVAL  = 1.1        # global minimum gap between requests (s)
_SAVE_EVERY    = 25         # persist cache every N artists
_ARTIST_LIMIT  = 100        # recordings fetched per artist per page
_USER_AGENT    = (
    "AIPlaylistGenerator/1.0 "
    "(https://github.com/polenekap-cpu/1111111111)"
)

# ---------------------------------------------------------------------------
# Instrument tag normalisation (50+ English MB tag synonyms → canonical name)
# ---------------------------------------------------------------------------
_INSTR_NORM: dict[str, str] = {
    # Flute family
    "flute": "flute", "transverse flute": "flute", "concert flute": "flute",
    "alto flute": "flute", "bass flute": "flute", "piccolo": "flute",
    "pan flute": "flute", "recorder": "flute", "ocarina": "flute",
    # Violin / bowed strings
    "violin": "violin", "fiddle": "violin",
    "viola": "viola",
    "cello": "cello", "violoncello": "cello",
    "double bass": "double bass", "upright bass": "double bass",
    "strings": "strings", "string section": "strings",
    "string quartet": "strings",
    # Piano / keyboard
    "piano": "piano", "grand piano": "piano", "upright piano": "piano",
    "electric piano": "piano", "keyboard": "piano",
    "harpsichord": "harpsichord", "clavichord": "harpsichord",
    "organ": "organ", "hammond organ": "organ", "pipe organ": "organ",
    "accordion": "accordion", "bandoneon": "accordion",
    # Guitar
    "guitar": "guitar", "acoustic guitar": "guitar",
    "electric guitar": "guitar", "classical guitar": "guitar",
    "twelve-string guitar": "guitar", "steel guitar": "guitar",
    "bass guitar": "bass guitar", "electric bass": "bass guitar",
    # Brass
    "trumpet": "trumpet", "cornet": "trumpet", "flugelhorn": "trumpet",
    "trombone": "trombone", "french horn": "horn", "horn": "horn",
    "tuba": "tuba", "euphonium": "tuba",
    # Woodwind
    "saxophone": "saxophone", "alto saxophone": "saxophone",
    "tenor saxophone": "saxophone", "soprano saxophone": "saxophone",
    "baritone saxophone": "saxophone", "bass saxophone": "saxophone",
    "clarinet": "clarinet", "bass clarinet": "clarinet",
    "oboe": "oboe", "english horn": "oboe",
    "bassoon": "bassoon", "contrabassoon": "bassoon",
    # Percussion
    "drums": "drums", "drum kit": "drums", "drum set": "drums",
    "percussion": "percussion", "timpani": "percussion",
    "xylophone": "percussion", "marimba": "percussion",
    "vibraphone": "percussion",
    # Other
    "harp": "harp", "banjo": "banjo", "mandolin": "mandolin",
    "ukulele": "ukulele", "lute": "lute",
    "synthesizer": "synthesizer", "synth": "synthesizer",
    "theremin": "theremin", "bagpipes": "bagpipes",
    "sitar": "sitar", "tabla": "tabla",
}

# Suffix patterns that appear in local files but not in MB canonical titles
_SUFFIX_RE = re.compile(
    r"\s*[\(\[](live|concert|remaster(?:ed)?|acoustic|bonus|demo|"
    r"radio\s*edit|single\s*version|deluxe|extended|"
    r"концерт|ремастер|акустика)[\)\]]?\s*$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Rate limiter (global, thread-safe)
# ---------------------------------------------------------------------------

class _RateLimiter:
    """Ensures at most one request every _REQ_INTERVAL seconds."""

    def __init__(self, interval: float = _REQ_INTERVAL):
        self._lock     = threading.Lock()
        self._last     = 0.0
        self._interval = interval

    def wait(self) -> None:
        with self._lock:
            now     = time.monotonic()
            elapsed = now - self._last
            if elapsed < self._interval:
                time.sleep(self._interval - elapsed)
            self._last = time.monotonic()


_limiter = _RateLimiter()


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
    """Normalised cache key."""
    return f"{artist.strip().lower()}|{title.strip().lower()}"


# ---------------------------------------------------------------------------
# Title normalisation for fuzzy matching
# ---------------------------------------------------------------------------

def _norm_title(title: str) -> str:
    """Strip remaster/live suffixes and punctuation for matching."""
    t = _SUFFIX_RE.sub("", title).lower().strip()
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


# ---------------------------------------------------------------------------
# MusicBrainz API helpers
# ---------------------------------------------------------------------------

def _escape_lucene(s: str) -> str:
    return re.sub(r'([+\-&|!(){}\[\]^"~*?:\\/])', r"\\\1", s)


def _mb_get(endpoint: str, params: dict) -> dict | None:
    """Rate-limited GET against the MB API. Returns parsed JSON or None."""
    _limiter.wait()
    try:
        resp = requests.get(
            MB_API_URL + endpoint,
            params={**params, "fmt": "json"},
            headers={"User-Agent": _USER_AGENT},
            timeout=15,
        )
        if resp.status_code == 503:
            # MB overload — wait extra and retry once
            time.sleep(5)
            _limiter.wait()
            resp = requests.get(
                MB_API_URL + endpoint,
                params={**params, "fmt": "json"},
                headers={"User-Agent": _USER_AGENT},
                timeout=15,
            )
        return resp.json() if resp.status_code == 200 else None
    except Exception:
        return None


def _parse_recording(rec: dict) -> dict:
    """Extract instruments, tags and live status from a MB recording dict."""
    raw_tags   = [t["name"].lower() for t in rec.get("tags", [])]
    instruments: list[str] = []
    seen: set[str] = set()
    for tag in raw_tags:
        norm = _INSTR_NORM.get(tag)
        if norm and norm not in seen:
            instruments.append(norm)
            seen.add(norm)

    releases = rec.get("releases", [])
    is_live  = False
    for rel in releases[:5]:
        rg_type = (rel.get("release-group") or {}).get("primary-type", "")
        if "live" in rg_type.lower() or "live" in rel.get("title", "").lower():
            is_live = True
            break

    return {
        "title":       rec.get("title", ""),
        "instruments": instruments,
        "tags":        raw_tags[:10],
        "is_live":     is_live,
        "mb_id":       rec.get("id", ""),
    }


def _fetch_artist_recordings(artist_name: str,
                              limit: int = _ARTIST_LIMIT) -> list[dict]:
    """Fetch up to *limit* recordings for *artist_name* in one MB request.

    Returns list of _parse_recording() dicts.
    """
    data = _mb_get("recording/", {
        "query": f'artist:"{_escape_lucene(artist_name)}"',
        "limit": limit,
    })
    if not data:
        return []
    return [_parse_recording(r) for r in data.get("recordings", [])]


# ---------------------------------------------------------------------------
# Per-artist batch enrichment (main entry point)
# ---------------------------------------------------------------------------

def enrich_catalog_mb(
    catalog_index: dict,
    data_dir: str,
    progress_cb=None,
    max_artists: int = 400,
) -> dict:
    """Incrementally enrich catalog using per-artist batch strategy.

    ONE request per unique artist returns up to 100 recordings with tags.
    For a 40 000-track / 2 000-artist library this takes ~36 minutes
    (vs 12+ hours with the naïve per-track approach).

    Runs in batches of *max_artists* per call so the user can pause and
    resume — already-cached artists are skipped on subsequent calls.

    Args:
        catalog_index: {index: {artist, title, path, ...}}
        data_dir:      directory for mb_cache.json
        progress_cb:   optional callback(current, total, message)
        max_artists:   max new artists to fetch per call (~1.1 s each)

    Returns:
        Full cache dict.
    """
    cache = load_mb_cache(data_dir)
    today = time.strftime("%Y-%m-%d")

    # Build per-artist track index:  artist_lower → {name, title_norm→(orig, idx)}
    artists: dict[str, dict] = {}
    for idx, track in catalog_index.items():
        a = track.get("artist", "").strip()
        t = track.get("title",  "").strip()
        if not a or not t:
            continue
        key = a.lower()
        if key not in artists:
            artists[key] = {"name": a, "by_norm": {}}
        norm = _norm_title(t)
        # Keep first occurrence of each normalised title
        if norm not in artists[key]["by_norm"]:
            artists[key]["by_norm"][norm] = (t, idx)

    # Filter to artists not yet fully cached
    pending = [
        info for info in artists.values()
        if not _artist_fully_cached(info, cache)
    ]
    total     = len(pending)
    fetched   = 0

    if progress_cb:
        progress_cb(0, total,
            f"MusicBrainz: {len(artists) - total} артистов уже в кэше, "
            f"{total} ожидают обогащения.")

    for i, artist_info in enumerate(pending):
        artist_name = artist_info["name"]
        by_norm     = artist_info["by_norm"]    # norm_title → (orig_title, idx)

        mb_recs = _fetch_artist_recordings(artist_name)

        for mb_rec in mb_recs:
            mb_norm = _norm_title(mb_rec["title"])
            match   = by_norm.get(mb_norm)

            # If exact norm match fails, try prefix containment
            if not match:
                for local_norm, local_val in by_norm.items():
                    if mb_norm and (mb_norm in local_norm or local_norm in mb_norm):
                        match = local_val
                        break

            if match:
                orig_title, _ = match
                k = cache_key(artist_name, orig_title)
                if not _is_fresh(cache.get(k, {})):
                    cache[k] = {
                        "instruments": mb_rec["instruments"],
                        "tags":        mb_rec["tags"],
                        "is_live":     mb_rec["is_live"],
                        "mb_id":       mb_rec["mb_id"],
                        "fetched":     today,
                    }

        # Mark all unmatched local tracks as attempted (prevents re-fetching)
        for orig_title, _ in by_norm.values():
            k = cache_key(artist_name, orig_title)
            if k not in cache:
                cache[k] = {
                    "instruments": [], "tags": [], "is_live": False,
                    "mb_id": "", "fetched": today,
                }

        fetched += 1

        if fetched % _SAVE_EVERY == 0:
            _save_mb_cache(data_dir, cache)

        if progress_cb:
            with_data = sum(
                1 for v in cache.values() if v.get("instruments") or v.get("tags")
            )
            progress_cb(
                i + 1, total,
                f"MusicBrainz: {i+1}/{total} артистов — {artist_name} "
                f"({with_data} треков с данными)",
            )

        if fetched >= max_artists:
            if progress_cb:
                remaining = total - fetched
                progress_cb(i + 1, total,
                    f"MusicBrainz: пауза. Запустите снова для продолжения "
                    f"({remaining} артистов осталось).")
            break

    _save_mb_cache(data_dir, cache)

    if progress_cb:
        with_data = sum(1 for v in cache.values() if v.get("instruments") or v.get("tags"))
        progress_cb(total, total,
            f"MusicBrainz: готово. {with_data} треков с данными "
            f"(всего записей в кэше: {len(cache):,})")

    return cache


def _artist_fully_cached(artist_info: dict, cache: dict) -> bool:
    """True when every track of this artist has a fresh cache entry."""
    for orig_title, _ in artist_info["by_norm"].values():
        k = cache_key(artist_info["name"], orig_title)
        if not _is_fresh(cache.get(k, {})):
            return False
    return True


# ---------------------------------------------------------------------------
# On-demand global instrument search (instant, no pre-enrichment needed)
# ---------------------------------------------------------------------------

def search_instrument_globally(
    instrument: str,
    catalog_index: dict,
    limit: int = 200,
) -> list[int]:
    """Find catalog tracks that feature *instrument* via a global MB tag search.

    Does NOT require prior enrichment — searches MB for "tag:<instrument>"
    and cross-references results with the local catalog.  Uses at most
    ceil(limit/100) API requests (≈ 1-3 seconds).

    Returns list of matching catalog indices.  May return [] if the local
    library has no well-known tracks for this instrument.
    """
    found: list[int] = []
    # Build normalised lookup: (artist_norm, title_norm) → index
    lookup: dict[tuple, int] = {}
    for idx, track in catalog_index.items():
        a = track.get("artist", "").strip().lower()
        t = _norm_title(track.get("title", ""))
        if a and t:
            lookup[(a, t)] = idx

    pages     = max(1, (limit + 99) // 100)
    seen_idx: set[int] = set()

    for page in range(pages):
        data = _mb_get("recording/", {
            "query": f"tag:{_escape_lucene(instrument)}",
            "limit": 100,
            "offset": page * 100,
        })
        if not data:
            break
        recs = data.get("recordings", [])
        if not recs:
            break

        for rec in recs:
            mb_title = _norm_title(rec.get("title", ""))
            for credit in rec.get("artist-credit", []):
                if not isinstance(credit, dict):
                    continue
                artist_obj = credit.get("artist", {})
                mb_artist  = artist_obj.get("name", "").lower().strip()
                if not mb_artist:
                    continue
                idx = lookup.get((mb_artist, mb_title))
                if idx is not None and idx not in seen_idx:
                    seen_idx.add(idx)
                    found.append(idx)
                    break   # don't add the same track twice for multi-artist

    return found


# ---------------------------------------------------------------------------
# Query-time helpers (use pre-built cache)
# ---------------------------------------------------------------------------

def get_tracks_with_instrument(
    instrument: str,
    catalog_index: dict,
    mb_cache: dict,
) -> list[int]:
    """Return catalog indices confirmed by the *local cache* to have *instrument*.

    Fast (no network).  Returns [] when cache is empty or instrument not found.
    Complement with search_instrument_globally() for on-demand live results.
    """
    if not mb_cache:
        return []
    result = []
    for idx, track in catalog_index.items():
        k = cache_key(track.get("artist", ""), track.get("title", ""))
        if instrument in (mb_cache.get(k) or {}).get("instruments", []):
            result.append(idx)
    return result


def get_studio_track_indices(catalog_index: dict, mb_cache: dict) -> set[int]:
    """Indices of tracks confirmed as studio recordings (not live)."""
    return {
        idx for idx, track in catalog_index.items()
        if not (mb_cache.get(
            cache_key(track.get("artist", ""), track.get("title", "")),
            {}
        ) or {}).get("is_live", False)
    }
