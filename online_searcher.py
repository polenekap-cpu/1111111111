#!/usr/bin/env python3
"""Online music service search for local-catalog cross-referencing.

Queries free public music APIs (no API key required for most) to find
tracks that match the user's query, then cross-references the results
with the local catalog.  Tracks confirmed by an online service get a
score boost so the AI receives them near the top of Step-2 candidates.

Services used (in priority order):
  1. Deezer public API  — no key, 50 req/5 s, ranked by popularity
  2. iTunes Search API  — no key, ~20 req/min, Apple global catalog
  3. Last.fm tag search — uses existing lastfm_api_key from config

WHY THIS HELPS
--------------
• "100 самых популярных песен в мире"
    TF-IDF score = 0 for all tracks (no literal keyword in artist/title)
    Last.fm listener count helps for famous artists but not enough.
    Deezer search "most popular songs" → Beatles, Queen, MJ → found
    in local catalog → boosted → AI picks them.

• "флейта без слов"
    Deezer/iTunes "flute instrumental" → Jethro Tull, Jean-Pierre Rampal
    → found locally → boosted.  Complements MusicBrainz tag lookup.

• "грустная инди-музыка"
    AI decomposition → genres=["indie"], mood="melancholic"
    Deezer "melancholic indie" → ranked results → local matches boosted.

INTEGRATION
-----------
Called from query_engine.create_playlist() as Step 2e, after TF-IDF
and MusicBrainz.  Results merged via `online_boost_map` parameter added
to _merge_tracks_scored().  Runs at most 2 HTTP requests per query
(fast, ~0.3–0.8 s each).  Requires only `requests` (already in deps).
"""

import re
import time
from typing import Optional

import requests

# Seconds to wait between API calls (gentle rate limiting)
_DEEZER_INTERVAL = 0.15    # 50/5 s = 10/s → 0.15 s is safe
_ITUNES_INTERVAL = 0.25

# Score assigned to the #1 ranked result; scales linearly to 0 at last rank.
# Chosen to sit between TF-IDF typical range (1–8) and MB boost (20).
ONLINE_BOOST_MAX = 12.0

# How many results to fetch per service call
_DEEZER_LIMIT = 100
_ITUNES_LIMIT = 100


# ---------------------------------------------------------------------------
# Title / artist normalisation
# ---------------------------------------------------------------------------

_PUNCT_RE    = re.compile(r"[^\w\s]")
_SPACE_RE    = re.compile(r"\s+")
_THE_PREFIX  = re.compile(r"^the\s+", re.IGNORECASE)
_FEAT_RE     = re.compile(r"\s*(feat\.?|ft\.?|featuring)\s+.*$", re.IGNORECASE)
_SUFFIX_RE   = re.compile(
    r"\s*[\(\[].*(remaster|live|demo|acoustic|bonus|radio|single|edit|"
    r"version|концерт|ремастер|акустика).*[\)\]]?\s*$",
    re.IGNORECASE,
)


def _norm(text: str) -> str:
    """Normalise for fuzzy matching: lowercase, no punctuation, no 'The' prefix."""
    t = text.lower().strip()
    t = _SUFFIX_RE.sub("", t)      # strip (Live), (Remaster) etc.
    t = _FEAT_RE.sub("", t)        # strip "feat. X"
    t = _THE_PREFIX.sub("", t)     # "the beatles" → "beatles"
    t = _PUNCT_RE.sub(" ", t)      # remove punctuation
    return _SPACE_RE.sub(" ", t).strip()


# ---------------------------------------------------------------------------
# Service calls
# ---------------------------------------------------------------------------

def search_deezer(
    query: str,
    limit: int = _DEEZER_LIMIT,
) -> list[tuple[str, str]]:
    """Search Deezer (no API key).

    Returns list of (artist_name, track_title) sorted by Deezer ranking.
    Returns [] on network error or empty results.
    """
    try:
        resp = requests.get(
            "https://api.deezer.com/search",
            params={"q": query, "order": "RANKING", "limit": limit},
            timeout=8,
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
        results = []
        for item in data.get("data", []):
            artist = item.get("artist", {}).get("name", "").strip()
            title  = item.get("title", "").strip()
            if artist and title:
                results.append((artist, title))
        return results
    except Exception:
        return []


def search_itunes(
    query: str,
    limit: int = _ITUNES_LIMIT,
    country: str = "us",
) -> list[tuple[str, str]]:
    """Search Apple iTunes (no API key).

    Returns list of (artist_name, track_title).
    Returns [] on network error or empty results.
    """
    try:
        resp = requests.get(
            "https://itunes.apple.com/search",
            params={
                "term":    query,
                "media":   "music",
                "entity":  "song",
                "limit":   limit,
                "country": country,
            },
            timeout=8,
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
        results = []
        for item in data.get("results", []):
            artist = item.get("artistName", "").strip()
            title  = item.get("trackName",  "").strip()
            if artist and title:
                results.append((artist, title))
        return results
    except Exception:
        return []


def lastfm_tag_top_tracks(
    tag: str,
    api_key: str,
    limit: int = 50,
) -> list[tuple[str, str]]:
    """Get top tracks for a genre/mood tag from Last.fm.

    Uses the existing Last.fm API key.  Returns [] if no key provided.
    """
    if not api_key or not tag:
        return []
    try:
        resp = requests.get(
            "https://ws.audioscrobbler.com/2.0/",
            params={
                "method":  "tag.getTopTracks",
                "tag":     tag,
                "api_key": api_key,
                "format":  "json",
                "limit":   limit,
            },
            timeout=8,
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
        tracks = (data.get("tracks") or {}).get("track", [])
        results = []
        for t in tracks:
            artist = (t.get("artist") or {}).get("name", "").strip()
            title  = t.get("name", "").strip()
            if artist and title:
                results.append((artist, title))
        return results
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Catalog cross-reference
# ---------------------------------------------------------------------------

def _build_catalog_lookup(
    catalog_index: dict,
) -> tuple[dict, dict]:
    """Build two lookups for fast matching.

    Returns:
        exact_lookup:  {(artist_norm, title_norm): idx}
        artist_lookup: {artist_norm: [(title_norm, idx), ...]}
    """
    exact:  dict[tuple[str, str], int]     = {}
    by_art: dict[str, list[tuple[str, int]]] = {}

    for idx, track in catalog_index.items():
        a = _norm(track.get("artist", ""))
        t = _norm(track.get("title",  ""))
        if not a or not t:
            continue
        exact[(a, t)] = idx
        by_art.setdefault(a, []).append((t, idx))

    return exact, by_art


def match_to_catalog(
    online_tracks: list[tuple[str, str]],
    catalog_index: dict,
    boost_max: float = ONLINE_BOOST_MAX,
) -> dict[int, float]:
    """Cross-reference online results with the local catalog.

    Assigns a linearly decreasing boost score based on result rank:
        rank 1 → boost_max,  rank N → boost_max/N

    Matching strategy (in decreasing precision):
        1. Exact normalised (artist + title)
        2. Artist exact  + title contains online title (or vice versa)
        3. Both artist and title share a long common prefix (≥5 chars)

    Args:
        online_tracks: list of (artist, title) from search_deezer() etc.
        catalog_index: {index: {artist, title, ...}}
        boost_max:     score for the #1 ranked result

    Returns:
        {catalog_idx: boost_score}  — empty dict if no matches found.
    """
    if not online_tracks or not catalog_index:
        return {}

    exact, by_art = _build_catalog_lookup(catalog_index)
    total  = len(online_tracks)
    result: dict[int, float] = {}

    for rank, (raw_artist, raw_title) in enumerate(online_tracks, start=1):
        score   = boost_max * (total - rank + 1) / total
        a_norm  = _norm(raw_artist)
        t_norm  = _norm(raw_title)

        # 1. Exact match
        idx = exact.get((a_norm, t_norm))
        if idx is not None:
            result[idx] = max(result.get(idx, 0.0), score)
            continue

        # 2. Artist exact + title containment
        artist_tracks = by_art.get(a_norm, [])
        matched = False
        for local_t, idx in artist_tracks:
            if t_norm in local_t or local_t in t_norm:
                result[idx] = max(result.get(idx, 0.0), score * 0.85)
                matched = True
                break
        if matched:
            continue

        # 3. Common prefix heuristic (handles slight artist name differences)
        if len(a_norm) >= 5:
            a_prefix = a_norm[:5]
            for (cat_a, cat_t), idx in exact.items():
                if cat_a.startswith(a_prefix) and (t_norm in cat_t or cat_t in t_norm):
                    result[idx] = max(result.get(idx, 0.0), score * 0.70)
                    break   # limit search cost

    return result


# ---------------------------------------------------------------------------
# Query formation
# ---------------------------------------------------------------------------

def form_search_queries(
    user_query: str,
    structured_intent: Optional[dict] = None,
) -> list[str]:
    """Build 1–3 search queries from user request + AI structured intent.

    Always includes the original query.  When the structured intent has
    useful English genre/mood terms, adds a second English-language query
    (better for Deezer/iTunes which skew towards English results).

    Returns list of queries to try (first = highest priority).
    """
    queries = [user_query]

    if not structured_intent:
        return queries

    # Build English query from decomposed intent
    parts: list[str] = []
    genres = structured_intent.get("genres") or []
    parts.extend(genres[:2])                              # top 2 genres

    mood = structured_intent.get("mood")
    if mood:
        parts.append(mood)

    energy = structured_intent.get("energy", "medium")
    if energy == "high":
        parts.append("energetic")
    elif energy == "low":
        parts.append("calm")

    vocal = structured_intent.get("vocal", "any")
    if vocal == "instrumental":
        parts.append("instrumental")

    eng_query = " ".join(parts).strip()

    # Only add if it's meaningfully different from the original query
    if (eng_query
            and eng_query.lower() not in user_query.lower()
            and len(eng_query) >= 5):
        queries.append(eng_query)

    # For genre-tagged Last.fm queries, add per-genre entries
    for genre in genres[:2]:
        if genre not in queries:
            queries.append(genre)

    return queries[:3]   # cap at 3 to limit API calls
