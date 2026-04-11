#!/usr/bin/env python3
"""OpenRouter API query engine — two-step artist+track pipeline.

Pipeline:
  Step 1 (AI): Send full artist list → LLM picks relevant artists
  Step 2 (AI): Send real tracks from those artists → LLM picks final playlist
  Step 2b (parallel): TF-IDF over full catalog for literal keyword matches

API key loading priority:
  1. Environment variable OPENROUTER_API_KEY
  2. secrets.json in the same directory as config.json (not in git)
  3. config.json field "api_key" (legacy, will log a warning)

IMPORTANT: Never commit api_key into config.json or any tracked file.
"""

import json
import os
import re
import sys
from datetime import datetime

import requests

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

DECOMPOSE_PROMPT = """\
Parse the user's music query (may be in Russian or English) into a JSON structure.
Respond with ONLY a single JSON object on one line, no explanation, no markdown.

Schema (all fields required; use null when unknown):
{"genres":[],"mood":null,"energy":"medium","bpm_min":null,"bpm_max":null,"vocal":"any","mode":"any","language":null,"search_hint":null}

Field rules:
- genres:      list of genre names in English, e.g. ["rock","heavy metal"]
- mood:        single English word/phrase, e.g. "melancholic", "energetic", "relaxing"
- energy:      "low" | "medium" | "high"
- bpm_min/max: integer BPM bounds or null; for "fastest tempo" set bpm_min:180
- vocal:       "any" | "vocal" | "instrumental"
- mode:        "any" | "major" | "minor"
- language:    null | "ru" | "en" | "fr" | "ja" | "ko" | "es"
- search_hint: SHORT English phrase optimised for a music search engine.
  Translate idioms, cultural references and vibes into searchable terms.
  This is the single most important field — make it specific and evocative.
  Examples:
    "противный дед"           → "grumpy raspy male vocalist russian bard folk"
    "вайб ночного города"     → "dark city night atmospheric synthwave"
    "100 самых популярных"    → "greatest hits all time most popular"
    "медитация перед сном"    → "sleep meditation ambient calm"
    "быстрый темп"            → "high bpm fast tempo thrash death metal"
    "неофолк спокойный"       → "neofolk pagan folk calm medieval acoustic"

Examples (full):
  Query: "расслабляющая инструментальная музыка с флейтой"
  → {"genres":["new age","classical"],"mood":"relaxing","energy":"low","bpm_min":null,"bpm_max":90,"vocal":"instrumental","mode":"any","language":null,"search_hint":"flute instrumental relaxing new age classical"}

  Query: "energetic metal from the 80s"
  → {"genres":["heavy metal","hard rock"],"mood":"energetic","energy":"high","bpm_min":140,"bpm_max":null,"vocal":"vocal","mode":"any","language":"en","search_hint":"heavy metal hard rock 80s energetic"}

  Query: "спокойный неофолк"
  → {"genres":["neofolk","pagan folk","medieval folk"],"mood":"calm","energy":"low","bpm_min":null,"bpm_max":100,"vocal":"any","mode":"any","language":null,"search_hint":"neofolk pagan folk calm acoustic medieval"}"""

ARTIST_SELECT_PROMPT = """\
Тебе дан пронумерованный список артистов из локальной музыкальной библиотеки.
Формат строки: INDEX|АРТИСТ|КОЛ-ВО ТРЕКОВ|ГОДЫ|ПРИМЕРЫ ТРЕКОВ|ЖАНРЫ|ПОПУЛЯРНОСТЬ
(поля ЖАНРЫ и ПОПУЛЯРНОСТЬ присутствуют только если есть данные Last.fm)

Задача: выбери артистов, у которых НАИБОЛЕЕ ВЕРОЯТНО есть треки, подходящие под запрос пользователя.
Учитывай жанр, стиль, язык, настроение и эпоху. Выбирай щедро — лучше взять лишних, чем пропустить нужных.
Целевое количество: 30–100 артистов (больше для широких запросов, меньше для конкретных).

Ответ: верни ТОЛЬКО номера через запятую внутри тегов <PLAYLIST> и </PLAYLIST>.
Пример: <PLAYLIST>3,17,42,88,103</PLAYLIST>"""

TRACK_SELECT_PROMPT = """\
Ты — музыкальный куратор. Тебе дан список реальных треков из локального каталога.
Формат: INDEX|АРТИСТ|НАЗВАНИЕ|ГОД  (иногда с дополнительной колонкой BPM:XXX — темп трека)

Задача: выбери до N треков, которые НАИЛУЧШИМ ОБРАЗОМ соответствуют запросу.

Правила:
- Используй свои знания о каждом треке и артисте: жанр, инструменты, наличие вокала,
  настроение, язык — даже если это явно не указано в названии.
- Если в строке трека есть колонка BPM — используй реальные значения темпа для сортировки,
  особенно для запросов типа «быстрый темп», «самый медленный» и т.д.
- Если запрос про инструментальную музыку или конкретный инструмент (флейта, скрипка и т.д.)
  — выбирай треки, которые, по твоим знаниям, действительно содержат этот инструмент или
  не содержат вокала. Если таких нет в списке — верни пустой плейлист.
- Если запрос называет конкретного исполнителя — включай ТОЛЬКО треки этого исполнителя.
- Соблюдай язык запроса: «на русском» → кириллические треки, «in english» → латиница.
- НЕ дублируй: если есть «Song (Live)» и «Song» — только студийную версию.
- Разнообразие: не концентрируй больше 20–25% треков у одного артиста, если не запрошен конкретный исполнитель.
- Лучше меньше, но точнее. Лучше 5 релевантных треков, чем 30 сомнительных.

Ответ: верни ТОЛЬКО индексы через запятую внутри тегов <PLAYLIST> и </PLAYLIST>.
Пример: <PLAYLIST>1452,891,23044,7821,445</PLAYLIST>"""

STRICT_PROMPT = """\
Ответь ТОЛЬКО индексами через запятую внутри тегов <PLAYLIST></PLAYLIST>.
Пример: <PLAYLIST>1452,891,23044</PLAYLIST>"""

DECOMPOSE_MIN_WORDS = 3   # minimum query words to trigger decomposition

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
GOOGLE_API_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
DEFAULT_MODEL = "qwen/qwen3-235b-a22b:free"
DEFAULT_GOOGLE_MODEL = "gemini-2.0-flash"

ARTIST_SELECT_TARGET = 80
TRACK_CONTEXT_LIMIT = 1500
TFIDF_CANDIDATES = 300
MAX_TRACKS_PER_ARTIST = 30  # prevents any single artist from dominating the prompt


# ---------------------------------------------------------------------------
# Config & API key
# ---------------------------------------------------------------------------

def load_config(config_path=None):
    """Load config.json and expand paths. Does NOT require api_key in config."""
    if config_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json не найден: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    for key in ("catalog_path", "ai_catalog_path", "output_dir"):
        if key in cfg:
            cfg[key] = os.path.expanduser(cfg[key])
    cfg["music_dirs"] = [
        os.path.realpath(os.path.expanduser(d)) for d in cfg.get("music_dirs", [])
    ]
    return cfg


def load_api_key(config_path=None):
    """Load API key from env → secrets.json → config.json (legacy, warns).

    Returns:
        API key string.
    Raises:
        ValueError if no key found anywhere.
    """
    # 1. Environment variable (recommended)
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key:
        return key

    # 2. secrets.json next to config.json
    if config_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config.json")
    secrets_path = os.path.join(os.path.dirname(config_path), "secrets.json")
    if os.path.exists(secrets_path):
        with open(secrets_path, "r", encoding="utf-8") as f:
            secrets = json.load(f)
        key = secrets.get("api_key", secrets.get("openrouter_api_key", "")).strip()
        if key:
            return key

    # 3. Legacy: api_key in config.json (deprecated)
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        key = cfg.get("api_key", cfg.get("gemini_api_key", "")).strip()
        if key:
            print(
                "[ПРЕДУПРЕЖДЕНИЕ] api_key в config.json устарел и небезопасен. "
                "Переместите ключ в переменную окружения OPENROUTER_API_KEY "
                "или в файл secrets.json (добавлен в .gitignore).",
                file=sys.stderr,
            )
            return key

    raise ValueError(
        "API-ключ не найден. Установите переменную окружения OPENROUTER_API_KEY "
        "или создайте файл secrets.json с полем \"api_key\"."
    )


# ---------------------------------------------------------------------------
# Catalog loading
# ---------------------------------------------------------------------------

def load_catalog_index(catalog_path):
    """Load catalog.tsv into {index: {artist, title, year, path}}."""
    catalog = {}
    if not os.path.exists(catalog_path):
        return catalog
    with open(catalog_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 5:
                continue
            try:
                catalog[int(parts[0])] = {
                    "artist": parts[1],
                    "title": parts[2],
                    "year": parts[3],
                    "path": parts[4],
                }
            except (ValueError, IndexError):
                continue
    return catalog


def build_artist_index(catalog_index):
    """Build {artist_name_lower: {name, indices}} from catalog."""
    artist_map = {}
    for idx, track in catalog_index.items():
        artist = track.get("artist", "").strip()
        if not artist:
            continue
        key = artist.lower()
        if key not in artist_map:
            artist_map[key] = {"name": artist, "indices": []}
        artist_map[key]["indices"].append(idx)
    return artist_map


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def _call_openrouter(api_key, model, system_prompt, user_message, max_tokens=2048):
    """Single OpenRouter API call. Returns response text."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.7,
        "max_tokens": max_tokens,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/polenekap-cpu/111111111111111111111111111111111sdwew",
        "X-Title": "AI Playlist Generator",
    }
    try:
        resp = requests.post(OPENROUTER_API_URL, json=payload, headers=headers, timeout=180)
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Ошибка сети: {e}") from e

    try:
        data = resp.json()
    except json.JSONDecodeError:
        raise ValueError(f"Невалидный ответ ({resp.status_code}): {resp.text[:300]}")

    if resp.status_code != 200:
        error_msg = data.get("error", {}).get("message", resp.text[:300])
        error_code = data.get("error", {}).get("code", resp.status_code)
        if resp.status_code == 429 or error_code == 429:
            raise ConnectionError("Превышен лимит запросов OpenRouter. Подождите и попробуйте снова.")
        if error_code == 404:
            raise ValueError(
                f"Модель недоступна: {error_msg}\n"
                "Проверьте модель и настройки на openrouter.ai/settings/privacy"
            )
        raise ValueError(f"Ошибка API ({error_code}): {error_msg}")

    if "error" in data:
        raise ValueError(f"Ошибка OpenRouter: {data['error'].get('message', str(data['error']))}")

    choices = data.get("choices", [])
    if not choices:
        raise ValueError("OpenRouter не вернул ответ")

    message = choices[0].get("message") or {}
    text = message.get("content")
    if text is None:
        text = message.get("reasoning_content")
    text = (text or "").strip()

    if not text:
        finish = choices[0].get("finish_reason", "?")
        raise ValueError(
            f"Модель вернула пустой ответ. (finish_reason: {finish}). "
            "Попробуйте другую модель или повторите запрос."
        )
    return text


def _call_google(api_key, model, system_prompt, user_message, max_tokens=2048):
    """Single Google Gemini API call. Returns response text."""
    url = GOOGLE_API_URL_TEMPLATE.format(model=model, key=api_key)
    payload = {
        "contents": [{"parts": [{"text": f"{system_prompt}\n\n{user_message}"}]}],
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": max_tokens},
    }
    try:
        resp = requests.post(url, json=payload, timeout=180)
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Ошибка сети: {e}") from e

    try:
        data = resp.json()
    except json.JSONDecodeError:
        raise ValueError(f"Невалидный ответ Google ({resp.status_code}): {resp.text[:300]}")

    if resp.status_code != 200:
        err = data.get("error", {}).get("message", resp.text[:300])
        raise ValueError(f"Ошибка Google API ({resp.status_code}): {err}")

    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError):
        raise ValueError(f"Неожиданный ответ Google: {str(data)[:300]}")

    if not text:
        raise ValueError("Google Gemini вернул пустой ответ.")
    return text


def _call_ai(api_provider, api_key, model, system_prompt, user_message, max_tokens=2048):
    """Dispatch AI call to the correct provider."""
    if api_provider == "google":
        return _call_google(api_key, model, system_prompt, user_message, max_tokens)
    return _call_openrouter(api_key, model, system_prompt, user_message, max_tokens)


# ---------------------------------------------------------------------------
# AI query decomposition (pre-step)
# ---------------------------------------------------------------------------

_DECOMPOSE_DEFAULT = {
    "genres": [], "mood": None, "energy": "medium",
    "bpm_min": None, "bpm_max": None,
    "vocal": "any", "mode": "any", "language": None,
    "search_hint": None,
}


def decompose_query(
    api_provider: str,
    api_key: str,
    model: str,
    user_query: str,
) -> dict:
    """Pre-step: convert free-text query to structured intent via AI.

    Uses a very compact prompt (≈150 input tokens, ≤100 output tokens) to
    minimise API-call quota usage.  On any error returns the default dict
    so the rest of the pipeline can continue unaffected.

    The returned dict has keys:
        genres (list[str]), mood (str|None), energy ("low"|"medium"|"high"),
        bpm_min (int|None), bpm_max (int|None),
        vocal ("any"|"vocal"|"instrumental"),
        mode ("any"|"major"|"minor"),
        language (str|None — same codes as search_local).

    Only called when the query has >= DECOMPOSE_MIN_WORDS meaningful words
    and no explicit artist names were detected (to save API calls).
    """
    try:
        raw = _call_ai(
            api_provider, api_key, model,
            DECOMPOSE_PROMPT,
            f'Query: "{user_query}"',
            max_tokens=160,
        )
        # Strip markdown fences if the model wrapped the JSON
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```\w*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw.strip())

        parsed = json.loads(raw.strip())

        result = dict(_DECOMPOSE_DEFAULT)
        result["genres"]   = [str(g) for g in (parsed.get("genres") or [])]
        result["mood"]     = parsed.get("mood") or None
        energy = parsed.get("energy", "medium")
        result["energy"]   = energy if energy in ("low", "medium", "high") else "medium"
        result["bpm_min"]     = int(parsed["bpm_min"])  if parsed.get("bpm_min")  else None
        result["bpm_max"]     = int(parsed["bpm_max"])  if parsed.get("bpm_max")  else None
        vocal = parsed.get("vocal", "any")
        result["vocal"]       = vocal  if vocal  in ("any", "vocal", "instrumental") else "any"
        mode  = parsed.get("mode",  "any")
        result["mode"]        = mode   if mode   in ("any", "major", "minor")        else "any"
        result["language"]    = parsed.get("language") or None
        result["search_hint"] = (parsed.get("search_hint") or "").strip() or None
        return result

    except Exception as exc:
        print(f"[decompose_query] skipped: {exc}", file=sys.stderr)
        return dict(_DECOMPOSE_DEFAULT)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_indices(response_text, strict_playlist_tags=True):
    """Extract unique integer indices from AI response.

    With strict_playlist_tags=True (default): ONLY parses numbers inside
    <PLAYLIST>...</PLAYLIST> tags. This prevents accidental inclusion of
    numbers from reasoning text like "I found 15 tracks out of 300 candidates".

    Falls back to broader parsing only when tags are completely absent.
    """
    # Primary: extract from <PLAYLIST> tags
    playlist_match = re.search(
        r'<PLAYLIST>(.*?)</PLAYLIST>', response_text, re.DOTALL | re.IGNORECASE
    )
    if playlist_match:
        tag_content = playlist_match.group(1)
        numbers = re.findall(r'(?<!\d)(\d+)(?!\d)', tag_content)
        result = _clean_indices(numbers)
        if result:
            return result

    if not strict_playlist_tags:
        # Full permissive parse — only on explicit retry
        numbers = re.findall(r'(?<!\d)(\d+)(?!\d)', response_text)
        return _clean_indices(numbers)

    # Fallback: try JSON extraction
    try:
        data = json.loads(response_text.strip())
        if isinstance(data, (dict, list)):
            values = _extract_numbers_from_json(data)
            if values:
                return _deduplicate_preserve_order(values)
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Last resort: numbers separated by whitespace/punctuation only
    numbers = re.findall(r'(?:^|[\s,;:\[\(])(\d{1,6})(?:[\s,;:\]\)]|$)', response_text)
    return _clean_indices(numbers)


def _clean_indices(number_strings):
    seen = set()
    result = []
    for s in number_strings:
        try:
            idx = int(s)
        except ValueError:
            continue
        if idx <= 0 or idx > 999_999:
            continue
        if idx not in seen:
            seen.add(idx)
            result.append(idx)
    return result


def _extract_numbers_from_json(data):
    numbers = []
    if isinstance(data, int):
        numbers.append(data)
    elif isinstance(data, float) and data == int(data):
        numbers.append(int(data))
    elif isinstance(data, str):
        for m in re.findall(r'\d+', data):
            numbers.append(int(m))
    elif isinstance(data, list):
        for item in data:
            numbers.extend(_extract_numbers_from_json(item))
    elif isinstance(data, dict):
        for key in ("indices", "tracks", "results", "ids"):
            if key in data:
                numbers.extend(_extract_numbers_from_json(data[key]))
                if numbers:
                    return numbers
        for v in data.values():
            numbers.extend(_extract_numbers_from_json(v))
    return numbers


def _deduplicate_preserve_order(numbers):
    seen = set()
    result = []
    for n in numbers:
        try:
            val = int(n)
        except (ValueError, TypeError):
            continue
        if val <= 0 or val > 999_999:
            continue
        if val not in seen:
            seen.add(val)
            result.append(val)
    return result


# ---------------------------------------------------------------------------
# M3U8 generation
# ---------------------------------------------------------------------------

def generate_m3u8(tracks, output_dir, user_query):
    os.makedirs(output_dir, exist_ok=True)
    safe_query = re.sub(r"[^\w\s-]", "", user_query, flags=re.UNICODE)[:30].strip()
    safe_query = re.sub(r"\s+", "_", safe_query)
    if not safe_query:
        safe_query = "playlist"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"{safe_query}_{timestamp}.m3u"
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("#EXTM3U\n")
        for track in tracks:
            artist = track["artist"] or ""
            title = track["title"] or ""
            display = f"{artist} - {title}" if artist else title
            f.write(f"#EXTINF:0,{display}\n")
            f.write(f"{track['path']}\n")
    return output_path


# ---------------------------------------------------------------------------
# Two-step pipeline helpers
# ---------------------------------------------------------------------------

_VARIANT_SUFFIX_RE = re.compile(
    r"\s*[\(\[](live|concert|remaster(?:ed)?|acoustic|bonus|demo|"
    r"концерт|вживую|акустика|ремастер|instrumental|radio[\s\-]edit|"
    r"single[\s\-]version|deluxe|extended)[\)\]]?\s*$",
    re.IGNORECASE,
)


def _load_rich_artist_data(data_dir):
    """Load artists_for_ai.txt → {artist_name_lower: {count, years, samples}}.

    Format: ID|NAME|TRACK_COUNT|YEARS|SAMPLE_TITLES
    Built by catalog_builder.write_catalogs().
    """
    artists_file = os.path.join(data_dir, "artists_for_ai.txt")
    rich = {}
    if not os.path.exists(artists_file):
        return rich
    with open(artists_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|", 4)
            if len(parts) >= 2:
                name = parts[1]
                rich[name.lower()] = {
                    "name": name,
                    "count": parts[2] if len(parts) > 2 else "",
                    "years": parts[3] if len(parts) > 3 else "",
                    "samples": parts[4] if len(parts) > 4 else "",
                }
    return rich


def _build_artist_list_text(artist_map, rich_data=None, lastfm_cache=None):
    """Format numbered artist list for AI.

    Columns (all optional after INDEX|NAME):
      N тр. | YEARS | sample titles | lastfm tags | listeners

    With Last.fm data the AI can make genre-aware and popularity-aware
    decisions even for artists it doesn't recognise by name.
    """
    try:
        from lastfm_enricher import format_listeners
    except ImportError:
        def format_listeners(n):
            return f"{n // 1_000}K слуш." if n >= 1_000 else ""

    lines = []
    num_to_key = {}
    for i, (key, info) in enumerate(sorted(artist_map.items()), start=1):
        num_to_key[i] = key
        parts = [str(i), info["name"]]

        # Catalog-derived context (years, track count, sample titles)
        if rich_data and key in rich_data:
            r = rich_data[key]
            if r.get("count"):
                parts.append(f"{r['count']} тр.")
            if r.get("years"):
                parts.append(r["years"])
            if r.get("samples"):
                parts.append(r["samples"])

        # Last.fm enrichment (genre tags + popularity)
        if lastfm_cache:
            lfm = lastfm_cache.get(key, {})
            tags = lfm.get("tags", [])
            listeners = lfm.get("listeners", 0)
            if tags:
                parts.append(", ".join(tags[:5]))
            ls = format_listeners(listeners)
            if ls:
                parts.append(ls)

        lines.append("|".join(parts))
    return "\n".join(lines), num_to_key


def _prededup_tracks(tracks):
    """Remove Live/Remaster/Acoustic variants before sending to AI.

    Keeps studio version when both studio and variant exist for the same
    (artist, normalized_title) pair. If only a variant exists, keeps it.
    Preserves the original order of studio tracks, then appends orphaned
    variants (no studio equivalent found).
    """
    def _norm(title):
        return _VARIANT_SUFFIX_RE.sub("", title).lower().strip()

    studio, variants = [], []
    for t in tracks:
        if _VARIANT_SUFFIX_RE.search(t.get("title", "")):
            variants.append(t)
        else:
            studio.append(t)

    seen = set()
    result = []
    for t in studio:
        key = f"{t.get('artist', '').lower()}|{_norm(t.get('title', ''))}"
        if key not in seen:
            seen.add(key)
            result.append(t)
    for t in variants:
        key = f"{t.get('artist', '').lower()}|{_norm(t.get('title', ''))}"
        if key not in seen:
            seen.add(key)
            result.append(t)
    return result


def _get_tracks_for_artists(selected_keys, artist_map, catalog_index,
                             max_per_artist=MAX_TRACKS_PER_ARTIST):
    """Collect tracks for selected artists with per-artist cap.

    No global limit here — caller (``_merge_tracks_scored``) handles that.
    Per-artist cap ensures no single artist dominates the final prompt even
    when an artist has hundreds of tracks in the catalog.
    """
    tracks = []
    for key in selected_keys:
        info = artist_map.get(key)
        if not info:
            continue
        artist_tracks = []
        for idx in info["indices"]:
            track = catalog_index.get(idx)
            if track:
                artist_tracks.append({"index": idx, **track})
        tracks.extend(artist_tracks[:max_per_artist])
    return tracks


def _merge_tracks_scored(artist_tracks, tfidf_tracks, limit,
                          max_per_artist=MAX_TRACKS_PER_ARTIST,
                          lastfm_cache=None,
                          mb_priority_indices=None,
                          online_boost_map=None):
    """Merge artist tracks + TF-IDF results into a ranked, diverse candidate list.

    Strategy:
    1. Assign TF-IDF scores to all artist tracks (0.0 if not in TF-IDF results).
    2. Boost MusicBrainz-confirmed tracks (instrument matches) with a high score
       so they appear near the top of the candidate list for AI Step 2.
    3. Pre-deduplicate: remove Live/Remaster/Acoustic variants.
    4. Group by artist; sort each group by score (desc); cap at max_per_artist.
    5. Sort artist groups by their best track's score so most relevant artists
       appear first in the interleaved output.
    6. Round-robin interleave: take slot-0 from each artist, then slot-1, etc.
       This guarantees diversity even after hard truncation to ``limit``.

    Result: a balanced list where no artist dominates AND the most relevant
    tracks for the query appear near the top.
    """
    from collections import defaultdict

    tfidf_score_map   = {t["index"]: t.get("score", 0.0) for t in tfidf_tracks}
    mb_priority_set   = set(mb_priority_indices or [])
    online_map        = online_boost_map or {}
    MB_BOOST          = 20.0   # score assigned to MB-confirmed tracks

    # Merge, preserving scores
    seen = set()
    all_tracks = []
    for t in artist_tracks:
        idx = t["index"]
        if idx not in seen:
            seen.add(idx)
            base_score = tfidf_score_map.get(idx, 0.0)
            if idx in mb_priority_set:
                base_score = max(base_score, MB_BOOST)
            if idx in online_map:
                base_score = max(base_score, online_map[idx])
            all_tracks.append({**t, "score": base_score})
    for t in tfidf_tracks:
        idx = t["index"]
        if idx not in seen:
            seen.add(idx)
            score = t.get("score", 0.0)
            if idx in mb_priority_set:
                score = max(score, MB_BOOST)
            if idx in online_map:
                score = max(score, online_map[idx])
            all_tracks.append({**t, "score": score})

    # Pre-deduplicate variants
    all_tracks = _prededup_tracks(all_tracks)

    # Group by artist
    by_artist = defaultdict(list)
    for t in all_tracks:
        by_artist[t.get("artist", "").lower()].append(t)

    # Per-artist: sort by score, cap at max_per_artist
    for key in by_artist:
        by_artist[key].sort(key=lambda t: t.get("score", 0.0), reverse=True)
        by_artist[key] = by_artist[key][:max_per_artist]

    # Sort artist groups: primary = best TF-IDF/MB score, secondary = Last.fm
    # listener count.  When TF-IDF scores are all 0 (e.g. "самые популярные"),
    # listener count decides the order → Beatles (7.8M) before niche acts.
    def _sort_key(group):
        tfidf_score = group[0].get("score", 0.0) if group else 0.0
        artist_key  = group[0].get("artist", "").lower() if group else ""
        listeners   = (lastfm_cache or {}).get(artist_key, {}).get("listeners", 0)
        return (tfidf_score, listeners)

    artist_groups = sorted(by_artist.values(), key=_sort_key, reverse=True)

    # Round-robin interleave across all artists
    result = []
    for slot in range(max_per_artist):
        for group in artist_groups:
            if slot < len(group):
                result.append(group[slot])
            if len(result) >= limit:
                return result

    return result[:limit]


def _enforce_diversity(tracks: list, max_per_artist: int) -> list:
    """Hard cap: no more than max_per_artist tracks from the same artist.

    Applied as a post-processing step on the AI-selected playlist so that
    a single well-known artist can never crowd out others regardless of
    how the AI ranked them.
    """
    from collections import defaultdict
    counts: dict = defaultdict(int)
    result = []
    for t in tracks:
        a = t.get("artist", "").lower().strip()
        if counts[a] < max_per_artist:
            counts[a] += 1
            result.append(t)
    return result


def _build_track_text(tracks, audio_features=None):
    """Format track list for the Step-2 AI prompt.

    Appends a BPM column when audio_features are available and any track
    has BPM data.  The AI can use this to sort/filter by actual tempo.
    Format: INDEX|ARTIST|TITLE|YEAR  or  INDEX|ARTIST|TITLE|YEAR|BPM:120
    """
    lines = []
    for t in tracks:
        line = f"{t['index']}|{t.get('artist','')}|{t.get('title','')}|{t.get('year','')}"
        if audio_features:
            path  = t.get("path", "")
            feats = audio_features.get(path) or {}
            bpm   = feats.get("bpm", 0)
            if bpm and bpm > 0:
                line += f"|BPM:{bpm:.0f}"
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def create_playlist(config_path=None, user_query="", progress_cb=None):
    """Two-step playlist generation: artist selection → track curation.

    Returns:
        dict with keys: output_path, valid, total, skipped, tracks,
                        step1_artists, step2_candidates
    """
    cfg = load_config(config_path)
    catalog_path = cfg["catalog_path"]

    if not os.path.exists(catalog_path):
        raise FileNotFoundError("Каталог не найден. Сначала выполните сканирование библиотеки.")

    api_key = load_api_key(config_path)
    api_provider = cfg.get("api_provider", "openrouter")
    model = cfg.get("ai_model", cfg.get("model", cfg.get("gemini_model",
        DEFAULT_GOOGLE_MODEL if api_provider == "google" else DEFAULT_MODEL)))
    playlist_size = cfg.get("playlist_size", 40)
    output_dir = cfg["output_dir"]

    if progress_cb:
        progress_cb("Загрузка каталога...")

    catalog_index = load_catalog_index(catalog_path)
    if not catalog_index:
        raise FileNotFoundError("Каталог пуст. Выполните сканирование библиотеки.")

    artist_map = build_artist_index(catalog_index)
    num_artists = len(artist_map)
    data_dir = os.path.dirname(os.path.abspath(catalog_path))

    rich_artist_data = _load_rich_artist_data(data_dir)
    try:
        from lastfm_enricher import load_cache as _load_lastfm
        lastfm_cache = _load_lastfm(data_dir)
    except ImportError:
        lastfm_cache = {}

    # Load optional enrichment caches (audio features + MusicBrainz)
    audio_features: dict = {}
    mb_cache:        dict = {}
    try:
        from audio_analyzer import load_audio_features
        audio_features = load_audio_features(data_dir)
    except ImportError:
        pass
    try:
        from musicbrainz_enricher import load_mb_cache
        mb_cache = load_mb_cache(data_dir)
    except ImportError:
        pass

    # ---- Pre-step A: detect specific artists and musical attributes ---------
    # If the user explicitly names artists from the catalog, skip the AI
    # artist-selection step and go directly to their tracks.  This prevents
    # "лучшие песни Короля и Шута" from pulling in Noize MC or PHARAOH.
    mentioned_artists = []
    query_attributes = []  # e.g. ["instrumental", "instrument:flute"]
    try:
        from search_local import detect_mentioned_artists, expand_query_tokens
        mentioned_artists = detect_mentioned_artists(user_query, artist_map)
        _, intent_pre = expand_query_tokens(user_query)
        query_attributes = intent_pre.get("attributes", [])
    except ImportError:
        pass

    # ---- Pre-step B: AI query decomposition --------------------------------
    # Convert the free-text query into a structured intent (genres, mood,
    # energy, BPM range, vocal/instrumental, major/minor, language).
    # Only called when:
    #   • no specific artist names detected (saves an API call otherwise)
    #   • query is long enough to warrant decomposition
    #   • config does not disable it ("ai_query_decomposition": false)
    structured_intent: dict = dict(_DECOMPOSE_DEFAULT)
    run_decompose = (
        not mentioned_artists
        and len([w for w in user_query.split() if len(w) >= 3]) >= DECOMPOSE_MIN_WORDS
        and cfg.get("ai_query_decomposition", True)
    )
    if run_decompose:
        if progress_cb:
            progress_cb("Анализ запроса (AI)...")
        structured_intent = decompose_query(
            api_provider, api_key, model, user_query
        )
        if progress_cb:
            genres_str = ", ".join(structured_intent.get("genres", [])) or "—"
            progress_cb(
                f"Запрос: жанры={genres_str}, "
                f"энергия={structured_intent.get('energy','?')}, "
                f"вокал={structured_intent.get('vocal','?')}"
            )

    # ---- Step 1: select relevant artists ------------------------------------
    if mentioned_artists:
        # Specific artists named → skip AI call, use them directly
        selected_artist_keys = set(mentioned_artists)
        step1_artists = len(selected_artist_keys)
        names = ", ".join(artist_map[k]["name"] for k in mentioned_artists if k in artist_map)
        if progress_cb:
            progress_cb(f"Найдены конкретные артисты: {names}")
    else:
        if progress_cb:
            progress_cb(f"Шаг 1: отбор артистов из {num_artists} в каталоге...")

        artist_list_text, num_to_key = _build_artist_list_text(
            artist_map, rich_artist_data, lastfm_cache
        )

        # Build structured hint lines from AI decomposition
        hint_lines = []
        if structured_intent.get("genres"):
            hint_lines.append("Жанры: " + ", ".join(structured_intent["genres"]))
        if structured_intent.get("mood"):
            hint_lines.append(f"Настроение: {structured_intent['mood']}")
        energy_level = structured_intent.get("energy", "medium")
        if energy_level != "medium":
            energy_ru = {"low": "тихая/спокойная", "high": "энергичная/динамичная"}.get(
                energy_level, energy_level
            )
            hint_lines.append(f"Энергетика: {energy_ru}")
        if structured_intent.get("vocal") == "instrumental":
            hint_lines.append("Тип: инструментальная музыка (без вокала)")
        if structured_intent.get("language"):
            hint_lines.append(f"Язык: {structured_intent['language']}")
        if structured_intent.get("search_hint"):
            hint_lines.append(f"Поисковая подсказка: {structured_intent['search_hint']}")

        hint_block = ("\nКонтекст запроса:\n" + "\n".join(hint_lines) + "\n") if hint_lines else ""

        artist_user_msg = (
            f"Список артистов:\n{artist_list_text}\n\n"
            f"Запрос: \"{user_query}\"\n"
            f"{hint_block}"
            f"Выбери ~{ARTIST_SELECT_TARGET} артистов, у которых наиболее вероятно "
            f"есть треки под этот запрос."
        )

        artist_response = _call_ai(api_provider,
            api_key, model, ARTIST_SELECT_PROMPT, artist_user_msg, max_tokens=1024
        )

        artist_numbers = parse_indices(artist_response)
        selected_artist_keys = set()
        for num in artist_numbers:
            key = num_to_key.get(num)
            if key:
                selected_artist_keys.add(key)

        step1_artists = len(selected_artist_keys)
        if progress_cb:
            progress_cb(f"Шаг 1 завершён: выбрано {step1_artists} артистов.")

        if not selected_artist_keys:
            if progress_cb:
                progress_cb("Шаг 1 не вернул артистов. Используем весь каталог.")
            selected_artist_keys = set(artist_map.keys())

    # ---- Step 2b: TF-IDF for literal keyword matches ------------------------
    tfidf_tracks = []
    try:
        from search_local import SearchIndex
        if progress_cb:
            progress_cb("TF-IDF: поиск буквальных совпадений...")
        search_idx = SearchIndex.from_catalog_dict(catalog_index)
        tfidf_tracks = search_idx.search(user_query, topn=TFIDF_CANDIDATES)
    except ImportError:
        pass

    # ---- Step 2c: MusicBrainz priority tracks --------------------------------
    # Strategy A (fast, cached): look up pre-built mb_cache.json.
    # Strategy B (on-demand, ~2 s): global MB tag search — used as fallback
    #   when the local cache has no instrument matches yet.  This means the
    #   "флейта" query works even before the user runs full MB enrichment.
    mb_priority_indices: set[int] = set()
    if query_attributes:
        try:
            from musicbrainz_enricher import (
                get_tracks_with_instrument,
                search_instrument_globally,
            )
            for attr in query_attributes:
                if not attr.startswith("instrument:"):
                    continue
                instr = attr.split(":", 1)[1]

                # A: from local cache (instant)
                if mb_cache:
                    confirmed = get_tracks_with_instrument(instr, catalog_index, mb_cache)
                    mb_priority_indices.update(confirmed)

                # B: live global search if cache gave nothing
                if not mb_priority_indices:
                    if progress_cb:
                        progress_cb(f"MusicBrainz: поиск треков с '{instr}'...")
                    live_hits = search_instrument_globally(
                        instr, catalog_index, limit=200
                    )
                    mb_priority_indices.update(live_hits)

            if mb_priority_indices and progress_cb:
                progress_cb(
                    f"MusicBrainz: {len(mb_priority_indices)} треков с подтверждённым инструментом"
                )
        except ImportError:
            pass

    # ---- Step 2d: Audio feature pre-filter ----------------------------------
    # Build acoustic constraints from AI decomposition result, then apply them
    # to the artist-track pool.  Only active when audio_features.json has data.
    audio_constraints: dict = {}
    if audio_features:
        try:
            from audio_analyzer import build_audio_constraints, filter_by_audio_features
            audio_constraints = build_audio_constraints(structured_intent)
        except ImportError:
            pass

    # ---- Step 2: Collect + score + merge tracks -----------------------------
    # Per-artist cap + TF-IDF scoring + round-robin interleave ensure that:
    # - No single artist dominates (AC/DC doesn't crowd out Beatles)
    # - Most keyword-relevant tracks appear first (better for tight context windows)
    # - Artist diversity is maintained even after truncation to TRACK_CONTEXT_LIMIT
    artist_tracks = _get_tracks_for_artists(
        selected_artist_keys, artist_map, catalog_index
    )

    # Apply audio pre-filter when we have both feature data and constraints
    if audio_constraints and audio_features:
        try:
            from audio_analyzer import filter_by_audio_features
            before = len(artist_tracks)
            artist_tracks = filter_by_audio_features(
                artist_tracks, audio_features, audio_constraints
            )
            if progress_cb and len(artist_tracks) < before:
                progress_cb(
                    f"Акустический фильтр: {len(artist_tracks)} из {before} треков"
                )
        except ImportError:
            pass

    # ---- Step 2e: Online search cross-reference ----------------------------
    # Query Deezer/iTunes (no API key) with the user's request and/or the
    # AI-generated search_hint.  Any online result that matches a local
    # track gets an ONLINE_BOOST score so it rises in the candidate list.
    # This helps queries like "100 популярных песен" or "вайб противного деда"
    # where TF-IDF returns nothing useful (no keyword in artist/title).
    online_boost_map: dict = {}
    if cfg.get("online_search_enabled", True):
        try:
            import online_searcher
            import time as _time

            search_hint = structured_intent.get("search_hint") or ""
            raw_queries = online_searcher.form_search_queries(
                user_query, structured_intent
            )
            # Prepend the AI-generated search_hint as the highest-priority query
            if search_hint and search_hint not in raw_queries:
                raw_queries = [search_hint] + raw_queries[:2]
            online_queries = raw_queries[:2]   # cap at 2 API calls

            if progress_cb:
                progress_cb(f"Онлайн-поиск: {', '.join(online_queries[:1])}...")

            online_tracks: list = []
            for i, q in enumerate(online_queries):
                if i > 0:
                    _time.sleep(online_searcher._DEEZER_INTERVAL)
                hits = online_searcher.search_deezer(q)
                if not hits:
                    hits = online_searcher.search_itunes(q)
                online_tracks.extend(hits[:50])

            if online_tracks:
                online_boost_map = online_searcher.match_to_catalog(
                    online_tracks, catalog_index
                )
                if progress_cb and online_boost_map:
                    progress_cb(
                        f"Онлайн: {len(online_boost_map)} совпадений в каталоге"
                    )
        except Exception as _exc:
            print(f"[online_searcher] skipped: {_exc}", file=sys.stderr)

    # ---- BPM sort for extreme-tempo queries --------------------------------
    # When the user asks for "fastest/slowest tempo" and audio_features.json
    # has data, pre-sort artist_tracks by BPM so the AI sees the extremes
    # near the top of the candidate list even without knowing each band.
    has_max_bpm = "max_bpm" in query_attributes
    has_min_bpm = "min_bpm" in query_attributes
    if audio_features and (has_max_bpm or has_min_bpm):
        def _bpm_sort_key(t):
            feats = audio_features.get(t.get("path", "")) or {}
            b = feats.get("bpm", 0.0)
            return b if b > 0 else (0.0 if has_max_bpm else 9999.0)
        artist_tracks.sort(key=_bpm_sort_key, reverse=has_max_bpm)

    merged_tracks = _merge_tracks_scored(
        artist_tracks, tfidf_tracks, TRACK_CONTEXT_LIMIT,
        lastfm_cache=lastfm_cache,
        mb_priority_indices=mb_priority_indices,
        online_boost_map=online_boost_map,
    )
    step2_candidates = len(merged_tracks)

    if progress_cb:
        progress_cb(f"Шаг 2: AI выбирает из {step2_candidates} реальных треков...")

    track_text = _build_track_text(merged_tracks, audio_features=audio_features)

    # Language / era / attribute / artist hints for Step 2 prompt
    extra_instructions = ""
    try:
        from search_local import expand_query_tokens
        _, intent = expand_query_tokens(user_query)
        # Language: prefer rule-based detection (more reliable than AI for
        # short phrases like "на русском"); fall back to AI decomposition.
        detected_lang = intent.get("language") or structured_intent.get("language")
        if detected_lang == "ru":
            extra_instructions += "\nВАЖНО: выбирай ТОЛЬКО русскоязычные треки (кириллица).\n"
        elif detected_lang == "en":
            extra_instructions += "\nВАЖНО: выбирай ТОЛЬКО англоязычные треки.\n"
        elif detected_lang:
            extra_instructions += f"\nПредпочтение трекам на языке: {detected_lang}.\n"
        if intent.get("era_ranges"):
            eras_str = ", ".join(f"{s}е" for s, _ in intent["era_ranges"])
            extra_instructions += f"Предпочтение трекам {eras_str}.\n"
    except ImportError:
        pass

    # Structured intent hints from AI decomposition
    if structured_intent.get("mood"):
        extra_instructions += f"\nНастроение/атмосфера: {structured_intent['mood']}.\n"
    ai_vocal = structured_intent.get("vocal", "any")
    if ai_vocal == "instrumental" and "инструментальн" not in extra_instructions:
        extra_instructions += "\nВАЖНО: предпочтение инструментальным трекам (без вокала).\n"
    elif ai_vocal == "vocal":
        extra_instructions += "\nПредпочтение трекам с вокалом.\n"
    ai_mode = structured_intent.get("mode", "any")
    if ai_mode == "major":
        extra_instructions += "\nПредпочтение мажорным, жизнерадостным трекам.\n"
    elif ai_mode == "minor":
        extra_instructions += "\nПредпочтение минорным, меланхоличным трекам.\n"
    if structured_intent.get("energy") == "high":
        extra_instructions += "\nПредпочтение энергичным, динамичным трекам.\n"
    elif structured_intent.get("energy") == "low":
        extra_instructions += "\nПредпочтение тихим, спокойным трекам.\n"

    # MusicBrainz confirmation note (helps AI know instrument data is available)
    if mb_priority_indices:
        extra_instructions += (
            f"\nМузыкальные данные: {len(mb_priority_indices)} треков в списке "
            "подтверждены базой MusicBrainz как содержащие запрошенный инструмент. "
            "Отдай им приоритет.\n"
        )

    # Specific artist constraint
    if mentioned_artists:
        names = ", ".join(
            artist_map[k]["name"] for k in mentioned_artists if k in artist_map
        )
        extra_instructions += (
            f"\nВАЖНО: запрос относится к конкретным исполнителям: {names}.\n"
            f"Выбирай ТОЛЬКО треки этих артистов. Треки других артистов не включать.\n"
        )

    # Musical attribute constraints (instrumental, specific instruments, tempo)
    if query_attributes:
        attrs_ru = []
        for attr in query_attributes:
            if attr == "instrumental":
                attrs_ru.append("инструментальные (без вокала)")
            elif attr == "acoustic":
                attrs_ru.append("акустические")
            elif attr.startswith("instrument:"):
                instr = attr.split(":")[1]
                attrs_ru.append(f"с {instr}")
        if attrs_ru:
            extra_instructions += (
                f"\nВАЖНО: запрос требует конкретных музыкальных характеристик: "
                f"{', '.join(attrs_ru)}.\n"
                f"Используй своё знание о треках. Выбирай ТОЛЬКО те треки, которые "
                f"действительно обладают этими характеристиками.\n"
                f"Если подходящих треков нет — верни пустой плейлист <PLAYLIST></PLAYLIST>.\n"
            )

        # Tempo extremes — add genre hints so AI considers less-famous bands
        if has_max_bpm:
            extra_instructions += (
                "\nВАЖНО: выбирай треки с МАКСИМАЛЬНО ВЫСОКИМ темпом (BPM).\n"
                "Если в строке трека указан BPM — ориентируйся на него.\n"
                "Предпочитай жанры с объективно высоким темпом: death metal, black metal, "
                "grindcore, thrash metal, speedcore, drum and bass, hardcore.\n"
                "Не ограничивайся самыми известными исполнителями — "
                "малоизвестные дэт/блэк-метал группы часто быстрее Iron Maiden.\n"
            )
        elif has_min_bpm:
            extra_instructions += (
                "\nВАЖНО: выбирай треки с МИНИМАЛЬНО НИЗКИМ темпом (BPM).\n"
                "Если в строке трека указан BPM — ориентируйся на него.\n"
                "Предпочитай: dark ambient, drone, funeral doom, slow blues, ballads, "
                "post-rock (медленные части).\n"
            )

    # Per-artist diversity cap (soft, in prompt)
    if not mentioned_artists and playlist_size > 5:
        max_per_ai = max(3, playlist_size // 5)
        extra_instructions += (
            f"\nРАЗНООБРАЗИЕ: не включай более {max_per_ai} треков от одного исполнителя. "
            "Старайся охватить как можно больше разных артистов из списка.\n"
        )

    num_requested = min(playlist_size * 2, step2_candidates)
    track_user_msg = (
        f"Треки:\n{track_text}\n\n"
        f"Запрос: \"{user_query}\"\n"
        f"Выбери до {num_requested} наиболее подходящих треков."
        f"{extra_instructions}"
    )

    response = _call_ai(api_provider,
        api_key, model, TRACK_SELECT_PROMPT, track_user_msg, max_tokens=2048
    )

    if progress_cb:
        progress_cb("Обработка ответа...")

    indices = parse_indices(response)

    # Retry if no indices found
    if not indices:
        if progress_cb:
            progress_cb("AI не вернул индексы. Повторная попытка...")
        retry_msg = (
            f"Треки:\n{track_text}\n\n"
            f"Запрос: \"{user_query}\"\n"
            f"Выбери до {num_requested} треков.\n"
            f"{STRICT_PROMPT}"
        )
        response = _call_ai(api_provider,
            api_key, model, STRICT_PROMPT, retry_msg, max_tokens=2048
        )
        indices = parse_indices(response, strict_playlist_tags=False)

    if not indices:
        raise ValueError(f"AI не выбрал ни одного трека. Ответ: {response[:200]}")

    valid_tracks = _validate_tracks(indices, catalog_index)
    if not valid_tracks:
        raise ValueError("Ни один трек не прошёл валидацию.")

    # Hard diversity cap — no artist can dominate the playlist regardless of
    # what the AI returned.  Only applied when no specific artist was named.
    if not mentioned_artists:
        max_per = max(3, playlist_size // 5)
        valid_tracks = _enforce_diversity(valid_tracks, max_per)
        if not valid_tracks:
            raise ValueError("Ни один трек не прошёл валидацию (после фильтра разнообразия).")

    if progress_cb:
        progress_cb("Создание плейлиста...")

    output_path = generate_m3u8(valid_tracks, output_dir, user_query)

    return {
        "output_path": output_path,
        "valid": len(valid_tracks),
        "total": len(indices),
        "skipped": len(indices) - len(valid_tracks),
        "tracks": valid_tracks,
        "step1_artists": step1_artists,
        "step2_candidates": step2_candidates,
    }


def _validate_tracks(indices, catalog_index):
    valid = []
    seen_titles = set()
    for idx in indices:
        if idx not in catalog_index:
            continue
        track = catalog_index[idx]
        if not os.path.exists(track["path"]):
            continue
        artist_lower = track.get("artist", "").lower().strip()
        title_norm = re.sub(
            r"\s*[\(\[](live|concert|remaster(ed)?|acoustic|bonus|demo|"
            r"концерт|вживую|акустика|ремастер|instrumental|radio\s*edit)[\)\]]?\s*$",
            "", track.get("title", ""), flags=re.IGNORECASE,
        ).lower().strip()
        key = f"{artist_lower}|{title_norm}"
        if key in seen_titles:
            continue
        seen_titles.add(key)
        valid.append(track)
    return valid


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Запрос: ").strip()
        if not query:
            print("Запрос не может быть пустым.")
            sys.exit(1)

    try:
        result = create_playlist(user_query=query, progress_cb=print)
        print(f"\nПлейлист создан: {result['valid']} из {result['total']} треков")
        print(f"Артистов в шаге 1: {result['step1_artists']}")
        print(f"Треков в шаге 2: {result['step2_candidates']}")
        if result["skipped"] > 0:
            print(f"Пропущено: {result['skipped']}")
        print(f"Файл: {result['output_path']}")
    except (FileNotFoundError, ValueError, ConnectionError) as e:
        print(f"Ошибка: {e}")
        sys.exit(1)
