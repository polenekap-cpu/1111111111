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

ARTIST_SELECT_PROMPT = """\
Тебе дан пронумерованный список артистов из локальной музыкальной библиотеки.
Формат: INDEX|АРТИСТ

Задача: выбери артистов, у которых НАИБОЛЕЕ ВЕРОЯТНО есть треки, подходящие под запрос пользователя.
Учитывай жанр, стиль, язык, настроение и эпоху. Выбирай щедро — лучше взять лишних, чем пропустить нужных.
Целевое количество: 30–100 артистов (больше для широких запросов, меньше для конкретных).

Ответ: верни ТОЛЬКО номера через запятую внутри тегов <PLAYLIST> и </PLAYLIST>.
Пример: <PLAYLIST>3,17,42,88,103</PLAYLIST>"""

TRACK_SELECT_PROMPT = """\
Ты — музыкальный куратор. Тебе дан список треков из реального каталога (без галлюцинаций).
Формат: INDEX|АРТИСТ|НАЗВАНИЕ|ГОД

Задача: выбери до N треков, идеально подходящих под запрос.

Правила:
- Выбирай только треки, точно подходящие по жанру, настроению и эпохе.
- Соблюдай язык запроса: «на русском» → ТОЛЬКО кириллические треки, «in english» → ТОЛЬКО латиница.
- НЕ дублируй: если есть «Song (Live)» и «Song» — только студийную версию.
- Не ставь подряд треки одного артиста.
- Лучше меньше, но точнее.

Ответ: верни ТОЛЬКО индексы через запятую внутри тегов <PLAYLIST> и </PLAYLIST>.
Пример: <PLAYLIST>1452,891,23044,7821,445</PLAYLIST>"""

STRICT_PROMPT = """\
Ответь ТОЛЬКО индексами через запятую внутри тегов <PLAYLIST></PLAYLIST>.
Пример: <PLAYLIST>1452,891,23044</PLAYLIST>"""

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
GOOGLE_API_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
DEFAULT_MODEL = "qwen/qwen3-235b-a22b:free"
DEFAULT_GOOGLE_MODEL = "gemini-2.0-flash"

ARTIST_SELECT_TARGET = 80
TRACK_CONTEXT_LIMIT = 1500
TFIDF_CANDIDATES = 300


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
    filename = f"{safe_query}_{timestamp}.m3u8"
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

def _build_artist_list_text(artist_map):
    """Format numbered artist list. Returns (text, {number: artist_key})."""
    lines = []
    num_to_key = {}
    for i, (key, info) in enumerate(sorted(artist_map.items()), start=1):
        lines.append(f"{i}|{info['name']}")
        num_to_key[i] = key
    return "\n".join(lines), num_to_key


def _get_tracks_for_artists(selected_keys, artist_map, catalog_index, limit):
    """Collect real catalog tracks for selected artists, capped at limit."""
    tracks = []
    for key in selected_keys:
        info = artist_map.get(key)
        if not info:
            continue
        for idx in info["indices"]:
            track = catalog_index.get(idx)
            if track:
                tracks.append({"index": idx, **track})
    tracks.sort(key=lambda t: (t.get("artist", "").lower(), t.get("title", "").lower()))
    return tracks[:limit]


def _build_track_text(tracks):
    lines = []
    for t in tracks:
        lines.append(f"{t['index']}|{t.get('artist','')}|{t.get('title','')}|{t.get('year','')}")
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

    # ---- Step 1: AI selects relevant artists --------------------------------
    if progress_cb:
        progress_cb(f"Шаг 1: отбор артистов из {num_artists} в каталоге...")

    artist_list_text, num_to_key = _build_artist_list_text(artist_map)
    artist_user_msg = (
        f"Список артистов:\n{artist_list_text}\n\n"
        f"Запрос: \"{user_query}\"\n"
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

    # Fallback if step 1 returned nothing
    if not selected_artist_keys:
        if progress_cb:
            progress_cb("Шаг 1 не вернул артистов. Используем весь каталог.")
        selected_artist_keys = set(artist_map.keys())

    # ---- Step 2b: TF-IDF for literal keyword matches (parallel) -------------
    tfidf_indices = set()
    tfidf_tracks = []
    try:
        from search_local import SearchIndex
        if progress_cb:
            progress_cb("TF-IDF: поиск буквальных совпадений...")
        search_idx = SearchIndex.from_catalog_dict(catalog_index)
        tfidf_results = search_idx.search(user_query, topn=TFIDF_CANDIDATES)
        tfidf_indices = {r["index"] for r in tfidf_results}
        tfidf_tracks = tfidf_results
    except ImportError:
        pass

    # ---- Step 2: Collect real tracks + merge TF-IDF -------------------------
    artist_tracks = _get_tracks_for_artists(
        selected_artist_keys, artist_map, catalog_index, TRACK_CONTEXT_LIMIT
    )
    artist_track_indices = {t["index"] for t in artist_tracks}
    merged_tracks = list(artist_tracks)
    for t in tfidf_tracks:
        if t["index"] not in artist_track_indices:
            merged_tracks.append(t)
    merged_tracks = merged_tracks[:TRACK_CONTEXT_LIMIT]
    step2_candidates = len(merged_tracks)

    if progress_cb:
        progress_cb(f"Шаг 2: AI выбирает из {step2_candidates} реальных треков...")

    track_text = _build_track_text(merged_tracks)

    # Language/era hints from intent parser
    extra_instructions = ""
    try:
        from search_local import expand_query_tokens
        _, intent = expand_query_tokens(user_query)
        if intent.get("language") == "ru":
            extra_instructions = "\nВАЖНО: выбирай ТОЛЬКО русскоязычные треки (кириллица).\n"
        elif intent.get("language") == "en":
            extra_instructions = "\nВАЖНО: выбирай ТОЛЬКО англоязычные треки.\n"
        if intent.get("era_ranges"):
            eras_str = ", ".join(f"{s}е" for s, _ in intent["era_ranges"])
            extra_instructions += f"Предпочтение трекам {eras_str}.\n"
    except ImportError:
        pass

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
