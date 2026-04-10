#!/usr/bin/env python3
"""Builds a music catalog from local audio files for AI playlist generation.

Can be used as a module (import build_catalog) or run standalone.
Progress is reported via an optional callback(current, total, message).
"""

import json
import logging
import os
import re
import sys
import tempfile
import time

from mutagen import File as MutagenFile

AUDIO_EXTENSIONS = {".mp3", ".flac", ".ogg", ".m4a", ".wav"}

logger = logging.getLogger("catalog_builder")
_logging_configured = False


def load_config(config_path=None):
    """Load config.json and expand paths."""
    if config_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json не найден: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    for key in ("catalog_path", "ai_catalog_path", "output_dir"):
        cfg[key] = os.path.expanduser(cfg[key])
    cfg["music_dirs"] = [
        os.path.realpath(os.path.expanduser(d)) for d in cfg["music_dirs"]
    ]
    return cfg


def setup_logging(log_dir=None):
    """Configure error logging to file. Only adds handler once."""
    global _logging_configured
    if _logging_configured:
        return
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    handler = logging.FileHandler(
        os.path.join(log_dir, "build_log.txt"), encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.ERROR)
    _logging_configured = True


def sanitize_field(value):
    """Remove tab, pipe, newline from a field to prevent TSV/pipe injection."""
    if not value:
        return ""
    return value.replace("\t", " ").replace("|", "/").replace("\n", " ").replace("\r", "")


def discover_audio_files(music_dirs, progress_cb=None):
    """Walk music directories and return {absolute_path: mtime}.

    Reports progress during discovery via callback.
    """
    files = {}
    last_report = time.monotonic()
    for music_dir in music_dirs:
        if not os.path.isdir(music_dir):
            if progress_cb:
                progress_cb(0, 0, f"Папка не найдена: {music_dir}")
            continue
        for root, _, filenames in os.walk(music_dir):
            for fname in filenames:
                ext = os.path.splitext(fname)[1].lower()
                if ext in AUDIO_EXTENSIONS:
                    full_path = os.path.join(root, fname)
                    try:
                        files[full_path] = os.path.getmtime(full_path)
                    except OSError:
                        pass

            # Report discovery progress at most every 0.3 seconds
            now = time.monotonic()
            if progress_cb and now - last_report > 0.3:
                progress_cb(0, 0, f"Поиск файлов... {len(files):,}")
                last_report = now

    return files


def read_tags(filepath):
    """Read artist, title, year from audio file tags via mutagen."""
    try:
        audio = MutagenFile(filepath, easy=True)
        if audio is None:
            raise ValueError("mutagen returned None")
        artist = (audio.get("artist") or [None])[0]
        title = (audio.get("title") or [None])[0]
        year_raw = (audio.get("date") or audio.get("year") or [None])[0]
        year = year_raw[:4] if year_raw and len(year_raw) >= 4 else None
        return artist, title, year
    except Exception as e:
        logger.error("Tag read failed: %s: %s", filepath, e)
        return None, None, None


def parse_filename(filepath):
    """Extract artist and title from filename pattern 'Artist - Title.ext'."""
    stem = os.path.splitext(os.path.basename(filepath))[0]
    if " - " in stem:
        parts = stem.split(" - ", 1)
        return parts[0].strip(), parts[1].strip()
    return None, stem.strip()


def process_file(filepath):
    """Read tags for a file, with filename fallback. Returns sanitized fields."""
    artist, title, year = read_tags(filepath)
    fn_artist, fn_title = parse_filename(filepath)
    if not artist:
        artist = fn_artist or ""
    if not title:
        title = fn_title or os.path.splitext(os.path.basename(filepath))[0]
    if not year:
        year = ""
    return sanitize_field(artist), sanitize_field(title), sanitize_field(year)


def load_existing_catalog(catalog_path):
    """Load existing catalog.tsv into a dict {path: entry_dict}."""
    catalog = {}
    if not os.path.exists(catalog_path):
        return catalog
    with open(catalog_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 6:
                continue
            idx, artist, title, year, path, mtime_str = parts
            try:
                catalog[path] = {
                    "index": int(idx),
                    "artist": artist,
                    "title": title,
                    "year": year,
                    "mtime": float(mtime_str),
                }
            except (ValueError, IndexError):
                continue
    return catalog


def _atomic_write(filepath, write_func):
    """Write to a temp file then rename for crash safety."""
    dir_name = os.path.dirname(filepath) or "."
    os.makedirs(dir_name, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            write_func(f)
        os.replace(tmp_path, filepath)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


_LIVE_SUFFIXES = re.compile(
    r"\s*[\(\[](live|concert|концерт|вживую|acoustic|акустика|"
    r"remaster(ed)?|ремастер|bonus\s*track|demo|instrumental|"
    r"radio\s*edit|single\s*version|deluxe)[\)\]]?\s*$",
    re.IGNORECASE,
)


def _normalize_title(title):
    """Strip live/remaster/acoustic suffixes for deduplication."""
    return _LIVE_SUFFIXES.sub("", title).strip()


def _normalize_artist(artist):
    """Lowercase and strip whitespace for grouping."""
    return artist.strip().lower()


def _build_artist_index(catalog):
    """Build artist-level summary for Phase 1 of two-phase query.

    Returns:
        artists_by_id: {artist_id: {name, track_count, years, sample_titles}}
        tracks_by_artist: {artist_id: [track_indices]}
    """
    artist_groups = {}  # normalized_name -> {name, indices, years, titles}

    for entry in catalog.values():
        artist = entry.get("artist", "").strip()
        if not artist:
            continue
        key = _normalize_artist(artist)
        if key not in artist_groups:
            artist_groups[key] = {
                "name": artist,  # keep original casing from first encounter
                "indices": [],
                "years": set(),
                "titles": [],
            }
        group = artist_groups[key]
        group["indices"].append(entry["index"])
        if entry.get("year"):
            group["years"].add(entry["year"])
        group["titles"].append(entry.get("title", ""))

    # Assign artist IDs, build compact summaries
    artists_by_id = {}
    tracks_by_artist = {}
    for aid, (_, group) in enumerate(sorted(artist_groups.items()), start=1):
        years = sorted(group["years"])
        year_range = ""
        if years:
            if len(years) == 1:
                year_range = years[0]
            else:
                year_range = f"{years[0]}-{years[-1]}"

        # Pick up to 3 sample titles for context
        unique_titles = []
        seen_normalized = set()
        for t in group["titles"]:
            norm = _normalize_title(t)
            if norm.lower() not in seen_normalized:
                seen_normalized.add(norm.lower())
                unique_titles.append(t)
            if len(unique_titles) >= 3:
                break

        artists_by_id[aid] = {
            "name": group["name"],
            "track_count": len(group["indices"]),
            "year_range": year_range,
            "sample_titles": unique_titles,
        }
        tracks_by_artist[aid] = group["indices"]

    return artists_by_id, tracks_by_artist


def write_catalogs(catalog, catalog_path, ai_catalog_path):
    """Write catalog.tsv, catalog_for_ai.txt, artists_for_ai.txt, and tracks_by_artist.json."""
    sorted_entries = sorted(catalog.values(), key=lambda e: e["index"])
    base_dir = os.path.dirname(catalog_path) or "."

    def write_tsv(f):
        for entry in sorted_entries:
            f.write(
                f"{entry['index']}\t{entry['artist']}\t{entry['title']}\t"
                f"{entry['year']}\t{entry['path']}\t{entry['mtime']}\n"
            )

    def write_ai(f):
        for entry in sorted_entries:
            f.write(
                f"{entry['index']}|{entry['artist']}|{entry['title']}|"
                f"{entry['year']}\n"
            )

    _atomic_write(catalog_path, write_tsv)
    _atomic_write(ai_catalog_path, write_ai)

    # Build and write artist index for two-phase query
    artists_by_id, tracks_by_artist = _build_artist_index(catalog)

    artists_path = os.path.join(base_dir, "artists_for_ai.txt")

    def write_artists(f):
        for aid in sorted(artists_by_id.keys()):
            a = artists_by_id[aid]
            samples = "; ".join(a["sample_titles"])
            # Format: ID|ARTIST|TRACKS_COUNT|YEARS|SAMPLE_TITLES
            f.write(f"{aid}|{a['name']}|{a['track_count']}|{a['year_range']}|{samples}\n")

    _atomic_write(artists_path, write_artists)

    # Write tracks-by-artist mapping as JSON
    mapping_path = os.path.join(base_dir, "tracks_by_artist.json")
    def write_mapping(f):
        json.dump(
            {str(k): v for k, v in tracks_by_artist.items()},
            f, ensure_ascii=False,
        )
    _atomic_write(mapping_path, write_mapping)


def build_catalog(config_path=None, progress_cb=None):
    """Build or update the music catalog.

    Args:
        config_path: Path to config.json. None = auto-detect next to script.
        progress_cb: Optional callback(current, total, message) for progress.
                     Called at most every ~100 files to avoid flooding the UI.

    Returns:
        dict with keys: total, new, changed, removed, catalog_path
    """
    cfg = load_config(config_path)
    setup_logging()

    catalog_path = cfg["catalog_path"]
    ai_catalog_path = cfg["ai_catalog_path"]

    existing = load_existing_catalog(catalog_path)

    if existing:
        next_index = max(e["index"] for e in existing.values()) + 1
    else:
        next_index = 1

    if progress_cb:
        progress_cb(0, 0, "Сканирование папок...")

    discovered = discover_audio_files(cfg["music_dirs"], progress_cb)

    if progress_cb:
        progress_cb(0, 0, f"Найдено аудиофайлов: {len(discovered):,}")

    existing_paths = set(existing.keys())
    discovered_paths = set(discovered.keys())

    new_paths = discovered_paths - existing_paths
    removed_paths = existing_paths - discovered_paths
    changed_paths = {
        p
        for p in discovered_paths & existing_paths
        if discovered[p] > existing[p]["mtime"]
    }

    for p in removed_paths:
        del existing[p]

    to_process = list(changed_paths) + sorted(new_paths)
    total = len(to_process)

    # Throttle progress: report every BATCH_SIZE files or 0.5 seconds
    BATCH_SIZE = 100
    last_report_time = time.monotonic()

    for i, p in enumerate(to_process):
        artist, title, year = process_file(p)

        if p in changed_paths:
            entry = existing[p]
            entry["artist"] = artist
            entry["title"] = title
            entry["year"] = year
            entry["mtime"] = discovered[p]
        else:
            existing[p] = {
                "index": next_index,
                "artist": artist,
                "title": title,
                "year": year,
                "path": p,
                "mtime": discovered[p],
            }
            next_index += 1

        # Throttled progress reporting
        if progress_cb:
            now = time.monotonic()
            if (i + 1) % BATCH_SIZE == 0 or now - last_report_time > 0.5 or i + 1 == total:
                progress_cb(i + 1, total, f"Обработка: {i + 1:,} / {total:,}")
                last_report_time = now

    for path, entry in existing.items():
        if "path" not in entry:
            entry["path"] = path

    if progress_cb:
        progress_cb(total, total, "Запись каталога...")

    write_catalogs(existing, catalog_path, ai_catalog_path)

    result = {
        "total": len(existing),
        "new": len(new_paths),
        "changed": len(changed_paths),
        "removed": len(removed_paths),
        "catalog_path": catalog_path,
    }

    if progress_cb:
        progress_cb(
            total, total,
            f"Готово: {result['total']:,} треков "
            f"({result['new']:,} новых, {result['changed']:,} обновлённых, "
            f"{result['removed']:,} удалённых)",
        )

    return result


if __name__ == "__main__":
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    bar = [None]

    def cli_progress(current, total, message):
        if tqdm and total > 0:
            if bar[0] is None:
                bar[0] = tqdm(total=total, desc="Обработка")
            bar[0].n = current
            bar[0].refresh()
            if current >= total:
                bar[0].close()
                bar[0] = None
                print(message)
        else:
            print(message)

    try:
        result = build_catalog(progress_cb=cli_progress)
        print(
            f"Каталог обновлён: {result['total']:,} треков "
            f"({result['new']:,} новых, {result['changed']:,} обновлённых, "
            f"{result['removed']:,} удалённых)"
        )
    except FileNotFoundError as e:
        print(f"Ошибка: {e}")
        sys.exit(1)
