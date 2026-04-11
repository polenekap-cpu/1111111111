#!/usr/bin/env python3
"""Acoustic feature extraction via librosa (optional dependency).

Extracts BPM, energy level, mode (major/minor), spectral centroid and
acousticness from audio files.  Results are cached in audio_features.json
and loaded at query time to pre-filter track candidates.

IMPORTANT: librosa is an OPTIONAL dependency.  All public functions
degrade gracefully when librosa is not installed — they return empty dicts
or the unfiltered input list.

Analysis reads only the first ANALYSIS_SECONDS of each file for speed:
~3-8 seconds per track instead of 60-120 seconds for a full analysis.

Feature definitions:
  bpm              float  Tempo in beats per minute (typically 60–220)
  energy           float  Normalised RMS energy 0.0–1.0 (0=silent, 1=loud)
  mode             int    0 = minor,  1 = major  (Krumhansl-Schmuckler)
  spectral_centroid float  Normalised spectral centroid 0.0–1.0 (brightness)
  acousticness     float  Estimated acousticness 0.0–1.0 (1=fully acoustic)
"""

import json
import os

CACHE_FILENAME   = "audio_features.json"
ANALYSIS_SECONDS = 30    # seconds of audio to load per file
_SAVE_EVERY      = 50    # write cache every N newly analysed files

# Krumhansl-Schmuckler key profiles (major and natural minor)
_MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                  2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                  2.54, 4.75, 3.98, 2.69, 3.34, 3.17]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _try_import():
    """Import librosa + numpy. Returns (librosa, np) or (None, None)."""
    try:
        import librosa          # noqa: F401
        import numpy as np      # noqa: F401
        return librosa, np
    except ImportError:
        return None, None


def _pearson(x, y, np):
    """Pearson correlation between two equal-length arrays."""
    x_c = x - np.mean(x)
    y_c = y - np.mean(y)
    denom = np.std(x) * np.std(y)
    if denom < 1e-12:
        return 0.0
    return float(np.mean(x_c * y_c) / denom)


def _estimate_mode(chroma_mean, np):
    """Return 1 (major) or 0 (minor) via Krumhansl-Schmuckler profiles."""
    major = np.array(_MAJOR_PROFILE)
    minor = np.array(_MINOR_PROFILE)
    best_major = max(_pearson(np.roll(chroma_mean, i), major, np)
                     for i in range(12))
    best_minor = max(_pearson(np.roll(chroma_mean, i), minor, np)
                     for i in range(12))
    return 1 if best_major >= best_minor else 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_audio_file(path: str) -> dict:
    """Extract acoustic features from a single audio file.

    Returns a dict with keys: bpm, energy, mode, spectral_centroid,
    acousticness.  Returns {} when librosa is not installed or analysis
    fails for any reason.
    """
    librosa, np = _try_import()
    if librosa is None:
        return {}

    try:
        # Load mono at 22 050 Hz, read at most ANALYSIS_SECONDS
        y, sr = librosa.load(
            path, sr=22050, mono=True,
            duration=ANALYSIS_SECONDS,
            res_type="kaiser_fast",
        )
        if len(y) < sr * 2:          # less than 2 s — skip
            return {}

        # --- BPM ---
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        # librosa ≥ 0.10 returns a scalar; older versions a 1-element array
        bpm = float(tempo[0]) if hasattr(tempo, "__len__") else float(tempo)

        # --- Energy (normalised RMS) ---
        rms = librosa.feature.rms(y=y)[0]
        energy_raw = float(np.mean(rms))
        # Typical music RMS ≈ 0.01–0.25; clip to 0–1
        energy = min(1.0, max(0.0, energy_raw / 0.15))

        # --- Spectral centroid (brightness, normalised) ---
        sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        sc_mean = float(np.mean(sc))
        # Typical range 500–8 000 Hz → normalise linearly
        brightness = min(1.0, max(0.0, (sc_mean - 500.0) / 7500.0))

        # --- Acousticness via spectral flatness ---
        # Low flatness = tonal/acoustic; high flatness = noise-like/electronic
        sf = librosa.feature.spectral_flatness(y=y)[0]
        sf_mean = float(np.mean(sf))
        # Typical range 0.001–0.5; invert and clip to 0–1
        acousticness = min(1.0, max(0.0, 1.0 - sf_mean * 10.0))

        # --- Mode (major / minor) ---
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        mode = _estimate_mode(chroma_mean, np)

        return {
            "bpm":               round(bpm, 1),
            "energy":            round(energy, 3),
            "mode":              mode,
            "spectral_centroid": round(brightness, 3),
            "acousticness":      round(acousticness, 3),
        }

    except Exception:        # noqa: BLE001
        return {}


def load_audio_features(data_dir: str) -> dict:
    """Load audio_features.json.  Returns {} if missing or corrupt."""
    path = os.path.join(data_dir, CACHE_FILENAME)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_features(data_dir: str, features: dict) -> None:
    os.makedirs(data_dir, exist_ok=True)
    tmp = os.path.join(data_dir, CACHE_FILENAME + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(features, f, ensure_ascii=False)
    os.replace(tmp, os.path.join(data_dir, CACHE_FILENAME))


def analyze_catalog(catalog_index: dict, data_dir: str,
                    progress_cb=None) -> dict:
    """Incrementally analyse all tracks in catalog_index.

    Skips files already present in audio_features.json (keyed by file path).
    Saves the cache every _SAVE_EVERY newly analysed files.

    Args:
        catalog_index: {index: {artist, title, year, path, ...}}
        data_dir:      directory for audio_features.json
        progress_cb:   optional callback(current, total, message)

    Returns:
        Full features dict (including pre-existing entries).
    """
    librosa, _ = _try_import()
    if librosa is None:
        if progress_cb:
            progress_cb(0, 0,
                "librosa не установлена. "
                "Установите командой: pip install librosa")
        return {}

    features = load_audio_features(data_dir)
    tracks   = list(catalog_index.items())
    total    = len(tracks)
    new_cnt  = 0

    for i, (_, track) in enumerate(tracks):
        path = track.get("path", "")
        if not path or not os.path.exists(path):
            continue
        if path in features:
            continue                 # already analysed

        result = analyze_audio_file(path)
        # Store result or empty dict (marks the file as "attempted")
        features[path] = result if result else {}
        new_cnt += 1

        if new_cnt % _SAVE_EVERY == 0:
            _save_features(data_dir, features)

        if progress_cb and (i + 1) % 10 == 0:
            progress_cb(i + 1, total,
                f"Анализ аудио: {i+1:,}/{total:,} — "
                f"{os.path.basename(path)}")

    _save_features(data_dir, features)

    if progress_cb:
        progress_cb(total, total,
            f"Акустический анализ завершён: {new_cnt:,} новых треков")

    return features


def build_audio_constraints(structured_intent: dict) -> dict:
    """Convert a structured query intent into audio filter constraints.

    Args:
        structured_intent: dict from query_engine.decompose_query()
            with keys: energy, bpm_min, bpm_max, mode

    Returns:
        Dict with optional keys: energy_min, energy_max,
        bpm_min, bpm_max, mode, acousticness_min.
        Empty dict when no constraints can be derived.
    """
    if not structured_intent:
        return {}

    constraints = {}
    energy_level = structured_intent.get("energy", "medium")

    if energy_level == "high":
        constraints["energy_min"] = 0.5
        # Don't force BPM unless AI also said something about it
    elif energy_level == "low":
        constraints["energy_max"] = 0.35

    if structured_intent.get("bpm_min"):
        try:
            constraints["bpm_min"] = float(structured_intent["bpm_min"])
        except (TypeError, ValueError):
            pass
    if structured_intent.get("bpm_max"):
        try:
            constraints["bpm_max"] = float(structured_intent["bpm_max"])
        except (TypeError, ValueError):
            pass

    mode = structured_intent.get("mode", "any")
    if mode == "major":
        constraints["mode"] = 1
    elif mode == "minor":
        constraints["mode"] = 0

    if structured_intent.get("vocal") == "instrumental":
        # Instrumental tracks are often quieter live recordings or
        # orchestral — slightly prefer more acoustic profile
        constraints.setdefault("acousticness_min", 0.0)   # no hard floor

    return constraints


def filter_by_audio_features(
    track_list: list,
    audio_features: dict,
    constraints: dict,
) -> list:
    """Pre-filter a list of track dicts by acoustic constraints.

    Each element of track_list must have a "path" key.

    Strategy: if filtering eliminates more than 80 % of the tracks that
    *have* audio data, the filter is too aggressive — fall back to the
    unfiltered list (prevents empty results when the library has not
    been fully analysed yet).

    Args:
        track_list:     list of track dicts (each with "path")
        audio_features: {path: {bpm, energy, mode, ...}}
        constraints:    from build_audio_constraints()

    Returns:
        Filtered list, or original list when filtering is not applicable.
    """
    if not constraints or not audio_features:
        return track_list

    passing      = []
    no_data      = []
    analysed_cnt = 0

    for track in track_list:
        path  = track.get("path", "")
        feats = audio_features.get(path)

        if feats is None or not feats:
            no_data.append(track)
            continue

        analysed_cnt += 1
        bpm          = feats.get("bpm", 0.0)
        energy       = feats.get("energy", 0.5)
        mode         = feats.get("mode", -1)
        acousticness = feats.get("acousticness", 0.5)

        if "bpm_min" in constraints and bpm > 0 and bpm < constraints["bpm_min"]:
            continue
        if "bpm_max" in constraints and bpm > 0 and bpm > constraints["bpm_max"]:
            continue
        if "energy_min" in constraints and energy < constraints["energy_min"]:
            continue
        if "energy_max" in constraints and energy > constraints["energy_max"]:
            continue
        if "mode" in constraints and mode != -1 and mode != constraints["mode"]:
            continue
        if "acousticness_min" in constraints and acousticness < constraints["acousticness_min"]:
            continue

        passing.append(track)

    # If fewer than 20 % of analysed tracks pass, fall back to avoid
    # an empty or overly thin candidate list.
    if analysed_cnt > 0 and len(passing) < max(5, analysed_cnt // 5):
        return track_list

    # Always include un-analysed tracks (don't punish them for lack of data)
    return passing + no_data
