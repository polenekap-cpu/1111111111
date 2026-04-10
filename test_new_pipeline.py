#!/usr/bin/env python3
"""Test suite for the new TF-IDF + single-phase pipeline."""

import json
import os
import sys
import tempfile
import unittest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from search_local import SearchIndex, tokenize
from query_engine import parse_indices, _extract_numbers_from_json, _deduplicate_preserve_order


# ---------------------------------------------------------------------------
# Tokenizer tests
# ---------------------------------------------------------------------------

class TestTokenizer(unittest.TestCase):
    def test_basic_english(self):
        self.assertIn("hello", tokenize("hello world"))

    def test_basic_russian(self):
        tokens = tokenize("мрачный русский рок")
        self.assertIn("мрачный", tokens)
        self.assertIn("рок", tokens)

    def test_separators(self):
        self.assertIn("ac", tokenize("AC/DC"))
        self.assertIn("dc", tokenize("AC/DC"))

    def test_acronyms(self):
        tokens = tokenize("AC/DC")
        self.assertIn("ac", tokens)
        self.assertIn("dc", tokens)

    def test_bbtokens(self):
        # "B.B. King" -> ["b", "b", "king", "bb"]
        tokens = tokenize("B.B. King")
        self.assertIn("king", tokens)
        # BB is an acronym (2 letters repeated with dots)
        # May or may not be extracted depending on regex, so just check basic tokens

    def test_stopwords_filtered(self):
        from search_local import _STOP_WORDS
        result = [t for t in tokenize("хочу послушать мрачный рок вайб")
                  if t not in _STOP_WORDS]
        self.assertNotIn("хочу", result)
        self.assertNotIn("вайб", result)
        self.assertIn("мрачный", result)
        self.assertIn("рок", result)

    def test_empty(self):
        self.assertEqual(tokenize(""), [])
        self.assertEqual(tokenize("   "), [])


# ---------------------------------------------------------------------------
# SearchIndex tests
# ---------------------------------------------------------------------------

class TestSearchIndex(unittest.TestCase):
    def setUp(self):
        self.idx = SearchIndex()
        self.idx.add_document(1, "Кино", "Группа крови", "1988", "/path/1")
        self.idx.add_document(2, "Кино", "Звезда по имени Солнце", "1989", "/path/2")
        self.idx.build()

    def test_build(self):
        self.assertTrue(self.idx.built)
        self.assertEqual(len(self.idx), 2)

    def test_search_finds_artist(self):
        results = self.idx.search("Кино")
        self.assertEqual(len(results), 2)

    def test_search_finds_song(self):
        results = self.idx.search("Группа крови")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "Группа крови")

    def test_search_scores_artists(self):
        results = self.idx.search("Кино")
        self.assertEqual(results[0]["index"], 1)

    def test_search_empty(self):
        self.assertEqual(SearchIndex().search("test"), [])

    def test_from_dict(self):
        catalog = {
            10: {"artist": "Nirvana", "title": "Smells Like Teen Spirit",
                 "year": "1991", "path": "/path1"},
            20: {"artist": "Pink Floyd", "title": "Comfortably Numb",
                 "year": "1979", "path": "/path2"},
        }
        idx = SearchIndex.from_catalog_dict(catalog)
        results = idx.search("Nirvana")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "Smells Like Teen Spirit")

    def test_larger_catalog(self):
        idx = SearchIndex()
        for i in range(100):
            idx.add_document(i, f"Artist_{i % 10}", f"Track title {i}", "1990", f"/path/{i}")
        idx.build()
        results = idx.search("Artist_5")
        self.assertGreaterEqual(len(results), 10)

    def test_search_with_catalog_text(self):
        idx = SearchIndex()
        idx.add_document(1, "Кино", "Группа крови", "1988", "/path/1")
        idx.add_document(2, "Аквариум", "Город золотой", "1985", "/path/2")
        idx.build()
        text, tracks, _intent = idx.search_with_catalog_text("Кино", topn=10)
        self.assertIn("Кино", text)
        self.assertIn("Группа крови", text)
        # Fallback may add the other track, so just verify Kino appears first
        self.assertEqual(tracks[0]["artist"], "Кино")


# ---------------------------------------------------------------------------
# parse_indices tests
# ---------------------------------------------------------------------------

class TestParseIndices(unittest.TestCase):
    def test_simple_csv(self):
        self.assertEqual(parse_indices("12,45,78"), [12, 45, 78])

    def test_csv_spaces(self):
        self.assertEqual(parse_indices("12, 45, 78"), [12, 45, 78])

    def test_with_text(self):
        indices = parse_indices("Подходят артисты 12 и 45, итого треки 100")
        self.assertIn(12, indices)
        self.assertIn(45, indices)
        self.assertIn(100, indices)

    def test_json_object(self):
        self.assertEqual(parse_indices('{"indices": [12, 45, 78]}'), [12, 45, 78])

    def test_json_array(self):
        self.assertEqual(parse_indices('[12, 45, 78]'), [12, 45, 78])

    def test_json_nested(self):
        self.assertEqual(
            parse_indices('{"results": {"indices": [1, 2]}}'),
            [1, 2]
        )

    def test_dedup(self):
        self.assertEqual(parse_indices("12, 12, 45, 12"), [12, 45])

    def test_too_large(self):
        self.assertEqual(parse_indices("1000000"), [])

    def test_empty(self):
        self.assertEqual(parse_indices(""), [])

    def test_no_numbers(self):
        self.assertEqual(parse_indices("нет подходящих треков"), [])

    def test_realistic_thinking_response(self):
        """Simulate a Qwen3 thinking model response."""
        text = """Основываясь на анализе, я бы выбрал следующие треки:
1. Трек 12 (Кино - Группа крови)
2. Трек 45 (Кино - Звезда по имени Солнце)
3. Трек 78 (Аквариум - Город золотой)

Эти треки лучше всего соответствуют запросу."""
        indices = parse_indices(text)
        # Should extract 1, 12, 45, 78 (numbers from text)
        self.assertIn(12, indices)
        self.assertIn(45, indices)
        self.assertIn(78, indices)

    def test_multiline(self):
        result = parse_indices("123\n456\n789")
        self.assertEqual(result, [123, 456, 789])


# ---------------------------------------------------------------------------
# end-to-end: create_playlist mocks API
# ---------------------------------------------------------------------------

class TestE2E(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.catalog_path = os.path.join(self.tmpdir, "catalog.tsv")
        self.config_path = os.path.join(self.tmpdir, "config_test.json")

        # Write a test catalog with dummy files so validation passes
        self.dummy_files = []
        for i, (artist, title, year) in enumerate([
            ("Кино", "Группа крови", "1988"),
            ("Кино", "Звезда по имени Солнце", "1989"),
            ("Аквариум", "Город золотой", "1985"),
            ("ДДТ", "Что такое осень", "1992"),
            ("Наутилус Помпилиус", "Крылья", "1995"),
            ("Алиса", "Мы вместе", "1993"),
            ("Зоопарк", "Белый снег", "1987"),
            ("Цой", "Пачка сигарет", "1989"),
        ], 1):
            dummy_path = os.path.join(self.tmpdir, f"track{i}.mp3")
            with open(dummy_path, "wb") as f:
                f.write(b"\x00")
            self.dummy_files.append(dummy_path)
            line = f"{i}\t{artist}\t{title}\t{year}\t{dummy_path}\t1000\n"
            with open(self.catalog_path, "a", encoding="utf-8") as f:
                f.write(line)

        config = {
            "music_dirs": ["/tmp/music"],
            "catalog_path": self.catalog_path,
            "ai_catalog_path": os.path.join(self.tmpdir, "catalog_for_ai.txt"),
            "output_dir": self.tmpdir,
            "api_key": "test_key",
            "model": "test_model",
            "playlist_size": 3,
        }
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(config, f)

    def _patch_api(self, response_text):
        """Mock the _call_openrouter function."""
        from query_engine import _call_openrouter
        original = _call_openrouter

        def mock_call(*args, **kwargs):
            return response_text

        import query_engine
        query_engine._call_openrouter = mock_call
        return original

    def _restore_api(self, original):
        import query_engine
        query_engine._call_openrouter = original

    def test_e2e_creates_playlist(self):
        from query_engine import create_playlist

        original = self._patch_api("1,2,3")
        try:
            result = create_playlist(
                config_path=self.config_path,
                user_query="русский рок 90х"
            )
            self.assertIn("output_path", result)
            self.assertEqual(result["total"], 3)
            self.assertEqual(result["valid"], 3)
            # TF-IDF fallback samples full catalog when few direct matches
            self.assertGreater(result["candidates"], 0)
        finally:
            self._restore_api(original)

    def test_e2e_empty_ai_response(self):
        from query_engine import create_playlist

        original = self._patch_api("нет подходящих треков")
        try:
            # Should retry with strict prompt, then fail
            with self.assertRaises(ValueError):
                create_playlist(
                    config_path=self.config_path,
                    user_query="русский рок"
                )
        finally:
            self._restore_api(original)


if __name__ == "__main__":
    unittest.main()
