#!/usr/bin/env python3
"""Local TF-IDF search engine for music catalog. Zero external dependencies.

Builds a lightweight search index from catalog.tsv using artist/title/genre
tokenization and returns top-N most relevant tracks for a natural-language
query. Used as a pre-filter before sending to the AI, so only relevant
tracks (~200-500) are included in the prompt instead of the full catalog
(40 000+ tracks).

Features:
- Query intent parsing: extracts language, atmosphere, genre, era, artists
- Atmosphere word expansion: "осень" → search for "осень", "дождь", "листья"
- Language detection: "на русском" → filters to Cyrillic artists/titles
- TF-IDF scoring with bonus points for atmosphere + language matches
"""

import math
import os
import re
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

# Words to ignore in queries (RU + EN stop-words for music context)
_STOP_WORDS = frozenset({
    # English
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "it", "its", "this", "that", "these", "those", "i", "me", "my", "we",
    "our", "you", "your", "he", "she", "they", "them", "his", "her",
    "not", "no", "so", "if", "all", "as", "do", "just", "want", "need",
    "can", "will", "would", "could", "should", "like", "get", "got",
    "give", "make", "made", "has", "had", "have", "does", "did", "am",
    "some", "any", "than", "too", "very", "much", "more", "most",
    # Russian
    "а", "без", "более", "больше", "большой", "будет", "бы", "был", "была",
    "были", "быть", "в", "во", "вот", "всю", "где", "да", "давай", "для",
    "до", "его", "ее", "если", "есть", "еще", "ж", "же", "за", "здесь",
    "и", "из", "им", "их", "к", "как", "когда", "кто", "ли", "мне", "меня",
    "много", "может", "мои", "мой", "на", "над", "наш", "не", "него",
    "нее", "нет", "ни", "них", "но", "ну", "о", "об", "он", "она", "они",
    "оно", "от", "очень", "по", "под", "при", "про", "с", "со", "та",
    "так", "такой", "там", "те", "тебе", "тебя", "то", "тоже", "тому",
    "тот", "ты", "у", "уже", "что", "чтобы", "эта", "эти", "это", "я",
    "мне", "нам", "все", "вот", "его", "ее",
    # Music context
    "music", "музыка", "песни", "треки", "трек", "песня", "скачать",
    "скачать", "слушать", "хочу", "дай", "дайте", "подбери", "подберёшь",
    "сделай", "создай", "найти", "найди", "включи", "запусти",
    "playlist", "плейлист",
    # Atmosphere / filler words replaced by expansion dictionary
    "вайб", "атмосферная", "атмосферный", "вайбовая", "вайбовый",
    "настроение", "настроения",
})

# ---------------------------------------------------------------------------
# Query Intent Parser
# ---------------------------------------------------------------------------

# Language preference detection patterns: keyword → language code
_LANGUAGE_PATTERNS = [
    # Russian
    (re.compile(r'\b(на\s+русс(?:к(?:ом|и|ой)|ском|ских))\b', re.IGNORECASE), "ru"),
    (re.compile(r'\b(русскоязычн[a-zа-яё]+)\b', re.IGNORECASE), "ru"),
    (re.compile(r'\b(?:советск[a-zа-яё]+)\b', re.IGNORECASE), "ru"),
    # English
    (re.compile(r'\b(in\s+english|english)\b', re.IGNORECASE), "en"),
    (re.compile(r'\b(английск[a-zа-яё]+|на\s+английском)\b', re.IGNORECASE), "en"),
    # French
    (re.compile(r'\b(french|на\s+французском|французск[a-zа-яё]+)\b', re.IGNORECASE), "fr"),
    # Japanese
    (re.compile(r'\b(japanese|j[- ]?pop|на\s+японском|японск[a-zа-яё]+)\b', re.IGNORECASE), "ja"),
    # Korean
    (re.compile(r'\b(k[- ]?pop|korean|на\s+корейском|корейск[a-zа-яё]+)\b', re.IGNORECASE), "ko"),
    # Spanish
    (re.compile(r'\b(spanish|испанск[a-zа-яё]+)\b', re.IGNORECASE), "es"),
]

# Atmosphere / mood word expansion: atmosphere word → related searchable words
# These words may actually appear in artist names or song titles
_ATMOSPHERE_EXPANSIONS = {
    # Seasons
    "осень": ["осень", "осен", "листья", "дождь", "дожд", "ноябрь", "октябрь",
              "сентябрь", "поздн", "холод", "хмур", "туман", "ветер"],
    "зима": ["зима", "зимн", "снег", "декабрь", "январь", "февраль",
             "мороз", "холод", "лёд", "лед"],
    "весна": ["весна", "весенн", "март", "апрель", "май", "цвет", "капель",
              "солнц", "тепло"],
    "лето": ["лето", "летн", "июнь", "июль", "август", "солнц", "жар",
             "тепло", "море", "пляж"],
    # Moods
    "грусть": ["грусть", "грустн", "печаль", "печальн", "тоска", "тоскл",
               "слёз", "слез", "плач"],
    "счастье": ["радость", "радостн", "счасть", "счастл", "весел", "смех",
                "солнц", "светл", "любовь"],
    "любовь": ["любовь", "любим", "люб", "сердц", "серд", "чувств", "твоё",
               "твоей", "любимый", "нежность"],
    "злость": ["злость", "злой", "гнев", "агресс", "бешен", "ярость"],
    "тоска": ["тоска", "тоскл", "одиночеств", "один", "пустот", "пуст"],
    # Time of day
    "ночь": ["ночь", "ночн", "ночью", "лун", "темн", "полночь", "звезд"],
    "вечер": ["вечер", "вечерн", "закат", "сумерк", "вечером"],
    "утро": ["утро", "утренн", "рассвет", "зоря", "заря", "солнц"],
    "день": ["день", "солнц", "свет", "ясн"],
    # Atmosphere words
    "мрак": ["мрак", "мрачн", "тьм", "темн", "тьма", "демон", "смерть",
             "ад", "ужас", "чёрн", "черн"],
    "свет": ["свет", "светл", "сиян", "блик", "ярк", "солнц"],
    "мечты": ["мечт", "мечта", "грёз", "греза", "фантаз", "сон", "небо"],
    "дорога": ["дорог", "путь", "шоссе", "трасс", "путешеств", "странств",
               "вокзал", "поезд", "машин"],
    "город": ["город", "улиц", "асфальт", "бетон", "небоскрёб", "небоскреб",
              "район", "мегаполис"],
    "природа": ["лес", "гора", "река", "озеро", "поле", "сад", "цвет",
                "дерево", "птиц", "земля"],
    # Energy
    "энергичн": ["энергичн", "быстр", "ритм", "драйв", "танц", "кача",
                 "огонь", "взрыв"],
    "спокойн": ["тих", "спокойн", "мягк", "нежн", "лад", "мирн", "тишин"],
    # Instruments / vocal presence
    "флейт":    ["флейт", "flute", "флут"],
    "гитар":    ["гитар", "guitar", "акустич"],
    "фортепиан": ["фортепиан", "piano", "пианино", "клавир"],
    "скрипк":   ["скрипк", "violin", "виолин", "струнн"],
    "виолончел": ["виолончел", "cello", "чело"],
    "труб":     ["труб", "trumpet", "brass"],
    "саксофон": ["саксофон", "saxophone", "sax", "джаз"],
    "орган":    ["орган", "organ", "органн"],
}

# Patterns that signal the query is about musical ATTRIBUTES (not searchable keywords)
# Used to activate AI world-knowledge mode in Step 2
_ATTRIBUTE_PATTERNS = [
    # Instrumental / no vocals — Russian stems without trailing \b (handles inflection)
    (re.compile(r'без\s+слов|без\s+вокала|инструментальн|instrumental\b|karaoke\b|backing\s+track', re.IGNORECASE), "instrumental"),
    # Specific instruments (stem match for Russian, exact for English)
    (re.compile(r'флейт|flute\b',           re.IGNORECASE), "instrument:flute"),
    (re.compile(r'скрипк|скрипич|violin\b|виолин',  re.IGNORECASE), "instrument:violin"),
    (re.compile(r'фортепиан|piano\b|пианин', re.IGNORECASE), "instrument:piano"),
    (re.compile(r'гитар|guitar\b',          re.IGNORECASE), "instrument:guitar"),
    (re.compile(r'саксофон|saxophone\b|\bsax\b', re.IGNORECASE), "instrument:saxophone"),
    # Acoustic
    (re.compile(r'акустическ|acoustic\b',   re.IGNORECASE), "acoustic"),
    # Tempo extremes — "самый быстрый темп", "быстрее всего", "high bpm"
    (re.compile(
        r'самы[йх]\s+быстр|быстрей?ш[а-яё]*|максимальн[а-яё]*\s+темп|'
        r'fastest\s+tempo?\b|highest?\s+bpm\b|high\s+bpm\b|speed\s+metal\b',
        re.IGNORECASE,
    ), "max_bpm"),
    (re.compile(
        r'самы[йх]\s+медленн|медленнейш[а-яё]*|минимальн[а-яё]*\s+темп|'
        r'slowest\s+tempo?\b|lowest?\s+bpm\b|low\s+bpm\b',
        re.IGNORECASE,
    ), "min_bpm"),
]

# Genre/era keywords that should be kept in query_tokens (not stop-words)
# These are already NOT in _STOP_WORDS, but we make them explicit here
# for the query parser to recognize them as valid genres.

_CYRILLIC_RE = re.compile(r'[а-яёА-ЯЁ]')


def detect_language_preference(query):
    """Detect if user specified a language preference.

    Args:
        query: The user's natural language query.

    Returns:
        str or None: Language code ('ru', 'en', 'fr', 'ja', 'ko', 'es') or None.
    """
    for pattern, lang in _LANGUAGE_PATTERNS:
        if pattern.search(query):
            return lang
    return None


def is_cyrillic_track(entry):
    """Check if a track's artist or title is primarily in Cyrillic.

    Used for language filtering: "на русском" → keep only Cyrillic tracks.

    Args:
        entry: dict with 'artist' and 'title' keys.

    Returns:
        float: Fraction of alphabetic characters that are Cyrillic.
    """
    text = (entry.get("artist", "") + " " + entry.get("title", "")).lower()
    total_alpha = len([c for c in text if c.isalpha()])
    if total_alpha == 0:
        return False
    cyrillic_count = len([c for c in text if _CYRILLIC_RE.match(c)])
    return cyrillic_count / total_alpha > 0.5


def is_latin_track(entry):
    """Check if a track's artist or title is primarily in Latin script."""
    text = (entry.get("artist", "") + " " + entry.get("title", "")).lower()
    total_alpha = len([c for c in text if c.isalpha()])
    if total_alpha == 0:
        return False
    latin_count = len([c for c in text if c.isascii() and c.isalpha()])
    return latin_count / total_alpha > 0.5


def expand_query_tokens(query):
    """Parse query and return expanded search tokens + intent info.

    Returns:
        (tokens, intent):
            tokens: list of search-relevant words (expanded with atmosphere dictionary)
            intent: dict with keys 'language' (optional), 'era_ranges' (optional)
    """
    intent = {}

    # 1. Detect language preference
    lang = detect_language_preference(query)
    if lang:
        intent["language"] = lang

    # 1b. Detect musical attribute requirements (instrumental, specific instruments)
    attributes = []
    for pattern, attr in _ATTRIBUTE_PATTERNS:
        if pattern.search(query):
            attributes.append(attr)
    if attributes:
        intent["attributes"] = attributes

    # 2. Extract era/years from query
    years_match = re.findall(r'\b(19|20)?(\d{2})[хxеs]*\b', query)
    if years_match:
        eras = []
        for prefix, year_num in years_match:
            if len(year_num) == 2:
                year = int(year_num)
                if prefix and year < 50:
                    year += 2000
                elif prefix:
                    year += 1900
                else:
                    # Ambiguous: "80е" → assume 1980s
                    full_year = 1900 + year if year >= 50 else year
                    eras.append((full_year, full_year + 9))
        if eras:
            intent["era_ranges"] = eras

    # 3. Tokenize query (remove stop words)
    base_tokens = [t for t in tokenize(query) if t not in _STOP_WORDS]

    # 4. Expand atmosphere words
    expanded_tokens = list(base_tokens)
    for base_token in base_tokens:
        for mood_word, related_words in _ATMOSPHERE_EXPANSIONS.items():
            if mood_word in base_token or base_token in mood_word:
                # Add related words that might appear in artist/title
                expanded_tokens.extend(related_words)
                break

    return list(dict.fromkeys(expanded_tokens)), intent  # unique, preserving order


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_RE_WORD = re.compile(r"[\w']+", re.UNICODE)
_RE_ACRONYM = re.compile(r'[A-ZЁА-Я]{2,}')


def tokenize(text, lower=True):
    """Extract tokens from text. Keeps acronyms (AC/DC, B.B. King -> ac, dc, bb, king).

    Splits multi-part words on common separators: -, _, с, /
    """
    if not text:
        return []

    result = []
    # Normalize separators
    normalized = text.replace("-", " ").replace("_", " ").replace(
        "/", " "
    ).replace("—", " ")
    raw_tokens = _RE_WORD.findall(normalized)

    for tok in raw_tokens:
        tok = tok.lower() if lower else tok
        result.append(tok)

    # Add acronym tokens for better matching
    acronyms = _RE_ACRONYM.findall(text.upper())
    for acr in acronyms:
        acr_lower = acr.lower()
        if acr_lower not in result and len(acr_lower) >= 2:
            result.append(acr_lower)

    return result


# ---------------------------------------------------------------------------
# Search Index
# ---------------------------------------------------------------------------


class SearchIndex:
    """Lightweight TF-IDF search index built from catalog.tsv."""

    def __init__(self):
        self.documents = []  # list of {index, artist, title, year, path}
        self.idf = {}        # token -> log(1 + N / df)
        self.doc_index = defaultdict(Counter)  # doc_id -> {token: tf}
        self.built = False

    # ---- building ----

    def add_document(self, index, artist, title, year, path):
        """Add a single track to the index (before build)."""
        doc_id = len(self.documents)
        self.documents.append({
            "index": index,
            "artist": artist,
            "title": title,
            "year": year,
            "path": path,
        })

    def build(self):
        """Compute IDF across all documents. Call after add_document."""
        N = len(self.documents)
        if N == 0:
            return

        # Count document frequency for each token
        df = Counter()
        for doc_id, doc in enumerate(self.documents):
            tokens = set()
            # Artist tokens (weighted more by adding them twice)
            for t in tokenize(doc["artist"]):
                tokens.add(t)
                tokens.add(t)  # double weight for artist
            for t in tokenize(doc["title"]):
                tokens.add(t)
            for t in tokenize(doc.get("year", "")):
                tokens.add(t)

            for t in tokens:
                df[t] += 1

            # Build TF for this document
            doc_tokens = []
            for _ in range(2):  # artist weight
                doc_tokens.extend(tokenize(doc["artist"]))
            doc_tokens.extend(tokenize(doc["title"]))
            doc_tokens.extend(tokenize(doc.get("year", "")))
            self.doc_index[doc_id] = Counter(doc_tokens)

        # Compute IDF: log(1 + N / df)
        self.idf = {}
        for token, freq in df.items():
            self.idf[token] = math.log(1 + N / freq)

        self.built = True

    @classmethod
    def from_catalog(cls, catalog_path):
        """Build an index directly from catalog.tsv file.

        Args:
            catalog_path: path to catalog.tsv (format: index\tartist\ttitle\tyear\tpath\tmtime)

        Returns:
            SearchIndex instance
        """
        index = cls()
        if not os.path.exists(catalog_path):
            return index

        with open(catalog_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 5:
                    continue
                try:
                    idx = int(parts[0])
                    artist = parts[1]
                    title = parts[2]
                    year = parts[3]
                    path = parts[4]
                    index.add_document(idx, artist, title, year, path)
                except (ValueError, IndexError):
                    continue

        index.build()
        return index

    @classmethod
    def from_catalog_dict(cls, catalog_dict):
        """Build an index from a catalog dict {index: {artist, title, year, path}}.

        Used when loading from query_engine's in-memory catalog structure.
        """
        index = cls()
        for idx, entry in sorted(catalog_dict.items()):
            index.add_document(
                idx,
                entry.get("artist", ""),
                entry.get("title", ""),
                entry.get("year", ""),
                entry.get("path", ""),
            )
        index.build()
        return index

    def search(self, query, topn=300):
        """Search for top-N most relevant tracks.

        Uses query intent parsing for:
        - Language filtering: "на русском" → only Cyrillic tracks
        - Atmosphere expansion: "осень" → expanded to related words
        - Era filtering: "90е" → filter to 1990-1999
        - Fuzzy matching: "осени" matches tracks containing "осен"

        Args:
            query: Natural language query string.
            topn: Maximum number of results to return.

        Returns:
            List of {index, artist, title, year, path, score} sorted by score desc.
        """
        if not self.built or not self.documents:
            return []

        # Parse query intent
        query_tokens, intent = expand_query_tokens(query)

        # Remove stop words from query tokens (after expansion)
        query_tokens = [t for t in query_tokens if t not in _STOP_WORDS]

        # Apply language filter to all candidates
        lang = intent.get("language")
        filtered_docs = {}
        for doc_id, doc_tokens in self.doc_index.items():
            doc = self.documents[doc_id]
            if lang == "ru" and not is_cyrillic_track(doc):
                continue
            elif lang == "en" and not is_latin_track(doc):
                continue
            filtered_docs[doc_id] = doc_tokens

        if not filtered_docs:
            return []

        # Build cache of doc_tokens keys for fast lookup
        all_doc_token_keys = set()
        for dt in self.doc_index.values():
            all_doc_token_keys.update(dt.keys())

        # Also build inverse fuzzy index: for each query token, pre-compute
        # which catalog tokens it fuzzy-matches
        # Matching strategies (in priority order):
        # 1. Exact match: "осень" == "осень"
        # 2. Substring: "осен" ⊂ "осенний", "осень" ⊂ "осенние"
        # 3. Stem match (for inflected languages): "осени" → "осен" vs "осень" → "осен"
        # 4. Levenshtein-1 for short words
        fuzzy_matches = {}  # qt -> list of (catalog_token, idf_score, weight)
        for qt in query_tokens:
            fuzzy_matches[qt] = []
            # 1. Exact match
            if qt in self.idf:
                fuzzy_matches[qt].append((qt, self.idf[qt], 1.0))

            # 2. Substring match
            for existing_token in all_doc_token_keys:
                if existing_token == qt:
                    continue
                if qt in existing_token or existing_token in qt:
                    fuzzy_matches[qt].append(
                        (existing_token, self.idf.get(existing_token, 0) * 0.7, 0.7)
                    )
                # 3. Stem match: for words >= 4 chars, compare first N-2 chars
                elif len(qt) >= 4 and len(existing_token) >= 4:
                    qt_stem = qt[:-2]  # cut suffix
                    tok_stem = existing_token[:-2]  # cut suffix
                    if qt_stem == tok_stem and len(qt_stem) >= 3:
                        fuzzy_matches[qt].append(
                            (existing_token, self.idf.get(existing_token, 0) * 0.4, 0.4)
                        )

        # Score each filtered document
        scores = {}
        for doc_id, doc_tokens in filtered_docs.items():
            score = 0.0

            doc = self.documents[doc_id]
            art_lower = doc["artist"].lower()
            title_lower = doc["title"].lower()

            for qt in query_tokens:
                matched = False
                for match_token, idf_score, weight in fuzzy_matches.get(qt, []):
                    tf = doc_tokens.get(match_token, 0)
                    if tf == 0:
                        continue

                    tf_score = tf / (tf + 1.0)
                    score += tf_score * idf_score * weight

                    # Substring bonus
                    if qt in art_lower:
                        score += idf_score * 0.5 * weight
                    if qt in title_lower:
                        score += idf_score * 0.3 * weight

                    matched = True

                # Direct string containment (for words with no token/IDF at all)
                if not matched:
                    if qt in art_lower:
                        score += len(qt) * 0.3
                    if qt in title_lower:
                        score += len(qt) * 0.2

            # Era scoring bonus
            track_year = self._parse_year(doc.get("year", ""))
            era_ranges = intent.get("era_ranges", [])
            for era_start, era_end in era_ranges:
                if track_year and era_start <= track_year <= era_end:
                    score += 2.0
                elif track_year:
                    distance = min(
                        abs(track_year - era_start), abs(track_year - era_end)
                    )
                    if distance < 20:
                        score -= 0.1 * distance

            if score > 0:
                scores[doc_id] = score

        # Sort by score descending, return top-N
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topn]

        results = []
        for doc_id, score in ranked:
            doc = self.documents[doc_id].copy()
            doc["score"] = round(score, 4)
            results.append(doc)

        return results

    def _parse_year(self, year_str):
        """Parse year from string like '1990', '2000-е', '1989/90'."""
        if not year_str:
            return None
        match = re.search(r'(\d{4})', year_str)
        if match:
            return int(match.group(1))
        return None

    def _get_atmosphere_tokens_for_doc(self, doc):
        """Extract atmosphere-related tokens that match the document."""
        text = (doc.get("artist", "") + " " + doc.get("title", "") +
                " " + doc.get("year", "")).lower()

        matching_tokens = []
        for mood_word, related in _ATMOSPHERE_EXPANSIONS.items():
            if mood_word in text:
                matching_tokens.append(mood_word)
            for rw in related:
                if rw in text:
                    matching_tokens.append(rw)

        return list(set(matching_tokens))

    def search_with_catalog_text(self, query, topn=300):
        """Search and return formatted text block suitable for AI prompt + track list.

        Returns:
            (text_block, track_list, intent):
                text_block: compact text for AI prompt (INDEX|ARTIST|TITLE|YEAR)
                track_list: list of track dicts
                intent: dict with parsed intent (language, era, etc.)

        If TF-IDF finds too few matches, falls back to filtered sampling
        based on detected intent (language filter, atmosphere tokens).
        """
        results = self.search(query, topn=topn)
        _, intent = expand_query_tokens(query)

        lang = intent.get("language")

        # If TF-IDF found very few, fall back to intent-aware sampling
        if len(results) < max(topn // 3, 30):
            import random
            all_docs = self.documents.copy()
            random.seed(42)
            random.shuffle(all_docs)

            result_indices = {r["index"] for r in results}

            for doc in all_docs:
                if doc["index"] in result_indices:
                    continue
                # If language specified, only add tracks that MATCH the language
                if lang and not self._matches_language(doc, lang):
                    continue
                results.append({"score": 0.0, **doc})
                if len(results) >= topn:
                    break

        lines = []
        for r in results:
            year_part = r.get("year", "")
            lines.append(f"{r['index']}|{r['artist']}|{r['title']}|{year_part}")

        return "\n".join(lines), results, intent

    def _check_doc_language(self, doc, lang):
        """Check if a document matches the specified language code."""
        if lang == "ru":
            return is_cyrillic_track(doc)
        elif lang == "en":
            return is_latin_track(doc)
        # Unknown language — accept all
        return True

    def _matches_language(self, doc, lang):
        """Alias for compatibility. Checks language match."""
        return self._check_doc_language(doc, lang)


# ---------------------------------------------------------------------------
# Artist mention detection
# ---------------------------------------------------------------------------

_ARTIST_FUNC_WORDS = frozenset({
    "и", "или", "с", "в", "на", "о", "а", "но", "из", "до", "по", "от",
    "the", "and", "or", "of", "a", "an", "feat", "ft",
})


def detect_mentioned_artists(query: str, artist_map: dict) -> list[str]:
    """Find artist keys from artist_map that are explicitly named in the query.

    Uses two matching strategies (tried in order per artist):
    1. Exact substring match:  "Noize MC" ∈ "лучшее от Noize MC"
    2. Stem-based match for Russian inflection:
       "Король и Шут" → stems ["коро", "шут"]
       both match in "песни короля и шута" ("коро"⊂"короля", "шут"⊂"шута")

    Longest artist names are tried first to prevent "Шут" stealing the match
    before "Король и Шут".  Matched spans are blanked so they can't match twice.

    Stem matching is only applied when significant words are long enough to
    avoid false positives from common short words.

    Args:
        query:      The user's natural-language query.
        artist_map: {artist_name_lower: {name, indices}}.

    Returns:
        List of matched artist keys (lowercased), longest match first.
        Empty list if no catalog artist is mentioned.

    Examples:
        "лучшие песни Короля и Шута" → ["король и шут"]
        "Metallica и AC/DC"          → ["metallica", "ac/dc"]
        "энергичный рок 80-х"        → []
    """
    query_lower = query.lower()
    remaining = query_lower  # blanked progressively to avoid double-matches

    sorted_keys = sorted(artist_map.keys(), key=len, reverse=True)
    found = []

    for key in sorted_keys:
        if len(key) < 3:
            continue

        # --- Strategy 1: exact substring ---
        if key in remaining:
            found.append(key)
            remaining = remaining.replace(key, " " * len(key), 1)
            continue

        # --- Strategy 2: stem-based (handles Russian case inflection) ---
        # Extract content words (skip function words and very short particles)
        content_words = [
            w for w in key.split()
            if w not in _ARTIST_FUNC_WORDS and len(w) >= 3
        ]
        if not content_words:
            continue

        # For a single short word skip stem matching (too many false positives).
        # E.g. "шут" (3 chars) must match exactly, not via stem "шу".
        if len(content_words) == 1 and len(content_words[0]) < 5:
            continue

        # Stem = first max(3, len-2) chars: drops typical Russian case endings
        # "король"(6) → "коро"(4), "шут"(3) → "шут"(3), "гришин"(6) → "гриш"(4)
        stems = [w[:max(3, len(w) - 2)] for w in content_words]

        # ALL stems must appear as substrings in remaining query
        if not all(stem in remaining for stem in stems):
            continue

        found.append(key)

        # Blank the matched tokens to prevent re-use
        for stem in stems:
            pos = remaining.find(stem)
            if pos < 0:
                continue
            # Expand to word boundaries
            start, end = pos, pos + len(stem)
            while start > 0 and remaining[start - 1].isalpha():
                start -= 1
            while end < len(remaining) and remaining[end].isalpha():
                end += 1
            remaining = remaining[:start] + " " * (end - start) + remaining[end:]

    return found

    def __len__(self):
        return len(self.documents)
