"""Supplementary vocabulary layer for Nahuatl translation.

Architecture: OpenAI (GPT) is the PRIMARY translator — it already knows
Nahuatl grammar and vocabulary. This module provides SUPPLEMENTARY word-level
lookups from the parallel corpus (biblical text) to help with rare or archaic
terms. The AI should never blindly copy these entries.

Includes a basic Nahuatl morpheme splitter for better dictionary matching.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from webapp.spanish_filter import count_spanish, strip_spanish


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DictEntry:
    """A single vocabulary entry mapping a term to its translation(s)."""
    term: str            # normalized form
    translation: str     # translation in the other language
    pos: str = ""        # part of speech hint (noun, verb, etc.) — optional
    source: str = ""     # where this entry came from


# ---------------------------------------------------------------------------
# Nahuatl morpheme splitter
# ---------------------------------------------------------------------------

# Common Nahuatl prefixes (subject, object, possessive, directional)
_NAH_PREFIXES = [
    "nech", "mitz", "tech", "amech",  # object prefixes (longer first)
    "kin", "kim",
    "ni", "ti", "an",                  # subject prefixes
    "no", "mo", "to", "amo",           # possessive prefixes
    "on", "hual",                       # directional prefixes
]

# Common Nahuatl suffixes (absolutive, verbal, locative, etc.)
_NAH_SUFFIXES = [
    "tzintli", "tzin",                  # reverential/diminutive
    "yotl", "otl",                      # abstract noun
    "tlan", "can", "pan", "co", "nahuac",  # locative
    "hua", "ehua",                      # possessive adjective
    "lia", "tia", "huia",              # applicative/causative
    "tli", "tl", "li", "in",           # absolutive
    "que", "quej", "mej", "meh", "tin", # plural
    "qui", "ki",                        # object prefix (sometimes suffix-like)
]

# Sort by length descending so longer affixes match first
_NAH_PREFIXES.sort(key=len, reverse=True)
_NAH_SUFFIXES.sort(key=len, reverse=True)


def split_morphemes(word: str) -> List[str]:
    """Attempt to split a Nahuatl word into morpheme-like segments.

    This is a heuristic splitter, not a full morphological analyzer.
    Even partial/imperfect splits help dictionary lookup by exposing
    recognizable stems that might match vocabulary entries.

    Returns a list of segments (prefix parts, stem, suffix parts).
    The original word is always included as the first element.
    """
    w = word.lower().strip()
    if len(w) < 4:
        return [w]

    segments = [w]  # always include the full word

    remaining = w

    # Strip prefixes
    prefix_parts = []
    for pfx in _NAH_PREFIXES:
        if remaining.startswith(pfx) and len(remaining) > len(pfx) + 2:
            prefix_parts.append(pfx)
            remaining = remaining[len(pfx):]
            break  # only strip one prefix layer

    # Strip suffixes
    suffix_parts = []
    for sfx in _NAH_SUFFIXES:
        if remaining.endswith(sfx) and len(remaining) > len(sfx) + 2:
            suffix_parts.append(sfx)
            remaining = remaining[:-len(sfx)]
            break  # only strip one suffix layer

    # The remaining part is the candidate stem
    if remaining != w and len(remaining) >= 3:
        segments.append(remaining)

    # Also try combining prefix-stripped form (without suffix stripping)
    if prefix_parts:
        no_prefix = w[len(prefix_parts[0]):]
        if no_prefix != remaining and len(no_prefix) >= 3:
            segments.append(no_prefix)

    return list(dict.fromkeys(segments))  # deduplicate preserving order


# ---------------------------------------------------------------------------
# Vocabulary index
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-záéíóúñüā-ž]+", re.IGNORECASE)

# English stop words — minimal so that important words like pronouns and
# possessives still get looked up (they map to Nahuatl prefixes).
_EN_STOP: Set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "can", "could", "of", "in", "to",
    "for", "with", "on", "at", "from", "by", "as", "into", "through",
    "during", "before", "after", "above", "below", "between", "out",
    "off", "over", "under", "again", "further", "then", "once", "and",
    "but", "or", "nor", "so", "if", "than", "too", "very",
    "just", "about", "up", "that", "this", "these", "those", "it", "its",
    "such", "only", "own", "same",
    # NOTE: "my", "your", "i", "he", "she", "we", "they", "not", "no",
    # "hello", "name", "what", "who", "where", "when", "how" are intentionally
    # NOT in this stop list because they have important Nahuatl equivalents.
}


def _extract_tokens(text: str) -> List[str]:
    """Extract lowercase tokens from text."""
    return [w for w in _WORD_RE.findall(text.lower()) if len(w) > 2]


def _content_tokens(text: str, lang: str = "en") -> List[str]:
    """Extract content-bearing tokens (skip stop words for English)."""
    tokens = _extract_tokens(text)
    if lang in ("en", "es"):
        return [t for t in tokens if t not in _EN_STOP]
    return tokens


class NahuatlDictionary:
    """Supplementary word-level vocabulary index built from parallel corpus data.

    This provides OPTIONAL reference vocabulary for the AI translator.
    The AI is the primary translator and should not blindly copy these entries.
    """

    def __init__(self) -> None:
        # token → list of DictEntry
        self._en_to_nah: Dict[str, List[DictEntry]] = {}
        self._nah_to_en: Dict[str, List[DictEntry]] = {}
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    def _add_entry(
        self,
        en_token: str,
        nah_token: str,
        en_context: str,
        nah_context: str,
        source: str = "",
    ) -> None:
        """Add a bidirectional vocabulary mapping."""
        en_entry = DictEntry(
            term=en_token,
            translation=nah_context,
            source=source,
        )
        nah_entry = DictEntry(
            term=nah_token,
            translation=en_context,
            source=source,
        )
        self._en_to_nah.setdefault(en_token, []).append(en_entry)
        self._nah_to_en.setdefault(nah_token, []).append(nah_entry)

    def load_from_clics_json(self, path: str) -> None:
        """Load clean vocabulary from CLICS (Cross-Linguistic Information on
        Conceptual Structures) dataset. This is high-quality, pure Nahuatl
        vocabulary without Spanish contamination — ideal supplementary data.
        """
        import json

        if not os.path.isfile(path):
            return

        with open(path, "r", encoding="utf-8") as f:
            entries = json.load(f)

        for entry in entries:
            nah = entry.get("nahuatl", "").strip()
            en = entry.get("english", "").strip()
            gloss = entry.get("gloss", en)
            if not nah or not en:
                continue

            # Index by English concept tokens
            for tok in set(_content_tokens(en, "en")):
                self._en_to_nah.setdefault(tok, []).append(DictEntry(
                    term=tok,
                    translation=nah,
                    source="clics",
                ))

            # Also index by gloss words (often more descriptive)
            if gloss and gloss != en:
                for tok in set(_content_tokens(gloss, "en")):
                    if tok not in _EN_STOP:
                        self._en_to_nah.setdefault(tok, []).append(DictEntry(
                            term=tok,
                            translation=nah,
                            source="clics",
                        ))

            # Index Nahuatl → English
            for tok in set(_extract_tokens(nah)):
                self._nah_to_en.setdefault(tok, []).append(DictEntry(
                    term=tok,
                    translation=en,
                    source="clics",
                ))

        self._loaded = True

    def load_from_corpus_xlsx(self, path: str) -> None:
        """Build supplementary vocabulary index from the parallel corpus Excel file.

        The corpus is primarily biblical text. These entries serve as additional
        vocabulary reference for the AI — not as the primary translation source.
        """
        try:
            import openpyxl
        except ImportError:
            return

        if not os.path.isfile(path):
            return

        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        ws = wb.active
        if ws is None:
            wb.close()
            return

        for row in ws.iter_rows(min_row=2, values_only=True):
            if len(row) < 6:
                continue
            en_text = str(row[4] or "").strip()
            nah_text = str(row[5] or "").strip()
            source = str(row[6] or "") if len(row) > 6 else ""

            if not en_text or not nah_text:
                continue

            # Skip entries with Spanish contamination in the Nahuatl text
            if count_spanish(nah_text) >= 2:
                continue

            # Index English content words → Nahuatl sentence
            for tok in set(_content_tokens(en_text, "en")):
                self._en_to_nah.setdefault(tok, []).append(DictEntry(
                    term=tok,
                    translation=nah_text,
                    source=source,
                ))

            # Index Nahuatl words (including morpheme segments) → English sentence
            for tok in set(_extract_tokens(nah_text)):
                self._nah_to_en.setdefault(tok, []).append(DictEntry(
                    term=tok,
                    translation=en_text,
                    source=source,
                ))
                # Also index morpheme segments for compound words
                if len(tok) > 5:
                    for segment in split_morphemes(tok):
                        if segment != tok:
                            self._nah_to_en.setdefault(segment, []).append(DictEntry(
                                term=segment,
                                translation=en_text,
                                source=source,
                            ))

        wb.close()
        self._loaded = True

    def lookup(
        self,
        query: str,
        src_lang: str = "en",
        max_per_term: int = 2,
        max_total: int = 10,
    ) -> List[Tuple[str, str]]:
        """Look up supplementary vocabulary for key terms in the query.

        Returns a list of (term, translation_snippet) tuples.
        These are supplementary references — the AI should use its own knowledge first.
        """
        index = self._en_to_nah if src_lang == "en" else self._nah_to_en

        if src_lang in ("en", "es"):
            tokens = _content_tokens(query, src_lang)
        else:
            # For Nahuatl input, also try morpheme splits
            raw_tokens = _extract_tokens(query)
            tokens = []
            for t in raw_tokens:
                tokens.extend(split_morphemes(t))

        results: List[Tuple[str, str]] = []
        seen_terms: Set[str] = set()

        for tok in tokens:
            if tok in seen_terms:
                continue
            entries = index.get(tok, [])
            if not entries:
                continue

            seen_terms.add(tok)

            # Pick the shortest translation entries (most focused/relevant)
            sorted_entries = sorted(entries, key=lambda e: len(e.translation))
            for entry in sorted_entries[:max_per_term]:
                trans = entry.translation
                if len(trans) > 80:
                    trans = trans[:77] + "..."
                results.append((tok, trans))

            if len(results) >= max_total:
                break

        return results[:max_total]

    def format_vocab_block(
        self,
        query: str,
        src_lang: str = "en",
        max_entries: int = 5,
    ) -> str:
        """Look up terms and format as supplementary vocabulary reference for prompts."""
        matches = self.lookup(query, src_lang=src_lang, max_total=max_entries)
        if not matches:
            return ""

        lines = []
        for term, translation in matches:
            # Scrub any remaining Spanish from translations
            clean = strip_spanish(translation)
            if not clean.strip():
                continue
            lines.append(f'- "{term}" → "{clean}"')
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Hardcoded conversational vocabulary (supplements the biblical corpus)
# ---------------------------------------------------------------------------
# The parallel corpus is primarily biblical text and lacks everyday vocabulary.
# This curated list ensures common words produce correct supplementary hits.
# NOTE: The AI already knows most of these — this just reinforces them.

CONVERSATIONAL_VOCAB: List[Tuple[str, str]] = [
    # Greetings
    ("hello", "Pialli"),
    ("hi", "Niltze"),
    ("good morning", "Cualli tlaneci"),
    ("good afternoon", "Cualli tonalli"),
    ("good day", "Cualli tonalli"),
    ("good night", "Cualli yohualli"),
    ("goodbye", "Timo ittazqueh"),
    ("thank you", "Tlazohcamati"),
    ("thanks", "Tlazohcamati"),
    ("please", "Nimitznotlatlauhtilia"),
    ("yes", "Quemah"),
    ("no", "Ahmo"),
    # Introductions & personal
    ("name", "toca (possessed: notoca = my name, motoca = your name)"),
    ("my name", "notoca"),
    ("your name", "motoca"),
    ("friend", "icniuhtli (possessed: nocniuh = my friend)"),
    ("my friend", "nocniuh"),
    # Questions
    ("how are you", "¿Quen tica?"),
    ("what is your name", "¿Tlein motoca?"),
    ("where", "canin"),
    ("when", "queman"),
    ("why", "tleica"),
    ("what", "tlein"),
    ("who", "aquin"),
    # Common nouns
    ("water", "atl"),
    ("fire", "tletl"),
    ("earth", "tlalli"),
    ("wind", "ehecatl"),
    ("sun", "tonatiuh"),
    ("moon", "metztli"),
    ("star", "citlalli"),
    ("rain", "quiahuitl"),
    ("house", "calli (possessed: nocal = my house)"),
    ("my house", "nocal"),
    ("food", "tlacualli"),
    ("tortilla", "tlaxcalli"),
    ("flower", "xochitl"),
    ("song", "cuicatl"),
    ("book", "amoxtli"),
    ("heart", "yollotl"),
    ("mountain", "tepetl"),
    ("river", "atoyatl"),
    ("road", "ohtli"),
    ("dog", "chichi / itzcuintli"),
    ("snake", "coatl"),
    ("bird", "tototl"),
    ("fish", "michin"),
    ("tree", "cuahuitl"),
    ("stone", "tetl"),
    # People
    ("man", "tlacatl"),
    ("woman", "cihuatl"),
    ("child", "conetl / piltzintli"),
    ("father", "tahtli (possessed: notah = my father)"),
    ("mother", "nantli (possessed: nonan = my mother)"),
    # Common verbs
    ("love", "tlazohtla (nimitztlazohtla = I love you)"),
    ("eat", "cua (nitlacua = I eat)"),
    ("drink", "atli / i (natli = I drink)"),
    ("speak", "tlahtoa (nitlahtoa = I speak)"),
    ("see", "itta (niquitta = I see it)"),
    ("go", "yauh (niyauh = I go)"),
    ("come", "huallauh (nihuallauh = I come)"),
    ("want", "nequi (nicnequi = I want it)"),
    ("know", "mati (nicmati = I know it)"),
    # Adjectives
    ("good", "cualli"),
    ("big", "huei"),
    ("small", "tepitzin"),
    ("cold", "itztic / cecec"),
    ("hot", "totonqui"),
    ("new", "yancuic"),
    # Numbers
    ("one", "ce"),
    ("two", "ome"),
    ("three", "yei / ei"),
    ("four", "nahui"),
    ("five", "macuilli"),
    ("ten", "mahtlactli"),
    ("twenty", "cempohualli"),
    # Abstract
    ("god", "teotl"),
    ("ruler", "tlatoani"),
    ("warrior", "yaoquizqui"),
]


def _load_conversational_vocab(dictionary: NahuatlDictionary) -> None:
    """Inject conversational vocabulary as supplementary entries.

    These are added with high priority (short translations) so they
    rank above long biblical corpus sentences in lookup results.
    """
    for en_term, nah_translation in CONVERSATIONAL_VOCAB:
        en_key = en_term.lower().strip()
        dictionary._en_to_nah.setdefault(en_key, []).insert(0, DictEntry(
            term=en_key,
            translation=nah_translation,
            source="conversational-vocab",
        ))
        # Also index the Nahuatl side (first word only for multi-word translations)
        nah_key = nah_translation.lower().split()[0].strip("()/")
        if len(nah_key) >= 3:
            dictionary._nah_to_en.setdefault(nah_key, []).insert(0, DictEntry(
                term=nah_key,
                translation=en_term,
                source="conversational-vocab",
            ))


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_dictionary: Optional[NahuatlDictionary] = None


def get_dictionary() -> NahuatlDictionary:
    """Return the singleton dictionary instance, loading lazily on first call."""
    global _dictionary
    if _dictionary is None:
        _dictionary = NahuatlDictionary()
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

        # 1. Load CLICS vocabulary first (clean, no Spanish contamination)
        clics_path = os.path.join(data_dir, "clics_nahuatl_vocab.json")
        _dictionary.load_from_clics_json(clics_path)

        # 2. Load biblical corpus vocabulary (may contain Spanish loanwords)
        corpus_path = os.path.join(data_dir, "english_to_nahuatl_parallel.xlsx")
        _dictionary.load_from_corpus_xlsx(corpus_path)

        # 3. Load curated conversational vocab (highest priority)
        _load_conversational_vocab(_dictionary)

        if not _dictionary._loaded:
            # Even without corpus/CLICS, conversational vocab makes it usable
            _dictionary._loaded = True
    return _dictionary
