"""Dictionary & vocabulary enhancement layer for Nahuatl translation.

Builds a fast word-level lookup index from the parallel corpus (and any
future vocabulary files). Before each translation call, key terms are
extracted from the input, looked up in the dictionary, and injected into
the OpenAI prompt as a vocabulary reference block.

Includes a basic Nahuatl morpheme splitter that breaks compound words
into recognizable stems for better dictionary matching.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


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

# English stop words (reuse concept from corpus.py but keep independent)
_EN_STOP: Set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "of", "in", "to",
    "for", "with", "on", "at", "from", "by", "as", "into", "through",
    "during", "before", "after", "above", "below", "between", "out",
    "off", "over", "under", "again", "further", "then", "once", "and",
    "but", "or", "nor", "not", "no", "so", "if", "than", "too", "very",
    "just", "about", "up", "that", "this", "these", "those", "it", "its",
    "i", "me", "my", "we", "our", "you", "your", "he", "him", "his",
    "she", "her", "they", "them", "their", "what", "which", "who",
    "when", "where", "how", "all", "each", "every", "both", "few",
    "more", "most", "other", "some", "such", "only", "own", "same",
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
    """Word-level vocabulary index built from parallel corpus data."""

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

    def load_from_corpus_xlsx(self, path: str) -> None:
        """Build vocabulary index from the parallel corpus Excel file.

        For each parallel pair, we extract aligned tokens and create
        word-level associations. Since we don't have word-level alignment,
        we store the full sentence pair as context for each key token.
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
        """Look up vocabulary for key terms in the query.

        Returns a list of (term, translation_snippet) tuples. Translation
        snippets are kept short — just the most relevant fragment.
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

            # Pick the shortest translation entries (most likely to be
            # focused/relevant rather than long biblical passages)
            sorted_entries = sorted(entries, key=lambda e: len(e.translation))
            for entry in sorted_entries[:max_per_term]:
                # Truncate long translations to a useful snippet
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
        max_entries: int = 8,
    ) -> str:
        """Look up terms and format as a vocabulary reference block for prompts."""
        matches = self.lookup(query, src_lang=src_lang, max_total=max_entries)
        if not matches:
            return ""

        lines = []
        for term, translation in matches:
            lines.append(f'- "{term}" → "{translation}"')
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_dictionary: Optional[NahuatlDictionary] = None


def get_dictionary() -> NahuatlDictionary:
    """Return the singleton dictionary instance, loading lazily on first call."""
    global _dictionary
    if _dictionary is None:
        _dictionary = NahuatlDictionary()
        data_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            "english_to_nahuatl_parallel.xlsx",
        )
        _dictionary.load_from_corpus_xlsx(data_path)
    return _dictionary
