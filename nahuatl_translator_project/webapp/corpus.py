"""Supplementary parallel corpus loader and keyword-based context retrieval.

Architecture: OpenAI (GPT) is the PRIMARY translator. This module provides
SUPPLEMENTARY sentence-level reference from the biblical parallel corpus.
The AI should use these as optional vocabulary guidance, not as the main
translation source.

At startup, parses data/english_to_nahuatl_parallel.xlsx into an in-memory
index for keyword-based lookup of relevant parallel pairs.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class ParallelEntry:
    english: str
    nahuatl: str
    source: str = ""


# Simple English stop words to exclude from keyword matching
_STOP_WORDS: Set[str] = {
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

_WORD_RE = re.compile(r"[a-záéíóúñü]+", re.IGNORECASE)


def _tokenize(text: str) -> List[str]:
    """Extract lowercase alphabetic tokens, filtering stop words."""
    return [
        w for w in _WORD_RE.findall(text.lower())
        if w not in _STOP_WORDS and len(w) > 2
    ]


class ParallelCorpus:
    """In-memory index over English↔Nahuatl parallel sentences."""

    def __init__(self) -> None:
        self.entries: List[ParallelEntry] = []
        # Inverted index: token → set of entry indices
        self._en_index: Dict[str, Set[int]] = {}
        self._nah_index: Dict[str, Set[int]] = {}
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load_xlsx(self, path: str) -> None:
        """Load parallel data from the project Excel file."""
        try:
            import openpyxl
        except ImportError:
            return  # openpyxl not installed — skip corpus loading

        if not os.path.isfile(path):
            return

        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        ws = wb.active
        if ws is None:
            wb.close()
            return

        for row in ws.iter_rows(min_row=2, values_only=True):
            # Columns: id, book, chapter, verse, english, nahuatl, source
            if len(row) < 6:
                continue
            en_text = str(row[4] or "").strip()
            nah_text = str(row[5] or "").strip()
            source = str(row[6] or "") if len(row) > 6 else ""
            if en_text and nah_text:
                idx = len(self.entries)
                self.entries.append(ParallelEntry(
                    english=en_text,
                    nahuatl=nah_text,
                    source=source,
                ))
                for tok in _tokenize(en_text):
                    self._en_index.setdefault(tok, set()).add(idx)
                for tok in _tokenize(nah_text):
                    self._nah_index.setdefault(tok, set()).add(idx)

        wb.close()
        self._loaded = True

    def search(
        self,
        query: str,
        src_lang: str = "en",
        max_results: int = 5,
    ) -> List[ParallelEntry]:
        """Find the most relevant parallel entries for a query string.

        Uses a simple TF-overlap scoring: entries that share more unique
        query tokens score higher. Short entries are slightly preferred
        to avoid injecting very long biblical passages.
        """
        if not self.entries:
            return []

        tokens = _tokenize(query)
        if not tokens:
            return []

        index = self._en_index if src_lang == "en" else self._nah_index

        # Count how many query tokens each entry matches
        scores: Dict[int, float] = {}
        for tok in set(tokens):
            for idx in index.get(tok, set()):
                scores[idx] = scores.get(idx, 0.0) + 1.0

        if not scores:
            return []

        # Normalize by entry length to prefer concise, relevant matches
        scored = []
        for idx, raw_score in scores.items():
            entry = self.entries[idx]
            text = entry.english if src_lang == "en" else entry.nahuatl
            word_count = max(1, len(text.split()))
            # Boost short-to-medium entries; penalize very long ones
            length_factor = min(1.0, 30.0 / word_count)
            scored.append((idx, raw_score * (0.5 + 0.5 * length_factor)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [self.entries[idx] for idx, _ in scored[:max_results]]

    def format_as_reference(
        self,
        entries: List[ParallelEntry],
        src_lang: str = "en",
    ) -> str:
        """Format matched entries as a reference block for prompt injection."""
        if not entries:
            return ""
        lines = []
        for e in entries:
            lines.append(f"- English: {e.english}")
            lines.append(f"  Nahuatl: {e.nahuatl}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Module-level singleton — loaded once at import time (or on first use)
# ---------------------------------------------------------------------------
_corpus: Optional[ParallelCorpus] = None


def get_corpus() -> ParallelCorpus:
    """Return the singleton corpus instance, loading it lazily on first call."""
    global _corpus
    if _corpus is None:
        _corpus = ParallelCorpus()
        # Resolve the data path relative to this file
        data_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            "english_to_nahuatl_parallel.xlsx",
        )
        _corpus.load_xlsx(data_path)
    return _corpus
