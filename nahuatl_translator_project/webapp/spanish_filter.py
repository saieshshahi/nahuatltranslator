"""Spanish contamination detection and filtering for Nahuatl translations.

Loads a comprehensive Spanish wordlist from data/spanish_common_words.txt
and provides functions to detect, count, and strip Spanish words from text.
This replaces hardcoded word lists with a data-driven approach.
"""

from __future__ import annotations

import os
import re
from typing import FrozenSet, List, Set

_WORD_RE = re.compile(r"[a-záéíóúñüà-ž]+", re.IGNORECASE)

# Nahuatl words that look like Spanish but are actually Nahuatl.
# These are excluded from Spanish detection to avoid false positives.
_NAHUATL_WHITELIST: Set[str] = {
    # Nahuatl function words / particles
    "amo", "ahmo", "can", "zan", "pan", "cal", "man", "san",
    "nican", "acan", "como",  # como can be Nahuatl particle in some dialects
    # Common Nahuatl morphemes that overlap with Spanish
    "moca", "toca", "cana", "poca", "noche",  # nochi = all
    "ica", "nica", "tlan", "tla",
    # Historical loanwords fully adopted into Nahuatl
    "cahuayo", "castilan", "caballo",
    # Nahuatl words that happen to match Spanish
    "calli", "malli", "colli", "pilli",
    "cualli",  # good
    "tonal", "tonalli",  # day/sun
    "cochi",  # sleep
    "huey",  # big
}

# Module-level cache
_spanish_words: FrozenSet[str] | None = None


def _load_wordlist() -> FrozenSet[str]:
    """Load Spanish wordlist from data file. Cached after first call."""
    global _spanish_words
    if _spanish_words is not None:
        return _spanish_words

    words: Set[str] = set()
    wordlist_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "spanish_common_words.txt",
    )

    if os.path.isfile(wordlist_path):
        with open(wordlist_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip().lower()
                if line and not line.startswith("#"):
                    words.add(line)

    _spanish_words = frozenset(words)
    return _spanish_words


def is_spanish(word: str) -> bool:
    """Check if a word is Spanish (and not a known Nahuatl word)."""
    w = word.strip().lower()
    if w in _NAHUATL_WHITELIST:
        return False
    return w in _load_wordlist()


def count_spanish(text: str) -> int:
    """Count the number of Spanish words in a text string."""
    spanish_set = _load_wordlist()
    count = 0
    for w in _WORD_RE.findall(text.lower()):
        if len(w) < 3:
            continue
        if w in _NAHUATL_WHITELIST:
            continue
        if w in spanish_set:
            count += 1
    return count


def spanish_ratio(text: str) -> float:
    """Return the fraction of tokens in text that are Spanish (0.0 to 1.0)."""
    tokens = [w for w in _WORD_RE.findall(text.lower()) if len(w) >= 3]
    if not tokens:
        return 0.0
    spanish_set = _load_wordlist()
    spanish_count = sum(
        1 for w in tokens
        if w not in _NAHUATL_WHITELIST and w in spanish_set
    )
    return spanish_count / len(tokens)


def detect_spanish_in_output(text: str) -> List[str]:
    """Return a list of Spanish words found in the text.

    Filters out known Nahuatl words to reduce false positives.
    """
    spanish_set = _load_wordlist()
    found = []
    seen = set()
    for w in _WORD_RE.findall(text.lower()):
        if len(w) < 3:
            continue
        if w in _NAHUATL_WHITELIST:
            continue
        if w in spanish_set and w not in seen:
            found.append(w)
            seen.add(w)
    return found


def strip_spanish(text: str) -> str:
    """Remove Spanish words from text, preserving sentence structure.

    Removes the Spanish word and cleans up extra spaces/punctuation.
    """
    spanish_set = _load_wordlist()

    def replace_word(match: re.Match) -> str:
        word = match.group(0)
        w = word.lower()
        if len(w) < 3:
            return word
        if w in _NAHUATL_WHITELIST:
            return word
        if w in spanish_set:
            return ""
        return word

    result = _WORD_RE.sub(replace_word, text)
    # Clean up multiple spaces and orphaned punctuation
    result = re.sub(r"  +", " ", result)
    result = re.sub(r" ([,.])", r"\1", result)
    result = re.sub(r"^ +| +$", "", result, flags=re.MULTILINE)
    return result
