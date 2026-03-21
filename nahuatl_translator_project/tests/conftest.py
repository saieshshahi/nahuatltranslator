"""Shared fixtures for the Nahuatl Translator test suite."""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

import pytest

# Ensure project root is on sys.path so `import webapp` works in tests.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
TESTS_DIR = Path(__file__).resolve().parent
GOLDEN_TRANSLATIONS_PATH = TESTS_DIR / "golden_translations.json"
CORPUS_XLSX_PATH = DATA_DIR / "english_to_nahuatl_parallel.xlsx"


# ---------------------------------------------------------------------------
# Golden translation pairs
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def golden_pairs() -> List[Dict[str, str]]:
    """Load golden translation pairs from JSON file."""
    with open(GOLDEN_TRANSLATIONS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["pairs"]


@pytest.fixture(scope="session")
def golden_en_to_nah(golden_pairs) -> List[Dict[str, str]]:
    """Filter golden pairs for English-to-Nahuatl direction."""
    return [p for p in golden_pairs if p["src_lang"] == "en" and p["tgt_lang"] == "nah"]


@pytest.fixture(scope="session")
def golden_nah_to_en(golden_pairs) -> List[Dict[str, str]]:
    """Filter golden pairs for Nahuatl-to-English direction."""
    return [p for p in golden_pairs if p["src_lang"] == "nah" and p["tgt_lang"] == "en"]


@pytest.fixture(scope="session")
def golden_es_to_nah(golden_pairs) -> List[Dict[str, str]]:
    """Filter golden pairs for Spanish-to-Nahuatl direction."""
    return [p for p in golden_pairs if p["src_lang"] == "es" and p["tgt_lang"] == "nah"]


# ---------------------------------------------------------------------------
# Corpus fixtures (no Excel dependency for fast unit tests)
# ---------------------------------------------------------------------------

@pytest.fixture
def mini_corpus_entries():
    """A small set of in-memory parallel entries for unit tests."""
    from webapp.corpus import ParallelCorpus, ParallelEntry

    corpus = ParallelCorpus()
    entries = [
        ParallelEntry(english="In the beginning God created the heaven and the earth.",
                      nahuatl="Ipan tlahtocapehualiztli in teotl oquiyocox in ilhuicatl ihuan in tlalticpactli.",
                      source="Genesis 1:1"),
        ParallelEntry(english="And God said, Let there be light.",
                      nahuatl="Ihuan in teotl oquihtoh, Ma tlanextli.",
                      source="Genesis 1:3"),
        ParallelEntry(english="The Lord is my shepherd; I shall not want.",
                      nahuatl="In Totecuiyo nopixcauh; amo nitlapoloz.",
                      source="Psalm 23:1"),
        ParallelEntry(english="The water is very cold.",
                      nahuatl="Atl huel itztic.",
                      source="common phrase"),
        ParallelEntry(english="I want to eat tortillas.",
                      nahuatl="Nicnequi nitlaxcalcuas.",
                      source="common phrase"),
        ParallelEntry(english="Good morning. How are you?",
                      nahuatl="Cualli tonalli. Quen otitlathuil?",
                      source="greeting"),
        ParallelEntry(english="I love flowers and songs.",
                      nahuatl="Nitlazohtla in xochitl ihuan in cuicatl.",
                      source="literary"),
        ParallelEntry(english="The warrior went to the mountain.",
                      nahuatl="In yaoquizqui oyah in tepetl.",
                      source="narrative"),
    ]
    for idx, entry in enumerate(entries):
        corpus.entries.append(entry)
        # Build inverted index
        from webapp.corpus import _tokenize
        for tok in _tokenize(entry.english):
            corpus._en_index.setdefault(tok, set()).add(idx)
        for tok in _tokenize(entry.nahuatl):
            corpus._nah_index.setdefault(tok, set()).add(idx)
    corpus._loaded = True
    return corpus


@pytest.fixture
def mini_dictionary():
    """A small in-memory dictionary for unit tests."""
    from webapp.dictionary import NahuatlDictionary, DictEntry

    d = NahuatlDictionary()
    word_pairs = [
        ("water", "atl"),
        ("house", "calli"),
        ("sun", "tonatiuh"),
        ("flower", "xochitl"),
        ("song", "cuicatl"),
        ("fire", "tletl"),
        ("earth", "tlalli"),
        ("mountain", "tepetl"),
        ("rain", "quiahuitl"),
        ("wind", "ehecatl"),
        ("serpent", "coatl"),
        ("heart", "yollotl"),
        ("death", "miquiztli"),
        ("love", "tlazohtlaliztli"),
        ("ruler", "tlatoani"),
        ("warrior", "yaoquizqui"),
    ]
    for en, nah in word_pairs:
        d._en_to_nah.setdefault(en, []).append(DictEntry(term=en, translation=nah))
        d._nah_to_en.setdefault(nah, []).append(DictEntry(term=nah, translation=en))
    d._loaded = True
    return d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def corpus_xlsx_path() -> str:
    """Return the path to the real corpus Excel file (may not exist in CI)."""
    return str(CORPUS_XLSX_PATH)


def has_openai_key() -> bool:
    """Check if OPENAI_API_KEY is set (for skipping live API tests)."""
    return bool(os.getenv("OPENAI_API_KEY"))


def has_corpus_xlsx() -> bool:
    """Check if the corpus Excel file exists (for integration tests)."""
    return CORPUS_XLSX_PATH.is_file()


skip_no_openai = pytest.mark.skipif(
    not has_openai_key(),
    reason="OPENAI_API_KEY not set — skipping live API tests",
)

skip_no_corpus = pytest.mark.skipif(
    not has_corpus_xlsx(),
    reason="Corpus Excel file not found — skipping corpus-dependent tests",
)
