"""Unit tests for webapp/dictionary.py — supplementary vocabulary index.

The dictionary is a SUPPLEMENTARY resource — the AI is the primary translator.
These tests verify the dictionary provides correct supplementary vocabulary
lookups, morpheme splitting, and conversational vocab entries.
"""

import pytest
from webapp.dictionary import (
    split_morphemes,
    NahuatlDictionary,
    DictEntry,
    _extract_tokens,
    _content_tokens,
    CONVERSATIONAL_VOCAB,
)


# ===================================================================
# Morpheme splitter tests
# ===================================================================

class TestSplitMorphemes:
    """Tests for the heuristic Nahuatl morpheme splitter."""

    def test_short_word_unchanged(self):
        """Words shorter than 4 chars should return unchanged."""
        assert split_morphemes("atl") == ["atl"]
        assert split_morphemes("in") == ["in"]

    def test_prefix_stripping_ni(self):
        """ni- subject prefix should be stripped."""
        segments = split_morphemes("nicchihua")
        assert "nicchihua" in segments
        assert any(s != "nicchihua" for s in segments)

    def test_prefix_stripping_ti(self):
        segments = split_morphemes("ticchihua")
        assert "ticchihua" in segments
        assert len(segments) > 1

    def test_suffix_stripping_tl(self):
        """Absolutive suffix -tl should be stripped."""
        segments = split_morphemes("chocolatl")
        assert "chocolatl" in segments
        assert any("chocola" in s for s in segments)

    def test_suffix_stripping_tli(self):
        segments = split_morphemes("miquiztli")
        assert "miquiztli" in segments
        assert len(segments) > 1

    def test_suffix_stripping_tzin(self):
        segments = split_morphemes("tlacatzin")
        assert "tlacatzin" in segments
        assert any("tlaca" in s for s in segments)

    def test_locative_suffix_tlan(self):
        segments = split_morphemes("cuauhtlan")
        assert "cuauhtlan" in segments
        assert len(segments) > 1

    def test_possessive_prefix_no(self):
        segments = split_morphemes("nocaltzin")
        assert "nocaltzin" in segments
        assert len(segments) > 1

    def test_possessive_prefix_mo(self):
        segments = split_morphemes("mocaltzin")
        assert "mocaltzin" in segments
        assert len(segments) > 1

    def test_object_prefix_nech(self):
        segments = split_morphemes("nechtlazohtla")
        assert "nechtlazohtla" in segments
        assert len(segments) > 1

    def test_object_prefix_mitz(self):
        segments = split_morphemes("mitztlazohtla")
        assert "mitztlazohtla" in segments
        assert len(segments) > 1

    def test_deduplication(self):
        segments = split_morphemes("xochitl")
        assert len(segments) == len(set(segments))

    def test_full_word_always_first(self):
        segments = split_morphemes("nitlaxcalchihua")
        assert segments[0] == "nitlaxcalchihua"

    def test_very_long_compound(self):
        segments = split_morphemes("nitlaxcalchihualtia")
        assert len(segments) >= 2

    def test_word_without_known_affixes(self):
        segments = split_morphemes("cualli")
        assert "cualli" in segments

    def test_empty_string(self):
        segments = split_morphemes("")
        assert segments == [""]

    def test_whitespace_only(self):
        segments = split_morphemes("   ")
        assert segments == [""]

    def test_prefix_not_stripped_if_remainder_too_short(self):
        segments = split_morphemes("nita")
        assert "nita" in segments


# ===================================================================
# Token extraction tests
# ===================================================================

class TestTokenExtraction:
    def test_extract_tokens_basic(self):
        tokens = _extract_tokens("Hello world test")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens

    def test_extract_tokens_short_filtered(self):
        tokens = _extract_tokens("I am ok")
        assert len(tokens) == 0

    def test_content_tokens_english(self):
        tokens = _content_tokens("The big house on the hill", lang="en")
        assert "big" in tokens
        assert "house" in tokens
        assert "hill" in tokens
        assert "the" not in tokens

    def test_content_tokens_nahuatl(self):
        """Nahuatl tokens should not filter stop words."""
        tokens = _content_tokens("in atl ihuan tepetl", lang="nah")
        assert "atl" in tokens
        assert "ihuan" in tokens
        assert "tepetl" in tokens


# ===================================================================
# NahuatlDictionary tests (supplementary vocabulary)
# ===================================================================

class TestNahuatlDictionary:
    """Tests for the NahuatlDictionary as a supplementary resource."""

    def test_dictionary_loaded(self, mini_dictionary):
        assert mini_dictionary.loaded is True

    def test_lookup_english_to_nahuatl(self, mini_dictionary):
        results = mini_dictionary.lookup("water", src_lang="en")
        assert len(results) > 0
        terms = [r[0] for r in results]
        translations = [r[1] for r in results]
        assert "water" in terms
        assert "atl" in translations

    def test_lookup_nahuatl_to_english(self, mini_dictionary):
        results = mini_dictionary.lookup("xochitl", src_lang="nah")
        assert len(results) > 0
        terms = [r[0] for r in results]
        translations = [r[1] for r in results]
        assert "xochitl" in terms
        assert "flower" in translations

    def test_lookup_unknown_word(self, mini_dictionary):
        results = mini_dictionary.lookup("xyzzy", src_lang="en")
        assert len(results) == 0

    def test_lookup_empty_query(self, mini_dictionary):
        results = mini_dictionary.lookup("", src_lang="en")
        assert len(results) == 0

    def test_lookup_max_total(self, mini_dictionary):
        results = mini_dictionary.lookup(
            "water house sun flower song fire earth mountain",
            src_lang="en", max_total=3,
        )
        assert len(results) <= 3

    def test_format_vocab_block_nonempty(self, mini_dictionary):
        block = mini_dictionary.format_vocab_block("water fire", src_lang="en")
        assert block
        assert "→" in block
        assert "water" in block.lower()

    def test_format_vocab_block_empty(self, mini_dictionary):
        block = mini_dictionary.format_vocab_block("xyzzy", src_lang="en")
        assert block == ""

    def test_format_vocab_block_max_entries(self, mini_dictionary):
        block = mini_dictionary.format_vocab_block(
            "water house sun flower song fire earth mountain rain wind",
            src_lang="en", max_entries=3,
        )
        lines = [l for l in block.strip().split("\n") if l.strip()]
        assert len(lines) <= 3

    def test_lookup_with_morpheme_splitting(self, mini_dictionary):
        mini_dictionary._nah_to_en["tepetl"] = [
            DictEntry(term="tepetl", translation="mountain")
        ]
        results = mini_dictionary.lookup("tepetl", src_lang="nah")
        assert len(results) > 0

    def test_unloaded_dictionary(self):
        d = NahuatlDictionary()
        assert d.loaded is False
        results = d.lookup("water", src_lang="en")
        assert results == []

    def test_load_nonexistent_file(self):
        d = NahuatlDictionary()
        d.load_from_corpus_xlsx("/nonexistent/path.xlsx")
        assert d.loaded is False


# ===================================================================
# Conversational vocabulary tests (supplements the biblical corpus)
# ===================================================================

class TestConversationalVocab:
    """Verify conversational vocab supplements are loaded correctly."""

    def test_conversational_vocab_exists(self):
        assert len(CONVERSATIONAL_VOCAB) >= 50, (
            f"Expected 50+ conversational entries, got {len(CONVERSATIONAL_VOCAB)}"
        )

    def test_hello_in_conversational_vocab(self):
        """'hello' should map to Pialli."""
        hello_entries = [v for k, v in CONVERSATIONAL_VOCAB if k == "hello"]
        assert len(hello_entries) > 0
        assert "Pialli" in hello_entries[0]

    def test_my_name_in_conversational_vocab(self):
        """'my name' should map to notoca."""
        name_entries = [v for k, v in CONVERSATIONAL_VOCAB if k == "my name"]
        assert len(name_entries) > 0
        assert "notoca" in name_entries[0]

    def test_hello_in_dictionary_lookup(self):
        """After loading, 'hello' should produce Pialli from lookup."""
        from webapp.dictionary import get_dictionary
        d = get_dictionary()
        results = d.lookup("hello", src_lang="en")
        translations = [r[1].lower() for r in results]
        assert any("pialli" in t for t in translations), (
            f"'hello' should map to 'Pialli', got: {results}"
        )

    def test_name_in_dictionary_lookup(self):
        """'name' should produce toca/notoca from lookup."""
        from webapp.dictionary import get_dictionary
        d = get_dictionary()
        results = d.lookup("name", src_lang="en")
        translations = " ".join(r[1].lower() for r in results)
        assert "toca" in translations or "notoca" in translations

    def test_vocab_block_for_hello(self):
        """format_vocab_block for 'hello' should include Pialli."""
        from webapp.dictionary import get_dictionary
        d = get_dictionary()
        block = d.format_vocab_block("hello my name", src_lang="en")
        assert "pialli" in block.lower() or "Pialli" in block


# ===================================================================
# Real corpus dictionary loading
# ===================================================================

class TestDictionaryFromCorpus:
    @pytest.mark.skipif(
        not __import__("os").path.isfile(
            str(__import__("pathlib").Path(__file__).resolve().parents[1] / "data" / "english_to_nahuatl_parallel.xlsx")
        ),
        reason="Corpus Excel file not found",
    )
    def test_load_real_dictionary(self, corpus_xlsx_path):
        d = NahuatlDictionary()
        d.load_from_corpus_xlsx(corpus_xlsx_path)
        assert d.loaded is True
        assert len(d._en_to_nah) > 100
        assert len(d._nah_to_en) > 100

    @pytest.mark.skipif(
        not __import__("os").path.isfile(
            str(__import__("pathlib").Path(__file__).resolve().parents[1] / "data" / "english_to_nahuatl_parallel.xlsx")
        ),
        reason="Corpus Excel file not found",
    )
    def test_real_dictionary_lookup(self, corpus_xlsx_path):
        d = NahuatlDictionary()
        d.load_from_corpus_xlsx(corpus_xlsx_path)
        results = d.lookup("God", src_lang="en")
        assert len(results) > 0, "Expected to find 'God' in biblical corpus"
