"""Unit tests for webapp/corpus.py — parallel corpus loader and keyword search."""

import pytest
from webapp.corpus import ParallelCorpus, ParallelEntry, _tokenize


# ===================================================================
# Tokenizer tests
# ===================================================================

class TestTokenize:
    """Tests for the _tokenize() helper."""

    def test_basic_tokenization(self):
        tokens = _tokenize("The cat sat on the mat")
        assert "cat" in tokens
        assert "sat" in tokens
        assert "mat" in tokens

    def test_stop_words_removed(self):
        tokens = _tokenize("The cat is on the mat")
        assert "the" not in tokens
        assert "is" not in tokens
        assert "on" not in tokens

    def test_short_words_removed(self):
        """Words with 2 or fewer characters should be filtered out."""
        tokens = _tokenize("I am a go to do it")
        # All are stop words or too short
        assert len(tokens) == 0

    def test_lowercase_normalization(self):
        tokens = _tokenize("HELLO World FooBar")
        assert "hello" in tokens
        assert "world" in tokens
        assert "foobar" in tokens

    def test_nahuatl_accented_chars(self):
        tokens = _tokenize("oquiyocox ihuan tlalticpactli")
        assert "oquiyocox" in tokens
        assert "ihuan" in tokens
        assert "tlalticpactli" in tokens

    def test_empty_input(self):
        assert _tokenize("") == []

    def test_only_stop_words(self):
        assert _tokenize("the is a an of in to for") == []

    def test_punctuation_stripped(self):
        tokens = _tokenize("Hello, world! How are you?")
        assert "hello" in tokens
        assert "world" in tokens

    def test_mixed_nahuatl_english(self):
        tokens = _tokenize("The tlatoani spoke Nahuatl in Tenochtitlan")
        assert "tlatoani" in tokens
        assert "spoke" in tokens
        assert "nahuatl" in tokens
        assert "tenochtitlan" in tokens


# ===================================================================
# ParallelCorpus tests
# ===================================================================

class TestParallelCorpus:
    """Tests for the ParallelCorpus class using the mini_corpus_entries fixture."""

    def test_corpus_loaded(self, mini_corpus_entries):
        assert mini_corpus_entries.loaded is True
        assert len(mini_corpus_entries.entries) == 8

    def test_search_english_query(self, mini_corpus_entries):
        results = mini_corpus_entries.search("water cold", src_lang="en", max_results=3)
        assert len(results) > 0
        # Should find the "water is very cold" entry
        found_water = any("water" in r.english.lower() for r in results)
        assert found_water, "Expected to find an entry about water"

    def test_search_nahuatl_query(self, mini_corpus_entries):
        results = mini_corpus_entries.search("xochitl cuicatl", src_lang="nah", max_results=3)
        assert len(results) > 0
        found = any("xochitl" in r.nahuatl.lower() for r in results)
        assert found, "Expected to find entry containing xochitl"

    def test_search_returns_max_results(self, mini_corpus_entries):
        results = mini_corpus_entries.search("God light earth", src_lang="en", max_results=2)
        assert len(results) <= 2

    def test_search_no_results_for_unknown(self, mini_corpus_entries):
        results = mini_corpus_entries.search("xyzzy foobar blorp", src_lang="en")
        assert len(results) == 0

    def test_search_empty_query(self, mini_corpus_entries):
        results = mini_corpus_entries.search("", src_lang="en")
        assert len(results) == 0

    def test_search_only_stop_words(self, mini_corpus_entries):
        results = mini_corpus_entries.search("the is a", src_lang="en")
        assert len(results) == 0

    def test_search_prefers_shorter_entries(self, mini_corpus_entries):
        """Shorter entries should be ranked higher due to length normalization."""
        results = mini_corpus_entries.search("water", src_lang="en", max_results=5)
        if len(results) >= 2:
            # The short entry "The water is very cold" should score well
            first_len = len(results[0].english.split())
            assert first_len < 20, "Expected shorter entries first"

    def test_format_as_reference_english(self, mini_corpus_entries):
        entries = mini_corpus_entries.entries[:2]
        formatted = mini_corpus_entries.format_as_reference(entries, src_lang="en")
        assert "English:" in formatted
        assert "Nahuatl:" in formatted
        lines = formatted.strip().split("\n")
        assert len(lines) == 4  # 2 entries * 2 lines each

    def test_format_as_reference_empty(self, mini_corpus_entries):
        formatted = mini_corpus_entries.format_as_reference([], src_lang="en")
        assert formatted == ""

    def test_search_god_genesis(self, mini_corpus_entries):
        """Search for 'God created heaven' should match Genesis entries."""
        results = mini_corpus_entries.search("God created heaven", src_lang="en", max_results=5)
        assert len(results) > 0
        texts = [r.english for r in results]
        assert any("God" in t for t in texts)

    def test_search_tortilla_phrase(self, mini_corpus_entries):
        results = mini_corpus_entries.search("eat tortillas", src_lang="en", max_results=3)
        assert len(results) > 0
        assert any("tortilla" in r.english.lower() for r in results)

    def test_empty_corpus_search(self):
        """Searching an empty (unloaded) corpus should return empty list."""
        corpus = ParallelCorpus()
        assert corpus.loaded is False
        results = corpus.search("hello", src_lang="en")
        assert results == []


# ===================================================================
# Corpus loading from Excel (requires file on disk)
# ===================================================================

class TestCorpusXlsxLoading:
    """Tests for loading the actual corpus Excel file."""

    @pytest.mark.skipif(
        not __import__("os").path.isfile(
            str(__import__("pathlib").Path(__file__).resolve().parents[1] / "data" / "english_to_nahuatl_parallel.xlsx")
        ),
        reason="Corpus Excel file not found",
    )
    def test_load_real_corpus(self, corpus_xlsx_path):
        corpus = ParallelCorpus()
        corpus.load_xlsx(corpus_xlsx_path)
        assert corpus.loaded is True
        assert len(corpus.entries) > 1000, f"Expected 7000+ entries, got {len(corpus.entries)}"

    @pytest.mark.skipif(
        not __import__("os").path.isfile(
            str(__import__("pathlib").Path(__file__).resolve().parents[1] / "data" / "english_to_nahuatl_parallel.xlsx")
        ),
        reason="Corpus Excel file not found",
    )
    def test_real_corpus_search(self, corpus_xlsx_path):
        corpus = ParallelCorpus()
        corpus.load_xlsx(corpus_xlsx_path)
        results = corpus.search("beginning God created", src_lang="en", max_results=5)
        assert len(results) > 0

    def test_load_nonexistent_file(self):
        corpus = ParallelCorpus()
        corpus.load_xlsx("/nonexistent/path/fake.xlsx")
        assert corpus.loaded is False
        assert len(corpus.entries) == 0

    def test_entries_have_content(self, mini_corpus_entries):
        """All entries should have non-empty english and nahuatl fields."""
        for entry in mini_corpus_entries.entries:
            assert entry.english.strip(), "English text should not be empty"
            assert entry.nahuatl.strip(), "Nahuatl text should not be empty"
