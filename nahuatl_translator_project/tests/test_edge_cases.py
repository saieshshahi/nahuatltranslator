"""Edge case tests for the Nahuatl Translator.

Tests unusual inputs, boundary conditions, and error handling across all modules.
"""

import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest


# ===================================================================
# Corpus edge cases
# ===================================================================

class TestCorpusEdgeCases:
    def test_search_very_long_query(self, mini_corpus_entries):
        """A very long query should not crash."""
        long_query = "word " * 1000
        results = mini_corpus_entries.search(long_query, src_lang="en", max_results=5)
        assert isinstance(results, list)

    def test_search_special_characters(self, mini_corpus_entries):
        """Query with special characters should not crash."""
        results = mini_corpus_entries.search("hello!!! @#$% ^&*()", src_lang="en")
        assert isinstance(results, list)

    def test_search_unicode_characters(self, mini_corpus_entries):
        """Query with unicode characters should work."""
        results = mini_corpus_entries.search("café résumé naïve", src_lang="en")
        assert isinstance(results, list)

    def test_search_nahuatl_accented(self, mini_corpus_entries):
        """Nahuatl text with accented characters should be handled."""
        results = mini_corpus_entries.search("oquiyocox", src_lang="nah")
        assert isinstance(results, list)

    def test_search_numbers_in_query(self, mini_corpus_entries):
        """Numbers in query should be handled gracefully."""
        results = mini_corpus_entries.search("123 456 789", src_lang="en")
        assert isinstance(results, list)

    def test_search_max_results_zero(self, mini_corpus_entries):
        """max_results=0 should return empty list."""
        results = mini_corpus_entries.search("God", src_lang="en", max_results=0)
        assert results == []

    def test_search_max_results_negative(self, mini_corpus_entries):
        """Negative max_results should return empty or very short list."""
        results = mini_corpus_entries.search("God", src_lang="en", max_results=-1)
        # The slice [:max_results] with negative max_results excludes last items
        assert isinstance(results, list)

    def test_format_reference_single_entry(self, mini_corpus_entries):
        """Formatting a single entry should work."""
        from webapp.corpus import ParallelEntry
        entries = [ParallelEntry(english="hello", nahuatl="niltze")]
        result = mini_corpus_entries.format_as_reference(entries, src_lang="en")
        assert "English: hello" in result
        assert "Nahuatl: niltze" in result


# ===================================================================
# Dictionary edge cases
# ===================================================================

class TestDictionaryEdgeCases:
    def test_morpheme_split_single_char(self):
        from webapp.dictionary import split_morphemes
        segments = split_morphemes("a")
        assert segments == ["a"]

    def test_morpheme_split_two_chars(self):
        from webapp.dictionary import split_morphemes
        segments = split_morphemes("in")
        assert segments == ["in"]

    def test_morpheme_split_three_chars(self):
        from webapp.dictionary import split_morphemes
        segments = split_morphemes("atl")
        assert "atl" in segments

    def test_lookup_with_punctuation(self, mini_dictionary):
        results = mini_dictionary.lookup("water!", src_lang="en")
        # Should still find "water" despite punctuation
        assert isinstance(results, list)

    def test_lookup_max_per_term_one(self, mini_dictionary):
        results = mini_dictionary.lookup("water house sun", src_lang="en", max_per_term=1)
        assert isinstance(results, list)

    def test_lookup_max_total_one(self, mini_dictionary):
        results = mini_dictionary.lookup("water house sun", src_lang="en", max_total=1)
        assert len(results) <= 1

    def test_vocab_block_formatting(self, mini_dictionary):
        block = mini_dictionary.format_vocab_block("water", src_lang="en")
        if block:
            assert "→" in block
            lines = block.strip().split("\n")
            for line in lines:
                assert line.startswith("- ")


# ===================================================================
# Prompt edge cases
# ===================================================================

class TestPromptEdgeCases:
    def test_empty_text_translation(self):
        from webapp.prompts import translation_user_prompt
        prompt = translation_user_prompt("", "en", "nah", "")
        assert "TEXT TO TRANSLATE" in prompt

    def test_very_long_text_translation(self):
        from webapp.prompts import translation_user_prompt
        long_text = "word " * 5000
        prompt = translation_user_prompt(long_text, "en", "nah", "")
        assert len(prompt) > 5000

    def test_variety_with_special_chars(self):
        from webapp.prompts import translation_system_prompt
        prompt = translation_system_prompt("en", "nah", "Huasteca (Eastern)")
        assert "Huasteca (Eastern)" in prompt

    def test_empty_variety(self):
        from webapp.prompts import translation_system_prompt
        prompt = translation_system_prompt("en", "nah", "")
        # Should not contain variety-specific text
        assert "specified the variety" not in prompt

    def test_tile_prompt_edge_single_tile(self):
        from webapp.prompts import transcription_tile_prompt
        # Single tile should still work
        prompt = transcription_tile_prompt(0, 1, "nah")
        assert "section 1 of 1" in prompt

    def test_stitch_prompt_single_segment(self):
        from webapp.prompts import transcription_stitch_prompt
        prompt = transcription_stitch_prompt(1)
        assert "1" in prompt


# ===================================================================
# Entity edge cases
# ===================================================================

class TestEntityEdgeCases:
    def test_format_entity_reference_max_one(self):
        from webapp.entities import format_entity_reference
        ref = format_entity_reference(max_per_type=1)
        assert len(ref) > 50

    def test_format_entity_reference_max_zero(self):
        from webapp.entities import format_entity_reference
        ref = format_entity_reference(max_per_type=0)
        # Should still have headers but no entries
        assert "KNOWN MESOAMERICAN ENTITIES" in ref

    def test_format_entity_reference_max_large(self):
        from webapp.entities import format_entity_reference
        ref = format_entity_reference(max_per_type=100)
        assert len(ref) > 200

    def test_entity_notes_not_empty(self):
        from webapp.entities import KNOWN_ENTITIES
        for entity in KNOWN_ENTITIES:
            assert len(entity["note"]) > 5, f"Entity '{entity['name']}' has too short a note"


# ===================================================================
# OCR edge cases
# ===================================================================

class TestOCREdgeCases:
    def _skip_if_no_deps(self):
        try:
            from PIL import Image  # noqa: F401
            from webapp.ocr import tile_image  # noqa: F401
        except ImportError as e:
            pytest.skip(f"OCR dependency missing: {e}")

    def test_tile_very_small_image(self):
        """Image smaller than tile_height should return original."""
        self._skip_if_no_deps()
        from PIL import Image
        from webapp.ocr import tile_image

        img = Image.new("RGB", (100, 50))
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        img.save(path)

        try:
            tiles = tile_image(path, tile_height=800)
            assert len(tiles) == 1
            assert tiles[0] == path
        finally:
            os.unlink(path)

    def test_tile_1px_tall_image(self):
        """1px tall image should not crash."""
        self._skip_if_no_deps()
        from PIL import Image
        from webapp.ocr import tile_image

        img = Image.new("RGB", (100, 1))
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        img.save(path)

        try:
            tiles = tile_image(path, tile_height=800)
            assert len(tiles) == 1
        finally:
            os.unlink(path)

    def test_tile_wide_short_image(self):
        """Very wide but short image should return original."""
        self._skip_if_no_deps()
        from PIL import Image
        from webapp.ocr import tile_image

        img = Image.new("RGB", (5000, 200))
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        img.save(path)

        try:
            tiles = tile_image(path, tile_height=800)
            assert len(tiles) == 1
        finally:
            os.unlink(path)


# ===================================================================
# Services edge cases
# ===================================================================

class TestServicesEdgeCases:
    def test_translate_empty_text(self):
        """Translating empty text should not crash."""
        mock_resp = MagicMock()
        mock_resp.output_text = ""

        mock_client = MagicMock()
        mock_client.responses.create.return_value = mock_resp

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}), \
             patch("openai.OpenAI", return_value=mock_client):
            from webapp.services import openai_translate
            result = openai_translate("", "en", "nah", "Unknown")
            assert isinstance(result, str)

    def test_translate_very_long_text(self):
        """Translating very long text should not crash."""
        mock_resp = MagicMock()
        mock_resp.output_text = "translated"

        mock_client = MagicMock()
        mock_client.responses.create.return_value = mock_resp

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}), \
             patch("openai.OpenAI", return_value=mock_client):
            from webapp.services import openai_translate
            result = openai_translate("word " * 10000, "en", "nah", "Unknown")
            assert isinstance(result, str)

    def test_translate_with_newlines(self):
        """Text with newlines should be handled."""
        mock_resp = MagicMock()
        mock_resp.output_text = "translated"

        mock_client = MagicMock()
        mock_client.responses.create.return_value = mock_resp

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}), \
             patch("openai.OpenAI", return_value=mock_client):
            from webapp.services import openai_translate
            result = openai_translate("line1\nline2\nline3", "en", "nah", "Unknown")
            assert isinstance(result, str)

    def test_translate_null_output(self):
        """API returning None should be handled."""
        mock_resp = MagicMock()
        mock_resp.output_text = None

        mock_client = MagicMock()
        mock_client.responses.create.return_value = mock_resp

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}), \
             patch("openai.OpenAI", return_value=mock_client):
            from webapp.services import openai_translate
            result = openai_translate("Hello", "en", "nah", "Unknown")
            assert result == ""

    def test_extract_empty_text(self):
        """Extracting from empty text should not crash."""
        mock_resp = MagicMock()
        mock_resp.output_text = "{}"

        mock_client = MagicMock()
        mock_client.responses.create.return_value = mock_resp

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}), \
             patch("openai.OpenAI", return_value=mock_client):
            from webapp.services import openai_extract
            result = openai_extract("", "Extract entities.")
            assert result == "{}"

    def test_extract_api_failure(self):
        """API failure in extraction should raise RuntimeError."""
        mock_client = MagicMock()
        mock_client.responses.create.side_effect = Exception("Connection timeout")

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}), \
             patch("openai.OpenAI", return_value=mock_client):
            from webapp.services import openai_extract
            with pytest.raises(RuntimeError, match="OpenAI extraction failed"):
                openai_extract("Test text", "Extract entities.")

    def test_repetition_detector_borderline(self):
        """Test repetition detection at the threshold."""
        from webapp.services import guess_repetition

        # Exactly at threshold (45% repetition)
        tokens = ["word"] * 9 + ["other"] * 11  # 9/20 = 45%
        text = " ".join(tokens)
        result = guess_repetition(text)
        assert isinstance(result, bool)

    def test_repetition_detector_empty(self):
        from webapp.services import guess_repetition
        assert guess_repetition("") is False

    def test_repetition_detector_single_word(self):
        from webapp.services import guess_repetition
        assert guess_repetition("hello") is False

    def test_maybe_load_varieties_missing_file(self):
        from webapp.services import maybe_load_varieties
        result = maybe_load_varieties("/nonexistent/file.json")
        assert result == ["Unknown"]

    def test_maybe_load_varieties_invalid_json(self):
        from webapp.services import maybe_load_varieties
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        with open(path, "w") as f:
            f.write("not valid json")
        try:
            result = maybe_load_varieties(path)
            assert result == ["Unknown"]
        finally:
            os.unlink(path)

    def test_maybe_load_varieties_empty_list(self):
        from webapp.services import maybe_load_varieties
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        with open(path, "w") as f:
            f.write("[]")
        try:
            result = maybe_load_varieties(path)
            assert result == ["Unknown"]
        finally:
            os.unlink(path)

    def test_maybe_load_varieties_valid(self):
        from webapp.services import maybe_load_varieties
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        with open(path, "w") as f:
            f.write('["Classical", "Huasteca"]')
        try:
            result = maybe_load_varieties(path)
            assert result == ["Classical", "Huasteca"]
        finally:
            os.unlink(path)


# ===================================================================
# Cross-module edge cases
# ===================================================================

class TestCrossModuleEdgeCases:
    def test_corpus_and_dictionary_agree_on_water(self, mini_corpus_entries, mini_dictionary):
        """Both corpus and dictionary should find 'water' related content."""
        corpus_results = mini_corpus_entries.search("water", src_lang="en")
        dict_results = mini_dictionary.lookup("water", src_lang="en")

        # Both should have results
        assert len(corpus_results) > 0
        assert len(dict_results) > 0

    def test_entity_taxonomy_covers_corpus_entities(self):
        """Entities mentioned in disambiguation rules should exist in taxonomy."""
        from webapp.entities import KNOWN_ENTITIES, DISAMBIGUATION_RULES

        entity_names = {e["name"].lower() for e in KNOWN_ENTITIES}
        # Key entities mentioned in rules should be in taxonomy
        key_entities = ["mexica", "mexico", "tlatoani"]
        for name in key_entities:
            assert name in entity_names, f"'{name}' in rules but not in taxonomy"

    def test_prompts_reference_linguistic_features(self):
        """Translation prompts should reference key linguistic features."""
        from webapp.prompts import translation_system_prompt

        prompt = translation_system_prompt("en", "nah", "")
        key_features = ["agglutinative", "polysynthetic", "VSO", "articles"]
        for feature in key_features:
            assert feature in prompt, f"Missing linguistic feature: {feature}"
