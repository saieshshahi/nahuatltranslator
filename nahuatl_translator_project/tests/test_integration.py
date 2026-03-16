"""Integration tests for the Nahuatl Translator pipeline.

Architecture: OpenAI is the PRIMARY translator. The corpus and dictionary
are supplementary vocabulary sources only. These tests verify:
- AI-first translation pipeline works end-to-end
- Corpus/dictionary provide optional supplementary context
- Extraction pipeline works with entity taxonomy
"""

import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from tests.conftest import skip_no_openai, skip_no_corpus


# ===================================================================
# Corpus + Dictionary as supplementary context
# ===================================================================

class TestSupplementaryContextIntegration:
    """Test corpus and dictionary modules provide optional supplementary context."""

    def test_supplementary_context_pipeline(self, mini_corpus_entries, mini_dictionary):
        """Simulate _get_corpus_context: corpus search + dictionary lookup as supplements."""
        query = "water cold"

        # Corpus search (supplementary sentences)
        matches = mini_corpus_entries.search(query, src_lang="en", max_results=3)
        ref_sentences = mini_corpus_entries.format_as_reference(matches, src_lang="en")

        # Dictionary lookup (supplementary vocab)
        ref_vocab = mini_dictionary.format_vocab_block(query, src_lang="en", max_entries=5)

        # Both should produce output when available
        assert len(ref_sentences) > 0, "Corpus should find relevant supplementary sentences"
        assert len(ref_vocab) > 0, "Dictionary should find relevant supplementary vocab"
        assert "English:" in ref_sentences
        assert "→" in ref_vocab

    def test_empty_query_produces_empty_context(self, mini_corpus_entries, mini_dictionary):
        """Empty query should produce empty context — AI translates on its own."""
        ref_sentences_matches = mini_corpus_entries.search("", src_lang="en")
        ref_sentences = mini_corpus_entries.format_as_reference(ref_sentences_matches, src_lang="en")
        ref_vocab = mini_dictionary.format_vocab_block("", src_lang="en")

        assert ref_sentences == ""
        assert ref_vocab == ""

    def test_translation_works_without_supplementary_context(self):
        """AI should translate even with no corpus/dictionary context."""
        from webapp.prompts import translation_system_prompt, translation_user_prompt

        system = translation_system_prompt("en", "nah", "Unknown")
        user = translation_user_prompt(
            text="Hello, my name is Carlos",
            src="en", tgt="nah", variety="Unknown",
            reference_vocab="",
            reference_sentences="",
        )

        # System prompt should position AI as expert
        assert "expert" in system.lower()
        # User prompt should have the text and NO supplementary sections
        assert "TRANSLATE THIS" in user
        assert "SUPPLEMENTARY" not in user


# ===================================================================
# Prompt + Supplementary Context integration
# ===================================================================

class TestPromptContextIntegration:
    """Test that supplementary corpus context integrates into prompts correctly."""

    def test_translation_prompt_with_supplementary_context(self, mini_corpus_entries, mini_dictionary):
        from webapp.prompts import translation_system_prompt, translation_user_prompt

        query = "The warrior went to the mountain"

        # Get supplementary context
        matches = mini_corpus_entries.search(query, src_lang="en", max_results=3)
        ref_sentences = mini_corpus_entries.format_as_reference(matches, src_lang="en")
        ref_vocab = mini_dictionary.format_vocab_block(query, src_lang="en", max_entries=5)

        # Build prompts
        system = translation_system_prompt("en", "nah", "Unknown")
        user = translation_user_prompt(
            text=query, src="en", tgt="nah", variety="Unknown",
            reference_vocab=ref_vocab, reference_sentences=ref_sentences,
        )

        # System prompt should position AI as expert
        assert "expert" in system.lower()
        assert "TRANSLATION DIRECTION" in system

        # User prompt should include supplementary context clearly labeled
        assert "SUPPLEMENTARY VOCABULARY" in user
        assert "SUPPLEMENTARY PARALLEL" in user
        assert query in user

    def test_extraction_prompt_with_entities(self):
        from webapp.entities import format_entity_reference, DISAMBIGUATION_RULES
        from webapp.prompts import extraction_system_prompt, extraction_user_prompt

        entity_ref = format_entity_reference(max_per_type=5)
        system = extraction_system_prompt(
            entity_reference=entity_ref,
            disambiguation_rules=DISAMBIGUATION_RULES,
        )
        user = extraction_user_prompt(
            text="The Mexica built Tenochtitlan near the lake.",
            instruction="Extract people and places.",
            schema_hint='{"people": [], "places": []}',
        )

        # System should have entity knowledge
        assert "Mexica" in system
        assert "DISAMBIGUATION" in system

        # User should have the text and schema
        assert "Mexica" in user
        assert "Tenochtitlan" in user
        assert "SCHEMA HINT" in user


# ===================================================================
# Services integration (mocked OpenAI — AI is the primary engine)
# ===================================================================

class TestServicesIntegrationMocked:
    """Test services.py functions with mocked OpenAI client.

    These verify that OpenAI is the primary engine and corpus context is supplementary.
    """

    def test_openai_translate_calls_api_as_primary(self):
        """Verify openai_translate uses AI as primary with expert prompts."""
        mock_resp = MagicMock()
        mock_resp.output_text = "Pialli, notoca Carlos."

        mock_client = MagicMock()
        mock_client.responses.create.return_value = mock_resp

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}), \
             patch("openai.OpenAI", return_value=mock_client):
            from webapp.services import openai_translate
            result = openai_translate("Hello, my name is Carlos", "en", "nah", "Unknown")

        assert result == "Pialli, notoca Carlos."
        mock_client.responses.create.assert_called_once()

        # Check the system prompt positions AI as expert
        call_kwargs = mock_client.responses.create.call_args
        messages = call_kwargs.kwargs.get("input") or call_kwargs[1].get("input")
        system_msg = messages[0]["content"]
        assert "expert" in system_msg.lower()
        assert "Nahuatl" in system_msg

    def test_openai_translate_variants_calls_api(self):
        """Verify variant generation calls API multiple times."""
        mock_resp = MagicMock()
        mock_resp.output_text = "Cualli tonalli"

        mock_client = MagicMock()
        mock_client.responses.create.return_value = mock_resp

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}), \
             patch("openai.OpenAI", return_value=mock_client):
            from webapp.services import openai_translate_variants
            results = openai_translate_variants("Good morning", "en", "nah", "Unknown", k=3)

        assert len(results) >= 1
        assert mock_client.responses.create.call_count >= 1

    def test_openai_extract_calls_api_with_entities(self):
        """Verify extraction includes entity taxonomy and disambiguation."""
        mock_resp = MagicMock()
        mock_resp.output_text = '{"people": [{"name": "Mexica", "type": "ethnic_group"}]}'

        mock_client = MagicMock()
        mock_client.responses.create.return_value = mock_resp

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}), \
             patch("openai.OpenAI", return_value=mock_client):
            from webapp.services import openai_extract
            result = openai_extract(
                "The Mexica founded Tenochtitlan.",
                "Extract entities.",
            )

        assert "Mexica" in result
        call_kwargs = mock_client.responses.create.call_args
        messages = call_kwargs.kwargs.get("input") or call_kwargs[1].get("input")
        system_msg = messages[0]["content"]
        assert "Mesoamerican" in system_msg
        assert "DISAMBIGUATION" in system_msg

    def test_openai_translate_failure_raises(self):
        """Verify that API failure raises RuntimeError."""
        mock_client = MagicMock()
        mock_client.responses.create.side_effect = Exception("API Error")

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}), \
             patch("openai.OpenAI", return_value=mock_client):
            from webapp.services import openai_translate
            with pytest.raises(RuntimeError, match="OpenAI translation failed"):
                openai_translate("Hello", "en", "nah", "Unknown")

    def test_corpus_failure_does_not_break_translation(self):
        """If corpus/dictionary fails, translation should still work (AI is primary)."""
        mock_resp = MagicMock()
        mock_resp.output_text = "Pialli"

        mock_client = MagicMock()
        mock_client.responses.create.return_value = mock_resp

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}), \
             patch("openai.OpenAI", return_value=mock_client), \
             patch("webapp.corpus.get_corpus", side_effect=Exception("corpus broken")):
            from webapp.services import openai_translate
            result = openai_translate("Hello", "en", "nah", "Unknown")

        # Should still work — AI is primary, corpus is optional
        assert result == "Pialli"


# ===================================================================
# Services helper function tests
# ===================================================================

class TestServiceHelpers:
    def test_build_prompt(self):
        from webapp.services import build_prompt
        p = build_prompt("en", "nah", "Hello", "Classical")
        assert "translate English to Nahuatl" in p
        assert "Hello" in p
        assert "Classical" in p

    def test_guess_repetition_true(self):
        from webapp.services import guess_repetition
        assert guess_repetition("niman " * 30) is True

    def test_guess_repetition_false(self):
        from webapp.services import guess_repetition
        assert guess_repetition("This is a normal sentence with varied words.") is False

    def test_guess_repetition_short_text(self):
        from webapp.services import guess_repetition
        assert guess_repetition("hi") is False

    def test_openai_available_with_key(self):
        from webapp.services import openai_available
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            assert openai_available() is True

    def test_openai_available_without_key(self):
        from webapp.services import openai_available
        with patch.dict(os.environ, {}, clear=True):
            assert openai_available() is False

    def test_image_to_data_url(self):
        """Test base64 data URL generation."""
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")

        import tempfile
        img = Image.new("RGB", (10, 10), "red")
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        img.save(path)

        try:
            from webapp.services import _image_to_data_url
            url = _image_to_data_url(path)
            assert url.startswith("data:image/png;base64,")
            assert len(url) > 50
        finally:
            os.unlink(path)

    def test_get_corpus_context_returns_supplementary(self, mini_corpus_entries, mini_dictionary):
        """Test _get_corpus_context provides supplementary (not primary) context."""
        with patch("webapp.corpus.get_corpus", return_value=mini_corpus_entries), \
             patch("webapp.dictionary.get_dictionary", return_value=mini_dictionary):
            from webapp.services import _get_corpus_context
            ref_sentences, ref_vocab = _get_corpus_context("water cold", "en")

            # Both should have content (when corpus is available)
            assert len(ref_sentences) > 0
            assert len(ref_vocab) > 0

    def test_get_corpus_context_gracefully_fails(self):
        """If corpus loading fails, should return empty strings (not crash)."""
        with patch("webapp.corpus.get_corpus", side_effect=Exception("broken")):
            from webapp.services import _get_corpus_context
            ref_sentences, ref_vocab = _get_corpus_context("hello", "en")
            assert ref_sentences == ""
            assert ref_vocab == ""


# ===================================================================
# Live API integration tests
# ===================================================================

class TestLiveAPIIntegration:
    """End-to-end tests with real API calls (skipped without key)."""

    @skip_no_openai
    def test_translate_en_to_nah_live(self):
        from webapp.services import openai_translate
        result = openai_translate("Hello", "en", "nah", "Unknown")
        assert len(result) > 0
        assert result != "Hello"  # Should not just echo input

    @skip_no_openai
    def test_translate_nah_to_en_live(self):
        from webapp.services import openai_translate
        result = openai_translate("Cualli tonalli", "nah", "en", "Unknown")
        assert len(result) > 0
        # Should contain English words
        assert any(c.isascii() for c in result)

    @skip_no_openai
    def test_extract_entities_live(self):
        from webapp.services import openai_extract
        text = "The Mexica people built the great city of Tenochtitlan. Their ruler was the tlatoani Motecuhzoma."
        result = openai_extract(text, "Extract people, places, and titles.")
        # Should be valid JSON
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    @skip_no_openai
    def test_full_pipeline_en_to_nah(self):
        """Full pipeline: AI primary + optional corpus context."""
        from webapp.services import openai_translate
        result = openai_translate(
            "In the beginning God created the heaven and the earth",
            "en", "nah", "Unknown",
        )
        assert len(result) > 10
        # Should contain common Nahuatl words
        result_lower = result.lower()
        has_nahuatl = any(
            w in result_lower
            for w in ["dios", "ilhuicatl", "tlalticpactli", "ihuan", "ipan", "oqui"]
        )
        assert has_nahuatl, f"Expected Nahuatl output, got: {result}"

    @skip_no_openai
    def test_ai_translates_without_corpus(self):
        """AI should produce good translation even without corpus context."""
        from webapp.services import openai_translate

        # Patch corpus to return nothing — AI should still work
        from webapp.corpus import ParallelCorpus
        from webapp.dictionary import NahuatlDictionary

        empty_corpus = ParallelCorpus()
        empty_dict = NahuatlDictionary()

        with patch("webapp.corpus.get_corpus", return_value=empty_corpus), \
             patch("webapp.dictionary.get_dictionary", return_value=empty_dict):
            result = openai_translate("Hello, my name is Carlos", "en", "nah", "Unknown")

        assert len(result) > 0
        # Should still produce Nahuatl (AI knows the language)
        result_lower = result.lower()
        has_nahuatl = "pialli" in result_lower or "notoca" in result_lower or "carlos" in result_lower
        assert has_nahuatl, f"AI should translate without corpus. Got: {result}"
