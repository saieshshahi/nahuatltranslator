"""Unit tests for webapp/prompts.py — prompt template builders."""

import pytest
from webapp.prompts import (
    LANG_LABELS,
    NAHUATL_LINGUISTIC_CONTEXT,
    FEWSHOT_TRANSLATION_EXAMPLES,
    COLONIAL_NAHUATL_TRANSCRIPTION_CONTEXT,
    translation_system_prompt,
    translation_user_prompt,
    translation_variants_system_prompt,
    transcription_system_prompt,
    transcription_overview_prompt,
    transcription_tile_prompt,
    transcription_stitch_prompt,
    extraction_system_prompt,
    extraction_user_prompt,
    _format_fewshot,
    _label,
)


# ===================================================================
# Language label tests
# ===================================================================

class TestLangLabels:
    def test_known_labels(self):
        assert _label("en") == "English"
        assert _label("es") == "Spanish"
        assert _label("nah") == "Nahuatl"

    def test_unknown_label_returns_code(self):
        assert _label("fr") == "fr"
        assert _label("xyz") == "xyz"

    def test_all_labels_defined(self):
        assert "en" in LANG_LABELS
        assert "es" in LANG_LABELS
        assert "nah" in LANG_LABELS


# ===================================================================
# Linguistic context tests
# ===================================================================

class TestLinguisticContext:
    def test_contains_morphology_rules(self):
        assert "agglutinative" in NAHUATL_LINGUISTIC_CONTEXT
        assert "polysynthetic" in NAHUATL_LINGUISTIC_CONTEXT

    def test_contains_verb_structure(self):
        assert "SUBJECT_PREFIX" in NAHUATL_LINGUISTIC_CONTEXT
        assert "OBJECT_PREFIX" in NAHUATL_LINGUISTIC_CONTEXT

    def test_contains_no_articles_rule(self):
        assert "NO definite or indefinite articles" in NAHUATL_LINGUISTIC_CONTEXT

    def test_contains_orthographic_conventions(self):
        assert "ORTHOGRAPHIC CONVENTIONS" in NAHUATL_LINGUISTIC_CONTEXT
        assert "hu = /w/" in NAHUATL_LINGUISTIC_CONTEXT

    def test_contains_pitfalls(self):
        assert "COMMON TRANSLATION PITFALLS" in NAHUATL_LINGUISTIC_CONTEXT
        assert "chocolatl" in NAHUATL_LINGUISTIC_CONTEXT

    def test_contains_dialect_notes(self):
        assert "DIALECT/VARIETY NOTES" in NAHUATL_LINGUISTIC_CONTEXT
        assert "Huasteca" in NAHUATL_LINGUISTIC_CONTEXT

    def test_colonial_context_has_long_s(self):
        assert "Long-s" in COLONIAL_NAHUATL_TRANSCRIPTION_CONTEXT

    def test_colonial_context_has_abbreviations(self):
        assert "xpo" in COLONIAL_NAHUATL_TRANSCRIPTION_CONTEXT


# ===================================================================
# Few-shot examples tests
# ===================================================================

class TestFewshot:
    def test_examples_exist(self):
        assert len(FEWSHOT_TRANSLATION_EXAMPLES) >= 5

    def test_examples_have_correct_structure(self):
        for ex in FEWSHOT_TRANSLATION_EXAMPLES:
            assert len(ex) == 4, f"Expected 4-tuple, got {len(ex)}"
            src, tgt, inp, out = ex
            assert src in ("en", "es", "nah")
            assert tgt in ("en", "es", "nah")
            assert len(inp) > 0
            assert len(out) > 0

    def test_format_fewshot_en_to_nah(self):
        formatted = _format_fewshot("en", "nah")
        assert "English:" in formatted
        assert "Nahuatl:" in formatted
        assert len(formatted) > 50

    def test_format_fewshot_nah_to_en(self):
        formatted = _format_fewshot("nah", "en")
        assert "Nahuatl:" in formatted
        assert "English:" in formatted

    def test_format_fewshot_es_to_nah(self):
        formatted = _format_fewshot("es", "nah")
        assert "Nahuatl:" in formatted

    def test_format_fewshot_unknown_direction_gets_fallback(self):
        """Unknown direction should still return some examples as fallback."""
        formatted = _format_fewshot("de", "nah")
        assert len(formatted) > 0


# ===================================================================
# Translation prompt tests
# ===================================================================

class TestTranslationPrompts:
    def test_system_prompt_contains_linguistic_context(self):
        prompt = translation_system_prompt("en", "nah", "")
        assert "agglutinative" in prompt
        assert "Nahuatl" in prompt

    def test_system_prompt_contains_direction(self):
        prompt = translation_system_prompt("en", "nah", "")
        assert "English" in prompt
        assert "Nahuatl" in prompt
        assert "TRANSLATION DIRECTION" in prompt

    def test_system_prompt_with_variety(self):
        prompt = translation_system_prompt("en", "nah", "Huasteca")
        assert "Huasteca" in prompt

    def test_system_prompt_unknown_variety_ignored(self):
        prompt = translation_system_prompt("en", "nah", "Unknown")
        assert "Unknown" not in prompt or "variety" not in prompt.lower().split("Unknown")[0][-50:]

    def test_system_prompt_contains_fewshot(self):
        prompt = translation_system_prompt("en", "nah", "")
        assert "REFERENCE TRANSLATION EXAMPLES" in prompt

    def test_system_prompt_contains_instructions(self):
        prompt = translation_system_prompt("en", "nah", "")
        assert "INSTRUCTIONS:" in prompt
        assert "ONLY the translation" in prompt

    def test_user_prompt_contains_text(self):
        prompt = translation_user_prompt("Hello world", "en", "nah", "")
        assert "Hello world" in prompt
        assert "TEXT TO TRANSLATE" in prompt

    def test_user_prompt_with_vocab(self):
        prompt = translation_user_prompt(
            "Hello", "en", "nah", "",
            reference_vocab='- "hello" → "niltze"',
        )
        assert "REFERENCE VOCABULARY" in prompt
        assert "niltze" in prompt

    def test_user_prompt_with_sentences(self):
        prompt = translation_user_prompt(
            "Hello", "en", "nah", "",
            reference_sentences="- English: Hello\n  Nahuatl: Niltze",
        )
        assert "REFERENCE PARALLEL SENTENCES" in prompt

    def test_user_prompt_no_context(self):
        prompt = translation_user_prompt("Hello", "en", "nah", "")
        assert "REFERENCE VOCABULARY" not in prompt
        assert "REFERENCE PARALLEL SENTENCES" not in prompt

    def test_variants_system_prompt_extends_base(self):
        base = translation_system_prompt("en", "nah", "")
        variants = translation_variants_system_prompt("en", "nah", "")
        assert len(variants) > len(base)
        assert "variant" in variants.lower() or "phrasing" in variants.lower()

    def test_all_directions_produce_prompts(self):
        directions = [("en", "nah"), ("nah", "en"), ("es", "nah"), ("nah", "es")]
        for src, tgt in directions:
            prompt = translation_system_prompt(src, tgt, "")
            assert len(prompt) > 100, f"Prompt for {src}->{tgt} seems too short"


# ===================================================================
# Transcription prompt tests
# ===================================================================

class TestTranscriptionPrompts:
    def test_system_prompt_default(self):
        prompt = transcription_system_prompt()
        assert "paleographer" in prompt.lower()
        assert "colonial" in prompt.lower()

    def test_system_prompt_nahuatl_hint(self):
        prompt = transcription_system_prompt(language_hint="nah")
        assert "Nahuatl" in prompt
        assert "agglutinative" in prompt

    def test_system_prompt_spanish_hint(self):
        prompt = transcription_system_prompt(language_hint="es")
        assert "Spanish" in prompt

    def test_system_prompt_with_alphabet_hint(self):
        prompt = transcription_system_prompt(alphabet_hint="Use modern orthography")
        assert "modern orthography" in prompt

    def test_transcription_rules_present(self):
        prompt = transcription_system_prompt()
        assert "TRANSCRIPTION RULES" in prompt
        assert "EXACTLY what you see" in prompt
        assert "[illegible]" in prompt

    def test_overview_prompt(self):
        prompt = transcription_overview_prompt("nah")
        assert "structure" in prompt.lower()
        assert "text lines" in prompt.lower()
        assert "Do NOT transcribe" in prompt

    def test_tile_prompt_first(self):
        prompt = transcription_tile_prompt(0, 5, "nah")
        assert "section 1 of 5" in prompt
        assert "top" in prompt

    def test_tile_prompt_middle(self):
        prompt = transcription_tile_prompt(2, 5, "es")
        assert "section 3 of 5" in prompt
        assert "middle" in prompt

    def test_tile_prompt_last(self):
        prompt = transcription_tile_prompt(4, 5, "nah")
        assert "section 5 of 5" in prompt
        assert "bottom" in prompt

    def test_stitch_prompt(self):
        prompt = transcription_stitch_prompt(4)
        assert "4" in prompt
        assert "overlap" in prompt.lower()
        assert "duplicate" in prompt.lower()


# ===================================================================
# Extraction prompt tests
# ===================================================================

class TestExtractionPrompts:
    def test_system_prompt_default(self):
        prompt = extraction_system_prompt()
        assert "Mesoamerican" in prompt
        assert "JSON" in prompt

    def test_system_prompt_with_entity_ref(self):
        prompt = extraction_system_prompt(entity_reference="PEOPLE: Mexica, Nahua")
        assert "Mexica" in prompt

    def test_system_prompt_with_disambiguation(self):
        prompt = extraction_system_prompt(
            disambiguation_rules="Mexica = people, Mexico = place"
        )
        assert "Mexica = people" in prompt

    def test_system_prompt_output_rules(self):
        prompt = extraction_system_prompt()
        assert "OUTPUT RULES" in prompt
        assert "valid JSON" in prompt

    def test_user_prompt(self):
        prompt = extraction_user_prompt(
            text="The Mexica built Tenochtitlan.",
            instruction="Extract entities.",
        )
        assert "INSTRUCTION:" in prompt
        assert "TEXT:" in prompt
        assert "Mexica" in prompt
        assert "Tenochtitlan" in prompt

    def test_user_prompt_with_schema(self):
        prompt = extraction_user_prompt(
            text="Test text",
            instruction="Extract entities.",
            schema_hint='{"people": [], "places": []}',
        )
        assert "SCHEMA HINT" in prompt
        assert '"people"' in prompt

    def test_user_prompt_without_schema(self):
        prompt = extraction_user_prompt(
            text="Test text",
            instruction="Extract entities.",
        )
        assert "SCHEMA HINT" not in prompt
