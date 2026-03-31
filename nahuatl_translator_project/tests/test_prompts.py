"""Unit tests for webapp/prompts.py — AI-first prompt template builders.

Tests verify that the prompts follow the AI-first architecture:
- OpenAI is the primary expert translator
- Corpus/dictionary are supplementary references only
- Key warnings (hello≠cualli, possessives) are present
- Few-shot examples guide the AI effectively
"""

import pytest
from webapp.prompts import (
    LANG_LABELS,
    NAHUATL_EXPERT_GUIDANCE,
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
# Expert guidance tests (AI-first approach)
# ===================================================================

class TestExpertGuidance:
    """Verify the expert guidance is concise and focused on critical rules."""

    def test_contains_anti_hallucination_rule(self):
        """Must warn against inventing words."""
        assert "invent" in NAHUATL_EXPERT_GUIDANCE.lower()

    def test_contains_possessive_rule(self):
        """Must enforce possessives as one word."""
        assert "notoca" in NAHUATL_EXPERT_GUIDANCE
        assert "nocal" in NAHUATL_EXPERT_GUIDANCE

    def test_contains_no_articles_rule(self):
        assert "no article" in NAHUATL_EXPERT_GUIDANCE.lower()

    def test_guidance_is_concise(self):
        """Expert guidance should be concise — let the model use its own knowledge."""
        assert len(NAHUATL_EXPERT_GUIDANCE) < 2000, (
            f"Guidance is too long ({len(NAHUATL_EXPERT_GUIDANCE)} chars). "
            "Keep it minimal — the model already knows Nahuatl."
        )

    def test_preserve_names_rule(self):
        assert "proper noun" in NAHUATL_EXPERT_GUIDANCE.lower() or "proper name" in NAHUATL_EXPERT_GUIDANCE.lower()

    def test_no_spanish_mixing_rule(self):
        """Must warn against mixing Spanish into Nahuatl output."""
        assert "spanish" in NAHUATL_EXPERT_GUIDANCE.lower()

    def test_colonial_context_for_transcription(self):
        assert "Long-s" in COLONIAL_NAHUATL_TRANSCRIPTION_CONTEXT
        assert "Do NOT expand" in COLONIAL_NAHUATL_TRANSCRIPTION_CONTEXT


# ===================================================================
# Few-shot examples tests
# ===================================================================

class TestFewshot:
    def test_examples_exist(self):
        assert len(FEWSHOT_TRANSLATION_EXAMPLES) >= 10

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

    def test_fewshot_includes_greeting_example(self):
        """Must have a greeting example showing Pialli, not cualli."""
        greeting_examples = [
            ex for ex in FEWSHOT_TRANSLATION_EXAMPLES
            if "hello" in ex[2].lower() or "pialli" in ex[3].lower()
        ]
        assert len(greeting_examples) >= 1, "Need at least one greeting example"

    def test_fewshot_includes_possessive_example(self):
        """Must have an example showing notoca as one word."""
        possessive_examples = [
            ex for ex in FEWSHOT_TRANSLATION_EXAMPLES
            if "notoca" in ex[3].lower() or "nocal" in ex[3].lower()
        ]
        assert len(possessive_examples) >= 1, "Need possessive example"


# ===================================================================
# Translation prompt tests (AI-first architecture)
# ===================================================================

class TestTranslationPrompts:
    def test_system_prompt_positions_ai_as_expert(self):
        """The system prompt should tell the AI it's the expert, not just a template filler."""
        prompt = translation_system_prompt("en", "nah", "")
        assert "expert" in prompt.lower()
        assert "Nahuatl" in prompt

    def test_system_prompt_contains_direction(self):
        prompt = translation_system_prompt("en", "nah", "")
        assert "English" in prompt
        assert "Nahuatl" in prompt
        assert "DIRECTION" in prompt

    def test_system_prompt_with_variety(self):
        prompt = translation_system_prompt("en", "nah", "Huasteca")
        assert "Huasteca" in prompt

    def test_system_prompt_unknown_variety_ignored(self):
        prompt = translation_system_prompt("en", "nah", "Unknown")
        # "Unknown" should not appear in variety instruction
        assert "specified the dialect: Unknown" not in prompt

    def test_system_prompt_contains_examples(self):
        prompt = translation_system_prompt("en", "nah", "")
        assert "EXAMPLES" in prompt

    def test_system_prompt_conservative_approach(self):
        """Must instruct AI to be conservative and avoid inventing words."""
        prompt = translation_system_prompt("en", "nah", "")
        assert "invent" in prompt.lower() or "conservative" in prompt.lower()

    def test_system_prompt_output_only(self):
        prompt = translation_system_prompt("en", "nah", "")
        assert "ONLY" in prompt and "translation" in prompt.lower()

    def test_user_prompt_contains_text(self):
        prompt = translation_user_prompt("Hello world", "en", "nah", "")
        assert "Hello world" in prompt
        assert "TRANSLATE THIS" in prompt

    def test_user_prompt_with_supplementary_vocab(self):
        prompt = translation_user_prompt(
            "Hello", "en", "nah", "",
            reference_vocab='- "hello" → "Pialli"',
        )
        assert "SUPPLEMENTARY VOCABULARY" in prompt
        assert "Pialli" in prompt
        # Must say it's optional/supplementary
        assert "optional" in prompt.lower() or "supplementary" in prompt.lower()

    def test_user_prompt_with_supplementary_sentences(self):
        prompt = translation_user_prompt(
            "Hello", "en", "nah", "",
            reference_sentences="- English: Hello\n  Nahuatl: Niltze",
        )
        assert "SUPPLEMENTARY" in prompt

    def test_user_prompt_no_context(self):
        prompt = translation_user_prompt("Hello", "en", "nah", "")
        assert "SUPPLEMENTARY" not in prompt

    def test_variants_system_prompt_extends_base(self):
        base = translation_system_prompt("en", "nah", "")
        variants = translation_variants_system_prompt("en", "nah", "", k=3)
        assert len(variants) > len(base)
        assert "3" in variants
        assert "variant" in variants.lower() or "distinct" in variants.lower()

    def test_variants_system_prompt_respects_k(self):
        v2 = translation_variants_system_prompt("en", "nah", "", k=2)
        v5 = translation_variants_system_prompt("en", "nah", "", k=5)
        assert "2" in v2
        assert "5" in v5

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
        assert "diplomatic" in prompt.lower()
        assert "exactly" in prompt.lower()

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
        assert "HARD RULES" in prompt
        assert "exactly" in prompt.lower()
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
        assert "SCHEMA" in prompt
        assert '"people"' in prompt

    def test_user_prompt_without_schema(self):
        prompt = extraction_user_prompt(
            text="Test text",
            instruction="Extract entities.",
        )
        assert "SCHEMA HINT" not in prompt
