"""Translation accuracy tests — AI-first architecture.

Architecture: OpenAI is the PRIMARY translator. These tests verify:
- Offline metrics (BLEU/chrF) work correctly
- Golden translation pairs are well-structured
- Live API tests confirm the AI produces correct translations
- Conversational accuracy (greetings, possessives, common phrases)

The AI should produce good translations using its own knowledge, with
the corpus providing only optional supplementary vocabulary.
"""

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import List

import pytest

from tests.conftest import skip_no_openai, skip_no_corpus


# ===================================================================
# Lightweight BLEU/chrF implementations (no nltk dependency)
# ===================================================================

def _ngrams(tokens: List[str], n: int) -> List[tuple]:
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def compute_bleu(reference: str, hypothesis: str, max_n: int = 4) -> float:
    """Compute simplified BLEU score (sentence-level, smoothed)."""
    import math

    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    if len(hyp_tokens) == 0:
        return 0.0

    effective_n = min(max_n, len(ref_tokens), len(hyp_tokens))
    if effective_n == 0:
        return 0.0

    precisions = []
    for n in range(1, effective_n + 1):
        ref_ngrams = Counter(_ngrams(ref_tokens, n))
        hyp_ngrams = Counter(_ngrams(hyp_tokens, n))

        if not hyp_ngrams:
            precisions.append(0.0)
            continue

        clipped = sum(min(count, ref_ngrams[ng]) for ng, count in hyp_ngrams.items())
        total = sum(hyp_ngrams.values())
        precisions.append((clipped + 1) / (total + 1))

    if not precisions or all(p == 0 for p in precisions):
        return 0.0

    log_avg = sum(math.log(max(p, 1e-10)) for p in precisions) / len(precisions)

    bp = 1.0
    if len(hyp_tokens) < len(ref_tokens):
        bp = math.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1))

    return bp * math.exp(log_avg)


def compute_chrf(reference: str, hypothesis: str, n: int = 6, beta: float = 2.0) -> float:
    """Compute chrF score (character n-gram F-score)."""
    def _char_ngrams(text: str, order: int) -> Counter:
        text = text.strip()
        grams = Counter()
        for i in range(len(text) - order + 1):
            grams[text[i:i + order]] += 1
        return grams

    ref = reference.lower()
    hyp = hypothesis.lower()

    if not hyp or not ref:
        return 0.0

    precisions = []
    recalls = []

    for order in range(1, n + 1):
        ref_ngrams = _char_ngrams(ref, order)
        hyp_ngrams = _char_ngrams(hyp, order)

        common = sum((ref_ngrams & hyp_ngrams).values())
        total_hyp = sum(hyp_ngrams.values())
        total_ref = sum(ref_ngrams.values())

        p = common / total_hyp if total_hyp > 0 else 0.0
        r = common / total_ref if total_ref > 0 else 0.0
        precisions.append(p)
        recalls.append(r)

    avg_p = sum(precisions) / len(precisions) if precisions else 0.0
    avg_r = sum(recalls) / len(recalls) if recalls else 0.0

    if avg_p + avg_r == 0:
        return 0.0

    beta_sq = beta ** 2
    return (1 + beta_sq) * avg_p * avg_r / (beta_sq * avg_p + avg_r)


def normalize_nahuatl(text: str) -> str:
    """Normalize Nahuatl text for comparison."""
    t = text.lower().strip()
    t = re.sub(r'[¿¡?.!,;:\'"()]', '', t)
    t = t.replace("hua", "wa").replace("hu", "w")
    t = re.sub(r'qu(?=[ei])', 'k', t)
    return t.strip()


# ===================================================================
# BLEU/chrF self-tests
# ===================================================================

class TestMetrics:
    def test_bleu_identical(self):
        score = compute_bleu("the cat sat on the mat", "the cat sat on the mat")
        assert score > 0.9

    def test_bleu_empty_hypothesis(self):
        score = compute_bleu("the cat sat", "")
        assert score == 0.0

    def test_bleu_partial_match(self):
        score = compute_bleu("the cat sat on the mat", "the cat sat")
        assert 0.0 < score < 1.0

    def test_bleu_no_match(self):
        score = compute_bleu("hello world", "foo bar baz")
        assert score < 0.3

    def test_chrf_identical(self):
        score = compute_chrf("hello world", "hello world")
        assert score > 0.99

    def test_chrf_empty(self):
        score = compute_chrf("hello", "")
        assert score == 0.0

    def test_chrf_similar(self):
        score = compute_chrf("chocolatl", "chocolate")
        assert score > 0.5

    def test_chrf_different(self):
        score = compute_chrf("hello", "xyzzy")
        assert score < 0.3

    def test_normalize_nahuatl(self):
        assert normalize_nahuatl("Cuauhtemoc") == normalize_nahuatl("cuauhtemoc")
        assert normalize_nahuatl("¿Quen tica?") == normalize_nahuatl("Quen tica")


# ===================================================================
# Golden pair structure tests (offline)
# ===================================================================

class TestGoldenPairsStructure:
    def test_golden_pairs_exist(self, golden_pairs):
        assert len(golden_pairs) >= 50

    def test_golden_pairs_have_required_fields(self, golden_pairs):
        required = {"src", "tgt", "src_lang", "tgt_lang", "category"}
        for pair in golden_pairs:
            missing = required - set(pair.keys())
            assert not missing, f"Pair missing fields {missing}: {pair.get('src', '?')}"

    def test_golden_pairs_valid_languages(self, golden_pairs):
        valid_langs = {"en", "es", "nah"}
        for pair in golden_pairs:
            assert pair["src_lang"] in valid_langs
            assert pair["tgt_lang"] in valid_langs

    def test_golden_pairs_have_content(self, golden_pairs):
        for pair in golden_pairs:
            assert pair["src"].strip()
            assert pair["tgt"].strip()

    def test_golden_pairs_have_categories(self, golden_pairs):
        categories = {p["category"] for p in golden_pairs}
        assert len(categories) >= 3

    def test_golden_pairs_en_to_nah_count(self, golden_en_to_nah):
        assert len(golden_en_to_nah) >= 30

    def test_golden_pairs_nah_to_en_count(self, golden_nah_to_en):
        assert len(golden_nah_to_en) >= 5

    def test_golden_pairs_include_vocabulary(self, golden_pairs):
        vocab_pairs = [p for p in golden_pairs if p["category"] == "vocabulary"]
        assert len(vocab_pairs) >= 15

    def test_golden_pairs_include_conversational(self, golden_pairs):
        conv_pairs = [p for p in golden_pairs if p["category"] == "conversational"]
        assert len(conv_pairs) >= 5, "Need conversational pairs for AI accuracy testing"


# ===================================================================
# Cross-checking golden pairs
# ===================================================================

class TestGoldenAgainstCorpus:
    def test_vocabulary_pairs_have_nahuatl_suffixes(self, golden_en_to_nah):
        vocab = [p for p in golden_en_to_nah if p["category"] == "vocabulary"]
        suffix_count = 0
        for p in vocab:
            nah = p["tgt"].lower()
            if any(nah.endswith(s) for s in ["tl", "tli", "li", "in", "tl.", "tli."]):
                suffix_count += 1
        ratio = suffix_count / max(len(vocab), 1)
        assert ratio >= 0.5

    def test_morphology_pairs_have_prefixes(self, golden_en_to_nah):
        morph = [p for p in golden_en_to_nah if p["category"] == "morphology"]
        prefix_count = 0
        for p in morph:
            nah = p["tgt"].lower()
            if any(nah.startswith(pfx) for pfx in ["no", "mo", "ni", "ti", "i"]):
                prefix_count += 1
        assert prefix_count >= 2


# ===================================================================
# Live API accuracy tests (AI as primary translator)
# ===================================================================

class TestLiveTranslationAccuracy:
    """Test that the AI produces accurate translations using its own knowledge."""

    @skip_no_openai
    def test_single_word_accuracy(self, golden_en_to_nah):
        from webapp.services import openai_translate

        vocab_pairs = [p for p in golden_en_to_nah if p["category"] == "vocabulary"][:10]
        correct = 0
        total = len(vocab_pairs)

        for pair in vocab_pairs:
            result = openai_translate(pair["src"], "en", "nah", "Unknown")
            result_norm = normalize_nahuatl(result)
            expected_norm = normalize_nahuatl(pair["tgt"])

            chrf = compute_chrf(expected_norm, result_norm)
            if result_norm == expected_norm or chrf > 0.7:
                correct += 1

        accuracy = correct / max(total, 1)
        assert accuracy >= 0.5, f"Single-word accuracy too low: {accuracy:.0%} ({correct}/{total})"

    @skip_no_openai
    def test_phrase_accuracy(self, golden_en_to_nah):
        from webapp.services import openai_translate

        phrase_pairs = [p for p in golden_en_to_nah if p["category"] == "phrase"][:5]
        scores = []

        for pair in phrase_pairs:
            result = openai_translate(pair["src"], "en", "nah", "Unknown")
            chrf = compute_chrf(pair["tgt"].lower(), result.lower())
            scores.append(chrf)

        avg_chrf = sum(scores) / max(len(scores), 1)
        assert avg_chrf >= 0.3, f"Phrase chrF too low: {avg_chrf:.3f}"

    @skip_no_openai
    def test_greeting_accuracy(self, golden_en_to_nah):
        from webapp.services import openai_translate

        greetings = [p for p in golden_en_to_nah if p["category"] == "greeting"][:3]
        for pair in greetings:
            result = openai_translate(pair["src"], "en", "nah", "Unknown")
            chrf = compute_chrf(pair["tgt"].lower(), result.lower())
            assert chrf >= 0.2, (
                f"Greeting '{pair['src']}' scored too low: chrF={chrf:.3f}, "
                f"expected='{pair['tgt']}', got='{result}'"
            )

    @skip_no_openai
    def test_nah_to_en_accuracy(self, golden_nah_to_en):
        from webapp.services import openai_translate

        pairs = golden_nah_to_en[:5]
        scores = []

        for pair in pairs:
            result = openai_translate(pair["src"], "nah", "en", "Unknown")
            bleu = compute_bleu(pair["tgt"].lower(), result.lower())
            chrf = compute_chrf(pair["tgt"].lower(), result.lower())
            scores.append(max(bleu, chrf))

        avg_score = sum(scores) / max(len(scores), 1)
        assert avg_score >= 0.2, f"Nah->En accuracy too low: {avg_score:.3f}"

    @skip_no_openai
    def test_variant_generation_produces_multiple(self):
        from webapp.services import openai_translate_variants

        variants = openai_translate_variants("Good morning", "en", "nah", "Unknown", k=3)
        assert len(variants) >= 2
        assert len(set(variants)) >= 2

    @skip_no_openai
    def test_translation_no_hallucination(self):
        from webapp.services import openai_translate

        result = openai_translate("The sun rises in the east", "en", "nah", "Unknown")
        commentary_markers = ["note:", "translation:", "meaning:", "literally:"]
        for marker in commentary_markers:
            assert marker not in result.lower(), f"Output contains commentary: '{marker}'"


# ===================================================================
# Conversational accuracy tests (AI should handle these natively)
# ===================================================================

class TestConversationalAccuracy:
    """Tests for everyday conversational Nahuatl — the AI should know these."""

    @skip_no_openai
    def test_hello_not_cualli(self):
        """'Hello' must be Pialli/Niltze, NOT 'cualli' (which means good)."""
        from webapp.services import openai_translate

        result = openai_translate("Hello", "en", "nah", "Unknown").lower()
        has_greeting = "pialli" in result or "niltze" in result
        assert has_greeting, (
            f"'Hello' should be 'Pialli' or 'Niltze', not '{result}'. "
            f"'Cualli' means 'good', NOT 'hello'."
        )

    @skip_no_openai
    def test_hello_my_name_is(self):
        """'Hello, my name is X' should use Pialli + notoca."""
        from webapp.services import openai_translate

        result = openai_translate("Hello, my name is Carlos", "en", "nah", "Unknown")
        result_lower = result.lower()

        has_greeting = "pialli" in result_lower or "niltze" in result_lower
        assert has_greeting, f"Missing proper greeting in: '{result}'"

        has_notoca = "notoca" in result_lower
        assert has_notoca, f"'my name' should be 'notoca' (one word), got: '{result}'"

        assert "carlos" in result_lower, f"Name 'Carlos' should be preserved, got: '{result}'"

    @skip_no_openai
    def test_hi_how_are_you(self):
        from webapp.services import openai_translate

        result = openai_translate("Hi, how are you?", "en", "nah", "Unknown")
        result_lower = result.lower()

        has_greeting = "pialli" in result_lower or "niltze" in result_lower
        has_question = "quen" in result_lower
        assert has_greeting, f"Missing greeting in: '{result}'"
        assert has_question, f"Missing 'quen' (how) in: '{result}'"

    @skip_no_openai
    def test_what_is_your_name(self):
        from webapp.services import openai_translate

        result = openai_translate("What is your name?", "en", "nah", "Unknown")
        result_lower = result.lower()

        has_motoca = "motoca" in result_lower
        assert has_motoca, f"'your name' should be 'motoca', got: '{result}'"

    @skip_no_openai
    def test_yes_and_no(self):
        from webapp.services import openai_translate

        yes_result = openai_translate("Yes", "en", "nah", "Unknown").lower()
        assert "quemah" in yes_result or "quema" in yes_result, (
            f"'Yes' should be 'Quemah', got: '{yes_result}'"
        )

        no_result = openai_translate("No", "en", "nah", "Unknown").lower()
        assert "ahmo" in no_result or "amo" in no_result, (
            f"'No' should be 'Ahmo', got: '{no_result}'"
        )

    @skip_no_openai
    def test_thank_you(self):
        from webapp.services import openai_translate

        result = openai_translate("Thank you", "en", "nah", "Unknown").lower()
        assert "tlazohcamati" in result or "tlazo" in result, (
            f"'Thank you' should be 'Tlazohcamati', got: '{result}'"
        )

    @skip_no_openai
    def test_my_house_possessive(self):
        """'my house' should be 'nocal' (one word, absolutive drops)."""
        from webapp.services import openai_translate

        result = openai_translate("my house", "en", "nah", "Unknown").lower()
        assert "nocal" in result, f"'my house' should be 'nocal', got: '{result}'"

    @skip_no_openai
    def test_i_love_you(self):
        from webapp.services import openai_translate

        result = openai_translate("I love you", "en", "nah", "Unknown").lower()
        has_love = "nimitztlazohtla" in result or "nimitznahuilia" in result
        assert has_love, f"'I love you' expected 'nimitztlazohtla', got: '{result}'"

    @skip_no_openai
    def test_pialli_to_english(self):
        """'Pialli' should translate to Hello/Hi, not 'Good'."""
        from webapp.services import openai_translate

        result = openai_translate("Pialli, notoca Maria.", "nah", "en", "Unknown").lower()
        has_hello = "hello" in result or "hi" in result or "greetings" in result
        assert has_hello, f"'Pialli' should translate to 'Hello/Hi', got: '{result}'"
        assert "maria" in result, f"Name 'Maria' should be preserved, got: '{result}'"
