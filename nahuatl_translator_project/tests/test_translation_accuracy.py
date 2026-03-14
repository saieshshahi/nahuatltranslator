"""Translation accuracy tests using BLEU and chrF metrics against golden pairs.

These tests verify that the AI translation pipeline produces outputs that
are semantically close to known-correct translations. Tests are split into:
- Offline metrics (BLEU/chrF) that work without API keys
- Live API tests (skipped without OPENAI_API_KEY) that call OpenAI
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
    """Extract n-grams from a token list."""
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def compute_bleu(reference: str, hypothesis: str, max_n: int = 4) -> float:
    """Compute a simplified BLEU score (sentence-level, smoothed).

    Returns a float in [0, 1]. For testing we use this lightweight
    implementation to avoid an nltk dependency. Uses add-1 smoothing
    so that short sentences still produce meaningful scores.
    """
    import math

    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    if len(hyp_tokens) == 0:
        return 0.0

    # Cap n-gram order at the length of the shorter sequence
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
        # Add-1 (Laplace) smoothing to avoid zero scores on short sentences
        precisions.append((clipped + 1) / (total + 1))

    # Geometric mean of precisions
    if not precisions or all(p == 0 for p in precisions):
        return 0.0

    log_avg = sum(math.log(max(p, 1e-10)) for p in precisions) / len(precisions)

    # Brevity penalty
    bp = 1.0
    if len(hyp_tokens) < len(ref_tokens):
        bp = math.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1))

    return bp * math.exp(log_avg)


def compute_chrf(reference: str, hypothesis: str, n: int = 6, beta: float = 2.0) -> float:
    """Compute chrF score (character n-gram F-score).

    A lightweight implementation for testing accuracy.
    """
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
    """Normalize Nahuatl text for comparison.

    Handles common orthographic variations: hu/w, qu/k, etc.
    """
    t = text.lower().strip()
    # Normalize common orthographic alternations
    t = re.sub(r'[¿¡?.!,;:\'"()]', '', t)
    t = t.replace("hua", "wa").replace("hu", "w")
    t = re.sub(r'qu(?=[ei])', 'k', t)
    return t.strip()


# ===================================================================
# BLEU/chrF self-tests
# ===================================================================

class TestMetrics:
    """Sanity checks for the scoring functions."""

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
        assert score < 0.3  # Very low due to smoothing but not necessarily zero

    def test_chrf_identical(self):
        score = compute_chrf("hello world", "hello world")
        assert score > 0.99

    def test_chrf_empty(self):
        score = compute_chrf("hello", "")
        assert score == 0.0

    def test_chrf_similar(self):
        score = compute_chrf("chocolatl", "chocolate")
        assert score > 0.5  # High char overlap

    def test_chrf_different(self):
        score = compute_chrf("hello", "xyzzy")
        assert score < 0.3

    def test_normalize_nahuatl(self):
        assert normalize_nahuatl("Cuauhtemoc") == normalize_nahuatl("cuauhtemoc")
        assert normalize_nahuatl("¿Quen tica?") == normalize_nahuatl("Quen tica")


# ===================================================================
# Golden pair accuracy tests (offline — no API)
# ===================================================================

class TestGoldenPairsStructure:
    """Validate the golden translations file structure."""

    def test_golden_pairs_exist(self, golden_pairs):
        assert len(golden_pairs) >= 50, f"Expected 50+ golden pairs, got {len(golden_pairs)}"

    def test_golden_pairs_have_required_fields(self, golden_pairs):
        required = {"src", "tgt", "src_lang", "tgt_lang", "category"}
        for pair in golden_pairs:
            missing = required - set(pair.keys())
            assert not missing, f"Pair missing fields {missing}: {pair.get('src', '?')}"

    def test_golden_pairs_valid_languages(self, golden_pairs):
        valid_langs = {"en", "es", "nah"}
        for pair in golden_pairs:
            assert pair["src_lang"] in valid_langs, f"Invalid src_lang: {pair['src_lang']}"
            assert pair["tgt_lang"] in valid_langs, f"Invalid tgt_lang: {pair['tgt_lang']}"

    def test_golden_pairs_have_content(self, golden_pairs):
        for pair in golden_pairs:
            assert pair["src"].strip(), f"Empty src in pair: {pair}"
            assert pair["tgt"].strip(), f"Empty tgt in pair: {pair}"

    def test_golden_pairs_have_categories(self, golden_pairs):
        categories = {p["category"] for p in golden_pairs}
        assert len(categories) >= 3, "Expected at least 3 different categories"

    def test_golden_pairs_en_to_nah_count(self, golden_en_to_nah):
        assert len(golden_en_to_nah) >= 30, f"Expected 30+ en->nah pairs, got {len(golden_en_to_nah)}"

    def test_golden_pairs_nah_to_en_count(self, golden_nah_to_en):
        assert len(golden_nah_to_en) >= 5, f"Expected 5+ nah->en pairs, got {len(golden_nah_to_en)}"

    def test_golden_pairs_include_vocabulary(self, golden_pairs):
        vocab_pairs = [p for p in golden_pairs if p["category"] == "vocabulary"]
        assert len(vocab_pairs) >= 15, f"Expected 15+ vocabulary pairs, got {len(vocab_pairs)}"

    def test_golden_pairs_include_morphology(self, golden_pairs):
        morph_pairs = [p for p in golden_pairs if p["category"] == "morphology"]
        assert len(morph_pairs) >= 3, f"Expected 3+ morphology pairs, got {len(morph_pairs)}"


# ===================================================================
# Cross-checking golden pairs with corpus
# ===================================================================

class TestGoldenAgainstCorpus:
    """Verify golden pairs are consistent with the corpus data."""

    def test_vocabulary_pairs_have_nahuatl_suffixes(self, golden_en_to_nah):
        """Nahuatl vocabulary words typically end in -tl, -tli, -li, or -in."""
        vocab = [p for p in golden_en_to_nah if p["category"] == "vocabulary"]
        suffix_count = 0
        for p in vocab:
            nah = p["tgt"].lower()
            if any(nah.endswith(s) for s in ["tl", "tli", "li", "in", "tl.", "tli."]):
                suffix_count += 1
        ratio = suffix_count / max(len(vocab), 1)
        assert ratio >= 0.5, f"Only {ratio:.0%} of vocab pairs have expected Nahuatl suffixes"

    def test_morphology_pairs_have_prefixes(self, golden_en_to_nah):
        """Morphology test pairs should demonstrate prefix usage."""
        morph = [p for p in golden_en_to_nah if p["category"] == "morphology"]
        prefix_count = 0
        for p in morph:
            nah = p["tgt"].lower()
            if any(nah.startswith(pfx) for pfx in ["no", "mo", "ni", "ti", "i"]):
                prefix_count += 1
        assert prefix_count >= 2, "Expected morphology pairs to show prefixes"


# ===================================================================
# Live API accuracy tests (require OPENAI_API_KEY)
# ===================================================================

class TestLiveTranslationAccuracy:
    """Test translation accuracy against golden pairs using the live API."""

    @skip_no_openai
    def test_single_word_accuracy(self, golden_en_to_nah):
        """Test accuracy on single-word translations (vocabulary)."""
        from webapp.services import openai_translate

        vocab_pairs = [p for p in golden_en_to_nah if p["category"] == "vocabulary"][:10]
        correct = 0
        total = len(vocab_pairs)

        for pair in vocab_pairs:
            result = openai_translate(pair["src"], "en", "nah", "Unknown")
            result_norm = normalize_nahuatl(result)
            expected_norm = normalize_nahuatl(pair["tgt"])

            # Check for exact match or high chrF similarity
            chrf = compute_chrf(expected_norm, result_norm)
            if result_norm == expected_norm or chrf > 0.7:
                correct += 1

        accuracy = correct / max(total, 1)
        assert accuracy >= 0.5, (
            f"Single-word accuracy too low: {accuracy:.0%} ({correct}/{total})"
        )

    @skip_no_openai
    def test_phrase_accuracy(self, golden_en_to_nah):
        """Test accuracy on phrase translations."""
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
        """Test accuracy on greeting translations."""
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
        """Test Nahuatl-to-English translation accuracy."""
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
        """Test that variant generation produces diverse outputs."""
        from webapp.services import openai_translate_variants

        variants = openai_translate_variants("Good morning", "en", "nah", "Unknown", k=3)
        assert len(variants) >= 2, f"Expected 2+ variants, got {len(variants)}"
        assert len(set(variants)) >= 2, "Variants should be diverse"

    @skip_no_openai
    def test_translation_no_hallucination(self):
        """Translated output should not contain untranslated English/commentary."""
        from webapp.services import openai_translate

        result = openai_translate("The sun rises in the east", "en", "nah", "Unknown")
        # Result should not contain common English commentary markers
        commentary_markers = ["note:", "translation:", "meaning:", "literally:"]
        for marker in commentary_markers:
            assert marker not in result.lower(), f"Output contains commentary: '{marker}'"
