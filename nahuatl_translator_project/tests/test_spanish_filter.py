"""Tests for the Spanish contamination detection and filtering module."""

import pytest
from webapp.spanish_filter import (
    is_spanish,
    count_spanish,
    spanish_ratio,
    detect_spanish_in_output,
    strip_spanish,
)


class TestIsSpanish:
    def test_common_spanish_words(self):
        assert is_spanish("pero") is True
        assert is_spanish("porque") is True
        assert is_spanish("universidad") is True
        assert is_spanish("iglesia") is True
        assert is_spanish("dios") is True

    def test_nahuatl_words_not_flagged(self):
        assert is_spanish("teopixqui") is False
        assert is_spanish("altepetl") is False
        assert is_spanish("teotl") is False
        assert is_spanish("nicmati") is False

    def test_nahuatl_whitelist(self):
        """Words in the Nahuatl whitelist should not be flagged as Spanish."""
        assert is_spanish("amo") is False  # Nahuatl negation
        assert is_spanish("cualli") is False
        assert is_spanish("tonalli") is False

    def test_case_insensitive(self):
        assert is_spanish("Pero") is True
        assert is_spanish("DIOS") is True

    def test_short_words_ignored(self):
        """Words under 3 chars should not be flagged (too ambiguous)."""
        assert is_spanish("de") is False
        assert is_spanish("en") is False


class TestCountSpanish:
    def test_clean_nahuatl(self):
        assert count_spanish("Nicmati teopixqui ihuan teotl in altepetl") == 0

    def test_contaminated_text(self):
        count = count_spanish("Pialli, pero titechpiaj cualli tonalli")
        assert count >= 1  # "pero" should be detected

    def test_heavily_contaminated(self):
        count = count_spanish("Dios iglesia sacerdote pero para nación")
        assert count >= 4

    def test_empty_text(self):
        assert count_spanish("") == 0


class TestSpanishRatio:
    def test_pure_nahuatl(self):
        ratio = spanish_ratio("Nicmati teopixqui ihuan teotl in altepetl")
        assert ratio == 0.0

    def test_mixed_text(self):
        ratio = spanish_ratio("Pialli pero cualli tonalli iglesia")
        assert 0.0 < ratio < 1.0

    def test_empty(self):
        assert spanish_ratio("") == 0.0


class TestDetectSpanishInOutput:
    def test_clean_output(self):
        found = detect_spanish_in_output("Nicmati teopixqui ihuan teotl in altepetl")
        assert len(found) == 0

    def test_finds_spanish(self):
        found = detect_spanish_in_output("Pialli, pero nican universidad")
        assert "pero" in found
        assert "universidad" in found

    def test_no_duplicates(self):
        found = detect_spanish_in_output("pero pero pero pero")
        assert found.count("pero") == 1

    def test_whitelist_excluded(self):
        found = detect_spanish_in_output("amo cualli tonalli")
        assert len(found) == 0


class TestStripSpanish:
    def test_removes_spanish_words(self):
        result = strip_spanish("Pialli, pero nican")
        assert "pero" not in result
        assert "Pialli" in result
        assert "nican" in result

    def test_preserves_pure_nahuatl(self):
        text = "Nicmati teopixqui ihuan teotl"
        assert strip_spanish(text) == text

    def test_handles_empty(self):
        assert strip_spanish("") == ""

    def test_cleans_whitespace(self):
        result = strip_spanish("Pialli  pero  nican")
        assert "  " not in result  # double spaces should be cleaned
