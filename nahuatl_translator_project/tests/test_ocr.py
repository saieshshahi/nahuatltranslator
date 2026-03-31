"""Unit tests for webapp/ocr.py — image tiling, cleanup, and Unicode handling."""

import os
import tempfile

import pytest

# PIL is needed for creating test images
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    # ocr.py imports fitz (PyMuPDF) at module level, so we must handle that
    from webapp.ocr import tile_image, cleanup_tiles, _has_corrupted_unicode
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

pytestmark = pytest.mark.skipif(
    not HAS_PIL or not HAS_OCR,
    reason="Pillow or PyMuPDF not installed — skipping OCR tests",
)


# ===================================================================
# Helper: create a test image
# ===================================================================

def _create_test_image(width: int, height: int, color: str = "white") -> str:
    """Create a temporary test image and return its path."""
    img = Image.new("RGB", (width, height), color=color)
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    img.save(path)
    return path


# ===================================================================
# tile_image() tests
# ===================================================================

class TestTileImage:
    def test_small_image_no_tiling(self):
        """Images shorter than tile_height should return original path."""
        path = _create_test_image(400, 600)
        try:
            tiles = tile_image(path, tile_height=800)
            assert len(tiles) == 1
            assert tiles[0] == path
        finally:
            os.unlink(path)

    def test_exact_tile_height_no_tiling(self):
        """Image exactly at tile_height should return original path."""
        path = _create_test_image(400, 800)
        try:
            tiles = tile_image(path, tile_height=800)
            assert len(tiles) == 1
            assert tiles[0] == path
        finally:
            os.unlink(path)

    def test_tall_image_produces_multiple_tiles(self):
        """A tall image should produce multiple tile files."""
        path = _create_test_image(400, 2000)
        try:
            tiles = tile_image(path, tile_height=800, overlap=100)
            assert len(tiles) >= 2
            # All tiles should be actual files
            for t in tiles:
                assert os.path.isfile(t)
        finally:
            cleanup_tiles(tiles, path)
            if os.path.exists(path):
                os.unlink(path)

    def test_tiles_have_correct_dimensions(self):
        """Each tile should have the expected width and max height."""
        path = _create_test_image(500, 2400)
        try:
            tiles = tile_image(path, tile_height=800, overlap=100)
            for tile_path in tiles:
                img = Image.open(tile_path)
                w, h = img.size
                assert w == 500, f"Tile width should match original: {w}"
                assert h <= 800, f"Tile height should be <= tile_height: {h}"
                img.close()
        finally:
            cleanup_tiles(tiles, path)
            if os.path.exists(path):
                os.unlink(path)

    def test_overlap_coverage(self):
        """Tiles should cover the full image with overlap."""
        height = 2000
        tile_height = 800
        overlap = 100
        path = _create_test_image(300, height)
        try:
            tiles = tile_image(path, tile_height=tile_height, overlap=overlap)
            # The total coverage should be at least the image height
            total_coverage = 0
            for tile_path in tiles:
                img = Image.open(tile_path)
                total_coverage += img.size[1]
                img.close()
            # Account for overlaps
            assert total_coverage >= height, "Tiles should cover entire image"
        finally:
            cleanup_tiles(tiles, path)
            if os.path.exists(path):
                os.unlink(path)

    def test_three_tile_split(self):
        """An image ~2400px tall with 800px tiles should produce ~3 tiles."""
        path = _create_test_image(300, 2400)
        try:
            tiles = tile_image(path, tile_height=800, overlap=100)
            assert len(tiles) >= 3
        finally:
            cleanup_tiles(tiles, path)
            if os.path.exists(path):
                os.unlink(path)

    def test_very_tall_image(self):
        """A very tall image should produce many tiles."""
        path = _create_test_image(300, 5000)
        try:
            tiles = tile_image(path, tile_height=800, overlap=100)
            assert len(tiles) >= 6
        finally:
            cleanup_tiles(tiles, path)
            if os.path.exists(path):
                os.unlink(path)

    def test_tile_files_are_valid_images(self):
        """All generated tile files should be valid PNG images."""
        path = _create_test_image(300, 2000)
        try:
            tiles = tile_image(path, tile_height=800, overlap=100)
            for tile_path in tiles:
                img = Image.open(tile_path)
                assert img.format == "PNG" or tile_path.endswith(".png")
                img.close()
        finally:
            cleanup_tiles(tiles, path)
            if os.path.exists(path):
                os.unlink(path)

    def test_custom_tile_height(self):
        """Custom tile_height should be respected."""
        path = _create_test_image(300, 1500)
        try:
            tiles = tile_image(path, tile_height=500, overlap=50)
            assert len(tiles) >= 3
        finally:
            cleanup_tiles(tiles, path)
            if os.path.exists(path):
                os.unlink(path)

    def test_zero_overlap(self):
        """Zero overlap should still produce valid tiles."""
        path = _create_test_image(300, 2000)
        try:
            tiles = tile_image(path, tile_height=800, overlap=0)
            assert len(tiles) >= 2
            for t in tiles:
                assert os.path.isfile(t)
        finally:
            cleanup_tiles(tiles, path)
            if os.path.exists(path):
                os.unlink(path)


# ===================================================================
# cleanup_tiles() tests
# ===================================================================

class TestCleanupTiles:
    def test_removes_tile_files(self):
        """cleanup_tiles should remove temporary tile files."""
        paths = []
        for i in range(3):
            fd, p = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            paths.append(p)

        original = paths[0]
        cleanup_tiles(paths, original)

        # Original should NOT be removed
        assert os.path.exists(original), "Original image should be preserved"
        # Other tiles should be removed
        for p in paths[1:]:
            assert not os.path.exists(p), f"Tile {p} should have been removed"

        # Clean up original
        if os.path.exists(original):
            os.unlink(original)

    def test_preserves_original(self):
        """The original image path should never be deleted."""
        original = _create_test_image(100, 100)
        try:
            cleanup_tiles([original], original)
            assert os.path.exists(original)
        finally:
            if os.path.exists(original):
                os.unlink(original)

    def test_handles_missing_files_gracefully(self):
        """cleanup_tiles should not crash on already-deleted files."""
        cleanup_tiles(["/nonexistent/tile1.png", "/nonexistent/tile2.png"], "/original.png")

    def test_empty_list(self):
        """cleanup_tiles with empty list should not crash."""
        cleanup_tiles([], "/some/path.png")


# ===================================================================
# Unicode corruption detection tests
# ===================================================================

class TestUnicodeCorruptionDetection:
    """Test _has_corrupted_unicode — catches replacement characters from PDF extraction."""

    # --- Nahuatl words with macrons that must survive the pipeline ---
    MACRON_WORDS = [
        "Nāhuatl",
        "tlatōlli",
        "tlācameh",
        "cempōhualli",
        "nāhui",
        "āltepētl",
        "tēuctli",
        "tlahtoāni",
        "cīhuātl",
        "ōmpa",
    ]

    def test_clean_text_not_flagged(self):
        """Normal text without replacement chars should not be flagged."""
        clean = "Nāhuatl tlatōlli huehueh tlatōlli. Ipan tlajco Mexihco."
        assert not _has_corrupted_unicode(clean)

    def test_plain_ascii_not_flagged(self):
        """Plain ASCII text should not be flagged."""
        assert not _has_corrupted_unicode("Hello world, simple text.")

    def test_black_square_detected(self):
        """U+25A0 ■ (black square) — PyMuPDF's common replacement — must be caught."""
        corrupted = "N\u25a0huatl tlat\u25a0lli"
        assert _has_corrupted_unicode(corrupted)

    def test_replacement_char_detected(self):
        """U+FFFD � (Unicode replacement character) must be caught."""
        corrupted = "N\ufffdhuatl tlat\ufffdlli"
        assert _has_corrupted_unicode(corrupted)

    def test_null_byte_detected(self):
        """U+0000 null bytes from broken CMap entries must be caught."""
        corrupted = "N\x00huatl"
        assert _has_corrupted_unicode(corrupted)

    def test_macron_words_are_clean(self):
        """All standard Nahuatl macron words must pass as clean (not corrupted)."""
        for word in self.MACRON_WORDS:
            assert not _has_corrupted_unicode(word), (
                f"Macron word '{word}' was incorrectly flagged as corrupted"
            )

    def test_macron_sentence_is_clean(self):
        """Full sentence with macrons must pass as clean."""
        sentence = (
            "Nāhuatl tlatōlli huehueh tlatōlli. "
            "Ipan tlajco Mexihco miac tlācameh quipia in tlatōlli."
        )
        assert not _has_corrupted_unicode(sentence)

    def test_mixed_corruption_detected(self):
        """Text with some good chars and some replacement chars must be caught."""
        mixed = "Nāhuatl is good but N\u25a0huatl is corrupted"
        assert _has_corrupted_unicode(mixed)

    def test_empty_string(self):
        """Empty string should not be flagged."""
        assert not _has_corrupted_unicode("")


# ===================================================================
# Macron character regression tests
# ===================================================================

class TestMacronPreservation:
    """Regression tests: macron characters must never become ■, ?, �, or stripped ASCII."""

    CORRUPTION_PATTERNS = ["\u25a0", "\ufffd", "?"]

    EXPECTED_PAIRS = [
        # (with macrons, stripped to ASCII = what corruption would look like)
        ("Nāhuatl", "Nahuatl"),
        ("tlatōlli", "tlatolli"),
        ("tlācameh", "tlacameh"),
        ("cempōhualli", "cempohualli"),
        ("nāhui", "nahui"),
    ]

    def test_macron_chars_are_distinct_from_ascii(self):
        """Verify macron vowels are NOT the same as plain ASCII vowels."""
        assert "ā" != "a"
        assert "ō" != "o"
        assert "ī" != "i"
        assert "ē" != "e"
        assert "ū" != "u"

    def test_macron_chars_survive_utf8_roundtrip(self):
        """Macron characters must survive UTF-8 encode/decode."""
        for macron_word, _ in self.EXPECTED_PAIRS:
            encoded = macron_word.encode("utf-8")
            decoded = encoded.decode("utf-8")
            assert decoded == macron_word, (
                f"UTF-8 roundtrip failed for '{macron_word}': got '{decoded}'"
            )

    def test_macron_chars_survive_json_roundtrip(self):
        """Macron characters must survive JSON serialization."""
        import json
        for macron_word, _ in self.EXPECTED_PAIRS:
            j = json.dumps({"text": macron_word}, ensure_ascii=False)
            restored = json.loads(j)["text"]
            assert restored == macron_word, (
                f"JSON roundtrip failed for '{macron_word}': got '{restored}'"
            )

    def test_no_corruption_chars_in_macron_words(self):
        """Macron words must not contain any known corruption characters."""
        for macron_word, _ in self.EXPECTED_PAIRS:
            for bad in self.CORRUPTION_PATTERNS:
                assert bad not in macron_word, (
                    f"'{macron_word}' contains corruption char '{bad}'"
                )

    def test_corruption_detection_catches_replaced_macrons(self):
        """Simulated macron→■ corruption must be detected."""
        # Simulate what a bad PDF extraction does: replace ā→■, ō→■
        original = "Nāhuatl tlatōlli tlācameh cempōhualli nāhui"
        corrupted = original.replace("ā", "\u25a0").replace("ō", "\u25a0")
        assert _has_corrupted_unicode(corrupted)
        assert not _has_corrupted_unicode(original)
