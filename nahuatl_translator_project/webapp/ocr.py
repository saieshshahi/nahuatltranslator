from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

import fitz  # PyMuPDF
import pytesseract
from PIL import Image


@dataclass
class OCRResult:
    text: str
    pages: int
    used_ocr: bool


def _tesseract_langs(lang_hint: str) -> str:
    # We ship English + Spanish packs in the Dockerfile.
    # Nahuatl uses Latin script; OCR works OK with eng/spa models.
    if (lang_hint or "").lower().startswith("es"):
        return "spa+eng"
    return "eng+spa"


def extract_text(path: str, lang_hint: str = "eng") -> OCRResult:
    """Extract text from PDF or image.

    Strategy:
      1) If PDF has selectable text, use it.
      2) Otherwise render pages to images and OCR.
      3) If image, OCR.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return _extract_pdf(path, lang_hint)
    return _ocr_image(path, lang_hint)


def render_pdf_page_to_png(path: str, page_num: int = 1, dpi: int = 220) -> str:
    """Render a 1-indexed PDF page to a temporary PNG file and return its path."""
    import tempfile

    doc = fitz.open(path)
    if page_num < 1 or page_num > len(doc):
        raise ValueError(f"page_num out of range (1..{len(doc)}): {page_num}")
    page = doc[page_num - 1]
    pix = page.get_pixmap(dpi=dpi)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    fd, out_path = tempfile.mkstemp(prefix="page_", suffix=".png")
    os.close(fd)
    img.save(out_path)
    return out_path


def _extract_pdf(path: str, lang_hint: str) -> OCRResult:
    doc = fitz.open(path)
    pages_text: List[str] = []

    # 1) Direct text
    for page in doc:
        t = (page.get_text("text") or "").strip()
        pages_text.append(t)

    joined = "\n\n".join([t for t in pages_text if t])
    if len(joined) >= 80:
        return OCRResult(text=joined, pages=len(doc), used_ocr=False)

    # 2) Render + OCR
    ocr_pages: List[str] = []
    langs = _tesseract_langs(lang_hint)
    for page in doc:
        pix = page.get_pixmap(dpi=220)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        ocr_pages.append(pytesseract.image_to_string(img, lang=langs))
    return OCRResult(text="\n\n".join(ocr_pages).strip(), pages=len(doc), used_ocr=True)


def _ocr_image(path: str, lang_hint: str) -> OCRResult:
    img = Image.open(path)
    langs = _tesseract_langs(lang_hint)
    text = pytesseract.image_to_string(img, lang=langs)
    return OCRResult(text=(text or "").strip(), pages=1, used_ocr=True)


# ---------------------------------------------------------------------------
# Image tiling for large manuscript pages (Phase 3)
# ---------------------------------------------------------------------------

def tile_image(
    image_path: str,
    tile_height: int = 800,
    overlap: int = 100,
) -> List[str]:
    """Split a tall image into overlapping horizontal strips (tiles).

    For images shorter than tile_height, returns a single-element list
    with the original path (no splitting needed).

    Each tile overlaps with the next by `overlap` pixels to ensure no
    text is lost at boundaries.

    Returns a list of temporary file paths for the tile images.
    The caller is responsible for cleaning up these files.
    """
    import tempfile

    img = Image.open(image_path)
    width, height = img.size

    # No tiling needed for small images
    if height <= tile_height:
        return [image_path]

    tiles: List[str] = []
    y = 0
    while y < height:
        bottom = min(y + tile_height, height)
        tile = img.crop((0, y, width, bottom))

        fd, tile_path = tempfile.mkstemp(prefix=f"tile_{len(tiles)}_", suffix=".png")
        os.close(fd)
        tile.save(tile_path)
        tiles.append(tile_path)

        # Advance by (tile_height - overlap) so tiles overlap
        y += tile_height - overlap

        # If the remaining strip would be very small, just stop
        if height - y < overlap and y < height:
            break

    return tiles


def cleanup_tiles(tile_paths: List[str], original_path: str) -> None:
    """Remove temporary tile files (but not the original image)."""
    for p in tile_paths:
        if p != original_path and os.path.exists(p):
            try:
                os.unlink(p)
            except Exception:
                pass
