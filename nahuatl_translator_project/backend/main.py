from __future__ import annotations

import os
import tempfile
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from webapp import services
from webapp.ocr import extract_text, render_pdf_page_to_png
from webapp.evaluation import (
    evaluate_golden_pairs,
    compute_corpus_stats,
    load_eval_history,
    save_eval_result,
)


APP_NAME = "Nahuatl Translator & Manuscript Transcriber API"

app = FastAPI(title=APP_NAME)

# CORS: allow GitHub Pages + local dev.
allowed = os.getenv("CORS_ORIGINS", "*")
origins = [o.strip() for o in allowed.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins else ["*"],
    allow_credentials=False,
    allow_methods=["*"] ,
    allow_headers=["*"] ,
)


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1)
    src: str = Field("en")
    tgt: str = Field("nah")
    variety: str = Field("Unknown")
    variants: int = Field(1, ge=1, le=5)


class TranslateResponse(BaseModel):
    engine: str
    variants: List[str]


@app.get("/health")
def health():
    return {"ok": True, "app": APP_NAME, "openai": services.openai_available()}


# Allowed translation directions — every pair must go through Nahuatl
_ALLOWED_PAIRS = {("en", "nah"), ("nah", "en"), ("es", "nah"), ("nah", "es")}


@app.post("/translate", response_model=TranslateResponse)
def translate(req: TranslateRequest):
    """OpenAI-powered translation (with variants)."""
    if (req.src, req.tgt) not in _ALLOWED_PAIRS:
        return TranslateResponse(
            engine="openai",
            variants=[f"ERROR: Unsupported direction {req.src}→{req.tgt}. "
                      "All translations must go through Nahuatl."],
        )

    if not services.openai_available():
        return TranslateResponse(engine="openai", variants=["ERROR: OPENAI_API_KEY not set"])

    try:
        outs = services.openai_translate_variants(
            text=req.text,
            src=req.src,
            tgt=req.tgt,
            variety=req.variety,
            k=req.variants,

            model=os.getenv("OPENAI_TRANSLATE_MODEL"),
        )
        return TranslateResponse(engine="openai", variants=outs)
    except Exception as e:
        return TranslateResponse(engine="openai", variants=[f"ERROR: {e}"])


class ExtractRequest(BaseModel):
    text: str
    instruction: str
    schema_hint: str = ""


@app.post("/extract")
def extract(req: ExtractRequest):
    if not services.openai_available():
        return {"engine": "openai", "output": "ERROR: OPENAI_API_KEY not set"}
    try:
        out = services.openai_extract(
            text=req.text,
            instruction=req.instruction,
            schema_hint=req.schema_hint,
            model=os.getenv("OPENAI_EXTRACT_MODEL"),
        )
        return {"engine": "openai", "output": out}
    except Exception as e:
        return {"engine": "openai", "output": f"ERROR: {e}"}


@app.post("/doc_text")
def doc_text(file: UploadFile = File(...), lang_hint: str = Form("es")):
    """Extract text from a PDF/image (direct text when available, else OCR)."""
    suffix = os.path.splitext(file.filename or "upload")[1].lower() or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    try:
        res = extract_text(tmp_path, lang_hint=lang_hint)
        return {
            "text": res.text,
            "pages": res.pages,
            "used_ocr": res.used_ocr,
        }
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@app.post("/transcribe")
def transcribe(
    file: UploadFile = File(...),
    page_num: int = Form(1),
    language_hint: str = Form("es"),
    alphabet_hint: str = Form(""),
    temperature: float = Form(0.0),
    handwriting_profile: bool = Form(False),
):
    """Transcribe a manuscript page.

    - If PDF: renders selected page to PNG.
    - If image: uses it directly.

    Uses OpenAI vision model (requires OPENAI_API_KEY).
    """
    if not services.openai_available():
        return {"engine": "openai_vision", "transcription": "ERROR: OPENAI_API_KEY not set"}

    suffix = os.path.splitext(file.filename or "upload")[1].lower() or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.file.read())
        src_path = tmp.name

    img_path: Optional[str] = None
    try:
        if suffix == ".pdf":
            img_path = render_pdf_page_to_png(src_path, page_num=page_num)
        else:
            img_path = src_path

        tx = services.openai_transcribe_image(
            image_path=img_path,
            language_hint=language_hint,
            alphabet_hint=alphabet_hint,
            model=os.getenv("OPENAI_TRANSCRIBE_MODEL"),
            temperature=float(temperature),
        )

        prof = None
        if handwriting_profile:
            prof = services.openai_handwriting_profile(
                image_path=img_path,
                model=os.getenv("OPENAI_TRANSCRIBE_MODEL"),
            )

        return {
            "engine": "openai_vision",
            "transcription": tx,
            "handwriting_profile": prof,
        }
    finally:
        for p in {src_path, img_path}:
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
                except Exception:
                    pass


# ===================================================================
# Evaluation endpoints
# ===================================================================

import json as _json
from dataclasses import asdict as _asdict
from pathlib import Path as _Path

_GOLDEN_PATH = _Path(__file__).resolve().parent.parent / "tests" / "golden_translations.json"

# Simple in-memory cache for evaluation results
_eval_cache: dict = {"result": None, "timestamp": 0.0}
_CACHE_TTL = 3600  # 1 hour


class EvalRequest(BaseModel):
    directions: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    max_pairs: int = Field(79, ge=1, le=200)


@app.post("/evaluate")
def evaluate(req: EvalRequest):
    """Run golden-pair evaluation and return full metrics."""
    import time as _time

    if not services.openai_available():
        return {"error": "OPENAI_API_KEY not set"}

    # Check cache
    now = _time.time()
    cache_key = f"{req.directions}:{req.categories}:{req.max_pairs}"
    if (
        _eval_cache["result"] is not None
        and now - _eval_cache["timestamp"] < _CACHE_TTL
        and _eval_cache.get("key") == cache_key
    ):
        return {"cached": True, **_eval_cache["result"]}

    # Load golden pairs
    try:
        with open(_GOLDEN_PATH, "r", encoding="utf-8") as f:
            golden_data = _json.load(f)
        golden_pairs = golden_data["pairs"]
    except Exception as e:
        return {"error": f"Failed to load golden translations: {e}"}

    model = os.getenv("OPENAI_TRANSLATE_MODEL", "gpt-5")

    result = evaluate_golden_pairs(
        translate_fn=services.openai_translate,
        golden_pairs=golden_pairs,
        directions=req.directions or None,
        categories=req.categories or None,
        max_pairs=req.max_pairs,
        model=model,
    )

    result_dict = result.to_dict()

    # Save to disk
    try:
        save_eval_result(result)
    except Exception:
        pass

    # Update cache
    _eval_cache["result"] = result_dict
    _eval_cache["timestamp"] = now
    _eval_cache["key"] = cache_key

    return {"cached": False, **result_dict}


@app.get("/evaluate/history")
def evaluate_history():
    """Return summaries of all saved evaluation runs."""
    return load_eval_history()


@app.get("/corpus/stats")
def corpus_stats():
    """Return corpus statistics (no API calls needed)."""
    stats = compute_corpus_stats()
    return _asdict(stats)
