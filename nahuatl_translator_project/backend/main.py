from __future__ import annotations

import os
import tempfile
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from webapp import services
from webapp.ocr import extract_text, render_pdf_page_to_png


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
    temperature: float = Field(0.6, ge=0.0, le=2.0)


class TranslateResponse(BaseModel):
    engine: str
    variants: List[str]


@app.get("/health")
def health():
    return {"ok": True, "app": APP_NAME, "openai": services.openai_available()}


@app.post("/translate", response_model=TranslateResponse)
def translate(req: TranslateRequest):
    """OpenAI-powered translation (with variants)."""
    if not services.openai_available():
        return TranslateResponse(engine="openai", variants=["ERROR: OPENAI_API_KEY not set"])

    try:
        outs = services.openai_translate_variants(
            text=req.text,
            src=req.src,
            tgt=req.tgt,
            variety=req.variety,
            k=req.variants,
            temperature=req.temperature,
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
