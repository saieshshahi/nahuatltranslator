"""Service layer used by the web UI.

This module intentionally lazy-loads heavy ML dependencies (torch/transformers)
so lightweight utilities can be imported without requiring those packages.
"""

import os
import json
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any, List


LANG_LABELS = {
    "en": "English",
    "es": "Spanish",
    "nah": "Nahuatl",
}


def _label(code: str) -> str:
    return LANG_LABELS.get(code, code)


def build_prompt(src_code: str, tgt_code: str, text: str, variety: str) -> str:
    # Keep the original prompt style used in the training script.
    return f"translate {_label(src_code)} to {_label(tgt_code)} [{variety}]: {text.strip()}"


@dataclass
class DecodeConfig:
    max_new_tokens: int = 128
    num_beams: int = 4
    repetition_penalty: float = 1.15
    no_repeat_ngram_size: int = 3
    length_penalty: float = 1.0


class LocalSeq2SeqTranslator:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()

    @property
    def device(self):
        return self.model.device

    def translate(self, text: str, src: str, tgt: str, variety: str, cfg: DecodeConfig) -> str:
        prompt = build_prompt(src, tgt, text, variety)
        enc = self.tok(prompt, return_tensors="pt", truncation=True).to(self.device)

        gen_kwargs = dict(
            max_new_tokens=int(cfg.max_new_tokens),
            num_beams=int(cfg.num_beams),
            repetition_penalty=float(cfg.repetition_penalty),
            no_repeat_ngram_size=int(cfg.no_repeat_ngram_size),
            length_penalty=float(cfg.length_penalty),
            early_stopping=True,
        )

        with self._torch.no_grad():
            out = self.model.generate(**enc, **gen_kwargs)
        return self.tok.decode(out[0], skip_special_tokens=True)

    def translate_variants(
        self,
        text: str,
        src: str,
        tgt: str,
        variety: str,
        cfg: DecodeConfig,
        k: int = 3,
    ) -> List[str]:
        """Return up to k candidate translations.

        Implemented via beam search n-best (num_return_sequences).
        Note: num_return_sequences must be <= num_beams.
        """
        k = max(1, int(k))
        prompt = build_prompt(src, tgt, text, variety)
        enc = self.tok(prompt, return_tensors="pt", truncation=True).to(self.device)

        num_beams = max(1, int(cfg.num_beams))
        num_return_sequences = min(k, num_beams)

        gen_kwargs = dict(
            max_new_tokens=int(cfg.max_new_tokens),
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            repetition_penalty=float(cfg.repetition_penalty),
            no_repeat_ngram_size=int(cfg.no_repeat_ngram_size),
            length_penalty=float(cfg.length_penalty),
            early_stopping=True,
        )

        with self._torch.no_grad():
            outs = self.model.generate(**enc, **gen_kwargs)
        cand = [self.tok.decode(o, skip_special_tokens=True).strip() for o in outs]
        # De-duplicate while preserving order
        seen = set()
        uniq: List[str] = []
        for c in cand:
            if c and c not in seen:
                uniq.append(c)
                seen.add(c)
        return uniq[:k] if uniq else [""]

def maybe_load_varieties(path: str) -> list[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            v = json.load(f)
        if isinstance(v, list) and v:
            return [str(x) for x in v]
    except Exception:
        pass
    return ["Unknown"]


def guess_repetition(text: str) -> bool:
    """Very rough repetition detector for degenerate outputs like 'niman niman niman ...'."""
    toks = [t for t in text.lower().split() if t.strip()]
    if len(toks) < 8:
        return False
    top = max((toks.count(t) for t in set(toks)), default=0)
    return (top / max(1, len(toks))) >= 0.45


def openai_available() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def openai_translate(
    text: str,
    src: str,
    tgt: str,
    variety: str,
    model: Optional[str] = None,
) -> str:
    """Optional translation fallback using the OpenAI Responses API.

    Requires OPENAI_API_KEY in the environment.
    """
    from openai import OpenAI

    model = model or os.getenv("OPENAI_TRANSLATE_MODEL", "gpt-4o-mini")
    client = OpenAI()
    system = (
        "You are a careful professional translator. "
        "Preserve meaning, names, and numbers. "
        "Keep the output only in the target language."
    )
    user = (
        f"Translate from {_label(src)} to {_label(tgt)}. "
        f"Variety/dialect hint (if applicable): {variety}.\n\n"
        f"TEXT:\n{text.strip()}"
    )
    try:
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=float(os.getenv("OPENAI_TRANSLATE_TEMPERATURE", "0.2")),
        )
        return (resp.output_text or "").strip()
    except Exception as e:
        # Caller can decide to fall back to local/offline.
        raise RuntimeError(f"OpenAI translation failed: {e}") from e


def openai_translate_variants(
    text: str,
    src: str,
    tgt: str,
    variety: str,
    k: int = 3,
    temperature: float = 0.6,
    model: Optional[str] = None,
) -> List[str]:
    """Generate k translation variants with OpenAI.

    The Responses API is optimized for a single output; for variant lists we issue
    multiple calls with a moderate temperature and de-duplicate results.
    """
    from openai import OpenAI

    if not openai_available():
        raise RuntimeError("OPENAI_API_KEY is not set")

    model = model or os.getenv("OPENAI_TRANSLATE_MODEL", "gpt-4o-mini")
    client = OpenAI()
    k = max(1, int(k))
    temperature = float(temperature)

    system = (
        "You are a careful professional translator. "
        "Preserve meaning, names, and numbers. "
        "Keep the output only in the target language. "
        "Return a faithful translation; phrasings may vary but must remain accurate."
    )
    user = (
        f"Translate from {_label(src)} to {_label(tgt)}. "
        f"Variety/dialect hint (if applicable): {variety}.\n\n"
        f"TEXT:\n{text.strip()}"
    )

    outs: List[str] = []
    seen = set()
    for _ in range(k):
        resp = client.responses.create(
            model=model,
            temperature=temperature,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        t = (resp.output_text or "").strip()
        if t and t not in seen:
            outs.append(t)
            seen.add(t)
        if len(outs) >= k:
            break
    return outs if outs else [""]


def openai_transcribe_image(
    image_path: str,
    language_hint: str = "es",
    alphabet_hint: str = "",
    model: Optional[str] = None,
    temperature: float = 0.0,
) -> str:
    """Transcribe a manuscript page image using an OpenAI vision-capable model."""
    import base64
    from io import BytesIO
    from PIL import Image
    from openai import OpenAI

    if not openai_available():
        raise RuntimeError("OPENAI_API_KEY is not set")

    model = model or os.getenv("OPENAI_TRANSCRIBE_MODEL", "gpt-4o-mini")
    client = OpenAI()

    # Normalize to PNG for consistent mime + smaller surprises.
    img = Image.open(image_path)
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    data_url = f"data:image/png;base64,{b64}"

    system = (
        "You are a paleography-focused transcription assistant. "
        "Transcribe EXACTLY what you see into plain text. "
        "Preserve line breaks and punctuation. Do not translate. "
        "If a character/word is unclear, mark it with [?]. "
        "Do not add commentary."
    )
    user = (
        f"Language hint: {language_hint}. "
        f"Alphabet/orthography hint: {alphabet_hint or '(none)'}\n\n"
        "Task: produce a faithful transcription." 
    )

    resp = client.responses.create(
        model=model,
        temperature=float(temperature),
        input=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user},
                    {"type": "input_image", "image_url": data_url},
                ],
            },
        ],
    )
    return (resp.output_text or "").strip()


def openai_handwriting_profile(
    image_path: str,
    model: Optional[str] = None,
) -> str:
    """Return a JSON-ish description of the handwriting (for scribe/hand comparison).

    This is a *heuristic* assistant output, useful for clustering/notes, not a
    definitive attribution.
    """
    import base64
    from openai import OpenAI
    from PIL import Image
    from io import BytesIO

    if not openai_available():
        raise RuntimeError("OPENAI_API_KEY is not set")

    img = Image.open(image_path)
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    data_url = f"data:image/png;base64,{b64}"

    model = model or os.getenv("OPENAI_TRANSCRIBE_MODEL", "gpt-4o-mini")
    client = OpenAI()

    system = (
        "You analyze handwriting features for research notes. "
        "Return ONLY valid JSON with keys: slant, spacing, stroke_weight, "
        "letterform_notes, likely_script_type, and a short hand_signature string."
    )
    user = "Describe handwriting characteristics visible in this image for later comparison." 

    resp = client.responses.create(
        model=model,
        temperature=0.2,
        input=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user},
                    {"type": "input_image", "image_url": data_url},
                ],
            },
        ],
    )
    return (resp.output_text or "").strip()


def openai_extract(
    text: str,
    instruction: str,
    schema_hint: str = "",
    model: Optional[str] = None,
) -> str:
    """Optional extraction using OpenAI. Returns a JSON string when possible."""
    from openai import OpenAI

    model = model or os.getenv("OPENAI_EXTRACT_MODEL", "gpt-4o-mini")
    client = OpenAI()
    system = (
        "You extract structured data from text. "
        "Return ONLY valid JSON. No markdown, no commentary."
    )
    user = (
        f"INSTRUCTION:\n{instruction.strip()}\n\n"
        f"SCHEMA_HINT (optional):\n{schema_hint.strip()}\n\n"
        f"TEXT:\n{text.strip()}\n"
    )
    try:
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return (resp.output_text or "").strip()
    except Exception as e:
        raise RuntimeError(f"OpenAI extraction failed: {e}") from e
