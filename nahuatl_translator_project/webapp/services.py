"""Service layer used by the web UI.

Architecture: OpenAI (GPT) is the PRIMARY translation engine. It already has
strong knowledge of Nahuatl. The parallel corpus (biblical text) and dictionary
are used only as OPTIONAL supplementary vocabulary — they help with rare/archaic
terms but should never override the AI's own knowledge.

This module intentionally lazy-loads heavy ML dependencies (torch/transformers)
so lightweight utilities can be imported without requiring those packages.
"""

import os
import json
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any, List

from webapp.spanish_filter import detect_spanish_in_output, strip_spanish


LANG_LABELS = {
    "en": "English",
    "es": "Spanish",
    "nah": "Nahuatl",
}


def _label(code: str) -> str:
    return LANG_LABELS.get(code, code)


def _supports_temperature(model: str) -> bool:
    """Check if a model supports the temperature parameter."""
    # GPT-5 and reasoning models (o-series) don't support temperature
    m = model.lower()
    if m.startswith("gpt-5") or m.startswith("o1") or m.startswith("o3") or m.startswith("o4"):
        return False
    return True


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


def _get_corpus_context(text: str, src: str) -> Tuple[str, str]:
    """Look up supplementary vocabulary from the parallel corpus.

    This is OPTIONAL context — the AI is the primary translator and already
    knows Nahuatl. The corpus (mostly biblical text) provides additional
    reference for rare or archaic terms only.

    Returns (reference_sentences, reference_vocab) strings. Either or both
    may be empty if the corpus isn't loaded or has no relevant matches.
    """
    from webapp.corpus import get_corpus
    from webapp.dictionary import get_dictionary

    ref_sentences = ""
    ref_vocab = ""

    try:
        # Light supplementary sentence context (just 3 matches, not 5)
        corpus = get_corpus()
        if corpus.loaded:
            matches = corpus.search(query=text, src_lang=src, max_results=3)
            ref_sentences = corpus.format_as_reference(matches, src_lang=src)

        # Light supplementary vocabulary (just 5 entries, not 8)
        dictionary = get_dictionary()
        if dictionary.loaded:
            ref_vocab = dictionary.format_vocab_block(
                query=text, src_lang=src, max_entries=5,
            )
    except Exception:
        # Corpus/dictionary errors should never break translation.
        # The AI can translate perfectly fine without them.
        pass

    return ref_sentences, ref_vocab


def _validate_translation(client: Any, source: str, translation: str, src: str, tgt: str) -> str:
    """Validation pass: check translation for hallucination, Spanish leakage, and errors.

    Uses a second model call to review the translation and correct issues.
    Only fires when translating TO Nahuatl.
    """
    if tgt != "nah":
        return translation

    # Quick Spanish check — strip minor contamination without an API call
    spanish_found = detect_spanish_in_output(translation)
    if spanish_found and len(spanish_found) <= 2:
        translation = strip_spanish(translation)
        spanish_found = detect_spanish_in_output(translation)

    # If clean and short input, skip the validation call to save cost
    if not spanish_found and len(source.split()) <= 5:
        return translation

    validation_model = os.getenv("OPENAI_TRANSLATE_MODEL", "gpt-5")
    try:
        val_kwargs = dict(
            model=validation_model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are reviewing a Nahuatl translation for accuracy. "
                        "Check for: invented/hallucinated words, Spanish leakage, "
                        "incorrect meanings, grammar problems. Be strict.\n\n"
                        "If the translation is acceptable, output it exactly as-is.\n"
                        "If there are problems, output ONLY the corrected Nahuatl translation. "
                        "No commentary, no explanations, no notes — just the corrected text."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Source ({_label(src)}):\n{source}\n\n"
                        f"Translation (Nahuatl):\n{translation}"
                    ),
                },
            ],
        )
        if _supports_temperature(validation_model):
            val_kwargs["temperature"] = 0.0
        resp = client.responses.create(**val_kwargs)
        corrected = (resp.output_text or "").strip()
        return corrected if corrected else translation
    except Exception:
        # If validation fails, return what we have
        if spanish_found:
            return strip_spanish(translation)
        return translation


def openai_translate(
    text: str,
    src: str,
    tgt: str,
    variety: str,
    model: Optional[str] = None,
) -> str:
    """Primary translation using the OpenAI Responses API.

    OpenAI is the primary translation engine — it already knows Nahuatl.
    The parallel corpus provides optional supplementary vocabulary for
    rare/archaic terms only.

    Requires OPENAI_API_KEY in the environment.
    """
    from openai import OpenAI
    from webapp.prompts import translation_system_prompt, translation_user_prompt

    model = model or os.getenv("OPENAI_TRANSLATE_MODEL", "gpt-5")
    client = OpenAI()

    system = translation_system_prompt(src, tgt, variety)

    # Optional supplementary context from the corpus (AI is the primary source)
    ref_sentences, ref_vocab = _get_corpus_context(text, src)
    user = translation_user_prompt(
        text=text,
        src=src,
        tgt=tgt,
        variety=variety,
        reference_vocab=ref_vocab,
        reference_sentences=ref_sentences,
    )

    try:
        kwargs = dict(
            model=model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        if _supports_temperature(model):
            kwargs["temperature"] = float(os.getenv("OPENAI_TRANSLATE_TEMPERATURE", "0.2"))
        resp = client.responses.create(**kwargs)
        raw = (resp.output_text or "").strip()
        return _validate_translation(client, text, raw, src, tgt)
    except Exception as e:
        # Caller can decide to fall back to local/offline.
        raise RuntimeError(f"OpenAI translation failed: {e}") from e


def _parse_numbered_variants(text: str, k: int) -> List[str]:
    """Parse numbered variant output like '1. ...\n2. ...\n3. ...' into a list."""
    import re
    lines = text.strip().splitlines()
    variants: List[str] = []
    seen = set()
    for line in lines:
        # Match lines starting with "1.", "2.", etc.
        m = re.match(r"^\s*\d+[\.\)]\s*(.+)", line)
        if m:
            t = m.group(1).strip()
            if t and t not in seen:
                variants.append(t)
                seen.add(t)
    # If parsing failed (AI didn't number them), treat the whole output as one variant
    if not variants:
        t = text.strip()
        if t:
            variants.append(t)
    return variants[:k]


def openai_translate_variants(
    text: str,
    src: str,
    tgt: str,
    variety: str,
    k: int = 3,
    temperature: float = 0.6,
    model: Optional[str] = None,
) -> List[str]:
    """Generate k translation variants with OpenAI in a single API call.

    OpenAI is the primary engine. Instead of making k separate identical calls
    and hoping temperature creates diversity, we ask the AI to produce all k
    distinct variants in one response. Temperature controls creative expression,
    not variant diversity.
    """
    from openai import OpenAI
    from webapp.prompts import translation_variants_system_prompt, translation_user_prompt

    if not openai_available():
        raise RuntimeError("OPENAI_API_KEY is not set")

    model = model or os.getenv("OPENAI_TRANSLATE_MODEL", "gpt-5")
    client = OpenAI()
    k = max(1, int(k))
    temperature = float(temperature)

    system = translation_variants_system_prompt(src, tgt, variety, k=k)

    # Optional supplementary context from the corpus
    ref_sentences, ref_vocab = _get_corpus_context(text, src)
    user = translation_user_prompt(
        text=text,
        src=src,
        tgt=tgt,
        variety=variety,
        reference_vocab=ref_vocab,
        reference_sentences=ref_sentences,
    )

    kwargs = dict(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    if _supports_temperature(model):
        kwargs["temperature"] = temperature
    resp = client.responses.create(**kwargs)
    raw = (resp.output_text or "").strip()
    variants = _parse_numbered_variants(raw, k)
    if not variants:
        return [""]
    # Only the first variant enforces pure Nahuatl (no Spanish).
    # Remaining variants allow natural modern loanwords, reflecting
    # how post-colonial and contemporary Nahuatl is actually spoken.
    cleaned = []
    for i, v in enumerate(variants):
        if i == 0:
            cleaned.append(_validate_translation(client, text, v, src, tgt))
        else:
            cleaned.append(v)
    return cleaned


def _image_to_data_url(image_path: str) -> str:
    """Convert an image file to a base64 data URL for the OpenAI API."""
    import base64
    from io import BytesIO
    from PIL import Image

    img = Image.open(image_path)
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _transcribe_single_image(
    client: Any,
    model: str,
    image_path: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
) -> str:
    """Send a single image to OpenAI vision and get transcription text back."""
    data_url = _image_to_data_url(image_path)
    kwargs = dict(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_prompt},
                    {"type": "input_image", "image_url": data_url},
                ],
            },
        ],
    )
    if _supports_temperature(model):
        kwargs["temperature"] = float(temperature)
    resp = client.responses.create(**kwargs)
    return (resp.output_text or "").strip()


def openai_transcribe_image(
    image_path: str,
    language_hint: str = "es",
    alphabet_hint: str = "",
    model: Optional[str] = None,
    temperature: float = 0.0,
) -> str:
    """Transcribe a manuscript page image using an OpenAI vision-capable model.

    Uses a two-pass strategy for large images:
      Pass 1 (Overview): full image at reduced resolution → structural layout.
      Pass 2 (Detail): tile the image into overlapping strips, transcribe each,
                        then stitch results together.

    For small images (< 1500px height), does a single direct transcription.
    """
    from PIL import Image
    from openai import OpenAI
    from webapp.ocr import tile_image, cleanup_tiles
    from webapp.prompts import (
        transcription_system_prompt,
        transcription_tile_prompt,
        transcription_stitch_prompt,
    )

    if not openai_available():
        raise RuntimeError("OPENAI_API_KEY is not set")

    model = model or os.getenv("OPENAI_TRANSCRIBE_MODEL", "gpt-4o")
    client = OpenAI()

    system = transcription_system_prompt(language_hint, alphabet_hint)

    # Check image dimensions to decide strategy
    img = Image.open(image_path)
    _, height = img.size
    img.close()

    # --- Small image: single-pass transcription ---
    if height <= 1500:
        user = (
            f"Language hint: {language_hint}. "
            f"Alphabet/orthography hint: {alphabet_hint or '(none)'}\n\n"
            "Transcribe ALL text visible on this page. Work from top to bottom."
        )
        return _transcribe_single_image(
            client, model, image_path, system, user, temperature,
        )

    # --- Large image: two-pass with tiling ---

    # Pass 1: Overview scan (optional, helps with ordering)
    # We skip this for speed and go straight to tiled detail transcription.

    # Pass 2: Tile and transcribe each strip
    tile_paths = tile_image(image_path, tile_height=800, overlap=100)

    try:
        if len(tile_paths) == 1:
            # Tiling returned the original (wasn't tall enough after all)
            user = (
                f"Language hint: {language_hint}. "
                f"Alphabet/orthography hint: {alphabet_hint or '(none)'}\n\n"
                "Transcribe ALL text visible on this page. Work from top to bottom."
            )
            return _transcribe_single_image(
                client, model, image_path, system, user, temperature,
            )

        # Transcribe each tile
        tile_texts: List[str] = []
        for i, tile_path in enumerate(tile_paths):
            user = transcription_tile_prompt(
                tile_index=i,
                total_tiles=len(tile_paths),
                language_hint=language_hint,
                alphabet_hint=alphabet_hint,
            )
            text = _transcribe_single_image(
                client, model, tile_path, system, user, temperature,
            )
            tile_texts.append(text)

        # Stitch tile transcriptions together
        if len(tile_texts) == 1:
            return tile_texts[0]

        # Use OpenAI to intelligently merge overlapping transcriptions
        stitch_system = transcription_stitch_prompt(len(tile_texts))
        segments = "\n\n".join(
            f"--- SEGMENT {i+1} of {len(tile_texts)} ---\n{t}"
            for i, t in enumerate(tile_texts)
        )
        stitch_user = f"Merge these transcription segments:\n\n{segments}"

        stitch_kwargs = dict(
            model=model,
            input=[
                {"role": "system", "content": stitch_system},
                {"role": "user", "content": stitch_user},
            ],
        )
        if _supports_temperature(model):
            stitch_kwargs["temperature"] = 0.0
        resp = client.responses.create(**stitch_kwargs)
        return (resp.output_text or "").strip()

    finally:
        cleanup_tiles(tile_paths, image_path)


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

    model = model or os.getenv("OPENAI_TRANSCRIBE_MODEL", "gpt-4o")
    client = OpenAI()

    system = (
        "You analyze handwriting features for research notes. "
        "Return ONLY valid JSON with keys: slant, spacing, stroke_weight, "
        "letterform_notes, likely_script_type, and a short hand_signature string."
    )
    user = "Describe handwriting characteristics visible in this image for later comparison." 

    hw_kwargs = dict(
        model=model,
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
    if _supports_temperature(model):
        hw_kwargs["temperature"] = 0.2
    resp = client.responses.create(**hw_kwargs)
    return (resp.output_text or "").strip()


def openai_extract(
    text: str,
    instruction: str,
    schema_hint: str = "",
    model: Optional[str] = None,
) -> str:
    """Structured extraction using OpenAI with Mesoamerican domain knowledge.

    Injects entity taxonomy and disambiguation rules so the model can
    correctly classify entities like Mexica (people) vs Mexico (place).
    Returns a JSON string.
    """
    from openai import OpenAI
    from webapp.entities import format_entity_reference, DISAMBIGUATION_RULES
    from webapp.prompts import extraction_system_prompt, extraction_user_prompt

    model = model or os.getenv("OPENAI_EXTRACT_MODEL", "gpt-5")
    client = OpenAI()

    entity_ref = format_entity_reference(max_per_type=8)
    system = extraction_system_prompt(
        entity_reference=entity_ref,
        disambiguation_rules=DISAMBIGUATION_RULES,
    )
    user = extraction_user_prompt(
        text=text,
        instruction=instruction,
        schema_hint=schema_hint,
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
