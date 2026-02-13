import os
from pathlib import Path

import gradio as gr

from webapp.ocr import extract_text, render_pdf_page_to_png
from webapp.services import (
    DecodeConfig,
    LocalSeq2SeqTranslator,
    maybe_load_varieties,
    guess_repetition,
    openai_available,
    openai_translate,
    openai_translate_variants,
    openai_extract,
    openai_transcribe_image,
    openai_handwriting_profile,
)


PORT = int(os.getenv("PORT", "7860"))
MODEL_DIR = os.getenv("MODEL_DIR", "runs/mt5_small")

# Optional: second local model (e.g., reverse direction) if you train separately.
MODEL_DIR_2 = os.getenv("MODEL_DIR_2", "").strip()


def _load_local(model_dir: str):
    if not model_dir:
        return None
    if not Path(model_dir).exists():
        return None
    return LocalSeq2SeqTranslator(model_dir)


LOCAL_1 = _load_local(MODEL_DIR)
LOCAL_2 = _load_local(MODEL_DIR_2)


VARIETIES_PATH = os.path.join(os.path.dirname(__file__), "varieties.json")
VARIETIES = maybe_load_varieties(VARIETIES_PATH)


LANGS = [
    ("English", "en"),
    ("Nahuatl", "nah"),
    ("Spanish", "es"),
]

PAIRS = [
    ("English → Nahuatl", ("en", "nah")),
    ("Nahuatl → English", ("nah", "en")),
    ("Spanish → Nahuatl", ("es", "nah")),
    ("Nahuatl → Spanish", ("nah", "es")),
    ("English → Spanish", ("en", "es")),
    ("Spanish → English", ("es", "en")),
]


def _local_translate(text: str, src: str, tgt: str, variety: str, cfg: DecodeConfig) -> str:
    # If a second model is provided, use it for the reverse direction.
    # Heuristic: if MODEL_DIR_2 is set, treat it as the reverse (nah->en) model.
    if src == "nah" and tgt == "en" and LOCAL_2 is not None:
        return LOCAL_2.translate(text, src, tgt, variety, cfg)
    if LOCAL_1 is None:
        raise RuntimeError(
            f"Local model not found. Set MODEL_DIR to a trained model folder. Current: {MODEL_DIR}"
        )
    return LOCAL_1.translate(text, src, tgt, variety, cfg)


def translate_ui(
    text: str,
    pair: tuple[str, str],
    variety: str,
    engine: str,
    max_new_tokens: int,
    num_beams: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    length_penalty: float,
    variants_k: int,
    openai_temperature: float,
) -> str:
    text = (text or "").strip()
    if not text:
        return "(empty input)"
    src, tgt = pair
    cfg = DecodeConfig(
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        length_penalty=length_penalty,
    )

    variants_k = max(1, int(variants_k))

    # Modes:
    # - "OpenAI only": always OpenAI (requires key)
    # - "Local only": never OpenAI
    # - "Auto": local first, then OpenAI if output looks degenerate or local errors
    if engine == "OpenAI only":
        if not openai_available():
            return "OpenAI is not configured. Set OPENAI_API_KEY (e.g., via .env) and restart."
        try:
            if variants_k > 1:
                outs = openai_translate_variants(text, src, tgt, variety, k=variants_k, temperature=openai_temperature)
                return "\n\n".join([f"{i+1}. {o}" for i, o in enumerate(outs)])
            return openai_translate(text, src, tgt, variety)
        except Exception as e:
            return str(e)

    if engine == "Local only":
        try:
            if variants_k > 1:
                if LOCAL_1 is None and LOCAL_2 is None:
                    raise RuntimeError("Local model not configured")
                # choose model via same heuristic as _local_translate
                model = LOCAL_2 if (src == "nah" and tgt == "en" and LOCAL_2 is not None) else LOCAL_1
                outs = model.translate_variants(text, src, tgt, variety, cfg, k=variants_k)
                return "\n\n".join([f"{i+1}. {o}" for i, o in enumerate(outs)])
            return _local_translate(text, src, tgt, variety, cfg)
        except Exception as e:
            return str(e)

    # Auto (recommended): local first, fallback to OpenAI if it looks degenerate.
    try:
        if variants_k > 1:
            model = LOCAL_2 if (src == "nah" and tgt == "en" and LOCAL_2 is not None) else LOCAL_1
            if model is None:
                raise RuntimeError("Local model not configured")
            local_variants = model.translate_variants(text, src, tgt, variety, cfg, k=variants_k)
            pred = local_variants[0] if local_variants else ""
        else:
            pred = _local_translate(text, src, tgt, variety, cfg)
    except Exception as e:
        # If local model isn't available, try OpenAI as a fallback.
        if openai_available():
            try:
                if variants_k > 1:
                    outs = openai_translate_variants(text, src, tgt, variety, k=variants_k, temperature=openai_temperature)
                    return "\n\n".join([f"{i+1}. {o}" for i, o in enumerate(outs)])
                return openai_translate(text, src, tgt, variety)
            except Exception as oe:
                return f"Local failed: {e}\nOpenAI failed: {oe}"
        return str(e)

    if guess_repetition(pred) and openai_available():
        try:
            if variants_k > 1:
                outs = openai_translate_variants(text, src, tgt, variety, k=variants_k, temperature=openai_temperature)
                return "\n\n".join([f"{i+1}. {o}" for i, o in enumerate(outs)])
            alt = openai_translate(text, src, tgt, variety)
            return alt or pred
        except Exception:
            return pred
    if variants_k > 1:
        # If we computed local variants, return them.
        try:
            return "\n\n".join([f"{i+1}. {o}" for i, o in enumerate(local_variants)])
        except Exception:
            pass
    return pred


def manuscript_ui(file_obj, page_num: int, engine: str, language_hint: str, alphabet_hint: str):
    if file_obj is None:
        return "", "No file uploaded", ""
    path = file_obj.name
    ext = os.path.splitext(path)[1].lower()
    img_path = path
    note = ""
    try:
        if ext == ".pdf":
            from webapp.ocr import render_pdf_page_to_png
            img_path = render_pdf_page_to_png(path, page_num=page_num)
            note = f"Rendered PDF page {page_num} to image for transcription."
    except Exception as e:
        return "", f"Failed to render PDF page: {e}", ""

    try:
        if engine == "OpenAI Vision":
            txt = openai_transcribe_image(img_path, language_hint=language_hint, alphabet_hint=alphabet_hint)
        else:
            # Tesseract is best for printed; may be weak on handwriting.
            from webapp.ocr import extract_text
            txt = extract_text(img_path, lang_hint=language_hint).text
        prof = ""
        if openai_available():
            try:
                prof = openai_handwriting_profile(img_path)
            except Exception:
                prof = ""
        return txt, (note or "OK"), prof
    except Exception as e:
        return "", str(e), ""


def ocr_ui(file_obj, lang_hint: str):
    if file_obj is None:
        return "", "No file uploaded"
    res = extract_text(file_obj.name, lang_hint=lang_hint)
    note = f"Pages: {res.pages} | Used OCR: {res.used_ocr}"
    return res.text, note


def extract_ui(text: str, instruction: str, schema_hint: str, engine: str):
    text = (text or "").strip()
    instruction = (instruction or "").strip()
    schema_hint = (schema_hint or "").strip()
    if not text:
        return "(empty input)"
    if not instruction:
        return "(add an instruction, e.g. 'extract names, dates, and amounts')"

    if engine == "OpenAI" and openai_available():
        try:
            return openai_extract(text, instruction, schema_hint)
        except Exception as e:
            # Fall back to offline heuristics
            pass

    # Simple offline fallback (regex-ish heuristics). Not perfect, but useful.
    import re
    from collections import OrderedDict

    out = OrderedDict()
    # emails
    out["emails"] = sorted(set(re.findall(r"\b[\w.+'-]+@[\w.-]+\.[A-Za-z]{2,}\b", text)))
    # phones (very loose)
    out["phones"] = sorted(
        set(re.findall(r"\b(?:\+?\d[\d\-().\s]{7,}\d)\b", text))
    )
    # dates (very loose)
    out["dates"] = sorted(
        set(
            re.findall(
                r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b",
                text,
            )
        )
    )
    out["note"] = (
        "Offline extraction is basic. Set OPENAI_API_KEY and choose OpenAI for instruction-following extraction."
    )

    import json

    return json.dumps(out, ensure_ascii=False, indent=2)


with gr.Blocks(title="Nahuatl Translator + Document OCR") as demo:
    key_status = "✅ OpenAI fallback enabled" if openai_available() else "⚪ OpenAI fallback disabled (set OPENAI_API_KEY)"
    gr.Markdown(
        "# Nahuatl Translator + Document OCR\n"
        "Translate text (English↔Nahuatl, Spanish↔Nahuatl) and extract text from uploaded documents.\n\n"
        f"**OpenAI status:** {key_status}\n\n"
        "**Tips:** If your local model outputs repetitive text (e.g., `niman niman niman ...`), try increasing beams or enable OpenAI fallback."
    )

    with gr.Tabs():
        with gr.Tab("Translate"):
            with gr.Row():
                src = gr.Textbox(
                    label="Input",
                    value="Good morning. Where is the school?",
                    lines=8,
                )
                out = gr.Textbox(label="Output", lines=8)

            with gr.Row():
                pair = gr.Dropdown(
                    label="Language pair",
                    choices=[p[0] for p in PAIRS],
                    value=PAIRS[0][0],
                )
                variety = gr.Dropdown(label="Variety", choices=VARIETIES, value=VARIETIES[0])
                engine = gr.Dropdown(
                    label="Engine",
                    choices=["Auto (local → OpenAI if needed)", "Local only", "OpenAI only"],
                    value="Auto (local → OpenAI if needed)",
                )

            with gr.Row():
                variants_k = gr.Slider(1, 5, value=1, step=1, label="Translation variants")
                openai_temperature = gr.Slider(0.0, 1.5, value=0.6, step=0.05, label="OpenAI temperature (for variants)")

            with gr.Accordion("Decoding controls", open=False):
                with gr.Row():
                    max_new_tokens = gr.Slider(16, 256, value=128, step=1, label="Max new tokens")
                    num_beams = gr.Slider(1, 8, value=4, step=1, label="Beams")
                with gr.Row():
                    repetition_penalty = gr.Slider(1.0, 2.0, value=1.15, step=0.01, label="Repetition penalty")
                    no_repeat_ngram_size = gr.Slider(0, 6, value=3, step=1, label="No repeat ngram")
                    length_penalty = gr.Slider(0.5, 2.0, value=1.0, step=0.01, label="Length penalty")

            btn = gr.Button("Translate")

            def _pair_value(label: str) -> tuple[str, str]:
                for name, codes in PAIRS:
                    if name == label:
                        return codes
                return ("en", "nah")

            btn.click(
                fn=lambda t, pl, v, e, m, b, rp, nr, lp, vk, ot: translate_ui(
                    t,
                    _pair_value(pl),
                    v,
                    "OpenAI only" if e == "OpenAI only" else ("Local only" if e == "Local only" else "Auto"),
                    m,
                    b,
                    rp,
                    nr,
                    lp,
                    vk,
                    ot,
                ),
                inputs=[src, pair, variety, engine, max_new_tokens, num_beams, repetition_penalty, no_repeat_ngram_size, length_penalty, variants_k, openai_temperature],
                outputs=[out],
            )

        with gr.Tab("Upload → OCR → Translate"):
            gr.Markdown(
                "Upload a PDF (text or scanned) or an image, extract text, then translate it."
            )
            with gr.Row():
                file_in = gr.File(label="Document", file_types=[".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"])
                lang_hint = gr.Dropdown(label="OCR hint", choices=["en", "es"], value="en")
            ocr_btn = gr.Button("Extract text (OCR)")
            with gr.Row():
                extracted = gr.Textbox(label="Extracted text", lines=12)
                ocr_note = gr.Textbox(label="OCR details", interactive=False)

            ocr_btn.click(fn=ocr_ui, inputs=[file_in, lang_hint], outputs=[extracted, ocr_note])

            gr.Markdown("### Translate extracted text")
            with gr.Row():
                pair2 = gr.Dropdown(label="Language pair", choices=[p[0] for p in PAIRS], value=PAIRS[0][0])
                variety2 = gr.Dropdown(label="Variety", choices=VARIETIES, value=VARIETIES[0])
                engine2 = gr.Dropdown(
                    label="Engine",
                    choices=["Auto (local → OpenAI if needed)", "Local only", "OpenAI only"],
                    value="Auto (local → OpenAI if needed)",
                )
            translate_doc_btn = gr.Button("Translate extracted text")
            translated_doc = gr.Textbox(label="Translation", lines=12)
            translate_doc_btn.click(
                fn=lambda t, pl, v, e: translate_ui(
                    t,
                    _pair_value(pl),
                    v,
                    "OpenAI only" if e == "OpenAI only" else ("Local only" if e == "Local only" else "Auto"),
                    256,
                    4,
                    1.15,
                    3,
                    1.0,
                    1,
                    0.6,
                ),
                inputs=[extracted, pair2, variety2, engine2],
                outputs=[translated_doc],
            )

        with gr.Tab("Manuscript transcription"):
            gr.Markdown(
                "Upload a manuscript page (image or PDF). Use OpenAI Vision for handwriting, or Tesseract for printed text. "
                "You can then copy the transcription into the Translate tab."
            )
            with gr.Row():
                ms_file = gr.File(label="Manuscript page", file_types=[".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"])
                page_num = gr.Number(label="PDF page (1-indexed)", value=1, precision=0)
            with gr.Row():
                ms_engine = gr.Dropdown(label="Transcription engine", choices=["OpenAI Vision", "Tesseract OCR"], value="OpenAI Vision")
                ms_lang = gr.Dropdown(label="Language hint", choices=["es", "nah", "en"], value="es")
            alphabet_hint = gr.Textbox(
                label="Alphabet / orthography hint (optional)",
                value="Translating Mesoamerica alphabet (if applicable)",
            )
            ms_btn = gr.Button("Transcribe")
            with gr.Row():
                ms_out = gr.Textbox(label="Transcription", lines=14)
                ms_note = gr.Textbox(label="Status", interactive=False)
            ms_profile = gr.Textbox(label="Handwriting profile (JSON, optional)", lines=10)

            ms_btn.click(
                fn=manuscript_ui,
                inputs=[ms_file, page_num, ms_engine, ms_lang, alphabet_hint],
                outputs=[ms_out, ms_note, ms_profile],
            )

        with gr.Tab("Manual extraction"):
            gr.Markdown(
                "Give an instruction (and optional schema hint). If you set `OPENAI_API_KEY`, you can do instruction-following extraction."
            )
            text_in = gr.Textbox(label="Text", lines=12)
            instruction = gr.Textbox(
                label="Instruction",
                value="Extract names, dates, organizations, and any monetary amounts.",
                lines=3,
            )
            schema_hint = gr.Textbox(
                label="Schema hint (optional)",
                value='{"names":[],"dates":[],"organizations":[],"amounts":[],"notes":""}',
                lines=3,
            )
            engine3 = gr.Dropdown(label="Engine", choices=["Offline", "OpenAI"], value="Offline")
            extract_btn = gr.Button("Extract")
            extracted_json = gr.Textbox(label="Output", lines=12)
            extract_btn.click(fn=extract_ui, inputs=[text_in, instruction, schema_hint, engine3], outputs=[extracted_json])

    gr.Markdown(
        "---\n"
        "**Configuration**\n\n"
        "- Local model: set `MODEL_DIR` (and optionally `MODEL_DIR_2` for reverse direction)\n"
        "- OpenAI fallback: set `OPENAI_API_KEY` (optional)\n"
        "\n"
        "If you want Spanish↔Nahuatl quality comparable to English↔Nahuatl, add parallel data and retrain (see scripts)."
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=PORT, show_api=False)
