"""Domain-specific prompt templates for Nahuatl translation, transcription, and extraction.

Architecture: OpenAI (GPT) is the PRIMARY translation engine. It already has
strong knowledge of Nahuatl grammar, vocabulary, and morphology. Our prompts
give it focused expert guidance and common-mistake warnings — NOT a full
grammar textbook. The parallel corpus (biblical text) is used only as
optional supplementary vocabulary for rare/archaic terms.
"""

from __future__ import annotations

from typing import List, Tuple


# ---------------------------------------------------------------------------
# Language labels
# ---------------------------------------------------------------------------
LANG_LABELS = {
    "en": "English",
    "es": "Spanish",
    "nah": "Nahuatl",
}


def _label(code: str) -> str:
    return LANG_LABELS.get(code, code)


# ---------------------------------------------------------------------------
# Core Nahuatl expert guidance (concise — the AI already knows the grammar)
# ---------------------------------------------------------------------------
NAHUATL_EXPERT_GUIDANCE = """\
CRITICAL RULES (these are the most common mistakes to avoid):

1. GREETINGS: "Hello"/"Hi" = Pialli or Niltze. NEVER use "cualli" for hello.
   "Cualli" means "good" — it is used in "Cualli tonalli" (good day) but NOT alone as a greeting.

2. POSSESSIVES ARE ONE WORD: Possessive prefixes (no-, mo-, i-, to-, amo-, in-)
   attach directly to the noun stem. The absolutive suffix (-tl, -tli, -li) drops.
   CORRECT: notoca (my name), nocal (my house), nocniuh (my friend)
   WRONG: "no itoca", "no toca", "no calli" — NEVER separate with a space.

3. NO ARTICLES: Nahuatl has no "the"/"a". Do not insert articles in Nahuatl output.

4. PRESERVE NAMES: Keep proper names as-is. Do not translate personal names.

5. ORTHOGRAPHY: Accept both classical (hu, qu) and modern (w, k) spellings.
   Default to classical orthography unless the user specifies otherwise.

6. NO SPANISH MIXING: When translating TO Nahuatl, output ONLY Nahuatl words.
   NEVER substitute Spanish words (like "país", "iglesia", "rey", "Dios") when
   a Nahuatl equivalent exists. Use the Nahuatl term:
   - nation/country → altepetl (or cemanahuac for "the world")
   - king → tlatoani
   - god → teotl
   - church → teocalli (or teopantli)
   - priest → teopixqui (or tlamacazqui)
   Only use Spanish loanwords if no Nahuatl equivalent exists AND the word was
   historically borrowed into Nahuatl (e.g., "caballo" → cahuayo).

COMMON VOCABULARY REMINDERS:
- Hello/Hi → Pialli (formal), Niltze (informal)
- My name is [X] → Notoca [X]
- What is your name? → ¿Tlein motoca?
- How are you? → ¿Quen tica?
- Thank you → Tlazohcamati
- Yes → Quemah   |   No → Ahmo
- I love you → Nimitztlazohtla
- Goodbye → Timo ittazqueh
- Water → Atl   |   House → Calli   |   My house → Nocal
- Good morning → Cualli tlaneci
- Good day → Cualli tonalli\
"""

# ---------------------------------------------------------------------------
# Few-shot translation examples (these are the main teaching tool for the AI)
# ---------------------------------------------------------------------------
FEWSHOT_TRANSLATION_EXAMPLES = [
    # --- Greetings & introductions (EN→NAH) ---
    ("en", "nah", "Hello, my name is Carlos.",
     "Pialli, notoca Carlos."),
    ("en", "nah", "Hi, how are you?",
     "Niltze, ¿quen tica?"),
    ("en", "nah", "Good morning. How are you?",
     "Cualli tlaneci. ¿Quen otitlathuil?"),
    ("en", "nah", "Thank you very much.",
     "Huel tlazohcamati."),
    # --- Everyday sentences (EN→NAH) ---
    ("en", "nah", "The water is very cold.",
     "Atl huel itztic."),
    ("en", "nah", "I want to eat tortillas.",
     "Nicnequi nitlaxcalcuas."),
    ("en", "nah", "My house is near the mountain.",
     "Nocal itzalan tepetl."),
    ("en", "nah", "What is your name?",
     "¿Tlein motoca?"),
    # --- Sentences with commonly Spanish-substituted words (EN→NAH) ---
    ("en", "nah", "I know the priest and the god of the nation.",
     "Nicmati teopixqui ihuan teotl in altepetl."),
    ("en", "nah", "The king went to the church.",
     "Tlatoani oyah teocalco."),
    # --- Nahuatl to English ---
    ("nah", "en", "Pialli, notoca Maria. Nimitztlazohtla.",
     "Hello, my name is Maria. I love you."),
    ("nah", "en", "Nicnequi atl.",
     "I want water."),
    ("nah", "en", "Cualli tonalli. ¿Quen tica?",
     "Good day. How are you?"),
    # --- Spanish to Nahuatl ---
    ("es", "nah", "Hola, ¿cómo te llamas?",
     "Pialli, ¿tlein motoca?"),
    ("es", "nah", "Buenos días. ¿Cómo estás?",
     "Cualli tlaneci. ¿Quen tica?"),
]


def _format_fewshot(src: str, tgt: str) -> str:
    """Select and format few-shot examples relevant to the language direction."""
    relevant = []
    for ex_src, ex_tgt, inp, out in FEWSHOT_TRANSLATION_EXAMPLES:
        if ex_src == src and ex_tgt == tgt:
            relevant.append(f"  {_label(src)}: {inp}\n  {_label(tgt)}: {out}")
        elif ex_src == tgt and ex_tgt == src:
            # Reverse direction example is also useful
            relevant.append(f"  {_label(src)}: {out}\n  {_label(tgt)}: {inp}")
    if not relevant:
        # Fallback: show any Nahuatl example for general guidance
        for ex_src, ex_tgt, inp, out in FEWSHOT_TRANSLATION_EXAMPLES[:2]:
            relevant.append(f"  {_label(ex_src)}: {inp}\n  {_label(ex_tgt)}: {out}")
    return "\n\n".join(relevant)


# ---------------------------------------------------------------------------
# Public prompt builders
# ---------------------------------------------------------------------------

def translation_system_prompt(src: str, tgt: str, variety: str) -> str:
    """Build a focused system prompt for Nahuatl translation.

    Philosophy: The AI is the primary expert. We give it concise guidance
    and critical warnings, NOT a full grammar textbook.
    """
    fewshot = _format_fewshot(src, tgt)
    variety_note = ""
    if variety and variety.lower() not in ("unknown", ""):
        variety_note = (
            f"\nThe user has specified the dialect: {variety}. "
            f"Adapt your orthography and vocabulary to match."
        )

    return f"""\
You are an expert Nahuatl (Aztec language) translator with deep knowledge of \
Classical and modern Nahuatl. You handle grammar, morphology, and vocabulary natively.

{NAHUATL_EXPERT_GUIDANCE}
{variety_note}

TRANSLATION DIRECTION: {_label(src)} → {_label(tgt)}

EXAMPLES:
{fewshot}

INSTRUCTIONS:
- LANGUAGE PURITY: When translating TO Nahuatl, every word in your output MUST be \
Nahuatl. NEVER insert Spanish words (país, nación, iglesia, rey, Dios, sacerdote, etc.). \
Use the Nahuatl equivalent: nation/country=altepetl, god=teotl, priest=teopixqui, \
king=tlatoani, church=teocalli. If you catch yourself writing a Spanish word, replace it.
- Translate faithfully, preserving meaning, tone, and intent.
- Preserve proper names, numbers, and dates as-is.
- Output ONLY the translation in {_label(tgt)}. No commentary, no explanations.
- Use your own knowledge as the primary source. If supplementary vocabulary is \
provided in the user message, treat it as optional reference — do not blindly copy it.
- If a word has no direct equivalent, use the closest culturally appropriate Nahuatl term.\
"""


def translation_user_prompt(
    text: str,
    src: str,
    tgt: str,
    variety: str,
    reference_vocab: str = "",
    reference_sentences: str = "",
) -> str:
    """Build the user prompt for a translation call, with optional supplementary context."""
    parts = []

    if reference_vocab:
        parts.append(
            f"SUPPLEMENTARY VOCABULARY (optional reference from corpus — use your own knowledge first):\n"
            f"{reference_vocab}"
        )

    if reference_sentences:
        parts.append(
            f"SUPPLEMENTARY PARALLEL EXAMPLES (from biblical corpus — adapt style to match the input):\n"
            f"{reference_sentences}"
        )

    parts.append(f"TRANSLATE THIS:\n{text.strip()}")

    return "\n\n".join(parts)


def translation_variants_system_prompt(src: str, tgt: str, variety: str, k: int = 3) -> str:
    """System prompt for variant generation.

    Asks the AI to produce all k variants in a single response, ensuring
    genuine diversity without relying on temperature for variance.
    """
    base = translation_system_prompt(src, tgt, variety)
    return (
        base + "\n\n"
        f"Provide EXACTLY {k} distinct translation variants, numbered 1 through {k}.\n"
        "Each variant must be a faithful, accurate translation but use DIFFERENT "
        "word choices, sentence structure, or phrasing. All variants must be equally "
        "correct — vary style, not accuracy.\n"
        "Format:\n"
        f"1. [first translation]\n"
        f"2. [second translation]\n"
        + (f"3. [third translation]\n" if k >= 3 else "")
        + ("..." if k > 3 else "")
        + "\nOutput ONLY the numbered translations. No commentary or explanations."
    )


# ---------------------------------------------------------------------------
# Transcription prompts (Phase 3)
# ---------------------------------------------------------------------------

COLONIAL_NAHUATL_TRANSCRIPTION_CONTEXT = """\
COLONIAL-ERA NAHUATL MANUSCRIPT CONVENTIONS:
- Long-s (ſ): commonly used for 's' in colonial texts. Transcribe as regular 's'.
- Abbreviations with tildes: ñ above a letter often marks a missing 'n' or 'm'
  (e.g., "cõ" = "con", "q̃" = "que"). Expand abbreviations when clearly identifiable.
- Ligatures: 'ct', 'st', 'ff' may appear as connected letterforms.
- Nahuatl words use Spanish colonial orthography:
  * "hu" for /w/ (modern: "w")
  * "qu" before e/i for /k/ (modern: "k")
  * "cu" for /kʷ/
  * "tz" for /ts/
  * "x" for /ʃ/ (like Spanish "x" in "México")
  * "tl" for the lateral affricate /tɬ/
- Common colonial abbreviations: "xpo" = "Cristo", "dios" = "Dios",
  "nro" = "nuestro", "sr" = "señor".
- Line continuation: a word split across lines may be marked with a hyphen
  or simply broken. Rejoin split words when the break is obvious.
- Marginal notes and interlinear glosses may appear; transcribe them separately
  if visible, marked with [margin:] or [gloss:].
- Numbers may appear as Roman numerals or Arabic numerals.
- Folio markers (e.g., "fol. 3r", "fol. 3v") indicate recto/verso pages.\
"""


def transcription_system_prompt(language_hint: str = "es", alphabet_hint: str = "") -> str:
    """Build a domain-aware system prompt for manuscript transcription."""
    lang_note = ""
    if language_hint == "nah":
        lang_note = (
            "The primary language is Nahuatl (Aztec). "
            "Expect agglutinative words with prefixes and suffixes. "
            "Common Nahuatl words include: in, ihuan, ica, ipan, niman, "
            "quimati, tlatoani, altepetl, calli, tlalli, atl."
        )
    elif language_hint == "es":
        lang_note = (
            "The primary language is Spanish, possibly mixed with Nahuatl. "
            "Colonial-era Spanish uses archaic spellings and abbreviations."
        )

    alphabet_note = ""
    if alphabet_hint:
        alphabet_note = f"\nAdditional orthography hint from user: {alphabet_hint}"

    return f"""\
You are an expert paleographer specializing in colonial-era Mesoamerican manuscripts.
You have deep expertise in reading 16th-18th century handwritten documents in
Spanish, Nahuatl, and mixed-language colonial texts.

{COLONIAL_NAHUATL_TRANSCRIPTION_CONTEXT}

{lang_note}{alphabet_note}

TRANSCRIPTION RULES:
- Transcribe EXACTLY what you see. Do NOT translate or paraphrase.
- Preserve original line breaks as they appear on the page.
- Preserve original punctuation, capitalization, and spacing.
- If a character or word is unclear, mark it with [?] or [illegible].
- If you can partially read a word, write what you can and mark unclear
  parts: e.g., "tlaca[?]li" for a partially legible word.
- Expand obvious abbreviations but mark them: e.g., "que" [expanded from q̃].
- If marginal notes exist, transcribe them as [margin: text here].
- Do NOT add commentary, analysis, or translation. Only transcribe.
- Transcribe ALL text visible on the page, not just the first few lines.
- Work systematically from top to bottom, left to right.\
"""


def transcription_overview_prompt(language_hint: str = "es") -> str:
    """Prompt for the overview pass — get structural layout info."""
    return (
        "Analyze the overall structure of this manuscript page. Report:\n"
        "1. Approximate number of text lines visible\n"
        "2. Number of text columns (1 or 2)\n"
        "3. Whether there are marginal notes\n"
        "4. Whether there are headers, titles, or section divisions\n"
        "5. General legibility (good / fair / poor)\n"
        "6. Primary script type (print / handwritten / mixed)\n\n"
        f"Language hint: {language_hint}.\n"
        "Be brief — this is a structural overview only. Do NOT transcribe the text."
    )


def transcription_tile_prompt(
    tile_index: int,
    total_tiles: int,
    language_hint: str = "es",
    alphabet_hint: str = "",
) -> str:
    """User prompt for transcribing a single tile of a larger page."""
    position = "top" if tile_index == 0 else ("bottom" if tile_index == total_tiles - 1 else "middle")
    return (
        f"This is section {tile_index + 1} of {total_tiles} ({position} portion) "
        f"of a manuscript page.\n"
        f"Language hint: {language_hint}. "
        f"Alphabet/orthography hint: {alphabet_hint or '(none)'}\n\n"
        "Transcribe ALL text visible in this section. "
        "If text is cut off at the top or bottom edge, transcribe what is visible "
        "and mark cut-off words with [...]. "
        "Produce ONLY the transcription, no commentary."
    )


def transcription_stitch_prompt(tile_count: int) -> str:
    """System prompt for stitching tile transcriptions together."""
    return (
        f"You are combining {tile_count} overlapping transcription segments from "
        "a single manuscript page. The segments overlap slightly, so some text "
        "appears in consecutive segments.\n\n"
        "RULES:\n"
        "- Merge the segments into one continuous transcription.\n"
        "- Remove duplicate lines that appear in the overlap zones.\n"
        "- Preserve the original line breaks and formatting.\n"
        "- Do NOT add, remove, or modify any text beyond de-duplicating overlaps.\n"
        "- If a word was cut off [...] in one segment but complete in the overlap "
        "of the next, use the complete version.\n"
        "- Output ONLY the merged transcription. No commentary."
    )


# ---------------------------------------------------------------------------
# Extraction prompts (Phase 4)
# ---------------------------------------------------------------------------

def extraction_system_prompt(entity_reference: str = "", disambiguation_rules: str = "") -> str:
    """Build a domain-aware system prompt for structured extraction from
    Nahuatl / colonial Mesoamerican texts."""

    entity_block = ""
    if entity_reference:
        entity_block = f"\n\n{entity_reference}"

    rules_block = ""
    if disambiguation_rules:
        rules_block = f"\n\n{disambiguation_rules}"

    return f"""\
You extract structured data from text, with specialization in colonial-era \
Mesoamerican and Nahuatl documents.

You have expert knowledge of Aztec/Nahua history, culture, and naming conventions. \
You understand the difference between ethnic groups, places, deities, titles, and \
individuals in Mesoamerican contexts.
{entity_block}{rules_block}

OUTPUT RULES:
- Return ONLY valid JSON. No markdown code fences, no commentary.
- Classify each entity with the correct type based on context and the reference above.
- When in doubt between two classifications, prefer the one supported by the \
disambiguation rules.
- If the text contains no extractable entities for a category, use an empty list [].
- Preserve original spelling of names from the source text.\
"""


def extraction_user_prompt(
    text: str,
    instruction: str,
    schema_hint: str = "",
) -> str:
    """Build the user prompt for an extraction call."""
    parts = [f"INSTRUCTION:\n{instruction.strip()}"]

    if schema_hint:
        parts.append(f"SCHEMA HINT (structure your output like this):\n{schema_hint.strip()}")

    parts.append(f"TEXT:\n{text.strip()}")

    return "\n\n".join(parts)
