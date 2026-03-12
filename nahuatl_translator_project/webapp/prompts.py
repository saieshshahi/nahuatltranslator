"""Domain-specific prompt templates for Nahuatl translation, transcription, and extraction.

This module centralizes all system/user prompts so they can embed linguistic
knowledge about Nahuatl (morphology, orthography, dialects) and inject
contextual vocabulary from the parallel corpus.
"""

from __future__ import annotations

from typing import List, Tuple


# ---------------------------------------------------------------------------
# Language labels (shared with services.py)
# ---------------------------------------------------------------------------
LANG_LABELS = {
    "en": "English",
    "es": "Spanish",
    "nah": "Nahuatl",
}


def _label(code: str) -> str:
    return LANG_LABELS.get(code, code)


# ---------------------------------------------------------------------------
# Core Nahuatl linguistic reference (injected into system prompts)
# ---------------------------------------------------------------------------
NAHUATL_LINGUISTIC_CONTEXT = """\
KEY NAHUATL GRAMMAR & MORPHOLOGY RULES:
1. Nahuatl is an agglutinative, polysynthetic language. Words are built by
   attaching multiple prefixes and suffixes to a root (e.g., ni-k-chihua =
   "I make it"; ni- = "I", k- = "it", chihua = "make").
2. Verb structure: SUBJECT_PREFIX + OBJECT_PREFIX + ROOT + TENSE/ASPECT SUFFIX.
   Subject prefixes: ni- (I), ti- (you sg), ∅/Ø (he/she/it), ti- (we),
   an- (you pl), ∅ (they). Object prefixes: nech- (me), mitz- (you),
   k-/ki- (him/her/it), tech- (us), amech- (you pl), kin-/kim- (them).
3. Nahuatl has NO definite or indefinite articles ("the"/"a"). Do NOT insert
   articles in Nahuatl output. When translating TO English/Spanish, you may
   need to add articles that have no Nahuatl equivalent.
4. Default word order is often VSO (verb-subject-object) or VOS, though SVO
   also occurs. Do NOT force English SVO order onto Nahuatl sentences.
5. Possessive prefixes on nouns: no- (my), mo- (your), i- (his/her/its),
   to- (our), amo- (your pl), in- (their). E.g., no-cal = "my house".
6. Noun incorporation: objects can merge into the verb (e.g., ni-tlaxcal-chihua
   = "I make tortillas" where tlaxcal- is incorporated).
7. Locative suffixes: -co, -pan, -tlan, -can, -nahuac indicate location.
   E.g., Mex-i-co = "place of the Mexica", Tlax-cal-lan = "place of tortillas".
8. Absolutive suffix: standalone nouns often end in -tl, -tli, -li, or -in.
   When possessed or incorporated, this suffix drops. E.g., cal-li (house)
   → no-cal (my house).
9. Pluralization: common patterns include reduplication of first syllable
   (calli → cacalli "houses") or suffix changes (-meh, -tin).
10. Reverential/honorific forms add -tzin (e.g., tlacatl → tlacatzintli).

ORTHOGRAPHIC CONVENTIONS:
- Classical orthography: hu = /w/, qu = /k/ before e/i, cu = /kʷ/, tz = /ts/,
  x = /ʃ/, tl = /tɬ/, ch = /tʃ/.
- Modern/SEP orthography: w replaces hu, k replaces qu/c, s may replace z,
  j may replace h. Both systems are valid; match the input's convention.
- The saltillo (glottal stop) may appear as h, ʼ, or be omitted entirely.
- Colonial-era texts use Spanish orthography conventions (e.g., "hu" for /w/).

COMMON TRANSLATION PITFALLS:
- Do NOT transliterate Spanish/English words into Nahuatl. Nahuatl has its own
  vocabulary. Only use Spanish/Nahuatl loanwords when they are genuinely used
  (e.g., "Dios" is commonly used in modern Nahuatl for "God").
- Reduplication changes meaning (intensity, plurality). Preserve it accurately.
- Many Nahuatl words have entered English/Spanish (chocolate, tomato, avocado,
  coyote). Use the original Nahuatl forms when translating back: chocolatl,
  tomatl, ahuacatl, coyotl.
- Verbal aspect (perfective vs. imperfective) is critical in Nahuatl and does
  not map 1:1 to English tenses.

DIALECT/VARIETY NOTES:
- Central Nahuatl (Huasteca): uses tl extensively, most documented variety.
- Huasteca Nahuatl: may simplify tl → t, different vocabulary items.
- Guerrero Nahuatl: distinct phonological shifts.
- Pipil (Nawat): spoken in El Salvador, significant divergence from Central.
- When a variety/dialect is specified, adapt orthography and vocabulary accordingly.
- If no variety is specified, default to Central/Classical Nahuatl conventions.\
"""

# ---------------------------------------------------------------------------
# Few-shot translation examples
# ---------------------------------------------------------------------------
FEWSHOT_TRANSLATION_EXAMPLES = [
    ("en", "nah", "Good morning. How are you?",
     "Cualli tonalli. ¿Quen otitlathuil?"),
    ("en", "nah", "The water is very cold.",
     "Atl huel itztic."),
    ("en", "nah", "I want to eat tortillas.",
     "Nicnequi nitlaxcalcuas."),
    ("nah", "en", "Nejhua niPablo, nimechtlajtlanilia.",
     "I, Paul, ask you all."),
    ("es", "nah", "Buenos días. ¿Cómo estás?",
     "Cualli tonalli. ¿Quen tica?"),
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
    """Build a linguistically-informed system prompt for Nahuatl translation."""
    fewshot = _format_fewshot(src, tgt)
    variety_note = ""
    if variety and variety.lower() not in ("unknown", ""):
        variety_note = (
            f"\nThe user has specified the variety/dialect: {variety}. "
            f"Adapt your orthography and vocabulary to match this variety."
        )

    return f"""\
You are an expert translator specializing in Nahuatl (Aztec language). You have \
deep knowledge of Classical and modern Nahuatl grammar, morphology, and vocabulary.

{NAHUATL_LINGUISTIC_CONTEXT}
{variety_note}

TRANSLATION DIRECTION: {_label(src)} → {_label(tgt)}

REFERENCE TRANSLATION EXAMPLES:
{fewshot}

INSTRUCTIONS:
- Translate faithfully, preserving the meaning, tone, and intent of the original.
- Preserve proper names, numbers, and dates as-is (do not translate names).
- Output ONLY the translation in {_label(tgt)}. No commentary, no explanations.
- If a word has no direct equivalent, use the closest culturally appropriate term \
and do NOT fall back to Spanish/English unless the loanword is established in Nahuatl.
- When vocabulary reference entries are provided in the user message, use them as \
guides to anchor your translation but do not copy them blindly if context differs.\
"""


def translation_user_prompt(
    text: str,
    src: str,
    tgt: str,
    variety: str,
    reference_vocab: str = "",
    reference_sentences: str = "",
) -> str:
    """Build the user prompt for a translation call, with optional corpus context."""
    parts = []

    if reference_vocab:
        parts.append(
            f"REFERENCE VOCABULARY (use these as guides, not strict templates):\n"
            f"{reference_vocab}"
        )

    if reference_sentences:
        parts.append(
            f"REFERENCE PARALLEL SENTENCES (for context — adapt, do not copy):\n"
            f"{reference_sentences}"
        )

    parts.append(f"TEXT TO TRANSLATE:\n{text.strip()}")

    return "\n\n".join(parts)


def translation_variants_system_prompt(src: str, tgt: str, variety: str) -> str:
    """System prompt for variant generation — same core knowledge, slightly
    different instructions to encourage diverse phrasings."""
    base = translation_system_prompt(src, tgt, variety)
    return (
        base + "\n\n"
        "You are generating one of several possible translations. Produce a "
        "faithful translation; your phrasing may vary but must remain accurate. "
        "Prefer natural, idiomatic expression over literal word-for-word rendering."
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
