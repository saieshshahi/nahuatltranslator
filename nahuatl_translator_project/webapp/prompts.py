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
