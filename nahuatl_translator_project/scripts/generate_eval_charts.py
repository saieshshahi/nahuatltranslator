#!/usr/bin/env python3
"""Generate comprehensive evaluation charts for the Nahuatl translation system.

Run from project root:
    python scripts/generate_eval_charts.py

Outputs all charts to eval_charts/ directory.
Does NOT require an OpenAI API key for static analysis charts.
Charts requiring live translation (BLEU/chrF/etc.) will prompt before running.
"""

import json
import math
import os
import sys
import re
from collections import Counter, defaultdict
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Project imports
from webapp.evaluation import (
    compute_bleu, compute_chrf, compute_meteor, compute_ter,
    normalize_nahuatl, load_eval_history,
)
from webapp.prompts import (
    translation_system_prompt, translation_variants_system_prompt,
    transcription_system_prompt, extraction_system_prompt,
    FEWSHOT_TRANSLATION_EXAMPLES, NEGATIVE_TRANSLATION_EXAMPLES,
    _NAHUATL_CORE, _NAHUATL_GENERATION_RULES, _NAHUATL_INTERPRETATION_RULES,
    _SPANISH_SOURCE_ADDENDUM, _SPANISH_TARGET_ADDENDUM,
)
from webapp.spanish_filter import detect_spanish_in_output, is_spanish
from webapp.prompt_guard import scan_input, _HIGH_RISK_PATTERNS, _LOW_RISK_PATTERNS
from webapp.entities import KNOWN_ENTITIES, DISAMBIGUATION_RULES

# Output directory
OUT_DIR = PROJECT_ROOT / "eval_charts"
OUT_DIR.mkdir(exist_ok=True)

# Style
plt.rcParams.update({
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor": "#16213e",
    "axes.edgecolor": "#e0e0e0",
    "axes.labelcolor": "#e0e0e0",
    "text.color": "#e0e0e0",
    "xtick.color": "#e0e0e0",
    "ytick.color": "#e0e0e0",
    "grid.color": "#2a2a4a",
    "figure.figsize": (12, 7),
    "font.size": 11,
})

COLORS = ["#00d2ff", "#7b2ff7", "#ff6b6b", "#feca57", "#48dbfb",
          "#ff9ff3", "#54a0ff", "#5f27cd", "#01a3a4", "#f368e0",
          "#c8d6e5", "#ff6348", "#2ed573", "#ffa502", "#70a1ff"]


def save(name: str):
    path = OUT_DIR / f"{name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {path}")


# ===================================================================
# 1. GOLDEN PAIR ANALYSIS (static — no API calls)
# ===================================================================

def load_golden():
    path = PROJECT_ROOT / "tests" / "golden_translations.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["pairs"]


def chart_golden_category_distribution(pairs):
    """Bar chart: number of golden pairs per category."""
    cats = Counter(p["category"] for p in pairs)
    cats_sorted = cats.most_common()
    labels = [c for c, _ in cats_sorted]
    counts = [n for _, n in cats_sorted]

    fig, ax = plt.subplots()
    bars = ax.barh(labels, counts, color=COLORS[:len(labels)])
    ax.set_xlabel("Number of Pairs")
    ax.set_title("Golden Translation Pairs by Category")
    ax.invert_yaxis()
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                str(count), va="center", fontsize=10)
    plt.tight_layout()
    save("01_golden_category_distribution")


def chart_golden_direction_distribution(pairs):
    """Pie chart: golden pairs by translation direction."""
    dirs = Counter(f"{p['src_lang']}→{p['tgt_lang']}" for p in pairs)
    labels = list(dirs.keys())
    sizes = list(dirs.values())

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.0f%%",
        colors=COLORS[:len(labels)], startangle=90,
        textprops={"fontsize": 12}
    )
    for t in autotexts:
        t.set_fontsize(11)
        t.set_color("white")
    ax.set_title("Golden Pairs by Translation Direction")
    save("02_golden_direction_distribution")


def chart_golden_source_lengths(pairs):
    """Histogram: source text lengths (word count) in golden pairs."""
    lengths = [len(p["src"].split()) for p in pairs]

    fig, ax = plt.subplots()
    ax.hist(lengths, bins=range(1, max(lengths) + 2), color=COLORS[0],
            edgecolor="#1a1a2e", alpha=0.85)
    ax.set_xlabel("Source Text Length (words)")
    ax.set_ylabel("Number of Pairs")
    ax.set_title("Distribution of Source Text Lengths in Golden Pairs")
    ax.axvline(np.mean(lengths), color=COLORS[3], linestyle="--",
               label=f"Mean: {np.mean(lengths):.1f}")
    ax.legend()
    save("03_golden_source_lengths")


def chart_golden_target_lengths(pairs):
    """Histogram: target text lengths."""
    lengths = [len(p["tgt"].split()) for p in pairs]

    fig, ax = plt.subplots()
    ax.hist(lengths, bins=range(1, max(lengths) + 2), color=COLORS[1],
            edgecolor="#1a1a2e", alpha=0.85)
    ax.set_xlabel("Target Text Length (words)")
    ax.set_ylabel("Number of Pairs")
    ax.set_title("Distribution of Target Text Lengths in Golden Pairs")
    ax.axvline(np.mean(lengths), color=COLORS[3], linestyle="--",
               label=f"Mean: {np.mean(lengths):.1f}")
    ax.legend()
    save("04_golden_target_lengths")


def chart_golden_length_ratio(pairs):
    """Scatter: source length vs target length, colored by direction."""
    fig, ax = plt.subplots()
    dir_colors = {}
    for p in pairs:
        d = f"{p['src_lang']}→{p['tgt_lang']}"
        if d not in dir_colors:
            dir_colors[d] = COLORS[len(dir_colors) % len(COLORS)]
        ax.scatter(len(p["src"].split()), len(p["tgt"].split()),
                   color=dir_colors[d], s=50, alpha=0.7)

    patches = [mpatches.Patch(color=c, label=d) for d, c in dir_colors.items()]
    ax.legend(handles=patches, loc="upper left")
    ax.set_xlabel("Source Length (words)")
    ax.set_ylabel("Target Length (words)")
    ax.set_title("Source vs Target Length by Direction")
    # Diagonal reference line
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], "--", color="#555", alpha=0.5, label="1:1")
    save("05_golden_length_ratio")


# ===================================================================
# 2. PROMPT ARCHITECTURE ANALYSIS (static)
# ===================================================================

def _count_tokens_approx(text: str) -> int:
    """Rough token count (~0.75 words per token for English/Nahuatl)."""
    return int(len(text.split()) / 0.75)


def chart_prompt_token_counts():
    """Bar chart: system prompt token counts by direction and endpoint."""
    data = {}
    for src, tgt in [("en", "nah"), ("nah", "en"), ("es", "nah"), ("nah", "es")]:
        label = f"{src}→{tgt}"
        single = translation_system_prompt(src, tgt, "Unknown")
        variant = translation_variants_system_prompt(src, tgt, "Unknown", k=3)
        data[label] = {
            "single": _count_tokens_approx(single),
            "variants": _count_tokens_approx(variant),
        }

    # Add transcription and extraction
    trans_es = transcription_system_prompt("es")
    trans_nah = transcription_system_prompt("nah")
    from webapp.entities import format_entity_reference
    extract = extraction_system_prompt(
        entity_reference=format_entity_reference(max_per_type=8),
        disambiguation_rules=DISAMBIGUATION_RULES,
    )

    labels = list(data.keys()) + ["transcribe-es", "transcribe-nah", "extraction"]
    single_vals = [data[l]["single"] for l in data] + [
        _count_tokens_approx(trans_es),
        _count_tokens_approx(trans_nah),
        _count_tokens_approx(extract),
    ]
    variant_vals = [data[l]["variants"] for l in data] + [0, 0, 0]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 7))
    bars1 = ax.bar(x - width/2, single_vals, width, label="Single Translation",
                   color=COLORS[0])
    bars2 = ax.bar(x + width/2, variant_vals, width, label="Variant (k=3)",
                   color=COLORS[1])

    # Cache threshold line
    ax.axhline(y=1024, color=COLORS[2], linestyle="--", alpha=0.7,
               label="OpenAI Cache Threshold (1024)")

    ax.set_xlabel("Endpoint / Direction")
    ax.set_ylabel("Approx Token Count")
    ax.set_title("System Prompt Size by Endpoint and Direction")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend()

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                str(int(bar.get_height())), ha="center", fontsize=9)
    for bar in bars2:
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    str(int(bar.get_height())), ha="center", fontsize=9)

    plt.tight_layout()
    save("06_prompt_token_counts")


def chart_prompt_component_breakdown():
    """Stacked bar: what components make up each prompt direction."""
    components = {
        "en→nah": {
            "Core guidance": len(_NAHUATL_CORE.split()),
            "Generation rules": len(_NAHUATL_GENERATION_RULES.split()),
            "Few-shot examples": sum(
                len(f"{inp} {out}".split())
                for s, t, inp, out in FEWSHOT_TRANSLATION_EXAMPLES
                if s == "en" and t == "nah"
            ),
            "Negative examples": sum(
                len(f"{src} {wrong} {correct} {reason}".split())
                for s, t, src, wrong, correct, reason in NEGATIVE_TRANSLATION_EXAMPLES
                if s == "en"
            ),
        },
        "es→nah": {
            "Core guidance": len(_NAHUATL_CORE.split()),
            "Generation rules": len(_NAHUATL_GENERATION_RULES.split()),
            "Spanish addendum": len(_SPANISH_SOURCE_ADDENDUM.split()),
            "Few-shot examples": sum(
                len(f"{inp} {out}".split())
                for s, t, inp, out in FEWSHOT_TRANSLATION_EXAMPLES
                if s == "es" and t == "nah"
            ),
            "Negative examples": sum(
                len(f"{src} {wrong} {correct} {reason}".split())
                for s, t, src, wrong, correct, reason in NEGATIVE_TRANSLATION_EXAMPLES
                if s == "es"
            ),
        },
        "nah→en": {
            "Core guidance": len(_NAHUATL_CORE.split()),
            "Interpretation rules": len(_NAHUATL_INTERPRETATION_RULES.split()),
            "Few-shot examples": sum(
                len(f"{inp} {out}".split())
                for s, t, inp, out in FEWSHOT_TRANSLATION_EXAMPLES
                if s == "nah" and t == "en"
            ),
        },
        "nah→es": {
            "Core guidance": len(_NAHUATL_CORE.split()),
            "Interpretation rules": len(_NAHUATL_INTERPRETATION_RULES.split()),
            "Spanish target addendum": len(_SPANISH_TARGET_ADDENDUM.split()),
            "Few-shot examples": sum(
                len(f"{inp} {out}".split())
                for s, t, inp, out in FEWSHOT_TRANSLATION_EXAMPLES
                if s == "nah" and t == "es"
            ),
        },
    }

    all_comp_names = sorted({k for d in components.values() for k in d})
    directions = list(components.keys())

    fig, ax = plt.subplots(figsize=(12, 7))
    bottoms = np.zeros(len(directions))
    for ci, comp in enumerate(all_comp_names):
        vals = [components[d].get(comp, 0) for d in directions]
        ax.bar(directions, vals, bottom=bottoms, label=comp,
               color=COLORS[ci % len(COLORS)])
        bottoms += np.array(vals)

    ax.set_ylabel("Word Count")
    ax.set_title("Prompt Component Breakdown by Direction")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    save("07_prompt_component_breakdown")


# ===================================================================
# 3. FEW-SHOT & NEGATIVE EXAMPLE ANALYSIS (static)
# ===================================================================

def chart_fewshot_coverage():
    """Grouped bar: few-shot example counts per direction."""
    pos_counts = Counter(f"{s}→{t}" for s, t, _, _ in FEWSHOT_TRANSLATION_EXAMPLES)
    neg_counts = Counter(f"{s}→{t}" for s, t, _, _, _, _ in NEGATIVE_TRANSLATION_EXAMPLES)

    all_dirs = sorted(set(list(pos_counts.keys()) + list(neg_counts.keys())))
    pos_vals = [pos_counts.get(d, 0) for d in all_dirs]
    neg_vals = [neg_counts.get(d, 0) for d in all_dirs]

    x = np.arange(len(all_dirs))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(x - width/2, pos_vals, width, label="Positive Examples", color=COLORS[4])
    ax.bar(x + width/2, neg_vals, width, label="Negative Examples", color=COLORS[2])

    ax.set_xlabel("Direction")
    ax.set_ylabel("Count")
    ax.set_title("Few-Shot Example Coverage by Direction")
    ax.set_xticks(x)
    ax.set_xticklabels(all_dirs)
    ax.legend()

    for i, (p, n) in enumerate(zip(pos_vals, neg_vals)):
        ax.text(i - width/2, p + 0.2, str(p), ha="center", fontsize=10)
        ax.text(i + width/2, n + 0.2, str(n), ha="center", fontsize=10)

    plt.tight_layout()
    save("08_fewshot_coverage")


def chart_negative_example_error_types():
    """Horizontal bar: distribution of error types in negative examples."""
    types = Counter(reason.split("—")[0].strip()
                    for _, _, _, _, _, reason in NEGATIVE_TRANSLATION_EXAMPLES)
    types_sorted = types.most_common()

    fig, ax = plt.subplots(figsize=(12, 6))
    labels = [t for t, _ in types_sorted]
    counts = [c for _, c in types_sorted]

    ax.barh(labels, counts, color=COLORS[2])
    ax.set_xlabel("Count")
    ax.set_title("Negative Example Error Types")
    ax.invert_yaxis()
    for i, c in enumerate(counts):
        ax.text(c + 0.1, i, str(c), va="center")
    plt.tight_layout()
    save("09_negative_example_error_types")


# ===================================================================
# 4. SPANISH CONTAMINATION ANALYSIS (static)
# ===================================================================

def chart_golden_spanish_contamination(pairs):
    """Bar chart: Spanish contamination in golden reference translations."""
    contaminated = []
    clean = []
    for p in pairs:
        if p["tgt_lang"] == "nah":
            found = detect_spanish_in_output(p["tgt"])
            if found:
                contaminated.append((p["tgt"], found))
            else:
                clean.append(p)

    nah_targets = [p for p in pairs if p["tgt_lang"] == "nah"]
    if not nah_targets:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Pie: clean vs contaminated
    ax1.pie([len(clean), len(contaminated)],
            labels=["Clean", "Spanish Detected"],
            autopct="%1.0f%%",
            colors=[COLORS[4], COLORS[2]],
            startangle=90, textprops={"fontsize": 12})
    ax1.set_title("Golden Nahuatl Targets: Spanish Contamination")

    # Bar: which Spanish words appear
    if contaminated:
        all_spanish = Counter()
        for _, words in contaminated:
            for w in words:
                all_spanish[w] += 1
        top = all_spanish.most_common(15)
        words = [w for w, _ in top]
        counts = [c for _, c in top]
        ax2.barh(words, counts, color=COLORS[2])
        ax2.invert_yaxis()
        ax2.set_xlabel("Occurrences")
        ax2.set_title("Most Common Spanish Words in Golden Nahuatl Targets")
    else:
        ax2.text(0.5, 0.5, "No contamination found", ha="center", va="center",
                 fontsize=14, transform=ax2.transAxes)
        ax2.set_title("Spanish Words Found")

    plt.tight_layout()
    save("10_spanish_contamination_golden")


# ===================================================================
# 5. CORPUS ANALYSIS (static)
# ===================================================================

def chart_corpus_stats():
    """Bar chart: corpus and vocabulary statistics."""
    try:
        from webapp.corpus import get_corpus
        corpus = get_corpus()
        if not corpus.loaded:
            print("  [SKIP] Corpus not loaded")
            return
    except Exception:
        print("  [SKIP] Corpus not available")
        return

    entries = corpus.entries
    if not entries:
        return

    # Length distributions (ParallelEntry dataclass: .english, .nahuatl, .source)
    en_lengths = [len(e.english.split()) for e in entries if e.english]
    nah_lengths = [len(e.nahuatl.split()) for e in entries if e.nahuatl]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # English lengths
    axes[0, 0].hist(en_lengths, bins=30, color=COLORS[0], edgecolor="#1a1a2e", alpha=0.8)
    axes[0, 0].set_title("English Sentence Lengths")
    axes[0, 0].set_xlabel("Words")
    axes[0, 0].axvline(np.mean(en_lengths), color=COLORS[3], linestyle="--",
                        label=f"Mean: {np.mean(en_lengths):.1f}")
    axes[0, 0].legend()

    # Nahuatl lengths
    axes[0, 1].hist(nah_lengths, bins=30, color=COLORS[1], edgecolor="#1a1a2e", alpha=0.8)
    axes[0, 1].set_title("Nahuatl Sentence Lengths")
    axes[0, 1].set_xlabel("Words")
    axes[0, 1].axvline(np.mean(nah_lengths), color=COLORS[3], linestyle="--",
                        label=f"Mean: {np.mean(nah_lengths):.1f}")
    axes[0, 1].legend()

    # Length ratio
    ratios = []
    for e in entries:
        en = e.english or ""
        nah = e.nahuatl or ""
        if en and nah:
            en_len = len(en.split())
            nah_len = len(nah.split())
            if en_len > 0:
                ratios.append(nah_len / en_len)
    axes[1, 0].hist(ratios, bins=30, color=COLORS[4], edgecolor="#1a1a2e", alpha=0.8)
    axes[1, 0].set_title("Nahuatl/English Length Ratio")
    axes[1, 0].set_xlabel("Ratio")
    axes[1, 0].axvline(np.mean(ratios), color=COLORS[3], linestyle="--",
                        label=f"Mean: {np.mean(ratios):.2f}")
    axes[1, 0].legend()

    # Spanish contamination in corpus
    contamination_rates = []
    sample = entries[:500]  # sample for speed
    for e in sample:
        nah = e.nahuatl or ""
        if nah:
            found = detect_spanish_in_output(nah)
            contamination_rates.append(len(found))

    axes[1, 1].hist(contamination_rates, bins=range(0, max(contamination_rates or [1]) + 2),
                    color=COLORS[2], edgecolor="#1a1a2e", alpha=0.8)
    axes[1, 1].set_title("Spanish Words per Corpus Entry (sample)")
    axes[1, 1].set_xlabel("Spanish Words Found")

    plt.suptitle(f"Parallel Corpus Analysis ({len(entries)} entries)", fontsize=14)
    plt.tight_layout()
    save("11_corpus_analysis")


def chart_corpus_book_distribution():
    """Bar chart: corpus entries by biblical book."""
    try:
        from webapp.corpus import get_corpus
        corpus = get_corpus()
        if not corpus.loaded:
            return
    except Exception:
        return

    books = Counter(getattr(e, "book", getattr(e, "source", "unknown")) for e in corpus.entries)
    top = books.most_common(20)

    fig, ax = plt.subplots(figsize=(14, 7))
    labels = [b for b, _ in top]
    counts = [c for _, c in top]
    ax.barh(labels, counts, color=COLORS[0])
    ax.invert_yaxis()
    ax.set_xlabel("Number of Parallel Entries")
    ax.set_title("Corpus Entries by Source Book (Top 20)")
    plt.tight_layout()
    save("12_corpus_book_distribution")


# ===================================================================
# 6. CLICS VOCABULARY ANALYSIS (static)
# ===================================================================

def chart_clics_vocab():
    """Analyze the CLICS Nahuatl vocabulary data."""
    path = PROJECT_ROOT / "data" / "clics_nahuatl_vocab.json"
    if not path.exists():
        print("  [SKIP] CLICS vocab not found")
        return

    with open(path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    # Word length distribution
    nah_lengths = [len(e.get("nahuatl", "")) for e in vocab if e.get("nahuatl")]
    en_lengths = [len(e.get("english", "").split()) for e in vocab if e.get("english")]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.hist(nah_lengths, bins=30, color=COLORS[0], edgecolor="#1a1a2e", alpha=0.8)
    ax1.set_title("Nahuatl Word Lengths (chars)")
    ax1.set_xlabel("Characters")
    ax1.axvline(np.mean(nah_lengths), color=COLORS[3], linestyle="--",
                label=f"Mean: {np.mean(nah_lengths):.1f}")
    ax1.legend()

    ax2.hist(en_lengths, bins=range(1, max(en_lengths or [5]) + 2),
             color=COLORS[1], edgecolor="#1a1a2e", alpha=0.8)
    ax2.set_title("English Gloss Lengths (words)")
    ax2.set_xlabel("Words")

    plt.suptitle(f"CLICS Nahuatl Vocabulary ({len(vocab)} entries)", fontsize=14)
    plt.tight_layout()
    save("13_clics_vocabulary")


# ===================================================================
# 7. ENTITY TAXONOMY ANALYSIS (static)
# ===================================================================

def chart_entity_taxonomy():
    """Bar chart: entity counts by type in the pre-classified database."""
    type_counts = Counter()
    for entity in KNOWN_ENTITIES:
        type_counts[entity.get("type", "unknown")] += 1

    fig, ax = plt.subplots()
    labels = [t for t, _ in type_counts.most_common()]
    counts = [c for _, c in type_counts.most_common()]

    bars = ax.bar(labels, counts, color=COLORS[:len(labels)])
    ax.set_ylabel("Count")
    ax.set_title("Pre-Classified Entity Database by Type")
    ax.set_xticklabels(labels, rotation=30, ha="right")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(count), ha="center", fontsize=10)
    plt.tight_layout()
    save("14_entity_taxonomy")


# ===================================================================
# 8. PROMPT INJECTION COVERAGE (static)
# ===================================================================

def chart_injection_pattern_coverage():
    """Bar chart: prompt injection pattern categories and counts."""
    high_cats = Counter(label for _, label in _HIGH_RISK_PATTERNS)
    low_cats = Counter(label for _, label in _LOW_RISK_PATTERNS)

    all_cats = sorted(set(list(high_cats.keys()) + list(low_cats.keys())))
    high_vals = [high_cats.get(c, 0) for c in all_cats]
    low_vals = [low_cats.get(c, 0) for c in all_cats]

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(all_cats))
    width = 0.35

    ax.barh(x - width/2, high_vals, width, label="High Risk (blocked)",
            color=COLORS[2])
    ax.barh(x + width/2, low_vals, width, label="Low Risk (flagged)",
            color=COLORS[3])

    ax.set_yticks(x)
    ax.set_yticklabels(all_cats, fontsize=9)
    ax.set_xlabel("Number of Patterns")
    ax.set_title("Prompt Injection Detection: Pattern Coverage")
    ax.legend()
    ax.invert_yaxis()
    plt.tight_layout()
    save("15_injection_pattern_coverage")


# ===================================================================
# 9. CROSS-METRIC GOLDEN PAIR ANALYSIS (static — no API needed)
# ===================================================================

def chart_golden_self_similarity(pairs):
    """Heatmap: how similar are golden pairs within each category?
    Uses chrF between all reference targets in the same direction.
    """
    # Only nah targets
    nah_pairs = [p for p in pairs if p["tgt_lang"] == "nah"]
    cats = sorted(set(p["category"] for p in nah_pairs))

    intra_scores = {}
    for cat in cats:
        cat_pairs = [p for p in nah_pairs if p["category"] == cat]
        if len(cat_pairs) < 2:
            intra_scores[cat] = 0.0
            continue
        scores = []
        for i in range(len(cat_pairs)):
            for j in range(i + 1, len(cat_pairs)):
                scores.append(compute_chrf(cat_pairs[i]["tgt"], cat_pairs[j]["tgt"]))
        intra_scores[cat] = np.mean(scores) if scores else 0.0

    fig, ax = plt.subplots()
    labels = list(intra_scores.keys())
    values = list(intra_scores.values())
    bars = ax.bar(labels, values, color=COLORS[5])
    ax.set_ylabel("Average Intra-Category chrF")
    ax.set_title("Reference Translation Similarity Within Categories")
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", fontsize=9)
    plt.tight_layout()
    save("16_golden_self_similarity")


def chart_golden_vocabulary_overlap(pairs):
    """Venn-style bar: vocabulary overlap between directions."""
    dir_vocabs = defaultdict(set)
    for p in pairs:
        d = f"{p['src_lang']}→{p['tgt_lang']}"
        dir_vocabs[d].update(p["tgt"].lower().split())

    dirs = sorted(dir_vocabs.keys())
    if len(dirs) < 2:
        return

    fig, ax = plt.subplots(figsize=(14, 7))

    # Calculate pairwise Jaccard similarity
    matrix = np.zeros((len(dirs), len(dirs)))
    for i, d1 in enumerate(dirs):
        for j, d2 in enumerate(dirs):
            inter = len(dir_vocabs[d1] & dir_vocabs[d2])
            union = len(dir_vocabs[d1] | dir_vocabs[d2])
            matrix[i][j] = inter / union if union > 0 else 0

    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(dirs)))
    ax.set_yticks(range(len(dirs)))
    ax.set_xticklabels(dirs, rotation=30, ha="right")
    ax.set_yticklabels(dirs)
    ax.set_title("Target Vocabulary Overlap (Jaccard Similarity) Between Directions")
    plt.colorbar(im, ax=ax, label="Jaccard Similarity")

    for i in range(len(dirs)):
        for j in range(len(dirs)):
            ax.text(j, i, f"{matrix[i][j]:.2f}", ha="center", va="center",
                    fontsize=10, color="black" if matrix[i][j] > 0.5 else "white")

    plt.tight_layout()
    save("17_golden_vocabulary_overlap")


# ===================================================================
# 10. MORPHOLOGICAL COMPLEXITY (static)
# ===================================================================

def chart_nahuatl_morphological_complexity(pairs):
    """Bar chart: average word length in Nahuatl targets by category.
    Proxy for morphological complexity (agglutinative = longer words).
    """
    cat_lengths = defaultdict(list)
    for p in pairs:
        if p["tgt_lang"] == "nah":
            words = p["tgt"].split()
            avg_word_len = np.mean([len(w) for w in words]) if words else 0
            cat_lengths[p["category"]].append(avg_word_len)
        elif p["src_lang"] == "nah":
            words = p["src"].split()
            avg_word_len = np.mean([len(w) for w in words]) if words else 0
            cat_lengths[p["category"]].append(avg_word_len)

    cats = sorted(cat_lengths.keys())
    means = [np.mean(cat_lengths[c]) for c in cats]
    stds = [np.std(cat_lengths[c]) for c in cats]

    fig, ax = plt.subplots()
    ax.bar(cats, means, yerr=stds, color=COLORS[6], capsize=3)
    ax.set_ylabel("Avg Characters per Nahuatl Word")
    ax.set_title("Nahuatl Morphological Complexity by Category")
    ax.set_xticklabels(cats, rotation=35, ha="right")
    plt.tight_layout()
    save("18_nahuatl_morphological_complexity")


# ===================================================================
# 11. COST ESTIMATION (static)
# ===================================================================

def chart_cost_estimates(pairs):
    """Bar chart: estimated translation cost by direction and model."""
    models = {
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-5": (1.25, 10.00),
    }

    # Estimate tokens per pair (rough: source words/0.75 for input, target/0.75 for output)
    dir_input_tokens = defaultdict(float)
    dir_output_tokens = defaultdict(float)
    dir_counts = Counter()

    for p in pairs:
        d = f"{p['src_lang']}→{p['tgt_lang']}"
        dir_input_tokens[d] += len(p["src"].split()) / 0.75
        dir_output_tokens[d] += len(p["tgt"].split()) / 0.75
        dir_counts[d] += 1

    # Add system prompt overhead (~1000 tokens per call)
    dirs = sorted(dir_counts.keys())

    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(dirs))
    width = 0.35

    for mi, (model, (inp_price, out_price)) in enumerate(models.items()):
        costs = []
        for d in dirs:
            total_input = dir_input_tokens[d] + (dir_counts[d] * 1000)  # + prompt overhead
            total_output = dir_output_tokens[d]
            cost = (total_input * inp_price + total_output * out_price) / 1_000_000
            costs.append(cost * 1000)  # convert to millicents for readability

        offset = (mi - 0.5) * width
        bars = ax.bar(x + offset, costs, width, label=model, color=COLORS[mi])
        for bar, cost in zip(bars, costs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"${cost/1000:.4f}", ha="center", fontsize=8, rotation=45)

    ax.set_xlabel("Direction")
    ax.set_ylabel("Estimated Cost (x $0.001)")
    ax.set_title("Estimated Translation Cost for Golden Pair Corpus")
    ax.set_xticks(x)
    ax.set_xticklabels(dirs)
    ax.legend()
    plt.tight_layout()
    save("19_cost_estimates")


# ===================================================================
# 12. SPANISH FILTER ANALYSIS (static)
# ===================================================================

def chart_spanish_filter_stats():
    """Analyze the Spanish filter word list and whitelist."""
    wordlist_path = PROJECT_ROOT / "data" / "spanish_common_words.txt"
    if not wordlist_path.exists():
        print("  [SKIP] Spanish word list not found")
        return

    with open(wordlist_path, "r", encoding="utf-8") as f:
        words = [line.strip().lower() for line in f if line.strip() and len(line.strip()) >= 3]

    lengths = [len(w) for w in words]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Word length distribution
    ax1.hist(lengths, bins=range(3, max(lengths) + 2), color=COLORS[0],
             edgecolor="#1a1a2e", alpha=0.8)
    ax1.set_title(f"Spanish Filter: Word Length Distribution ({len(words)} words)")
    ax1.set_xlabel("Word Length (chars)")
    ax1.set_ylabel("Count")

    # First-letter distribution
    first_letters = Counter(w[0] for w in words)
    top_letters = first_letters.most_common(15)
    letters = [l for l, _ in top_letters]
    counts = [c for _, c in top_letters]
    ax2.bar(letters, counts, color=COLORS[1])
    ax2.set_title("Spanish Words by First Letter (Top 15)")
    ax2.set_xlabel("First Letter")
    ax2.set_ylabel("Count")

    plt.suptitle("Spanish Contamination Filter Analysis", fontsize=14)
    plt.tight_layout()
    save("20_spanish_filter_stats")


# ===================================================================
# 13. EVALUATION HISTORY (if available)
# ===================================================================

def chart_eval_history():
    """Line charts from saved evaluation runs (if any exist)."""
    history = load_eval_history()
    if not history:
        print("  [SKIP] No evaluation history found")
        return

    timestamps = [h.get("timestamp", "") for h in history]
    bleus = [h.get("aggregate", {}).get("bleu", 0) for h in history]
    chrfs = [h.get("aggregate", {}).get("chrf", 0) for h in history]
    meteors = [h.get("aggregate", {}).get("meteor", 0) for h in history]
    ters = [h.get("aggregate", {}).get("ter", 0) for h in history]

    x = range(len(history))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    ax1.plot(x, bleus, "o-", label="BLEU", color=COLORS[0])
    ax1.plot(x, chrfs, "s-", label="chrF", color=COLORS[1])
    ax1.plot(x, meteors, "^-", label="METEOR", color=COLORS[4])
    ax1.set_ylabel("Score (higher = better)")
    ax1.set_title("Translation Quality Over Evaluation Runs")
    ax1.legend()
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    ax2.plot(x, ters, "D-", label="TER", color=COLORS[2])
    ax2.set_ylabel("TER (lower = better)")
    ax2.set_xlabel("Evaluation Run")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save("21_eval_history_trends")


# ===================================================================
# 14. NAHUATL CHARACTER FREQUENCY (static)
# ===================================================================

def chart_nahuatl_char_frequency(pairs):
    """Bar chart: character frequency in Nahuatl text from golden pairs."""
    nah_text = ""
    for p in pairs:
        if p["tgt_lang"] == "nah":
            nah_text += p["tgt"].lower() + " "
        if p["src_lang"] == "nah":
            nah_text += p["src"].lower() + " "

    chars = Counter(c for c in nah_text if c.isalpha())
    top = chars.most_common(25)

    fig, ax = plt.subplots(figsize=(14, 6))
    labels = [c for c, _ in top]
    counts = [n for _, n in top]
    ax.bar(labels, counts, color=COLORS[0])
    ax.set_xlabel("Character")
    ax.set_ylabel("Frequency")
    ax.set_title("Character Frequency in Nahuatl Golden Pair Text")
    plt.tight_layout()
    save("22_nahuatl_char_frequency")


# ===================================================================
# 15. NAHUATL BIGRAM FREQUENCY (static)
# ===================================================================

def chart_nahuatl_bigrams(pairs):
    """Bar chart: most common character bigrams in Nahuatl text."""
    nah_text = ""
    for p in pairs:
        if p["tgt_lang"] == "nah":
            nah_text += p["tgt"].lower() + " "
        if p["src_lang"] == "nah":
            nah_text += p["src"].lower() + " "

    # Character bigrams (only alphabetic)
    clean = re.sub(r"[^a-záéíóúñü]", " ", nah_text)
    words = clean.split()
    bigrams = Counter()
    for word in words:
        for i in range(len(word) - 1):
            bigrams[word[i:i+2]] += 1

    top = bigrams.most_common(25)

    fig, ax = plt.subplots(figsize=(14, 6))
    labels = [b for b, _ in top]
    counts = [n for _, n in top]
    ax.bar(labels, counts, color=COLORS[1])
    ax.set_xlabel("Bigram")
    ax.set_ylabel("Frequency")
    ax.set_title("Most Common Character Bigrams in Nahuatl Text")
    ax.set_xticklabels(labels, fontsize=9)
    plt.tight_layout()
    save("23_nahuatl_bigrams")


# ===================================================================
# 16. DIFFICULTY ESTIMATION (static)
# ===================================================================

def chart_estimated_difficulty(pairs):
    """Scatter: estimated translation difficulty based on source length
    and vocabulary rarity.
    """
    # Collect all source words for frequency baseline
    all_words = Counter()
    for p in pairs:
        all_words.update(p["src"].lower().split())

    fig, ax = plt.subplots()
    dir_colors = {}
    for p in pairs:
        d = f"{p['src_lang']}→{p['tgt_lang']}"
        if d not in dir_colors:
            dir_colors[d] = COLORS[len(dir_colors) % len(COLORS)]

        words = p["src"].lower().split()
        length = len(words)
        # Rarity = average inverse frequency
        rarity = np.mean([1 / all_words[w] for w in words]) if words else 0

        ax.scatter(length, rarity, color=dir_colors[d], s=50, alpha=0.7)

    patches = [mpatches.Patch(color=c, label=d) for d, c in dir_colors.items()]
    ax.legend(handles=patches, loc="upper right")
    ax.set_xlabel("Source Length (words)")
    ax.set_ylabel("Average Word Rarity (1/freq)")
    ax.set_title("Estimated Translation Difficulty")
    plt.tight_layout()
    save("24_estimated_difficulty")


# ===================================================================
# MAIN
# ===================================================================

def main():
    print(f"Generating evaluation charts -> {OUT_DIR}/\n")

    pairs = load_golden()
    print(f"Loaded {len(pairs)} golden pairs\n")

    print("--- Golden Pair Analysis ---")
    chart_golden_category_distribution(pairs)
    chart_golden_direction_distribution(pairs)
    chart_golden_source_lengths(pairs)
    chart_golden_target_lengths(pairs)
    chart_golden_length_ratio(pairs)

    print("\n--- Prompt Architecture ---")
    chart_prompt_token_counts()
    chart_prompt_component_breakdown()

    print("\n--- Few-Shot & Negative Examples ---")
    chart_fewshot_coverage()
    chart_negative_example_error_types()

    print("\n--- Spanish Contamination ---")
    chart_golden_spanish_contamination(pairs)
    chart_spanish_filter_stats()

    print("\n--- Corpus Analysis ---")
    chart_corpus_stats()
    chart_corpus_book_distribution()

    print("\n--- Vocabulary ---")
    chart_clics_vocab()

    print("\n--- Entity Taxonomy ---")
    chart_entity_taxonomy()

    print("\n--- Security ---")
    chart_injection_pattern_coverage()

    print("\n--- Cross-Metric Analysis ---")
    chart_golden_self_similarity(pairs)
    chart_golden_vocabulary_overlap(pairs)
    chart_nahuatl_morphological_complexity(pairs)

    print("\n--- Cost & Difficulty ---")
    chart_cost_estimates(pairs)
    chart_estimated_difficulty(pairs)

    print("\n--- Nahuatl Linguistics ---")
    chart_nahuatl_char_frequency(pairs)
    chart_nahuatl_bigrams(pairs)

    print("\n--- Historical Evaluation Runs ---")
    chart_eval_history()

    print(f"\nDone. {len(list(OUT_DIR.glob('*.png')))} charts saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
