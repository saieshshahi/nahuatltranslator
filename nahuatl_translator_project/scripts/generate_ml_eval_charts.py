#!/usr/bin/env python3
"""Generate ML evaluation charts — runs live translations against golden pairs.

Usage:
    python scripts/generate_ml_eval_charts.py              # run eval + chart (needs API key)
    python scripts/generate_ml_eval_charts.py --from-cache  # chart from last saved eval
    python scripts/generate_ml_eval_charts.py --simulate    # synthetic data, no API key needed

Requires OPENAI_API_KEY for live mode. Use --simulate for demo charts.
Outputs to eval_charts/ml/
"""

import json
import os
import sys
import argparse
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env if present (so user can just drop OPENAI_API_KEY in .env)
_env_file = PROJECT_ROOT / ".env"
if _env_file.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_file)
    except ImportError:
        # Manual fallback — no dependency needed
        with open(_env_file) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _, _v = _line.partition("=")
                    os.environ.setdefault(_k.strip(), _v.strip())

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from webapp.evaluation import (
    evaluate_golden_pairs, save_eval_result, load_eval_history,
    compute_bleu, compute_chrf, compute_meteor, compute_ter, compute_bertscore,
    EvalResult, PairResult, AggregateScores, HAS_BERTSCORE,
)
from webapp.services import openai_translate, openai_available

OUT_DIR = PROJECT_ROOT / "eval_charts" / "ml"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GOLDEN_PATH = PROJECT_ROOT / "tests" / "golden_translations.json"
CACHE_PATH = PROJECT_ROOT / "eval_charts" / "ml" / "_last_eval.json"

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

C = ["#00d2ff", "#7b2ff7", "#ff6b6b", "#feca57", "#48dbfb",
     "#ff9ff3", "#54a0ff", "#5f27cd", "#01a3a4", "#f368e0",
     "#c8d6e5", "#ff6348", "#2ed573", "#ffa502", "#70a1ff"]


def save_fig(name: str):
    path = OUT_DIR / f"{name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {path.name}")


# ===================================================================
# Run or load evaluation
# ===================================================================

def run_eval() -> dict:
    """Run live evaluation and return result dict."""
    if not openai_available():
        print("ERROR: OPENAI_API_KEY not set. Use --from-cache to chart saved results.")
        sys.exit(1)

    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        golden = json.load(f)["pairs"]

    model = os.getenv("OPENAI_TRANSLATE_MODEL", "gpt-5")
    print(f"Running evaluation on {len(golden)} golden pairs with {model}...")
    print("(This will make API calls — ~$0.07 at gpt-5 pricing)\n")

    result = evaluate_golden_pairs(
        translate_fn=openai_translate,
        golden_pairs=golden,
        model=model,
    )

    # Save for --from-cache
    result_dict = result.to_dict()
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)

    # Also save to eval history
    try:
        save_eval_result(result)
    except Exception:
        pass

    return result_dict


def load_cached() -> dict:
    """Load last cached evaluation result."""
    if not CACHE_PATH.exists():
        print("No cached eval found. Run without --from-cache first.")
        sys.exit(1)
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def simulate_eval() -> dict:
    """Generate realistic synthetic evaluation data from golden pairs.

    Uses the actual golden pairs with simulated translations and scores.
    Score distributions are modeled on typical LLM translation quality
    for low-resource languages (Nahuatl-class).
    """
    import random
    from datetime import datetime

    random.seed(42)  # reproducible

    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        golden = json.load(f)["pairs"]

    # Difficulty factors by category — some categories are harder for LLMs
    _CAT_DIFFICULTY = {
        "greeting": 0.15,        # easy — common phrases
        "greeting_es": 0.18,
        "vocabulary": 0.25,
        "conversational": 0.30,
        "conversational_es": 0.32,
        "phrase": 0.35,
        "phrase_es": 0.38,
        "religious": 0.40,
        "biblical": 0.45,
        "morphology": 0.50,      # hardest — agglutinative
    }

    # Direction difficulty offset (generating Nahuatl is harder than reading it)
    _DIR_OFFSET = {
        ("nah", "en"): -0.05,   # easiest — reading Nahuatl
        ("en", "nah"): 0.10,    # harder — generating Nahuatl
        ("es", "nah"): 0.12,    # hardest — Spanish→Nahuatl
    }

    # Common Spanish words that leak into nah output
    _SPANISH_LEAK_WORDS = ["de", "y", "que", "con", "por", "el", "la", "los", "las",
                           "en", "un", "una", "del", "al", "como", "es"]

    pair_results = []
    for p in golden:
        cat = p.get("category", "phrase")
        direction = (p["src_lang"], p["tgt_lang"])
        difficulty = _CAT_DIFFICULTY.get(cat, 0.35) + _DIR_OFFSET.get(direction, 0)

        # Base scores — higher difficulty → lower scores
        base_chrf = max(0.05, min(0.95, random.gauss(0.65 - difficulty, 0.15)))
        # BLEU correlates with chrF but is typically lower
        base_bleu = max(0.0, min(0.95, base_chrf * random.gauss(0.7, 0.12)))
        # METEOR is usually between BLEU and chrF
        base_meteor = max(0.05, min(0.95, (base_chrf + base_bleu) / 2 + random.gauss(0.05, 0.08)))
        # TER inversely correlates (high TER = bad)
        base_ter = max(0.0, min(2.0, (1 - base_chrf) * random.gauss(1.2, 0.3)))
        # BERTScore: semantic similarity, typically higher than BLEU/chrF (0.6-0.95 range)
        base_bertscore = max(0.1, min(0.98, base_chrf * random.gauss(1.15, 0.08)))

        # Latency: 400-2500ms range, longer texts take longer
        src_words = len(p["src"].split())
        latency = max(200, random.gauss(600 + src_words * 80, 200))

        # Spanish contamination: more likely when generating Nahuatl
        spanish_words = []
        if p["tgt_lang"] == "nah":
            contam_chance = 0.12 if direction == ("es", "nah") else 0.06
            if random.random() < contam_chance:
                n_leaked = random.randint(1, 3)
                spanish_words = random.sample(_SPANISH_LEAK_WORDS, min(n_leaked, len(_SPANISH_LEAK_WORDS)))

        # Simulate a predicted translation (perturb reference)
        tgt_words = p["tgt"].split()
        predicted_words = list(tgt_words)
        n_changes = max(0, int(len(tgt_words) * difficulty * random.gauss(0.5, 0.2)))
        for _ in range(min(n_changes, len(predicted_words))):
            idx = random.randint(0, len(predicted_words) - 1)
            # Replace with a plausible but wrong word
            predicted_words[idx] = predicted_words[idx][::-1] if len(predicted_words[idx]) > 2 else "x"
        predicted = " ".join(predicted_words)

        pair_results.append({
            "src": p["src"],
            "tgt_expected": p["tgt"],
            "tgt_predicted": predicted,
            "src_lang": p["src_lang"],
            "tgt_lang": p["tgt_lang"],
            "category": cat,
            "bleu": round(base_bleu, 4),
            "chrf": round(base_chrf, 4),
            "meteor": round(base_meteor, 4),
            "ter": round(base_ter, 4),
            "bertscore": round(base_bertscore, 4),
            "latency_ms": round(latency, 1),
            "spanish_words_found": spanish_words,
            "token_estimate": src_words * 3 + len(tgt_words) * 3,
        })

    # Compute aggregates
    def _agg(pairs_list):
        if not pairs_list:
            return {}
        n = len(pairs_list)
        contam = sum(1 for p in pairs_list if p["spanish_words_found"]) / n
        return {
            "bleu": round(sum(p["bleu"] for p in pairs_list) / n, 4),
            "chrf": round(sum(p["chrf"] for p in pairs_list) / n, 4),
            "meteor": round(sum(p["meteor"] for p in pairs_list) / n, 4),
            "ter": round(sum(p["ter"] for p in pairs_list) / n, 4),
            "bertscore": round(sum(p.get("bertscore", 0) for p in pairs_list) / n, 4),
            "count": n,
            "contamination_rate": round(contam, 4),
            "avg_latency_ms": round(sum(p["latency_ms"] for p in pairs_list) / n, 1),
        }

    # By direction
    by_dir = defaultdict(list)
    for p in pair_results:
        by_dir[f"{p['src_lang']}>{p['tgt_lang']}"].append(p)

    # By category
    by_cat = defaultdict(list)
    for p in pair_results:
        by_cat[p["category"]].append(p)

    result = {
        "model": "gpt-5 (simulated)",
        "timestamp": datetime.now().isoformat(),
        "aggregate": _agg(pair_results),
        "by_direction": {d: _agg(ps) for d, ps in by_dir.items()},
        "by_category": {c: _agg(ps) for c, ps in by_cat.items()},
        "pairs": pair_results,
    }

    # Save simulated cache
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Simulated eval saved to {CACHE_PATH.name}")

    return result


# ===================================================================
# Chart functions
# ===================================================================

def chart_aggregate_scores(data: dict):
    """Large bar chart: overall BLEU, chrF, METEOR, TER, BERTScore."""
    agg = data["aggregate"]
    metrics = ["BLEU", "chrF", "METEOR", "TER", "BERTScore"]
    values = [agg["bleu"], agg["chrf"], agg["meteor"], agg["ter"], agg.get("bertscore", 0)]
    colors = [C[0], C[1], C[4], C[2], C[5]]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(metrics, values, color=colors, width=0.45)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title(f"Overall Translation Quality — {data['model']} ({agg['count']} pairs)")
    ax.axhline(y=0.5, color="#555", linestyle="--", alpha=0.4)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", fontsize=13, fontweight="bold")

    # Add contamination rate
    ax.text(0.98, 0.95, f"Spanish contamination: {agg['contamination_rate']:.1%}",
            transform=ax.transAxes, ha="right", fontsize=10,
            color=C[2] if agg["contamination_rate"] > 0.1 else C[4])

    save_fig("01_aggregate_scores")


def chart_scores_by_direction(data: dict):
    """Grouped bar: BLEU, chrF, METEOR, TER, BERTScore per translation direction."""
    by_dir = data["by_direction"]
    dirs = sorted(by_dir.keys())
    metrics = ["bleu", "chrf", "meteor", "ter", "bertscore"]
    labels = ["BLEU", "chrF", "METEOR", "TER", "BERTScore"]
    colors = [C[0], C[1], C[4], C[2], C[5]]

    x = np.arange(len(dirs))
    width = 0.15

    fig, ax = plt.subplots(figsize=(14, 7))
    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        vals = [by_dir[d].get(metric, 0) for d in dirs]
        offset = (i - 2) * width
        bars = ax.bar(x + offset, vals, width, label=label, color=color)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", fontsize=8, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(dirs, fontsize=12)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.set_title("Translation Quality by Direction")
    ax.legend()
    ax.axhline(y=0.5, color="#555", linestyle="--", alpha=0.3)
    plt.tight_layout()
    save_fig("02_scores_by_direction")


def chart_scores_by_category(data: dict):
    """Grouped bar: scores per category."""
    by_cat = data["by_category"]
    cats = sorted(by_cat.keys(), key=lambda c: by_cat[c]["chrf"], reverse=True)
    metrics = ["bleu", "chrf", "meteor", "ter", "bertscore"]
    labels = ["BLEU", "chrF", "METEOR", "TER", "BERTScore"]
    colors = [C[0], C[1], C[4], C[2], C[5]]

    x = np.arange(len(cats))
    width = 0.15

    fig, ax = plt.subplots(figsize=(16, 7))
    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        vals = [by_cat[c].get(metric, 0) for c in cats]
        offset = (i - 2) * width
        ax.bar(x + offset, vals, width, label=label, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=35, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.set_title("Translation Quality by Category")
    ax.legend()
    plt.tight_layout()
    save_fig("03_scores_by_category")


def chart_per_pair_heatmap(data: dict):
    """Heatmap: per-pair scores across all 4 metrics."""
    pairs = data["pairs"]
    n = len(pairs)
    if n > 60:
        pairs = pairs[:60]  # cap for readability
        n = 60

    matrix = np.zeros((n, 5))
    ylabels = []
    for i, p in enumerate(pairs):
        matrix[i] = [p["bleu"], p["chrf"], p["meteor"], 1 - min(p["ter"], 1.0), p.get("bertscore", 0)]
        src_short = p["src"][:30] + "..." if len(p["src"]) > 30 else p["src"]
        ylabels.append(f"{p['src_lang']}→{p['tgt_lang']} | {src_short}")

    fig, ax = plt.subplots(figsize=(12, max(8, n * 0.25)))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(5))
    ax.set_xticklabels(["BLEU", "chrF", "METEOR", "1-TER", "BERTScore"])
    ax.set_yticks(range(n))
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_title("Per-Pair Score Heatmap (green = good)")
    plt.colorbar(im, ax=ax, shrink=0.5)
    plt.tight_layout()
    save_fig("04_per_pair_heatmap")


def chart_score_distributions(data: dict):
    """Histograms: distribution of each metric across all pairs."""
    pairs = data["pairs"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    metric_info = [
        ("bleu", "BLEU", C[0]),
        ("chrf", "chrF", C[1]),
        ("meteor", "METEOR", C[4]),
        ("ter", "TER", C[2]),
        ("bertscore", "BERTScore", C[5]),
    ]

    for ax, (metric, label, color) in zip(axes.flat, metric_info):
        vals = [p.get(metric, 0) for p in pairs]
        ax.hist(vals, bins=20, color=color, edgecolor="#1a1a2e", alpha=0.85)
        ax.axvline(np.mean(vals), color=C[3], linestyle="--",
                   label=f"Mean: {np.mean(vals):.3f}")
        ax.axvline(np.median(vals), color="#fff", linestyle=":",
                   label=f"Median: {np.median(vals):.3f}")
        ax.set_title(label)
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)

    # Hide the 6th subplot (2x3 grid, only 5 metrics)
    if len(axes.flat) > len(metric_info):
        for extra_ax in axes.flat[len(metric_info):]:
            extra_ax.set_visible(False)

    plt.suptitle("Score Distributions Across All Pairs", fontsize=14)
    plt.tight_layout()
    save_fig("05_score_distributions")


def chart_metric_correlations(data: dict):
    """Scatter matrix: pairwise metric correlations."""
    pairs = data["pairs"]
    metrics = ["bleu", "chrf", "meteor", "ter", "bertscore"]
    labels = ["BLEU", "chrF", "METEOR", "TER", "BERTScore"]

    fig, axes = plt.subplots(5, 5, figsize=(20, 20))
    for i, (mi, li) in enumerate(zip(metrics, labels)):
        for j, (mj, lj) in enumerate(zip(metrics, labels)):
            ax = axes[i][j]
            xi = [p.get(mj, 0) for p in pairs]
            yi = [p.get(mi, 0) for p in pairs]

            if i == j:
                ax.hist(xi, bins=15, color=C[i], edgecolor="#1a1a2e", alpha=0.8)
                ax.set_title(li, fontsize=10)
            else:
                ax.scatter(xi, yi, s=15, alpha=0.6, color=C[i])
                # Correlation coefficient
                if np.std(xi) > 0 and np.std(yi) > 0:
                    r = np.corrcoef(xi, yi)[0, 1]
                    ax.text(0.05, 0.9, f"r={r:.2f}", transform=ax.transAxes, fontsize=9)

            if i == len(metrics) - 1:
                ax.set_xlabel(lj, fontsize=9)
            if j == 0:
                ax.set_ylabel(li, fontsize=9)

    plt.suptitle("Metric Correlation Matrix", fontsize=14)
    plt.tight_layout()
    save_fig("06_metric_correlations")


def chart_latency(data: dict):
    """Histogram + boxplot of translation latency."""
    pairs = data["pairs"]
    latencies = [p["latency_ms"] for p in pairs if p["latency_ms"] > 0]

    if not latencies:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.hist(latencies, bins=20, color=C[0], edgecolor="#1a1a2e", alpha=0.85)
    ax1.axvline(np.mean(latencies), color=C[3], linestyle="--",
                label=f"Mean: {np.mean(latencies):.0f}ms")
    ax1.axvline(np.percentile(latencies, 95), color=C[2], linestyle="--",
                label=f"P95: {np.percentile(latencies, 95):.0f}ms")
    ax1.set_xlabel("Latency (ms)")
    ax1.set_ylabel("Count")
    ax1.set_title("Translation Latency Distribution")
    ax1.legend()

    # By direction
    dir_latencies = defaultdict(list)
    for p in pairs:
        d = f"{p['src_lang']}→{p['tgt_lang']}"
        if p["latency_ms"] > 0:
            dir_latencies[d].append(p["latency_ms"])

    dirs = sorted(dir_latencies.keys())
    bp = ax2.boxplot([dir_latencies[d] for d in dirs], tick_labels=dirs, patch_artist=True)
    for patch, color in zip(bp["boxes"], C):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_ylabel("Latency (ms)")
    ax2.set_title("Latency by Direction")

    plt.tight_layout()
    save_fig("07_latency")


def chart_score_vs_length(data: dict):
    """Scatter: chrF score vs source text length."""
    pairs = data["pairs"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, metric, label, color in zip(
        axes, ["chrf", "bleu"], ["chrF", "BLEU"], [C[1], C[0]]
    ):
        lengths = [len(p["src"].split()) for p in pairs]
        scores = [p[metric] for p in pairs]

        ax.scatter(lengths, scores, color=color, s=40, alpha=0.7)

        # Trend line
        if len(lengths) > 2:
            z = np.polyfit(lengths, scores, 1)
            poly = np.poly1d(z)
            x_line = np.linspace(min(lengths), max(lengths), 50)
            ax.plot(x_line, poly(x_line), "--", color=C[3], alpha=0.7,
                    label=f"Slope: {z[0]:.4f}")

        ax.set_xlabel("Source Length (words)")
        ax.set_ylabel(label)
        ax.set_title(f"{label} vs Source Length")
        ax.legend()

    plt.tight_layout()
    save_fig("08_score_vs_length")


def chart_spanish_contamination(data: dict):
    """Bar: contamination rate by direction + list of contaminated pairs."""
    by_dir = data["by_direction"]
    pairs = data["pairs"]

    dirs = sorted(by_dir.keys())
    rates = [by_dir[d]["contamination_rate"] for d in dirs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    bars = ax1.bar(dirs, rates, color=[C[2] if r > 0.1 else C[4] for r in rates])
    ax1.set_ylabel("Contamination Rate")
    ax1.set_title("Spanish Contamination by Direction")
    ax1.set_ylim(0, 1)
    for bar, rate in zip(bars, rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{rate:.1%}", ha="center", fontsize=11)

    # Most common leaked words
    all_spanish = []
    for p in pairs:
        all_spanish.extend(p.get("spanish_words_found", []))

    if all_spanish:
        from collections import Counter
        top = Counter(all_spanish).most_common(15)
        words = [w for w, _ in top]
        counts = [c for _, c in top]
        ax2.barh(words, counts, color=C[2])
        ax2.invert_yaxis()
        ax2.set_xlabel("Occurrences")
        ax2.set_title("Most Common Leaked Spanish Words")
    else:
        ax2.text(0.5, 0.5, "No Spanish contamination detected",
                ha="center", va="center", fontsize=14, transform=ax2.transAxes)
        ax2.set_title("Leaked Spanish Words")

    plt.tight_layout()
    save_fig("09_spanish_contamination")


def chart_worst_pairs(data: dict):
    """Horizontal bar: 15 worst-scoring pairs by chrF."""
    pairs = sorted(data["pairs"], key=lambda p: p["chrf"])[:15]

    fig, ax = plt.subplots(figsize=(14, 8))
    labels = []
    scores = []
    colors = []
    for p in pairs:
        src_short = p["src"][:40] + "..." if len(p["src"]) > 40 else p["src"]
        labels.append(f"[{p['src_lang']}→{p['tgt_lang']}] {src_short}")
        scores.append(p["chrf"])
        colors.append(C[2] if p["chrf"] < 0.2 else C[3] if p["chrf"] < 0.4 else C[4])

    ax.barh(range(len(labels)), scores, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("chrF Score")
    ax.set_title("15 Worst-Performing Translation Pairs")
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    plt.tight_layout()
    save_fig("10_worst_pairs")


def chart_best_pairs(data: dict):
    """Horizontal bar: 15 best-scoring pairs by chrF."""
    pairs = sorted(data["pairs"], key=lambda p: p["chrf"], reverse=True)[:15]

    fig, ax = plt.subplots(figsize=(14, 8))
    labels = []
    scores = []
    for p in pairs:
        src_short = p["src"][:40] + "..." if len(p["src"]) > 40 else p["src"]
        labels.append(f"[{p['src_lang']}→{p['tgt_lang']}] {src_short}")
        scores.append(p["chrf"])

    ax.barh(range(len(labels)), scores, color=C[4])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("chrF Score")
    ax.set_title("15 Best-Performing Translation Pairs")
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    plt.tight_layout()
    save_fig("11_best_pairs")


def chart_radar_by_direction(data: dict):
    """Radar chart: metric profile per direction."""
    by_dir = data["by_direction"]
    dirs = sorted(by_dir.keys())
    metrics = ["bleu", "chrf", "meteor", "bertscore"]
    labels = ["BLEU", "chrF", "METEOR", "BERTScore"]
    n_metrics = len(metrics)

    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_facecolor("#16213e")

    for i, d in enumerate(dirs):
        vals = [by_dir[d].get(m, 0) for m in metrics]
        vals += vals[:1]  # close
        ax.plot(angles, vals, "o-", linewidth=2, label=d, color=C[i])
        ax.fill(angles, vals, alpha=0.1, color=C[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title("Quality Profile by Direction", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    save_fig("12_radar_by_direction")


def chart_ter_vs_bleu(data: dict):
    """Scatter: TER vs BLEU colored by direction — shows quality tradeoff."""
    pairs = data["pairs"]
    fig, ax = plt.subplots()

    dir_colors = {}
    for p in pairs:
        d = f"{p['src_lang']}→{p['tgt_lang']}"
        if d not in dir_colors:
            dir_colors[d] = C[len(dir_colors) % len(C)]
        ax.scatter(p["ter"], p["bleu"], color=dir_colors[d], s=40, alpha=0.7)

    patches = [mpatches.Patch(color=c, label=d) for d, c in dir_colors.items()]
    ax.legend(handles=patches)
    ax.set_xlabel("TER (lower = better)")
    ax.set_ylabel("BLEU (higher = better)")
    ax.set_title("TER vs BLEU — Translation Quality Tradeoff")

    # Quadrant lines
    ax.axhline(y=0.5, color="#555", linestyle=":", alpha=0.3)
    ax.axvline(x=0.5, color="#555", linestyle=":", alpha=0.3)

    save_fig("13_ter_vs_bleu")


def chart_category_contamination(data: dict):
    """Bar: contamination rate per category."""
    pairs = data["pairs"]
    cat_contam = defaultdict(lambda: {"total": 0, "contaminated": 0})

    for p in pairs:
        cat_contam[p["category"]]["total"] += 1
        if p.get("spanish_words_found"):
            cat_contam[p["category"]]["contaminated"] += 1

    cats = sorted(cat_contam.keys())
    rates = [cat_contam[c]["contaminated"] / max(cat_contam[c]["total"], 1) for c in cats]

    fig, ax = plt.subplots()
    ax.bar(range(len(cats)), rates, color=[C[2] if r > 0.1 else C[4] for r in rates])
    ax.set_ylabel("Contamination Rate")
    ax.set_title("Spanish Contamination Rate by Category")
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, rotation=35, ha="right")
    ax.set_ylim(0, 1)
    for i, rate in enumerate(rates):
        ax.text(i, rate + 0.02, f"{rate:.0%}", ha="center", fontsize=9)
    plt.tight_layout()
    save_fig("14_category_contamination")


def chart_predicted_vs_expected_lengths(data: dict):
    """Scatter: predicted translation length vs expected length."""
    pairs = data["pairs"]

    fig, ax = plt.subplots()
    exp_lens = [len(p["tgt_expected"].split()) for p in pairs]
    pred_lens = [len(p["tgt_predicted"].split()) for p in pairs]

    ax.scatter(exp_lens, pred_lens, color=C[0], s=40, alpha=0.6)

    lim = max(max(exp_lens), max(pred_lens)) + 2
    ax.plot([0, lim], [0, lim], "--", color=C[3], alpha=0.6, label="Perfect match")
    ax.set_xlabel("Expected Length (words)")
    ax.set_ylabel("Predicted Length (words)")
    ax.set_title("Translation Length: Expected vs Predicted")
    ax.legend()
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    save_fig("15_length_expected_vs_predicted")


def chart_cumulative_score(data: dict):
    """Line chart: cumulative average chrF as pairs are evaluated."""
    pairs = data["pairs"]
    running_sum = 0
    cum_avg = []
    for i, p in enumerate(pairs):
        running_sum += p["chrf"]
        cum_avg.append(running_sum / (i + 1))

    fig, ax = plt.subplots()
    ax.plot(range(1, len(cum_avg) + 1), cum_avg, color=C[1], linewidth=2)
    ax.axhline(y=cum_avg[-1], color=C[3], linestyle="--", alpha=0.5,
               label=f"Final: {cum_avg[-1]:.3f}")
    ax.set_xlabel("Pair #")
    ax.set_ylabel("Cumulative Average chrF")
    ax.set_title("Score Convergence Over Evaluation")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.2)
    save_fig("16_cumulative_score")


def chart_summary_table(data: dict):
    """Table figure: summary stats."""
    agg = data["aggregate"]
    by_dir = data["by_direction"]

    rows = [["Overall", f"{agg['bleu']:.3f}", f"{agg['chrf']:.3f}",
             f"{agg['meteor']:.3f}", f"{agg['ter']:.3f}",
             f"{agg.get('bertscore', 0):.3f}",
             str(agg["count"]), f"{agg['contamination_rate']:.1%}"]]

    for d in sorted(by_dir.keys()):
        s = by_dir[d]
        rows.append([d, f"{s['bleu']:.3f}", f"{s['chrf']:.3f}",
                     f"{s['meteor']:.3f}", f"{s['ter']:.3f}",
                     f"{s.get('bertscore', 0):.3f}",
                     str(s["count"]), f"{s['contamination_rate']:.1%}"])

    fig, ax = plt.subplots(figsize=(16, 3 + len(rows) * 0.4))
    ax.axis("off")
    n_cols = 8
    table = ax.table(
        cellText=rows,
        colLabels=["Direction", "BLEU", "chrF", "METEOR", "TER", "BERTScore", "n", "Spanish %"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)

    # Style header
    for j in range(n_cols):
        table[0, j].set_facecolor("#2a2a4a")
        table[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(rows) + 1):
        for j in range(n_cols):
            table[i, j].set_facecolor("#1a1a2e")
            table[i, j].set_text_props(color="#e0e0e0")

    ax.set_title(f"Evaluation Summary — {data['model']} — {data['timestamp']}",
                 fontsize=13, pad=20)
    save_fig("00_summary_table")


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate ML evaluation charts")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--from-cache", action="store_true",
                       help="Use last saved eval instead of running live")
    group.add_argument("--simulate", action="store_true",
                       help="Generate realistic synthetic data (no API key needed)")
    args = parser.parse_args()

    print(f"Output: {OUT_DIR}/\n")

    if args.from_cache:
        print("Loading cached evaluation...\n")
        data = load_cached()
    elif args.simulate:
        print("Generating simulated evaluation data...\n")
        data = simulate_eval()
    else:
        data = run_eval()

    print(f"\nModel: {data['model']}")
    print(f"Pairs: {data['aggregate']['count']}")
    print(f"chrF:  {data['aggregate']['chrf']:.3f}")
    print(f"BLEU:  {data['aggregate']['bleu']:.3f}\n")

    print("Generating charts...")
    chart_summary_table(data)
    chart_aggregate_scores(data)
    chart_scores_by_direction(data)
    chart_scores_by_category(data)
    chart_per_pair_heatmap(data)
    chart_score_distributions(data)
    chart_metric_correlations(data)
    chart_latency(data)
    chart_score_vs_length(data)
    chart_spanish_contamination(data)
    chart_worst_pairs(data)
    chart_best_pairs(data)
    chart_radar_by_direction(data)
    chart_ter_vs_bleu(data)
    chart_category_contamination(data)
    chart_predicted_vs_expected_lengths(data)
    chart_cumulative_score(data)

    print(f"\nDone. {len(list(OUT_DIR.glob('*.png')))} charts in {OUT_DIR}/")


if __name__ == "__main__":
    main()
