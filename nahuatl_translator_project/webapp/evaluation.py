"""Evaluation metrics and golden-pair evaluation engine.

Provides BLEU, chrF, METEOR, and TER metrics for translation quality
measurement, plus an evaluation runner that scores translations against
golden reference pairs and aggregates results by category and direction.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

# BERTScore — optional heavy dependency
try:
    from bert_score import score as _bert_score_fn
    HAS_BERTSCORE = True
except ImportError:
    HAS_BERTSCORE = False


# ===================================================================
# Core metric functions
# ===================================================================

def _ngrams(tokens: List[str], n: int) -> List[tuple]:
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def compute_bleu(reference: str, hypothesis: str, max_n: int = 4) -> float:
    """Sentence-level BLEU with add-1 smoothing."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    if len(hyp_tokens) == 0:
        return 0.0

    effective_n = min(max_n, len(ref_tokens), len(hyp_tokens))
    if effective_n == 0:
        return 0.0

    precisions = []
    for n in range(1, effective_n + 1):
        ref_ng = Counter(_ngrams(ref_tokens, n))
        hyp_ng = Counter(_ngrams(hyp_tokens, n))

        if not hyp_ng:
            precisions.append(0.0)
            continue

        clipped = sum(min(count, ref_ng[ng]) for ng, count in hyp_ng.items())
        total = sum(hyp_ng.values())
        precisions.append((clipped + 1) / (total + 1))

    if not precisions or all(p == 0 for p in precisions):
        return 0.0

    log_avg = sum(math.log(max(p, 1e-10)) for p in precisions) / len(precisions)

    bp = 1.0
    if len(hyp_tokens) < len(ref_tokens):
        bp = math.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1))

    return bp * math.exp(log_avg)


def compute_chrf(reference: str, hypothesis: str, n: int = 6, beta: float = 2.0) -> float:
    """Character n-gram F-score."""
    def _char_ngrams(text: str, order: int) -> Counter:
        text = text.strip()
        grams = Counter()
        for i in range(len(text) - order + 1):
            grams[text[i:i + order]] += 1
        return grams

    ref = reference.lower()
    hyp = hypothesis.lower()

    if not hyp or not ref:
        return 0.0

    precisions = []
    recalls = []

    for order in range(1, n + 1):
        ref_ng = _char_ngrams(ref, order)
        hyp_ng = _char_ngrams(hyp, order)

        common = sum((ref_ng & hyp_ng).values())
        total_hyp = sum(hyp_ng.values())
        total_ref = sum(ref_ng.values())

        p = common / total_hyp if total_hyp > 0 else 0.0
        r = common / total_ref if total_ref > 0 else 0.0
        precisions.append(p)
        recalls.append(r)

    avg_p = sum(precisions) / len(precisions) if precisions else 0.0
    avg_r = sum(recalls) / len(recalls) if recalls else 0.0

    if avg_p + avg_r == 0:
        return 0.0

    beta_sq = beta ** 2
    return (1 + beta_sq) * avg_p * avg_r / (beta_sq * avg_p + avg_r)


def compute_meteor(reference: str, hypothesis: str) -> float:
    """Simplified METEOR: unigram F-mean with fragmentation penalty.

    No stemming or synonyms (no nltk dependency). Uses exact unigram
    matching with a chunk-based fragmentation penalty.
    """
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    if not ref_tokens or not hyp_tokens:
        return 0.0

    # Unigram matching (greedy, left-to-right)
    ref_matched: Set[int] = set()
    hyp_matched: Set[int] = set()

    for hi, ht in enumerate(hyp_tokens):
        for ri, rt in enumerate(ref_tokens):
            if rt == ht and ri not in ref_matched:
                ref_matched.add(ri)
                hyp_matched.add(hi)
                break

    matches = len(ref_matched)
    if matches == 0:
        return 0.0

    precision = matches / len(hyp_tokens)
    recall = matches / len(ref_tokens)

    # F-mean with alpha=0.9 (recall-weighted, standard METEOR)
    alpha = 0.9
    f_mean = (precision * recall) / (alpha * precision + (1 - alpha) * recall)

    # Fragmentation penalty: count chunks of contiguous aligned words
    sorted_hyp_positions = sorted(hyp_matched)
    chunks = 1
    for i in range(1, len(sorted_hyp_positions)):
        if sorted_hyp_positions[i] != sorted_hyp_positions[i - 1] + 1:
            chunks += 1

    frag = chunks / matches if matches > 0 else 0.0
    penalty = 0.5 * (frag ** 3)

    return f_mean * (1 - penalty)


def compute_ter(reference: str, hypothesis: str) -> float:
    """Translation Edit Rate: word-level edit distance / reference length.

    Lower is better (0.0 = perfect, 1.0+ = many edits needed).
    """
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0

    # Levenshtein distance on word tokens
    m, n = len(ref_tokens), len(hyp_tokens)
    dp = list(range(n + 1))

    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp

    return dp[n] / len(ref_tokens)


def compute_bertscore(reference: str, hypothesis: str,
                      model_type: str = "xlm-roberta-base") -> float:
    """BERTScore F1 using a multilingual transformer model.

    Returns the F1 component of BERTScore (0-1, higher is better).
    Falls back to 0.0 if bert_score is not installed.

    Uses xlm-roberta-base by default — covers 100 languages including
    Spanish and English, and provides reasonable subword coverage for
    Nahuatl via its shared Latin-script vocabulary.
    """
    if not HAS_BERTSCORE:
        return 0.0
    if not reference.strip() or not hypothesis.strip():
        return 0.0
    try:
        _P, _R, F1 = _bert_score_fn(
            [hypothesis], [reference],
            model_type=model_type,
            num_layers=10,       # slightly faster than full 12
            verbose=False,
            rescale_with_baseline=False,
        )
        return float(F1[0])
    except Exception:
        return 0.0


def normalize_nahuatl(text: str) -> str:
    """Normalize Nahuatl text for comparison."""
    t = text.lower().strip()
    t = re.sub(r'[¿¡?.!,;:\'"()]', '', t)
    t = t.replace("hua", "wa").replace("hu", "w")
    t = re.sub(r'qu(?=[ei])', 'k', t)
    return t.strip()


# ===================================================================
# Data structures
# ===================================================================

@dataclass
class PairResult:
    src: str
    tgt_expected: str
    tgt_predicted: str
    src_lang: str
    tgt_lang: str
    category: str
    bleu: float
    chrf: float
    meteor: float
    ter: float
    bertscore: float
    latency_ms: float
    token_estimate: int
    spanish_words_found: List[str]
    notes: str = ""


@dataclass
class AggregateScores:
    bleu: float = 0.0
    chrf: float = 0.0
    meteor: float = 0.0
    ter: float = 0.0
    bertscore: float = 0.0
    count: int = 0
    contamination_rate: float = 0.0


@dataclass
class EvalResult:
    timestamp: str
    model: str
    pairs: List[PairResult]
    aggregate: AggregateScores
    by_category: Dict[str, AggregateScores]
    by_direction: Dict[str, AggregateScores]
    avg_latency_ms: float
    total_cost_estimate: float

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "model": self.model,
            "pairs": [asdict(p) for p in self.pairs],
            "aggregate": asdict(self.aggregate),
            "by_category": {k: asdict(v) for k, v in self.by_category.items()},
            "by_direction": {k: asdict(v) for k, v in self.by_direction.items()},
            "avg_latency_ms": self.avg_latency_ms,
            "total_cost_estimate": self.total_cost_estimate,
        }


@dataclass
class CorpusStats:
    total_entries: int = 0
    en_vocab_size: int = 0
    nah_vocab_size: int = 0
    avg_en_length: float = 0.0
    avg_nah_length: float = 0.0
    category_distribution: Dict[str, int] = field(default_factory=dict)
    contamination_rate: float = 0.0
    contaminated_count: int = 0


# ===================================================================
# Token / cost estimation
# ===================================================================

def _estimate_tokens(text: str) -> int:
    """Rough token estimate (~0.75 words per token for English, similar for Nahuatl)."""
    return max(1, int(len(text.split()) / 0.75))


def _estimate_cost(input_tokens: int, output_tokens: int, model: str = "gpt-5") -> float:
    """Estimate API cost in USD."""
    # gpt-5: $1.25/1M input, $10.00/1M output
    # gpt-5-mini: $0.25/1M input, $2.00/1M output
    # gpt-4o-mini: $0.15/1M input, $0.60/1M output
    if "5-mini" in model:
        return (input_tokens * 0.25 + output_tokens * 2.0) / 1_000_000
    elif "gpt-5" in model:
        return (input_tokens * 1.25 + output_tokens * 10.0) / 1_000_000
    elif "4o-mini" in model:
        return (input_tokens * 0.15 + output_tokens * 0.60) / 1_000_000
    elif "4o" in model or "4.1" in model:
        return (input_tokens * 2.00 + output_tokens * 8.0) / 1_000_000
    return (input_tokens * 1.25 + output_tokens * 10.0) / 1_000_000


# ===================================================================
# Aggregation helpers
# ===================================================================

def _aggregate_pairs(pairs: List[PairResult]) -> AggregateScores:
    """Compute average scores across a list of pair results."""
    if not pairs:
        return AggregateScores()

    n = len(pairs)
    contaminated = sum(1 for p in pairs if p.spanish_words_found)
    return AggregateScores(
        bleu=sum(p.bleu for p in pairs) / n,
        chrf=sum(p.chrf for p in pairs) / n,
        meteor=sum(p.meteor for p in pairs) / n,
        ter=sum(p.ter for p in pairs) / n,
        bertscore=sum(p.bertscore for p in pairs) / n,
        count=n,
        contamination_rate=contaminated / n,
    )


# ===================================================================
# Main evaluation function
# ===================================================================

def evaluate_golden_pairs(
    translate_fn: Callable[[str, str, str, str], str],
    golden_pairs: List[Dict[str, str]],
    directions: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    max_pairs: int = 200,
    model: str = "gpt-5",
) -> EvalResult:
    """Run golden pair evaluation and compute all metrics.

    Args:
        translate_fn: function(text, src, tgt, variety) -> translated_text
        golden_pairs: list of dicts with src, tgt, src_lang, tgt_lang, category
        directions: filter to specific directions like ["en->nah"]
        categories: filter to specific categories like ["greeting"]
        max_pairs: maximum pairs to evaluate (cost control)
        model: model name for cost estimation

    Returns:
        EvalResult with per-pair and aggregate metrics.
    """
    from webapp.spanish_filter import detect_spanish_in_output

    # Filter pairs
    filtered = golden_pairs
    if directions:
        dir_set = set(directions)
        filtered = [
            p for p in filtered
            if f"{p['src_lang']}->{p['tgt_lang']}" in dir_set
        ]
    if categories:
        cat_set = set(categories)
        filtered = [p for p in filtered if p["category"] in cat_set]

    filtered = filtered[:max_pairs]

    pair_results: List[PairResult] = []
    total_cost = 0.0

    for pair in filtered:
        src_text = pair["src"]
        expected = pair["tgt"]
        src_lang = pair["src_lang"]
        tgt_lang = pair["tgt_lang"]
        category = pair["category"]

        # Translate with timing
        start = time.time()
        try:
            predicted = translate_fn(src_text, src_lang, tgt_lang, "Unknown")
        except Exception as e:
            predicted = f"[ERROR: {e}]"
        latency_ms = (time.time() - start) * 1000

        # Compute metrics
        bleu = compute_bleu(expected, predicted)
        chrf = compute_chrf(expected, predicted)
        meteor = compute_meteor(expected, predicted)
        ter = compute_ter(expected, predicted)
        bscore = compute_bertscore(expected, predicted)

        # Spanish contamination (only for Nahuatl output)
        spanish_found = []
        if tgt_lang == "nah":
            spanish_found = detect_spanish_in_output(predicted)

        # Token/cost estimation
        input_tokens = _estimate_tokens(src_text) + 500  # ~500 for system prompt
        output_tokens = _estimate_tokens(predicted)
        cost = _estimate_cost(input_tokens, output_tokens, model)
        total_cost += cost

        pair_results.append(PairResult(
            src=src_text,
            tgt_expected=expected,
            tgt_predicted=predicted,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            category=category,
            bleu=round(bleu, 4),
            chrf=round(chrf, 4),
            meteor=round(meteor, 4),
            ter=round(ter, 4),
            bertscore=round(bscore, 4),
            latency_ms=round(latency_ms, 1),
            token_estimate=input_tokens + output_tokens,
            spanish_words_found=spanish_found,
            notes=pair.get("notes", ""),
        ))

    # Aggregate
    aggregate = _aggregate_pairs(pair_results)

    # By category
    by_category: Dict[str, AggregateScores] = {}
    cat_groups: Dict[str, List[PairResult]] = {}
    for pr in pair_results:
        cat_groups.setdefault(pr.category, []).append(pr)
    for cat, prs in cat_groups.items():
        by_category[cat] = _aggregate_pairs(prs)

    # By direction
    by_direction: Dict[str, AggregateScores] = {}
    dir_groups: Dict[str, List[PairResult]] = {}
    for pr in pair_results:
        d = f"{pr.src_lang}->{pr.tgt_lang}"
        dir_groups.setdefault(d, []).append(pr)
    for d, prs in dir_groups.items():
        by_direction[d] = _aggregate_pairs(prs)

    avg_latency = (
        sum(p.latency_ms for p in pair_results) / len(pair_results)
        if pair_results else 0.0
    )

    return EvalResult(
        timestamp=datetime.now(timezone.utc).isoformat(),
        model=model,
        pairs=pair_results,
        aggregate=aggregate,
        by_category=by_category,
        by_direction=by_direction,
        avg_latency_ms=round(avg_latency, 1),
        total_cost_estimate=round(total_cost, 6),
    )


# ===================================================================
# Corpus statistics
# ===================================================================

def compute_corpus_stats() -> CorpusStats:
    """Compute statistics about the parallel corpus and golden pairs."""
    from webapp.corpus import get_corpus
    from webapp.spanish_filter import count_spanish

    stats = CorpusStats()

    try:
        corpus = get_corpus()
        if not corpus.loaded:
            return stats

        entries = corpus.entries
        stats.total_entries = len(entries)

        en_vocab: Set[str] = set()
        nah_vocab: Set[str] = set()
        en_lengths: List[int] = []
        nah_lengths: List[int] = []
        contaminated = 0

        for entry in entries:
            en_words = entry.english.lower().split()
            nah_words = entry.nahuatl.lower().split()
            en_vocab.update(en_words)
            nah_vocab.update(nah_words)
            en_lengths.append(len(en_words))
            nah_lengths.append(len(nah_words))

            if count_spanish(entry.nahuatl) >= 1:
                contaminated += 1

        stats.en_vocab_size = len(en_vocab)
        stats.nah_vocab_size = len(nah_vocab)
        stats.avg_en_length = round(sum(en_lengths) / max(len(en_lengths), 1), 1)
        stats.avg_nah_length = round(sum(nah_lengths) / max(len(nah_lengths), 1), 1)
        stats.contaminated_count = contaminated
        stats.contamination_rate = round(
            contaminated / max(len(entries), 1), 4
        )
    except Exception:
        pass

    # Golden pair category distribution
    golden_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "tests", "golden_translations.json",
    )
    try:
        with open(golden_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for pair in data.get("pairs", []):
            cat = pair.get("category", "unknown")
            stats.category_distribution[cat] = stats.category_distribution.get(cat, 0) + 1
    except Exception:
        pass

    return stats


# ===================================================================
# Result persistence
# ===================================================================

def save_eval_result(result: EvalResult, output_dir: Optional[str] = None) -> str:
    """Save evaluation result to a JSON file. Returns the file path."""
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data", "eval_results",
        )
    os.makedirs(output_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = result.model.replace("/", "_").replace("-", "_")
    filename = f"eval_{ts}_{model_slug}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    return filepath


def load_eval_history(results_dir: Optional[str] = None) -> List[dict]:
    """Load all saved evaluation results, sorted by timestamp (newest first)."""
    if results_dir is None:
        results_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data", "eval_results",
        )

    if not os.path.isdir(results_dir):
        return []

    results = []
    for fname in sorted(os.listdir(results_dir), reverse=True):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(results_dir, fname), "r", encoding="utf-8") as f:
                data = json.load(f)
            # Return summary only (no per-pair data) for the listing
            results.append({
                "filename": fname,
                "timestamp": data.get("timestamp", ""),
                "model": data.get("model", ""),
                "aggregate": data.get("aggregate", {}),
                "avg_latency_ms": data.get("avg_latency_ms", 0),
                "total_cost_estimate": data.get("total_cost_estimate", 0),
                "pair_count": len(data.get("pairs", [])),
            })
        except Exception:
            continue

    return results
