#!/usr/bin/env python3
"""Comprehensive evaluation script for the Nahuatl translator.

Modes:
  golden             -- Run golden pairs through OpenAI and compute full metrics
  temperature_sweep  -- Test different temperatures on a subset of pairs
  compare            -- Compare multiple models on the same pairs
  finetune           -- Evaluate a fine-tuned local model with sacrebleu

Results are saved to data/eval_results/ as JSON files.

Usage:
  python scripts/eval_full.py --mode golden
  python scripts/eval_full.py --mode temperature_sweep --temps 0.0,0.2,0.4,0.6,0.8,1.0
  python scripts/eval_full.py --mode compare --models gpt-5,gpt-4o
  python scripts/eval_full.py --mode finetune --model_dir runs/mt5 --data_dir data/splits
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_golden_pairs():
    golden_path = ROOT / "tests" / "golden_translations.json"
    with open(golden_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["pairs"]


def ensure_output_dir():
    d = ROOT / "data" / "eval_results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def mode_golden(args):
    """Run all golden pairs through OpenAI and compute full metrics."""
    from webapp.evaluation import evaluate_golden_pairs, save_eval_result
    from webapp.services import openai_translate

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    pairs = load_golden_pairs()
    model = args.model or os.getenv("OPENAI_TRANSLATE_MODEL", "gpt-5")

    print(f"Running evaluation on {len(pairs)} golden pairs with {model}...")
    result = evaluate_golden_pairs(
        translate_fn=openai_translate,
        golden_pairs=pairs,
        model=model,
    )

    filepath = save_eval_result(result, str(ensure_output_dir()))
    print(f"\nResults saved to: {filepath}")
    print(f"\nAggregate scores:")
    print(f"  BLEU:   {result.aggregate.bleu:.4f}")
    print(f"  chrF:   {result.aggregate.chrf:.4f}")
    print(f"  METEOR: {result.aggregate.meteor:.4f}")
    print(f"  TER:    {result.aggregate.ter:.4f}")
    print(f"  Contamination rate: {result.aggregate.contamination_rate:.1%}")
    print(f"  Avg latency: {result.avg_latency_ms:.0f}ms")
    print(f"  Estimated cost: ${result.total_cost_estimate:.4f}")

    print(f"\nBy direction:")
    for d, scores in result.by_direction.items():
        print(f"  {d}: BLEU={scores.bleu:.4f} chrF={scores.chrf:.4f} ({scores.count} pairs)")

    print(f"\nBy category:")
    for c, scores in result.by_category.items():
        print(f"  {c}: BLEU={scores.bleu:.4f} chrF={scores.chrf:.4f} ({scores.count} pairs)")


def mode_temperature_sweep(args):
    """Run a subset of pairs at different temperatures."""
    from webapp.evaluation import (
        compute_bleu, compute_chrf, compute_meteor, compute_ter,
    )
    from webapp.services import openai_translate
    import time

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    pairs = load_golden_pairs()[:args.max_pairs]
    temps = [float(t) for t in args.temps.split(",")]
    model = args.model or os.getenv("OPENAI_TRANSLATE_MODEL", "gpt-5")

    print(f"Temperature sweep: {len(pairs)} pairs x {len(temps)} temperatures")
    results = []

    for temp in temps:
        os.environ["OPENAI_TRANSLATE_TEMPERATURE"] = str(temp)
        scores = {"temperature": temp, "bleu": [], "chrf": [], "meteor": [], "ter": []}

        for pair in pairs:
            try:
                pred = openai_translate(pair["src"], pair["src_lang"], pair["tgt_lang"], "Unknown")
                scores["bleu"].append(compute_bleu(pair["tgt"], pred))
                scores["chrf"].append(compute_chrf(pair["tgt"], pred))
                scores["meteor"].append(compute_meteor(pair["tgt"], pred))
                scores["ter"].append(compute_ter(pair["tgt"], pred))
            except Exception as e:
                print(f"  Error at temp={temp}: {e}")

        n = max(len(scores["bleu"]), 1)
        result = {
            "temperature": temp,
            "avg_bleu": sum(scores["bleu"]) / n,
            "avg_chrf": sum(scores["chrf"]) / n,
            "avg_meteor": sum(scores["meteor"]) / n,
            "avg_ter": sum(scores["ter"]) / n,
            "count": len(scores["bleu"]),
        }
        results.append(result)
        print(f"  temp={temp:.1f}: BLEU={result['avg_bleu']:.4f} chrF={result['avg_chrf']:.4f}")

    # Save
    out_dir = ensure_output_dir()
    out_path = out_dir / f"temp_sweep_{model.replace('/', '_')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {out_path}")

    # Restore
    if "OPENAI_TRANSLATE_TEMPERATURE" in os.environ:
        del os.environ["OPENAI_TRANSLATE_TEMPERATURE"]


def mode_compare(args):
    """Compare multiple models on the same golden pairs."""
    from webapp.evaluation import evaluate_golden_pairs, save_eval_result
    from webapp.services import openai_translate

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    pairs = load_golden_pairs()
    models = [m.strip() for m in args.models.split(",")]

    print(f"Comparing {len(models)} models on {len(pairs)} pairs...")
    comparison = []

    for model in models:
        os.environ["OPENAI_TRANSLATE_MODEL"] = model
        print(f"\nEvaluating {model}...")
        result = evaluate_golden_pairs(
            translate_fn=openai_translate,
            golden_pairs=pairs,
            model=model,
        )
        save_eval_result(result, str(ensure_output_dir()))
        comparison.append({
            "model": model,
            "bleu": result.aggregate.bleu,
            "chrf": result.aggregate.chrf,
            "meteor": result.aggregate.meteor,
            "ter": result.aggregate.ter,
            "avg_latency_ms": result.avg_latency_ms,
            "cost": result.total_cost_estimate,
        })
        print(f"  BLEU={result.aggregate.bleu:.4f} chrF={result.aggregate.chrf:.4f}")

    print("\n--- Comparison Summary ---")
    for c in comparison:
        print(f"  {c['model']}: BLEU={c['bleu']:.4f} chrF={c['chrf']:.4f} METEOR={c['meteor']:.4f} TER={c['ter']:.4f} latency={c['avg_latency_ms']:.0f}ms cost=${c['cost']:.4f}")


def mode_finetune(args):
    """Evaluate a fine-tuned local model with sacrebleu (from original eval.py)."""
    import sacrebleu
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    LANG_LABELS = {"en": "English", "es": "Spanish", "nah": "Nahuatl"}

    data_dir = Path(args.data_dir)
    ds = load_dataset("json", data_files={"test": str(data_dir / "test.jsonl")})

    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    test = ds["test"]
    if "src_text" in test.column_names:
        src_text, tgt_text = test["src_text"], test["tgt_text"]
        src_lang = test["src_lang"] if "src_lang" in test.column_names else ["en"] * len(test)
        tgt_lang = test["tgt_lang"] if "tgt_lang" in test.column_names else ["nah"] * len(test)
        variety = test["variety"] if "variety" in test.column_names else ["Unknown"] * len(test)
    else:
        src_text, tgt_text = test["en"], test["nah"]
        src_lang, tgt_lang = ["en"] * len(test), ["nah"] * len(test)
        variety = test["variety"] if "variety" in test.column_names else ["Unknown"] * len(test)

    groups = {}
    for i, (s, t) in enumerate(zip(src_lang, tgt_lang)):
        groups.setdefault((s, t), []).append(i)

    results = {}
    for (s, t), idxs in groups.items():
        preds, refs = [], [tgt_text[j] for j in idxs]
        for k in range(0, len(idxs), args.batch_size):
            chunk = idxs[k:k + args.batch_size]
            prompts = [
                f"translate {LANG_LABELS.get(s, s)} to {LANG_LABELS.get(t, t)} [{variety[j]}]: {src_text[j]}"
                for j in chunk
            ]
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **enc, max_new_tokens=args.max_new_tokens,
                    num_beams=4, no_repeat_ngram_size=3,
                    repetition_penalty=1.15, early_stopping=True,
                )
            preds.extend(tok.batch_decode(out, skip_special_tokens=True))

        key = f"{s}->{t}"
        results[key] = {
            "n": len(idxs),
            "bleu": sacrebleu.corpus_bleu(preds, [refs]).score,
            "chrf": sacrebleu.corpus_chrf(preds, [refs]).score,
        }

    print("\nFine-tuned model evaluation:")
    for key, r in results.items():
        print(f"  {key}: BLEU={r['bleu']:.2f} chrF={r['chrf']:.2f} ({r['n']} pairs)")

    # Save
    out_dir = ensure_output_dir()
    model_slug = Path(args.model_dir).name
    out_path = out_dir / f"finetune_{model_slug}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Comprehensive evaluation for Nahuatl translator")
    ap.add_argument("--mode", required=True, choices=["golden", "temperature_sweep", "compare", "finetune"])
    ap.add_argument("--model", default=None, help="Model name (for golden/compare modes)")
    ap.add_argument("--models", default="gpt-5,gpt-4o", help="Comma-separated models (for compare mode)")
    ap.add_argument("--temps", default="0.0,0.2,0.4,0.6,0.8,1.0", help="Comma-separated temperatures")
    ap.add_argument("--max_pairs", type=int, default=20, help="Max pairs for temperature sweep")
    ap.add_argument("--model_dir", default=None, help="Local model directory (for finetune mode)")
    ap.add_argument("--data_dir", default=None, help="Data directory with test.jsonl (for finetune mode)")
    ap.add_argument("--batch_size", type=int, default=16, help="Batch size for finetune mode")
    ap.add_argument("--max_new_tokens", type=int, default=96, help="Max tokens for finetune mode")
    args = ap.parse_args()

    if args.mode == "golden":
        mode_golden(args)
    elif args.mode == "temperature_sweep":
        mode_temperature_sweep(args)
    elif args.mode == "compare":
        mode_compare(args)
    elif args.mode == "finetune":
        if not args.model_dir or not args.data_dir:
            print("ERROR: --model_dir and --data_dir required for finetune mode")
            sys.exit(1)
        mode_finetune(args)


if __name__ == "__main__":
    main()
