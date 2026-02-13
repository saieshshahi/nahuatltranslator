#!/usr/bin/env python3
"""Evaluate a trained model on the test split with BLEU + chrF.

Supports both the legacy schema (en/nah) and the newer bidirectional schema
(src_text/tgt_text + src_lang/tgt_lang).
"""

import argparse
from pathlib import Path

import sacrebleu
import torch
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


LANG_LABELS = {
    "en": "English",
    "es": "Spanish",
    "nah": "Nahuatl",
}


def _label(code: str) -> str:
    return LANG_LABELS.get(code, code)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=96)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    ds = load_dataset("json", data_files={
        "test": str(data_dir / "test.jsonl"),
    })

    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    test = ds["test"]

    # Normalize schema
    if "src_text" in test.column_names and "tgt_text" in test.column_names:
        src_text = test["src_text"]
        tgt_text = test["tgt_text"]
        src_lang = test.get("src_lang", ["en"] * len(test))
        tgt_lang = test.get("tgt_lang", ["nah"] * len(test))
        variety = test.get("variety", ["Unknown"] * len(test))
    else:
        src_text = test["en"]
        tgt_text = test["nah"]
        src_lang = ["en"] * len(test)
        tgt_lang = ["nah"] * len(test)
        variety = test.get("variety", ["Unknown"] * len(test))

    # Group by direction
    groups = {}
    for i, (s, t) in enumerate(zip(src_lang, tgt_lang)):
        groups.setdefault((s, t), []).append(i)

    results = {}
    for (s, t), idxs in groups.items():
        preds = []
        refs = [tgt_text[j] for j in idxs]
        for k in range(0, len(idxs), args.batch_size):
            chunk = idxs[k:k + args.batch_size]
            prompts = [
                f"translate {_label(s)} to {_label(t)} [{variety[j]}]: {src_text[j]}"
                for j in chunk
            ]
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=4,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.15,
                    early_stopping=True,
                )
            preds.extend(tok.batch_decode(out, skip_special_tokens=True))

        key = f"{s}->{t}"
        results[key] = {
            "n": len(idxs),
            "bleu": sacrebleu.corpus_bleu(preds, [refs]).score,
            "chrf": sacrebleu.corpus_chrf(preds, [refs]).score,
        }

    print(results)


if __name__ == "__main__":
    main()
