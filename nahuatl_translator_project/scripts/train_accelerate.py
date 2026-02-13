#!/usr/bin/env python3
"""Fine-tune a seq2seq translation model using Accelerate (no Trainer).

This avoids many Colab dependency/Trainer issues.

Example (bidirectional by default after running prepare_data.py):
  python scripts/train_accelerate.py --model t5-small --data_dir data/splits --out_dir runs/t5_small_demo --epochs 1 --train_examples 2000
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup


LANG_LABELS = {
    "en": "English",
    "es": "Spanish",
    "nah": "Nahuatl",
}


def _label(code: str) -> str:
    return LANG_LABELS.get(code, code)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/mt5-small")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--train_examples", type=int, default=0, help="If >0, train on a random subset for speed")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    ds = load_dataset("json", data_files={
        "train": str(data_dir / "train.jsonl"),
        "validation": str(data_dir / "valid.jsonl"),
        "test": str(data_dir / "test.jsonl"),
    })

    # Slow tokenizer is more robust in messy environments
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    max_len = args.max_len

    def preprocess(batch):
        # New schema (preferred): src_text/tgt_text + src_lang/tgt_lang
        if "src_text" in batch and "tgt_text" in batch:
            src_texts = batch["src_text"]
            tgt_texts = batch["tgt_text"]
            src_langs = batch.get("src_lang", ["en"] * len(src_texts))
            tgt_langs = batch.get("tgt_lang", ["nah"] * len(src_texts))
        else:
            # Legacy schema
            src_texts = batch["en"]
            tgt_texts = batch["nah"]
            src_langs = ["en"] * len(src_texts)
            tgt_langs = ["nah"] * len(src_texts)

        varieties = batch.get("variety", ["Unknown"] * len(src_texts))

        prompts = [
            f"translate {_label(s)} to {_label(t)} [{v}]: {x}"
            for x, s, t, v in zip(src_texts, src_langs, tgt_langs, varieties)
        ]

        model_inputs = tok(prompts, max_length=max_len, truncation=True, padding="max_length")
        labels_enc = tok(tgt_texts, max_length=max_len, truncation=True, padding="max_length")
        labels = labels_enc["input_ids"]
        labels = [[(t if t != tok.pad_token_id else -100) for t in row] for row in labels]
        model_inputs["labels"] = labels
        return model_inputs

    tds = ds.map(preprocess, batched=True, remove_columns=ds["train"].column_names)
    tds.set_format(type="torch")

    # Optional subset for faster debugging
    train_ds = tds["train"].shuffle(seed=args.seed)
    if args.train_examples and args.train_examples > 0:
        train_ds = train_ds.select(range(min(args.train_examples, len(train_ds))))

    val_ds = tds["validation"]

    accelerator = Accelerator()
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    num_training_steps = args.epochs * len(train_loader)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(10, num_training_steps // 20),
        num_training_steps=num_training_steps,
    )

    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_scheduler
    )

    accelerator.print(f"GPU available: {torch.cuda.is_available()} | training steps: {num_training_steps}")

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for batch in train_loader:
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            total += loss.item()
        accelerator.print(f"Epoch {epoch+1}/{args.epochs} loss: {total/max(1,len(train_loader)):.4f}")

        # quick validation loss
        model.eval()
        vtotal = 0.0
        with torch.no_grad():
            for vb in val_loader:
                vtotal += model(**vb).loss.item()
        accelerator.print(f"Epoch {epoch+1} val_loss: {vtotal/max(1,len(val_loader)):.4f}")

    # Save
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model).cpu()
        unwrapped.save_pretrained(out_dir, safe_serialization=True)
        tok.save_pretrained(out_dir)
        (out_dir / "training_args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    accelerator.print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
