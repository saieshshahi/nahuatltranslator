#!/usr/bin/env python3
"""Prepare English↔Nahuatl parallel data from Excel into JSONL splits.

Expected columns (minimum):
- english
- nahuatl
Optional:
- variety (dialect/variety label)

Outputs:
- train.jsonl, valid.jsonl, test.jsonl

By default this script creates **bidirectional** records so the same model can learn:
  * English → Nahuatl
  * Nahuatl → English

Each JSONL line looks like:
  {
    "src_text": "...",
    "tgt_text": "...",
    "src_lang": "en|nah|es",
    "tgt_lang": "en|nah|es",
    "variety": "..."
  }

You can disable bidirectional mode with `--bidirectional 0` to output legacy keys:
  {"en":..., "nah":..., "variety":...}
"""

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd


def clean(text: str) -> str:
    if text is None:
        return ""
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def tok_len(s: str) -> int:
    return len(s.split())


def write_jsonl(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", required=True, help="Path to Excel file")
    ap.add_argument("--out", required=True, help="Output directory for splits")
    ap.add_argument("--max_len", type=int, default=80, help="Max whitespace-token length per side")
    ap.add_argument("--bidirectional", type=int, default=1, help="If 1, write both en→nah and nah→en records")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_frac", type=float, default=0.90)
    ap.add_argument("--valid_frac", type=float, default=0.05)
    args = ap.parse_args()

    random.seed(args.seed)

    df = pd.read_excel(args.excel)

    # Accept either 'nahuatl' or 'nah'
    if "nahuatl" in df.columns and "nah" not in df.columns:
        df = df.rename(columns={"nahuatl": "nah"})

    if "english" in df.columns and "en" not in df.columns:
        df = df.rename(columns={"english": "en"})

    if "en" not in df.columns or "nah" not in df.columns:
        raise ValueError("Excel must contain english/en and nahuatl/nah columns")

    if "variety" not in df.columns:
        df["variety"] = "Unknown"

    df["en"] = df["en"].map(clean)
    df["nah"] = df["nah"].map(clean)
    df["variety"] = df["variety"].map(lambda x: clean(x) or "Unknown")

    df = df[(df["en"] != "") & (df["nah"] != "")].copy()
    df = df.drop_duplicates(subset=["en", "nah", "variety"]).reset_index(drop=True)

    # length filter
    df = df[(df["en"].map(tok_len) <= args.max_len) & (df["nah"].map(tok_len) <= args.max_len)].reset_index(drop=True)

    legacy_rows = df[["en", "nah", "variety"]].to_dict("records")

    # Build training rows
    rows = []
    for r in legacy_rows:
        rows.append(
            {
                "src_text": r["en"],
                "tgt_text": r["nah"],
                "src_lang": "en",
                "tgt_lang": "nah",
                "variety": r["variety"],
                # keep legacy fields too (helps with debugging)
                "en": r["en"],
                "nah": r["nah"],
            }
        )
        if args.bidirectional:
            rows.append(
                {
                    "src_text": r["nah"],
                    "tgt_text": r["en"],
                    "src_lang": "nah",
                    "tgt_lang": "en",
                    "variety": r["variety"],
                    "en": r["en"],
                    "nah": r["nah"],
                }
            )

    random.shuffle(rows)

    n = len(rows)
    n_train = int(n * args.train_frac)
    n_valid = int(n * args.valid_frac)

    train = rows[:n_train]
    valid = rows[n_train:n_train + n_valid]
    test = rows[n_train + n_valid:]

    out_dir = Path(args.out)
    if args.bidirectional:
        write_jsonl(train, out_dir / "train.jsonl")
        write_jsonl(valid, out_dir / "valid.jsonl")
        write_jsonl(test, out_dir / "test.jsonl")
    else:
        # Legacy mode: downcast to the original schema
        def to_legacy(rs):
            return [{"en": r["en"], "nah": r["nah"], "variety": r["variety"]} for r in rs]

        write_jsonl(to_legacy(train), out_dir / "train.jsonl")
        write_jsonl(to_legacy(valid), out_dir / "valid.jsonl")
        write_jsonl(to_legacy(test), out_dir / "test.jsonl")

    print(f"Wrote {len(train)} train, {len(valid)} valid, {len(test)} test to {out_dir}")


if __name__ == "__main__":
    main()
