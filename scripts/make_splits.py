"""Build leak-free train / val splits (90/10) keyed on FLEURS sentence_id.

WHY THIS EXISTS
---------------
FLEURS is a parallel corpus: the same sentence appears in many
languages. If we split naively on rows we end up with the *same
sentence* in both train and val, just spoken in a different
language -- a textbook leak. This script keys the split on
``sentence_id`` so all language variants of a given sentence land
in the same partition.

CPU-only utility; runs once before training.

Usage::

    python scripts/make_splits.py \\
        --accepted /path/to/accepted.jsonl \\
        --encoded-dir /path/to/data/encoded \
        --out-dir /path/to/data/splits \
        --val-frac 0.10 --seed 42
"""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

SENTENCE_RE = re.compile(r"fleurs_(\d+)")


def parse_sentence_id(pair_id: str) -> int:
    m = SENTENCE_RE.match(pair_id)
    if not m:
        raise ValueError(f"Cannot parse sentence_id from {pair_id}")
    return int(m.group(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accepted", required=True)
    ap.add_argument("--encoded-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    accepted = Path(args.accepted)
    encoded_dir = Path(args.encoded_dir).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    skipped_no_pt = 0
    for line in open(accepted):
        line = line.strip()
        if not line:
            continue
        e = json.loads(line)
        pair_id = e["pair_id"]
        src_lang = e["src_lang"]
        tgt_lang = e["tgt_lang"]
        # encode stage writes {pair_id}_{src_lang}{tgt_lang}.pt
        pt_name = f"{pair_id}_{src_lang}{tgt_lang}.pt"
        pt_path = encoded_dir / pt_name
        if not pt_path.exists():
            skipped_no_pt += 1
            continue
        sid = parse_sentence_id(pair_id)
        direction = f"{src_lang}->{tgt_lang}"
        rows.append(
            {
                "pt_path": str(pt_path),
                "src_align_path": str(
                    encoded_dir / f"{pair_id}_{src_lang}{tgt_lang}.src.alignments.json"
                ),
                "tgt_align_path": str(
                    encoded_dir / f"{pair_id}_{src_lang}{tgt_lang}.tgt.alignments.json"
                ),
                "direction": direction,
                "sentence_id": sid,
                "source_type": e.get("source_type", "unknown"),
                "tts_model": e.get("tts_model"),
                "tts_voice": e.get("tts_voice"),
            }
        )

    print(f"Rows: {len(rows)}  (skipped missing .pt: {skipped_no_pt})")

    sids = sorted({r["sentence_id"] for r in rows})
    rng = random.Random(args.seed)
    rng.shuffle(sids)
    n_val = max(1, int(round(len(sids) * args.val_frac)))
    val_sids = set(sids[:n_val])
    train_sids = set(sids[n_val:])

    assert len(val_sids & train_sids) == 0

    train_rows = [r for r in rows if r["sentence_id"] in train_sids]
    val_rows = [r for r in rows if r["sentence_id"] in val_sids]

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)

    def _write(rows_, p):
        with open(p, "w") as f:
            for r in rows_:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  wrote {len(rows_):>6d} → {p}")

    _write(train_rows, out_dir / "train.jsonl")
    _write(val_rows, out_dir / "val.jsonl")

    # Summary
    def summarize(name, rs):
        dirs = Counter(r["direction"] for r in rs)
        srcs = Counter(r["source_type"] for r in rs)
        print(f"\n[{name}] n={len(rs)} unique_sids={len({r['sentence_id'] for r in rs})}")
        print(f"  directions: {dict(dirs)}")
        print(f"  source_types: {dict(srcs)}")

    summarize("TOTAL", rows)
    summarize("TRAIN", train_rows)
    summarize("VAL", val_rows)
    overlap = {r["sentence_id"] for r in train_rows} & {r["sentence_id"] for r in val_rows}
    print(f"\nsentence_id overlap train∩val = {len(overlap)} (must be 0)")


if __name__ == "__main__":
    main()
