"""Push the encoded dataset (Mimi tokens + Whisper alignments + splits) to HF Hub.

WHY THIS EXISTS
---------------
Once the data pipeline has produced the ``.pt`` shards locally we
upload them as a private HF dataset so the TPU pod can pull from
HF instead of round-tripping through GCS. The script is a thin
wrapper around ``HfApi.upload_folder`` with skip rules for caches
and tarballs.

CPU-only; run on the data-prep host.

Usage::

    HF_TOKEN=... python scripts/upload_encoded_dataset.py \\
        --data-dir /path/to/phase-3-data-generation-pipeline/data \\
        --repo tiny-aya-translate/fleurs-tr-hi-mimi-encoded
"""

import argparse
import hashlib
import os
from pathlib import Path


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--repo", default="tiny-aya-translate/fleurs-tr-hi-mimi-encoded")
    ap.add_argument("--fallback-repo", default="ahmeterdempmk/fleurs-tr-hi-mimi-encoded")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    accepted_path = data_dir / "manifests" / "accepted.jsonl"
    accepted_hash = _sha256_file(accepted_path) if accepted_path.exists() else "?"

    n_encoded = len(list((data_dir / "encoded").glob("*.pt")))
    n_align = len(list((data_dir / "encoded").glob("*.alignments.json")))
    train_rows = sum(1 for _ in open(data_dir / "splits" / "train.jsonl"))
    val_rows = sum(1 for _ in open(data_dir / "splits" / "val.jsonl"))

    readme = f"""# fleurs-tr-hi-mimi-encoded

Mimi-encoded Turkish↔Hindi parallel speech pairs for TinyAya Stage 2
speech-to-speech translation training.

## Contents

- `encoded/*.pt` — {n_encoded} Mimi-encoded audio pairs (`kyutai/mimi`, 8 codebooks, 12.5 Hz, 24 kHz).
  Each file keys: `pair_id, src_lang, tgt_lang, src_text, tgt_text,
  src_codes[8, T_src], tgt_codes[8, T_tgt]`.
- `encoded/*.alignments.json` — {n_align} Whisper word-level alignment sidecars
  (`.src.alignments.json` / `.tgt.alignments.json`).
- `splits/train.jsonl` — {train_rows} rows (90%)
- `splits/val.jsonl`   — {val_rows} rows (10%)

## Provenance

Sourced from `tiny-aya-translate/fleurs-tr-hi-parallel-speech`
(`accepted.jsonl` sha256 `{accepted_hash}`). Mimi config: `kyutai/mimi`,
`mimi_num_codebooks=8`, `output_sample_rate=24000`. Splits built by
`scripts/make_splits.py` (seed=42, 90/10 keyed on FLEURS sentence_id — zero
overlap).
"""
    (data_dir / "README.md").write_text(readme)

    from huggingface_hub import HfApi, upload_folder

    api = HfApi(token=os.environ.get("HF_TOKEN"))

    repos = [args.repo, args.fallback_repo]
    for repo in repos:
        try:
            api.create_repo(repo, repo_type="dataset", exist_ok=True)
            upload_folder(
                repo_id=repo,
                folder_path=str(data_dir),
                allow_patterns=[
                    "encoded/*.pt",
                    "encoded/*.json",
                    "splits/*.jsonl",
                    "README.md",
                    "manifests/accepted.jsonl",
                    "manifests/stats.json",
                ],
                repo_type="dataset",
                token=os.environ.get("HF_TOKEN"),
            )
            print(f"Uploaded to {repo}")
            return
        except Exception as e:
            print(f"Upload to {repo} failed: {e}")
    raise SystemExit("All upload targets failed")


if __name__ == "__main__":
    main()
