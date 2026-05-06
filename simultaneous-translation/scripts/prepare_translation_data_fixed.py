"""Prepare strictly-matched translation pairs from FLEURS.

WHY THIS EXISTS
---------------
``prepare_translation_data.py`` had a subtle bug where some pairs
came from different sentences with similar IDs. This is the fixed
variant: it joins on ``sentence_id`` with strict equality so the
TR / HI / EN triplets are guaranteed to be the same content.

Use this script for any new data prep; the older one is kept only
for reproducing historical experiments.

CPU-only.

Usage:
    python scripts/prepare_translation_data_fixed.py --output_dir data/stage2_fixed --num_pairs 25
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


def create_matched_pairs(num_pairs=25):
    """Create TR↔HI pairs matched by FLEURS sentence ID."""
    from datasets import load_dataset

    print("Loading FLEURS TR, HI, EN...")
    tr_ds = load_dataset("google/fleurs", "tr_tr", split="train", trust_remote_code=True)
    hi_ds = load_dataset("google/fleurs", "hi_in", split="train", trust_remote_code=True)
    en_ds = load_dataset("google/fleurs", "en_us", split="train", trust_remote_code=True)

    # Index by sentence ID
    tr_by_id = {s["id"]: s for s in tr_ds}
    hi_by_id = {s["id"]: s for s in hi_ds}
    en_by_id = {s["id"]: s for s in en_ds}

    # Find common IDs
    common_ids = sorted(set(tr_by_id) & set(hi_by_id) & set(en_by_id))
    print(f"Common sentence IDs: {len(common_ids)}")

    num_pairs = min(num_pairs, len(common_ids))
    selected_ids = common_ids[:num_pairs]

    pairs = []
    for sid in selected_ids:
        tr_s = tr_by_id[sid]
        hi_s = hi_by_id[sid]
        en_s = en_by_id[sid]

        # TR → HI
        pairs.append(
            {
                "source": {
                    "audio": tr_s["audio"]["array"],
                    "sr": tr_s["audio"]["sampling_rate"],
                    "text": tr_s["transcription"],
                    "language": "tr",
                },
                "target": {
                    "audio": hi_s["audio"]["array"],
                    "sr": hi_s["audio"]["sampling_rate"],
                    "text": hi_s["transcription"],
                    "language": "hi",
                },
                "english": en_s["transcription"],
                "sentence_id": sid,
                "id": f"tr_hi_{sid:04d}",
            }
        )

        # HI → TR
        pairs.append(
            {
                "source": {
                    "audio": hi_s["audio"]["array"],
                    "sr": hi_s["audio"]["sampling_rate"],
                    "text": hi_s["transcription"],
                    "language": "hi",
                },
                "target": {
                    "audio": tr_s["audio"]["array"],
                    "sr": tr_s["audio"]["sampling_rate"],
                    "text": tr_s["transcription"],
                    "language": "tr",
                },
                "english": en_s["transcription"],
                "sentence_id": sid,
                "id": f"hi_tr_{sid:04d}",
            }
        )

    print(f"Created {len(pairs)} pairs ({num_pairs} TR→HI + {num_pairs} HI→TR)")

    # Show first 3 for verification
    for i in range(min(3, num_pairs)):
        p = pairs[i * 2]  # TR→HI
        print(f"\n  ID {p['sentence_id']}:")
        print(f"    EN: {p['english'][:70]}...")
        print(f"    TR: {p['source']['text'][:70]}...")
        print(f"    HI: {p['target']['text'][:70]}...")

    return pairs


def encode_pairs(pairs, output_dir: Path):
    """Encode with Mimi and save."""

    from src.data.mimi_encoder import MimiEncoder

    print("\nLoading Mimi...")
    encoder = MimiEncoder(device="cuda")
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, pair in enumerate(pairs):
        src_audio = torch.from_numpy(np.array(pair["source"]["audio"], dtype=np.float32))
        src_codes = encoder.encode(src_audio, sr=pair["source"]["sr"])

        tgt_audio = torch.from_numpy(np.array(pair["target"]["audio"], dtype=np.float32))
        tgt_codes = encoder.encode(tgt_audio, sr=pair["target"]["sr"])

        torch.save(
            {
                "source_audio_codes": src_codes,
                "target_audio_codes": tgt_codes,
                "source_text": pair["source"]["text"],
                "target_text": pair["target"]["text"],
                "english_text": pair["english"],
                "source_language": pair["source"]["language"],
                "target_language": pair["target"]["language"],
                "sentence_id": pair["sentence_id"],
            },
            output_dir / f"{pair['id']}.pt",
        )

        if (i + 1) % 10 == 0:
            print(f"  Encoded {i + 1}/{len(pairs)}")

    print(f"Saved {len(pairs)} pairs to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/stage2_fixed")
    parser.add_argument("--num_pairs", type=int, default=25)
    args = parser.parse_args()

    pairs = create_matched_pairs(args.num_pairs)
    encode_pairs(pairs, Path(args.output_dir))

    print(f"\n{'=' * 60}")
    print(f"Done! {len(pairs)} properly matched pairs in {args.output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
