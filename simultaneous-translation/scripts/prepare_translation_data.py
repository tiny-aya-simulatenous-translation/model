"""Prepare translation data by pairing FLEURS TR and HI samples.

WHY THIS EXISTS
---------------
Stage-2 needs *paired* audio -- the same sentence in two languages
-- to learn the translation direction. FLEURS is a parallel corpus
keyed on ``sentence_id``, so we just look up the matching ID across
languages and write paired ``.pt`` shards.

CPU-only (the heavy Mimi encode happens earlier in
``prepare_data.py``).

Usage:
    python scripts/prepare_translation_data.py --output_dir data/stage2 --num_pairs 25
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


def create_translation_pairs(num_pairs=25):
    """Create TR↔HI pairs from FLEURS parallel data."""
    from datasets import load_dataset

    print("Loading FLEURS Turkish and Hindi...")
    tr_ds = load_dataset("google/fleurs", "tr_tr", split="train")
    hi_ds = load_dataset("google/fleurs", "hi_in", split="train")

    # FLEURS samples are indexed by sentence ID — same index = same sentence
    num_pairs = min(num_pairs, len(tr_ds), len(hi_ds))

    pairs = []
    for i in range(num_pairs):
        tr_sample = tr_ds[i]
        hi_sample = hi_ds[i]
        pairs.append(
            {
                "source": {
                    "audio": tr_sample["audio"]["array"],
                    "sr": tr_sample["audio"]["sampling_rate"],
                    "text": tr_sample["transcription"],
                    "language": "tr",
                },
                "target": {
                    "audio": hi_sample["audio"]["array"],
                    "sr": hi_sample["audio"]["sampling_rate"],
                    "text": hi_sample["transcription"],
                    "language": "hi",
                },
                "id": f"tr_hi_{i:04d}",
            }
        )

    print(f"Created {len(pairs)} TR→HI pairs")

    # Also create reverse pairs (HI→TR)
    reverse_pairs = []
    for i in range(num_pairs):
        hi_sample = hi_ds[i]
        tr_sample = tr_ds[i]
        reverse_pairs.append(
            {
                "source": {
                    "audio": hi_sample["audio"]["array"],
                    "sr": hi_sample["audio"]["sampling_rate"],
                    "text": hi_sample["transcription"],
                    "language": "hi",
                },
                "target": {
                    "audio": tr_sample["audio"]["array"],
                    "sr": tr_sample["audio"]["sampling_rate"],
                    "text": tr_sample["transcription"],
                    "language": "tr",
                },
                "id": f"hi_tr_{i:04d}",
            }
        )

    print(f"Created {len(reverse_pairs)} HI→TR pairs")
    return pairs + reverse_pairs


def encode_pairs(pairs, output_dir: Path):
    """Encode translation pairs with Mimi and save."""
    from src.data.mimi_encoder import MimiEncoder

    print("\nLoading Mimi codec...")
    encoder = MimiEncoder(device="cuda")
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, pair in enumerate(pairs):
        # Encode source
        src_audio = torch.from_numpy(np.array(pair["source"]["audio"], dtype=np.float32))
        src_codes = encoder.encode(src_audio, sr=pair["source"]["sr"])

        # Encode target
        tgt_audio = torch.from_numpy(np.array(pair["target"]["audio"], dtype=np.float32))
        tgt_codes = encoder.encode(tgt_audio, sr=pair["target"]["sr"])

        # Save .pt
        pt_path = output_dir / f"{pair['id']}.pt"
        torch.save(
            {
                "source_audio_codes": src_codes,
                "target_audio_codes": tgt_codes,
                "source_text": pair["source"]["text"],
                "target_text": pair["target"]["text"],
                "source_language": pair["source"]["language"],
                "target_language": pair["target"]["language"],
            },
            pt_path,
        )

        if (i + 1) % 10 == 0:
            print(f"  Encoded {i + 1}/{len(pairs)} pairs")

    print(f"  Saved {len(pairs)} pairs to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/stage2")
    parser.add_argument("--num_pairs", type=int, default=25)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    pairs = create_translation_pairs(args.num_pairs)
    encode_pairs(pairs, output_dir)

    print(f"\n{'=' * 60}")
    print(f"Translation data ready: {len(pairs)} pairs in {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
