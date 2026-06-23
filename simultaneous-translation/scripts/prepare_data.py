"""Prepare FLEURS data for training -- download, Whisper-annotate, Mimi-encode.

WHY THIS EXISTS
---------------
Stage-1 / Stage-2 datasets read pre-encoded ``.pt`` shards. This
script builds those shards: download a FLEURS slice, run Whisper
to get word-level alignments, encode the audio with Mimi, and
write one ``.pt`` per sample.

GPU-strongly-preferred (Whisper + Mimi both have CUDA kernels).
Run once on a GPU host, upload outputs to GCS, then point the
training data dir at the GCS bucket.

Usage::

    python scripts/prepare_data.py --output_dir data/stage1 --num_samples 30
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).parent.parent))


def download_fleurs(languages=("tr_tr", "hi_in"), num_samples=30):
    """Download FLEURS samples for given languages."""
    from datasets import load_dataset

    all_samples = []
    for lang_code in languages:
        lang_short = lang_code.split("_")[0]  # "tr" or "hi"
        print(f"\nLoading FLEURS {lang_code}...")
        ds = load_dataset("google/fleurs", lang_code, split="train")

        for i, sample in enumerate(ds):
            if i >= num_samples:
                break
            all_samples.append(
                {
                    "audio": sample["audio"]["array"],
                    "sr": sample["audio"]["sampling_rate"],
                    "text": sample["transcription"],
                    "language": lang_short,
                    "id": f"{lang_short}_{i:04d}",
                }
            )
        print(f"  Loaded {min(num_samples, len(ds))} {lang_short} samples")

    return all_samples


def annotate_with_whisper(samples, whisper_model="base"):
    """Run whisper_timestamped on each sample to get word-level alignments."""
    import whisper_timestamped as whisper

    print(f"\nLoading Whisper {whisper_model}...")
    model = whisper.load_model(whisper_model, device="cuda")

    annotated = []
    for i, sample in enumerate(samples):
        audio = np.array(sample["audio"], dtype=np.float32)
        sr = sample["sr"]

        # Whisper expects 16kHz
        if sr != 16000:
            from math import gcd

            g = gcd(sr, 16000)
            audio = resample_poly(audio, 16000 // g, sr // g).astype(np.float32)

        try:
            result = whisper.transcribe(
                model,
                audio,
                language=sample["language"],
                vad="auditok" if len(audio) / 16000 > 10 else None,
                verbose=None,
            )

            alignments = []
            for segment in result.get("segments", []):
                for word in segment.get("words", []):
                    alignments.append(
                        [
                            word["text"],
                            [word["start"], word["end"]],
                            "SPEAKER_MAIN",
                        ]
                    )

            sample["alignments"] = alignments
            annotated.append(sample)

            if (i + 1) % 10 == 0:
                print(f"  Annotated {i + 1}/{len(samples)}")

        except Exception as e:
            print(f"  WARNING: Failed to annotate {sample['id']}: {e}")
            # Still include with empty alignments
            sample["alignments"] = []
            annotated.append(sample)

    print(f"  Annotated {len(annotated)} samples total")
    return annotated


def encode_with_mimi(samples, output_dir: Path):
    """Encode audio samples with Mimi codec and save as .pt + .json files."""
    from src.data.mimi_encoder import MimiEncoder

    print("\nLoading Mimi codec...")
    encoder = MimiEncoder(device="cuda")

    output_dir.mkdir(parents=True, exist_ok=True)

    for i, sample in enumerate(samples):
        audio = torch.from_numpy(np.array(sample["audio"], dtype=np.float32))
        sr = sample["sr"]

        # Encode
        codes = encoder.encode(audio, sr=sr)  # [num_codebooks, T]

        # Save .pt
        pt_path = output_dir / f"{sample['id']}.pt"
        torch.save(
            {
                "audio_codes": codes,
                "text": sample["text"],
                "language": sample["language"],
                "duration_s": len(sample["audio"]) / sr,
            },
            pt_path,
        )

        # Save .json alignment
        json_path = output_dir / f"{sample['id']}.json"
        with open(json_path, "w") as f:
            json.dump({"alignments": sample["alignments"]}, f, ensure_ascii=False)

        if (i + 1) % 10 == 0:
            print(f"  Encoded {i + 1}/{len(samples)}")

    print(f"  Saved {len(samples)} samples to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/stage1")
    parser.add_argument("--num_samples", type=int, default=30, help="Samples per language")
    parser.add_argument("--whisper_model", type=str, default="base")
    parser.add_argument("--languages", nargs="+", default=["tr_tr", "hi_in"])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Step 1: Download
    samples = download_fleurs(args.languages, args.num_samples)

    # Step 2: Annotate
    samples = annotate_with_whisper(samples, args.whisper_model)

    # Step 3: Encode and save
    encode_with_mimi(samples, output_dir)

    print(f"\n{'=' * 60}")
    print("Data preparation complete!")
    print(f"  {len(samples)} samples saved to {output_dir}")
    print("  Each sample has .pt (audio codes) + .json (alignments)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
