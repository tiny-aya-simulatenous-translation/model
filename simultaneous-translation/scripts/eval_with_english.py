"""Evaluate translation alongside an English reference wav.

WHY THIS EXISTS
---------------
A side-by-side qualitative listener: source / model output /
English reference. Useful when the trained TR<->HI direction
sounds plausible but we want to confirm intelligibility against
a known-good English baseline (the FLEURS triplet provides EN
parallel audio for free).

GPU-only.

For each sample, outputs three audio files:

1. Source (e.g. Turkish input).
2. Model-generated target (e.g. Hindi translation).
3. English reference (same sentence from FLEURS English)

Usage:
    python scripts/eval_with_english.py --checkpoint checkpoints/stage2/checkpoint_step_1000
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.mimi_encoder import MimiEncoder
from src.model.backbone import TinyAyaBackbone
from src.model.lora_setup import apply_lora, register_embedding_grad_mask


def autoregressive_generate(model, source_codes, device, max_new_tokens=150):
    """Generate target audio tokens autoregressively from source."""
    src = source_codes[0:1, :].to(device)  # codebook 0
    T_src = src.shape[1]
    generated = src.clone()
    text_ids = torch.full(
        (1, T_src),
        TinyAyaBackbone.ZERO_PADDING,
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        for _step in range(max_new_tokens):
            mask = torch.ones(1, generated.shape[1], dtype=torch.long, device=device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                output = model(text_ids=text_ids, audio_codes=generated, attention_mask=mask)
            logits = output["audio_logits"][0, -1, :]
            next_token = logits.argmax(dim=-1).unsqueeze(0).unsqueeze(0)
            generated = torch.cat([generated, next_token], dim=1)
            text_pad = torch.full(
                (1, 1), TinyAyaBackbone.ZERO_PADDING, dtype=torch.long, device=device
            )
            text_ids = torch.cat([text_ids, text_pad], dim=1)

    return generated[0, T_src:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="eval_comparison")
    parser.add_argument("--num_samples", type=int, default=3)
    args = parser.parse_args()

    device = "cuda"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("=== Loading model ===")
    backbone = TinyAyaBackbone(load_in_bf16=True)
    backbone = apply_lora(backbone, r=16)
    register_embedding_grad_mask(backbone)
    ckpt = torch.load(
        Path(args.checkpoint) / "training_state.pt", map_location="cpu", weights_only=False
    )
    backbone.load_state_dict(ckpt["model_state_dict"], strict=False)
    backbone = backbone.to(device).eval()
    print(f"Loaded checkpoint from step {ckpt['step']}")

    # Load Mimi
    print("=== Loading Mimi ===")
    encoder = MimiEncoder(device=device)

    # Load English FLEURS for reference
    print("=== Loading FLEURS English ===")
    from datasets import load_dataset

    en_ds = load_dataset("google/fleurs", "en_us", split="train")

    # Process TR→HI samples
    data_dir = Path("data/stage2")
    tr_hi_files = sorted([f for f in data_dir.glob("tr_hi_*.pt")])
    hi_tr_files = sorted([f for f in data_dir.glob("hi_tr_*.pt")])

    n = min(args.num_samples, len(tr_hi_files), len(hi_tr_files))

    for i in range(n):
        print(f"\n{'=' * 60}")
        print(f"Sample {i}")
        print(f"{'=' * 60}")

        # === TR → HI ===
        sample = torch.load(tr_hi_files[i], weights_only=False)
        print("\nTR→HI:")
        print(f"  Source (TR): {sample['source_text'][:80]}...")
        print(f"  Target (HI): {sample['target_text'][:80]}...")

        # Get English reference
        en_sample = en_ds[i]
        en_audio = np.array(en_sample["audio"]["array"], dtype=np.float32)
        en_sr = en_sample["audio"]["sampling_rate"]
        print(f"  English ref: {en_sample['transcription'][:80]}...")

        # Generate target
        tgt_len = sample["target_audio_codes"].shape[1]
        gen_tokens = autoregressive_generate(
            backbone,
            sample["source_audio_codes"],
            device,
            max_new_tokens=tgt_len,
        )
        gt_tokens = sample["target_audio_codes"][0, : len(gen_tokens)]
        accuracy = (gen_tokens.cpu() == gt_tokens).float().mean().item()
        print(f"  Token accuracy: {accuracy * 100:.1f}%")

        # Decode and save
        sample_dir = output_dir / f"sample_{i}_tr_to_hi"
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Source Turkish
        src_wav = encoder.decode(sample["source_audio_codes"])
        sf.write(str(sample_dir / "1_source_turkish.wav"), src_wav.numpy(), 24000)

        # Generated Hindi
        gen_full = torch.zeros(32, len(gen_tokens), dtype=torch.long)
        gen_full[0] = gen_tokens.cpu()
        gen_wav = encoder.decode(gen_full)
        sf.write(str(sample_dir / "2_generated_hindi.wav"), gen_wav.numpy(), 24000)

        # English reference
        sf.write(str(sample_dir / "3_english_reference.wav"), en_audio, en_sr)

        # Ground truth Hindi target
        tgt_wav = encoder.decode(sample["target_audio_codes"])
        sf.write(str(sample_dir / "4_groundtruth_hindi.wav"), tgt_wav.numpy(), 24000)

        print(f"  Saved to {sample_dir}/")

        # === HI → TR ===
        sample = torch.load(hi_tr_files[i], weights_only=False)
        print("\nHI→TR:")
        print(f"  Source (HI): {sample['source_text'][:80]}...")
        print(f"  Target (TR): {sample['target_text'][:80]}...")

        gen_tokens = autoregressive_generate(
            backbone,
            sample["source_audio_codes"],
            device,
            max_new_tokens=sample["target_audio_codes"].shape[1],
        )
        gt_tokens = sample["target_audio_codes"][0, : len(gen_tokens)]
        accuracy = (gen_tokens.cpu() == gt_tokens).float().mean().item()
        print(f"  Token accuracy: {accuracy * 100:.1f}%")

        sample_dir = output_dir / f"sample_{i}_hi_to_tr"
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Source Hindi
        src_wav = encoder.decode(sample["source_audio_codes"])
        sf.write(str(sample_dir / "1_source_hindi.wav"), src_wav.numpy(), 24000)

        # Generated Turkish
        gen_full = torch.zeros(32, len(gen_tokens), dtype=torch.long)
        gen_full[0] = gen_tokens.cpu()
        gen_wav = encoder.decode(gen_full)
        sf.write(str(sample_dir / "2_generated_turkish.wav"), gen_wav.numpy(), 24000)

        # English reference
        sf.write(str(sample_dir / "3_english_reference.wav"), en_audio, en_sr)

        # Ground truth Turkish target
        tgt_wav = encoder.decode(sample["target_audio_codes"])
        sf.write(str(sample_dir / "4_groundtruth_turkish.wav"), tgt_wav.numpy(), 24000)

        print(f"  Saved to {sample_dir}/")

    print(f"\n{'=' * 60}")
    print(f"All outputs saved to {output_dir}/")
    print("Each sample has:")
    print("  1_source_*.wav         - Input audio")
    print("  2_generated_*.wav      - Model output (translated)")
    print("  3_english_reference.wav - Same sentence in English")
    print("  4_groundtruth_*.wav    - Actual target from FLEURS")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
