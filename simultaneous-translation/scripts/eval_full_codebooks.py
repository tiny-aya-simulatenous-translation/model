"""Evaluate translation using all 32 audio codebooks for intelligible playback.

WHY THIS EXISTS
---------------
``eval_translation.py`` evaluates only codebook 0, which is enough
for a BLEU score but produces unintelligible audio. This script
runs the depth decoder over all 32 codebooks so a human reviewer
can listen to the generated wavs and judge naturalness.

GPU-only -- see ``eval_stage2.py`` for the rationale.

Usage::

    python scripts/eval_full_codebooks.py --checkpoint checkpoints/stage2_full/checkpoint_step_1000
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
    """Generate target audio tokens autoregressively — all 32 codebooks."""
    src_cb0 = source_codes[0:1, :].to(device)  # [1, T_src] codebook 0
    T_src = src_cb0.shape[1]
    generated_cb0 = src_cb0.clone()
    text_ids = torch.full(
        (1, T_src),
        TinyAyaBackbone.ZERO_PADDING,
        dtype=torch.long,
        device=device,
    )

    all_generated = []  # collect [32, 1] per step

    with torch.no_grad():
        for step in range(max_new_tokens):
            mask = torch.ones(1, generated_cb0.shape[1], dtype=torch.long, device=device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                output = model(text_ids=text_ids, audio_codes=generated_cb0, attention_mask=mask)

            # audio_logits: [1, 32, T, 2048] — take last timestep
            logits_all_cb = output["audio_logits"][0, :, -1, :]  # [32, 2048]
            next_tokens = logits_all_cb.argmax(dim=-1)  # [32]
            all_generated.append(next_tokens.cpu())

            # Feed back codebook 0 for next step
            next_cb0 = next_tokens[0].unsqueeze(0).unsqueeze(0).to(device)  # [1, 1]
            generated_cb0 = torch.cat([generated_cb0, next_cb0], dim=1)
            text_pad = torch.full(
                (1, 1), TinyAyaBackbone.ZERO_PADDING, dtype=torch.long, device=device
            )
            text_ids = torch.cat([text_ids, text_pad], dim=1)

            if (step + 1) % 50 == 0:
                print(f"  Generated {step + 1}/{max_new_tokens} frames")

    # Stack: [32, max_new_tokens]
    return torch.stack(all_generated, dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="eval_full")
    parser.add_argument("--num_samples", type=int, default=3)
    args = parser.parse_args()

    device = "cuda"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("=== Loading model ===")
    backbone = TinyAyaBackbone(load_in_bf16=True, num_codebooks=32)
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

    # Process samples
    data_dir = Path("data/stage2")
    tr_hi_files = sorted(data_dir.glob("tr_hi_*.pt"))
    hi_tr_files = sorted(data_dir.glob("hi_tr_*.pt"))

    n = min(args.num_samples, len(tr_hi_files), len(hi_tr_files))

    for i in range(n):
        print(f"\n{'=' * 60}")
        print(f"Sample {i}")
        print(f"{'=' * 60}")

        # English reference
        en_sample = en_ds[i]
        en_audio = np.array(en_sample["audio"]["array"], dtype=np.float32)
        en_sr = en_sample["audio"]["sampling_rate"]

        # === TR → HI ===
        sample = torch.load(tr_hi_files[i], weights_only=False)
        print(f"\nTR→HI: {sample['source_text'][:60]}... → {sample['target_text'][:60]}...")
        print(f"  English: {en_sample['transcription'][:80]}...")

        tgt_len = sample["target_audio_codes"].shape[1]
        gen_codes = autoregressive_generate(
            backbone,
            sample["source_audio_codes"],
            device,
            max_new_tokens=tgt_len,
        )

        # Check CB0 accuracy
        gt_cb0 = sample["target_audio_codes"][0, : gen_codes.shape[1]]
        acc = (gen_codes[0] == gt_cb0).float().mean().item()
        print(f"  CB0 accuracy: {acc * 100:.1f}%")

        sample_dir = output_dir / f"sample_{i}_tr_to_hi"
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Decode source (all 32 CB from GT)
        src_wav = encoder.decode(sample["source_audio_codes"])
        sf.write(str(sample_dir / "1_input_turkish.wav"), src_wav.numpy(), 24000)

        # Decode generated (all 32 CB from model)
        gen_wav = encoder.decode(gen_codes)
        sf.write(str(sample_dir / "2_output_hindi.wav"), gen_wav.numpy(), 24000)

        # English reference
        sf.write(str(sample_dir / "3_english_reference.wav"), en_audio, en_sr)

        # GT target for comparison
        gt_wav = encoder.decode(sample["target_audio_codes"])
        sf.write(str(sample_dir / "4_groundtruth_hindi.wav"), gt_wav.numpy(), 24000)

        print(f"  Saved to {sample_dir}/")

        # === HI → TR ===
        sample = torch.load(hi_tr_files[i], weights_only=False)
        print(f"\nHI→TR: {sample['source_text'][:60]}... → {sample['target_text'][:60]}...")

        tgt_len = sample["target_audio_codes"].shape[1]
        gen_codes = autoregressive_generate(
            backbone,
            sample["source_audio_codes"],
            device,
            max_new_tokens=tgt_len,
        )

        gt_cb0 = sample["target_audio_codes"][0, : gen_codes.shape[1]]
        acc = (gen_codes[0] == gt_cb0).float().mean().item()
        print(f"  CB0 accuracy: {acc * 100:.1f}%")

        sample_dir = output_dir / f"sample_{i}_hi_to_tr"
        sample_dir.mkdir(parents=True, exist_ok=True)

        src_wav = encoder.decode(sample["source_audio_codes"])
        sf.write(str(sample_dir / "1_input_hindi.wav"), src_wav.numpy(), 24000)

        gen_wav = encoder.decode(gen_codes)
        sf.write(str(sample_dir / "2_output_turkish.wav"), gen_wav.numpy(), 24000)

        sf.write(str(sample_dir / "3_english_reference.wav"), en_audio, en_sr)

        gt_wav = encoder.decode(sample["target_audio_codes"])
        sf.write(str(sample_dir / "4_groundtruth_turkish.wav"), gt_wav.numpy(), 24000)

        print(f"  Saved to {sample_dir}/")

    print(f"\n{'=' * 60}")
    print(f"All outputs in {output_dir}/")
    print("  1_input_*.wav        - Source language input")
    print("  2_output_*.wav       - Model translation (all 32 codebooks)")
    print("  3_english_ref.wav    - Same sentence in English")
    print("  4_groundtruth_*.wav  - Actual FLEURS target")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
