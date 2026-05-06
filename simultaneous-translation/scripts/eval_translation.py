"""Evaluate translation -- run a training sample through the model.

WHY THIS EXISTS
---------------
A lightweight per-sample debugger for the trained composite. Two
modes:

1. ``teacher_forced`` -- feed ground-truth tokens, check predicted
   tokens match. Useful for validating that the loss is actually
   minimised on a sample we *know* is in the training set.
2. ``autoregressive`` -- feed source only, sample target tokens,
   decode with Mimi. Closer to deployment behaviour.

GPU-only.

Usage:
    python scripts/eval_translation.py --checkpoint checkpoints/stage2/checkpoint_step_1000 \
        --data_dir data/stage2 --sample_idx 0
"""

import argparse
import sys
from pathlib import Path

import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.mimi_encoder import MimiEncoder
from src.model.backbone import TinyAyaBackbone
from src.model.lora_setup import apply_lora, register_embedding_grad_mask


def teacher_forced_eval(model, sample, device):
    """Run teacher-forced forward pass and check accuracy."""
    src_codes = sample["source_audio_codes"]  # [CB, T_src]
    tgt_codes = sample["target_audio_codes"]  # [CB, T_tgt]

    T_src = src_codes.shape[1]
    T_tgt = tgt_codes.shape[1]

    # Concatenate source + target (prefix-LM style)
    combined = torch.cat([src_codes, tgt_codes], dim=1)  # [CB, T_total]
    audio_input = combined[0:1, :].to(device)  # [1, T_total] codebook 0

    # Zero padding for text
    text_ids = torch.full(
        (1, audio_input.shape[1]),
        TinyAyaBackbone.ZERO_PADDING,
        dtype=torch.long,
        device=device,
    )
    mask = torch.ones(1, audio_input.shape[1], dtype=torch.long, device=device)

    # Forward
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        output = model(text_ids=text_ids, audio_codes=audio_input, attention_mask=mask)

    # Get predicted audio tokens (argmax)
    pred_audio = output["audio_logits"][0].argmax(dim=-1)  # [T_total, 2048] -> [T_total]
    gt_audio = audio_input[0]  # [T_total]

    # Check accuracy on target portion (shifted by 1 for next-token prediction)
    # Prediction at position t predicts token at t+1
    pred_target = pred_audio[T_src - 1 : -1]  # predictions for target positions
    gt_target = gt_audio[T_src:]  # actual target tokens

    T_eval = min(len(pred_target), len(gt_target))
    pred_target = pred_target[:T_eval]
    gt_target = gt_target[:T_eval]

    correct = (pred_target == gt_target).float().mean().item()

    print("\n=== Teacher-Forced Evaluation ===")
    print(f"Source length: {T_src} frames ({T_src / 12.5:.1f}s)")
    print(f"Target length: {T_tgt} frames ({T_tgt / 12.5:.1f}s)")
    print(f"Target token accuracy: {correct * 100:.1f}%")
    print(f"  Predicted tokens (first 20): {pred_target[:20].tolist()}")
    print(f"  Ground truth    (first 20): {gt_target[:20].tolist()}")

    return pred_target, gt_target


def autoregressive_generate(model, source_codes, device, max_new_tokens=150):
    """Generate target audio tokens autoregressively from source."""
    src = source_codes[0:1, :].to(device)  # [1, T_src] codebook 0
    T_src = src.shape[1]

    # Start with source as prefix
    generated = src.clone()  # [1, T_src]
    text_ids = torch.full(
        (1, T_src),
        TinyAyaBackbone.ZERO_PADDING,
        dtype=torch.long,
        device=device,
    )

    print(f"\nGenerating {max_new_tokens} target tokens from {T_src} source tokens...")

    with torch.no_grad():
        for step in range(max_new_tokens):
            mask = torch.ones(1, generated.shape[1], dtype=torch.long, device=device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                output = model(
                    text_ids=text_ids,
                    audio_codes=generated,
                    attention_mask=mask,
                )

            # Get logits for last position
            logits = output["audio_logits"][0, -1, :]  # [2048]

            # Sample (greedy for overfitting test)
            next_token = logits.argmax(dim=-1).unsqueeze(0).unsqueeze(0)  # [1, 1]

            # Append
            generated = torch.cat([generated, next_token], dim=1)
            text_pad = torch.full(
                (1, 1),
                TinyAyaBackbone.ZERO_PADDING,
                dtype=torch.long,
                device=device,
            )
            text_ids = torch.cat([text_ids, text_pad], dim=1)

            if (step + 1) % 50 == 0:
                print(f"  Generated {step + 1}/{max_new_tokens} tokens")

    # Extract generated target portion
    target_tokens = generated[0, T_src:]  # [max_new_tokens]
    return target_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/stage2")
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="eval_outputs")
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
        Path(args.checkpoint) / "training_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    backbone.load_state_dict(ckpt["model_state_dict"], strict=False)
    backbone = backbone.to(device).eval()
    print(f"Loaded checkpoint from step {ckpt['step']}")

    # Load sample
    data_dir = Path(args.data_dir)
    pt_files = sorted(data_dir.glob("*.pt"))
    sample_path = pt_files[args.sample_idx]
    sample = torch.load(sample_path, weights_only=False)
    print(f"\nSample: {sample_path.name}")
    print(f"  Source ({sample['source_language']}): {sample['source_text'][:80]}...")
    print(f"  Target ({sample['target_language']}): {sample['target_text'][:80]}...")

    # Teacher-forced evaluation
    pred_target, gt_target = teacher_forced_eval(backbone, sample, device)

    # Autoregressive generation
    tgt_len = sample["target_audio_codes"].shape[1]
    gen_tokens = autoregressive_generate(
        backbone,
        sample["source_audio_codes"],
        device,
        max_new_tokens=min(args.max_new_tokens, tgt_len),
    )

    # Compare generated vs ground truth
    gt = sample["target_audio_codes"][0, : len(gen_tokens)]
    gen_accuracy = (gen_tokens.cpu() == gt).float().mean().item()
    print("\n=== Autoregressive Generation ===")
    print(f"Generated {len(gen_tokens)} tokens")
    print(f"Token accuracy vs ground truth: {gen_accuracy * 100:.1f}%")
    print(f"  Generated (first 20): {gen_tokens[:20].tolist()}")
    print(f"  GT target (first 20): {gt[:20].tolist()}")

    # Decode with Mimi
    print("\n=== Decoding with Mimi ===")
    encoder = MimiEncoder(device=device)

    # Decode ground truth source
    src_full = sample["source_audio_codes"]  # [32, T_src]
    src_wav = encoder.decode(src_full)
    sf.write(str(output_dir / "source.wav"), src_wav.numpy(), 24000)
    print(f"  Saved source audio: {output_dir}/source.wav")

    # Decode ground truth target
    tgt_full = sample["target_audio_codes"]  # [32, T_tgt]
    tgt_wav = encoder.decode(tgt_full)
    sf.write(str(output_dir / "target_gt.wav"), tgt_wav.numpy(), 24000)
    print(f"  Saved GT target audio: {output_dir}/target_gt.wav")

    # Decode generated target (codebook 0 only — pad other codebooks with 0)
    gen_full = torch.zeros(32, len(gen_tokens), dtype=torch.long)
    gen_full[0] = gen_tokens.cpu()
    gen_wav = encoder.decode(gen_full)
    sf.write(str(output_dir / "target_generated.wav"), gen_wav.numpy(), 24000)
    print(f"  Saved generated target audio: {output_dir}/target_generated.wav")

    # Also decode teacher-forced prediction
    tf_full = torch.zeros(32, len(pred_target), dtype=torch.long)
    tf_full[0] = pred_target.cpu()
    tf_wav = encoder.decode(tf_full)
    sf.write(str(output_dir / "target_teacher_forced.wav"), tf_wav.numpy(), 24000)
    print(f"  Saved teacher-forced target audio: {output_dir}/target_teacher_forced.wav")

    print(f"\n{'=' * 60}")
    print(f"All outputs saved to {output_dir}/")
    print("  source.wav              - Original Turkish audio")
    print("  target_gt.wav           - Ground truth Hindi audio")
    print("  target_generated.wav    - Model-generated Hindi (autoregressive)")
    print("  target_teacher_forced.wav - Model-predicted Hindi (teacher-forced)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
