"""Translate a single wav: Mimi encode, autoregressive decode, Mimi decode.

WHY THIS EXISTS
---------------
End-to-end inference helper for ad-hoc demos. Pipeline:

1. Load the wav, resample to 24 kHz, encode with Mimi.
2. Build the prefix-LM input: source audio + special prompt token.
3. Generate target audio tokens autoregressively (codebook 0
   from the backbone, codebooks 1..N from the depth decoder).
4. Decode the resulting Mimi codes back to a wav and write to
   ``--output``.

GPU-only. Same reasoning as ``eval_stage2.py``: TPU SPMD does not
help sequential generation, and the Mimi codec is GPU-friendly.

Usage::

    python scripts/translate_wav.py --input my_turkish_audio.wav \\
        --checkpoint checkpoints/stage2_32cb/checkpoint_step_3000 \\
        --output translated_hindi.wav
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


def translate(model, encoder, input_wav, max_target_frames=200, device="cuda"):
    """Translate input audio to target language."""
    # Load and encode input
    audio, sr = sf.read(input_wav)
    audio_tensor = torch.from_numpy(audio).float()
    print(f"Input: {len(audio) / sr:.2f}s at {sr}Hz")

    # Encode with Mimi
    source_codes = encoder.encode(audio_tensor, sr=sr)  # [32, T_src]
    T_src = source_codes.shape[1]
    print(f"Encoded: {T_src} frames ({T_src / 12.5:.2f}s)")

    # Autoregressive generation
    src_cb0 = source_codes[0:1, :].to(device)
    generated_cb0 = src_cb0.clone()
    text_ids = torch.full(
        (1, T_src),
        TinyAyaBackbone.ZERO_PADDING,
        dtype=torch.long,
        device=device,
    )

    all_generated = []
    print(f"Generating {max_target_frames} target frames...")

    with torch.no_grad():
        for step in range(max_target_frames):
            mask = torch.ones(1, generated_cb0.shape[1], dtype=torch.long, device=device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                output = model(text_ids=text_ids, audio_codes=generated_cb0, attention_mask=mask)

            logits_all_cb = output["audio_logits"][0, :, -1, :]  # [32, 2048]
            next_tokens = logits_all_cb.argmax(dim=-1)  # [32]
            all_generated.append(next_tokens.cpu())

            next_cb0 = next_tokens[0].unsqueeze(0).unsqueeze(0).to(device)
            generated_cb0 = torch.cat([generated_cb0, next_cb0], dim=1)
            text_pad = torch.full(
                (1, 1), TinyAyaBackbone.ZERO_PADDING, dtype=torch.long, device=device
            )
            text_ids = torch.cat([text_ids, text_pad], dim=1)

            if (step + 1) % 50 == 0:
                print(f"  {step + 1}/{max_target_frames}")

    gen_codes = torch.stack(all_generated, dim=1)  # [32, T_target]
    print(f"Generated: {gen_codes.shape[1]} frames")

    # Decode
    target_audio = encoder.decode(gen_codes)
    return target_audio.numpy(), source_codes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input wav file")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="translated_output.wav")
    parser.add_argument("--max_frames", type=int, default=150)
    args = parser.parse_args()

    device = "cuda"

    print("=== Loading model ===")
    backbone = TinyAyaBackbone(load_in_bf16=True, num_codebooks=32)
    backbone = apply_lora(backbone, r=16)
    register_embedding_grad_mask(backbone)
    ckpt = torch.load(
        Path(args.checkpoint) / "training_state.pt", map_location="cpu", weights_only=False
    )
    backbone.load_state_dict(ckpt["model_state_dict"], strict=False)
    backbone = backbone.to(device).eval()
    print(f"Loaded step {ckpt['step']}")

    print("\n=== Loading Mimi ===")
    encoder = MimiEncoder(device=device)

    print("\n=== Translating ===")
    output_audio, source_codes = translate(
        backbone,
        encoder,
        args.input,
        max_target_frames=args.max_frames,
        device=device,
    )

    sf.write(args.output, output_audio, 24000)
    print(f"\nSaved: {args.output} ({len(output_audio) / 24000:.2f}s)")

    # Also re-encode the source through Mimi for comparison
    source_reconstructed = encoder.decode(source_codes)
    sf.write(
        args.output.replace(".wav", "_source_reencoded.wav"), source_reconstructed.numpy(), 24000
    )
    print(f"Saved source re-encoded: {args.output.replace('.wav', '_source_reencoded.wav')}")


if __name__ == "__main__":
    main()
