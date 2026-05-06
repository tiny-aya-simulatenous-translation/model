"""Inference only -- load a checkpoint and translate a list of wavs.

WHY THIS EXISTS
---------------
Minimal inference driver for users who already have a checkpoint
and want to translate a few wavs without standing up the full eval
harness. The companion of ``train_and_infer.py``: when training is
done, the resulting checkpoint can be re-run on different wavs
without paying the training cost again.

GPU-only.

Usage::

    python scripts/infer_only.py --checkpoint checkpoints/final/checkpoint.pt \\
        --infer_wavs deneme1.wav deneme2.wav
"""

import argparse
import sys
from pathlib import Path

import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.mimi_encoder import MimiEncoder
from src.model.backbone import TinyAyaBackbone
from src.model.composite import TinyAyaMoshiComposite
from src.model.lora_setup import apply_lora, register_embedding_grad_mask


@torch.no_grad()
def generate(model, mimi, source_codes, num_codebooks, device, max_frames=120):
    model.eval()
    src_cb0 = source_codes[0, :].unsqueeze(0).to(device)  # [1, T_src]
    generated_cb0 = src_cb0.clone()
    text_ids = torch.full(
        (1, src_cb0.shape[1]), TinyAyaBackbone.ZERO_PADDING, dtype=torch.long, device=device
    )

    all_generated = []
    for step in range(max_frames):
        mask = torch.ones(1, generated_cb0.shape[1], dtype=torch.long, device=device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            backbone_out = model.backbone(
                text_ids=text_ids,
                audio_codes=generated_cb0[0],
                attention_mask=mask,
            )
            projected = model.projection(backbone_out["hidden_states"])
            ctx = projected[:, -1:, :].expand(1, num_codebooks, -1).contiguous()

            depth_input = torch.zeros(1, num_codebooks, dtype=torch.long, device=device)
            frame_tokens = []
            for cb_idx in range(num_codebooks):
                depth_out = model.depth_decoder(
                    input_ids=depth_input,
                    last_hidden_state=ctx,
                    use_cache=False,
                    return_dict=True,
                )
                cb_token = depth_out.logits[0, cb_idx, :].argmax(dim=-1)
                frame_tokens.append(cb_token.cpu())
                if cb_idx + 1 < num_codebooks:
                    depth_input[0, cb_idx + 1] = cb_token

        all_generated.append(torch.stack(frame_tokens))
        next_cb0 = frame_tokens[0].unsqueeze(0).unsqueeze(0).to(device)
        generated_cb0 = torch.cat([generated_cb0, next_cb0], dim=1)
        text_ids = torch.cat(
            [
                text_ids,
                torch.full((1, 1), TinyAyaBackbone.ZERO_PADDING, dtype=torch.long, device=device),
            ],
            dim=1,
        )

        if (step + 1) % 25 == 0:
            print(f"  Generated {step + 1}/{max_frames} frames")

    gen_codes = torch.stack(all_generated, dim=1)
    gen_wav = mimi.decode(gen_codes)
    return gen_wav.numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--infer_wavs", nargs="+", required=True)
    parser.add_argument("--max_frames", type=int, default=120)
    parser.add_argument("--num_codebooks", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="inference_outputs")
    args = parser.parse_args()

    device = "cuda"

    print("=== Loading Model ===")
    model = TinyAyaMoshiComposite(num_codebooks=args.num_codebooks)
    model.backbone = apply_lora(model.backbone, r=16)
    register_embedding_grad_mask(model.backbone)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"Loaded checkpoint: step {ckpt['step']}")
    del ckpt
    model = model.to(device).eval()

    print("=== Loading Mimi ===")
    mimi = MimiEncoder(device=device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for wav_path in args.infer_wavs:
        print(f"\nTranslating: {wav_path}")
        audio, sr = sf.read(wav_path)
        source_codes = mimi.encode(torch.from_numpy(audio).float(), sr=sr)
        print(f"  Encoded: {source_codes.shape[1]} frames ({source_codes.shape[1] / 12.5:.1f}s)")

        gen_wav = generate(model, mimi, source_codes, args.num_codebooks, device, args.max_frames)

        stem = Path(wav_path).stem
        sf.write(str(output_dir / f"{stem}_translated.wav"), gen_wav, 24000)
        src_wav = mimi.decode(source_codes[: args.num_codebooks])
        sf.write(str(output_dir / f"{stem}_source.wav"), src_wav.numpy(), 24000)
        print(f"  Saved: {stem}_translated.wav, {stem}_source.wav")

    print(f"\nAll outputs in {output_dir}/")


if __name__ == "__main__":
    main()
