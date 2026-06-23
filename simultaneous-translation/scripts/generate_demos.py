"""Generate demo translation samples from FLEURS for showcase.

WHY THIS EXISTS
---------------
Produces a hand-curated demo bundle (4 wavs per sample: source,
generated translation, ground truth, English reference) suitable
for sharing in a write-up. Iterates over a slice of FLEURS,
re-encodes, runs autoregressive decoding, and saves the wavs to
``--out_dir``.

GPU-only.

Usage::
    python scripts/generate_demos.py --checkpoint checkpoints/final/checkpoint.pt --num_samples 5
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
from src.model.composite import TinyAyaMoshiComposite
from src.model.lora_setup import apply_lora, register_embedding_grad_mask


@torch.no_grad()
def generate(model, mimi, source_codes, num_codebooks, device, max_frames):
    model.eval()
    src_cb0 = source_codes[0, :].unsqueeze(0).to(device)
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
            print(f"    {step + 1}/{max_frames}", end="", flush=True)

    print()
    gen_codes = torch.stack(all_generated, dim=1)
    return mimi.decode(gen_codes).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--num_codebooks", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="demo_outputs")
    args = parser.parse_args()

    device = "cuda"

    # Load model
    print("=== Loading Model ===")
    model = TinyAyaMoshiComposite(num_codebooks=args.num_codebooks)
    model.backbone = apply_lora(model.backbone, r=16)
    register_embedding_grad_mask(model.backbone)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"Loaded step {ckpt['step']}")
    del ckpt
    model = model.to(device).eval()

    print("=== Loading Mimi ===")
    mimi = MimiEncoder(device=device)

    # Load English FLEURS for reference
    print("=== Loading FLEURS English ===")
    from datasets import load_dataset

    en_ds = load_dataset("google/fleurs", "en_us", split="train")
    en_by_id = {s["id"]: s for s in en_ds}

    data_dir = Path("data/stage2_fixed")
    output_dir = Path(args.output_dir)

    # TR -> HI samples
    tr_hi_files = sorted(data_dir.glob("tr_hi_*.pt"))[: args.num_samples]
    hi_tr_files = sorted(data_dir.glob("hi_tr_*.pt"))[: args.num_samples]

    for i, pt_path in enumerate(tr_hi_files):
        sample = torch.load(pt_path, weights_only=False)
        sid = sample["sentence_id"]
        tgt_len = sample["target_audio_codes"].shape[1]

        print(f"\n--- TR->HI Sample {i} (ID {sid}) ---")
        print(f"  TR: {sample['source_text'][:70]}...")
        print(f"  HI: {sample['target_text'][:70]}...")
        if sid in en_by_id:
            print(f"  EN: {en_by_id[sid]['transcription'][:70]}...")

        sample_dir = output_dir / f"sample_{i}_tr_to_hi"
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Source
        src_wav = mimi.decode(sample["source_audio_codes"][:8])
        sf.write(str(sample_dir / "1_source_turkish.wav"), src_wav.numpy(), 24000)

        # Generate
        print("  Generating...", end="", flush=True)
        gen_wav = generate(
            model,
            mimi,
            sample["source_audio_codes"],
            args.num_codebooks,
            device,
            max_frames=min(tgt_len, 150),
        )
        sf.write(str(sample_dir / "2_generated_hindi.wav"), gen_wav, 24000)

        # Ground truth
        gt_wav = mimi.decode(sample["target_audio_codes"][:8])
        sf.write(str(sample_dir / "3_groundtruth_hindi.wav"), gt_wav.numpy(), 24000)

        # English reference
        if sid in en_by_id:
            en_audio = np.array(en_by_id[sid]["audio"]["array"], dtype=np.float32)
            en_sr = en_by_id[sid]["audio"]["sampling_rate"]
            sf.write(str(sample_dir / "4_english_reference.wav"), en_audio, en_sr)

        # Save text
        with open(sample_dir / "text.txt", "w") as f:
            f.write(f"Turkish: {sample['source_text']}\n")
            f.write(f"Hindi: {sample['target_text']}\n")
            if sid in en_by_id:
                f.write(f"English: {en_by_id[sid]['transcription']}\n")

        print(f"  Saved to {sample_dir}/")

    for i, pt_path in enumerate(hi_tr_files):
        sample = torch.load(pt_path, weights_only=False)
        sid = sample["sentence_id"]
        tgt_len = sample["target_audio_codes"].shape[1]

        print(f"\n--- HI->TR Sample {i} (ID {sid}) ---")
        print(f"  HI: {sample['source_text'][:70]}...")
        print(f"  TR: {sample['target_text'][:70]}...")

        sample_dir = output_dir / f"sample_{i}_hi_to_tr"
        sample_dir.mkdir(parents=True, exist_ok=True)

        src_wav = mimi.decode(sample["source_audio_codes"][:8])
        sf.write(str(sample_dir / "1_source_hindi.wav"), src_wav.numpy(), 24000)

        print("  Generating...", end="", flush=True)
        gen_wav = generate(
            model,
            mimi,
            sample["source_audio_codes"],
            args.num_codebooks,
            device,
            max_frames=min(tgt_len, 150),
        )
        sf.write(str(sample_dir / "2_generated_turkish.wav"), gen_wav, 24000)

        gt_wav = mimi.decode(sample["target_audio_codes"][:8])
        sf.write(str(sample_dir / "3_groundtruth_turkish.wav"), gt_wav.numpy(), 24000)

        if sid in en_by_id:
            en_audio = np.array(en_by_id[sid]["audio"]["array"], dtype=np.float32)
            en_sr = en_by_id[sid]["audio"]["sampling_rate"]
            sf.write(str(sample_dir / "4_english_reference.wav"), en_audio, en_sr)

        with open(sample_dir / "text.txt", "w") as f:
            f.write(f"Hindi: {sample['source_text']}\n")
            f.write(f"Turkish: {sample['target_text']}\n")
            if sid in en_by_id:
                f.write(f"English: {en_by_id[sid]['transcription']}\n")

        print(f"  Saved to {sample_dir}/")

    print(f"\n{'=' * 60}")
    print(f"All demos saved to {output_dir}/")
    print(f"  {len(tr_hi_files)} TR->HI + {len(hi_tr_files)} HI->TR samples")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
