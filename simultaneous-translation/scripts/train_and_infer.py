"""Train + infer pipeline for the hierarchical composite (single-host).

WHY THIS EXISTS
---------------
``train_and_infer`` is the legacy single-host training driver, used
on a single GPU box. It trains for ``max_steps`` on the supplied
data dir, saves a final checkpoint, and immediately runs inference
on ``--infer_wavs`` so the user can listen to the result without a
separate eval invocation.

It is explicitly NOT the TPU entry point. Use
``scripts/train_hierarchical.py`` for TPU and any multi-host
training; that one wraps the model with the backend abstraction in
``src/backend/`` so SPMD strategies can be selected via config.

Usage::

    python scripts/train_and_infer.py --data_dir data/stage2_fixed --max_steps 3000 \\
        --infer_wavs deneme1.wav deneme2.wav
"""

import argparse
import os
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.collator import InterleavedCollator
from src.data.dataset import TranslationDataset
from src.data.mimi_encoder import MimiEncoder
from src.model.backbone import TinyAyaBackbone
from src.model.composite import TinyAyaMoshiComposite
from src.model.lora_setup import apply_lora, register_embedding_grad_mask


def freeze_depth_internals(model):
    frozen, kept = 0, 0
    for name, param in model.depth_decoder.named_parameters():
        if any(k in name for k in ("input_projections", "embed_tokens", "lm_heads")):
            param.requires_grad = True
            kept += param.numel()
        else:
            param.requires_grad = False
            frozen += param.numel()
    print(f"Depth decoder: frozen {frozen / 1e6:.0f}M, trainable I/O {kept / 1e6:.0f}M")


def get_param_groups(model, **kwargs):
    groups = {
        "lora": {"params": [], "lr": kwargs.get("lr_lora", 3e-4)},
        "full_ft": {"params": [], "lr": kwargs.get("lr_full_ft", 1e-4)},
        "projection": {"params": [], "lr": kwargs.get("lr_projection", 1e-3)},
        "depth": {"params": [], "lr": kwargs.get("lr_depth", 5e-4)},
        "audio_embed": {"params": [], "lr": kwargs.get("lr_audio_embed", 1e-3)},
        "text_embed": {"params": [], "lr": kwargs.get("lr_text_embed", 1e-3)},
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "projection" in name and "depth" not in name and "input_proj" not in name:
            groups["projection"]["params"].append(param)
        elif "depth_decoder" in name:
            groups["depth"]["params"].append(param)
        elif "text_embed" in name and "depth" not in name:
            groups["text_embed"]["params"].append(param)
        elif "lora_" in name:
            groups["lora"]["params"].append(param)
        elif "embed_tokens" in name:
            groups["audio_embed"]["params"].append(param)
        elif any(f"layers.{i}." in name for i in range(34, 36)):
            groups["full_ft"]["params"].append(param)
        else:
            groups["lora"]["params"].append(param)
    return [g for g in groups.values() if g["params"]]


def compute_loss(
    text_logits, audio_logits, text_targets, audio_targets, attention_mask, loss_mask=None
):
    text_logits = text_logits[:, :-1].contiguous()
    text_targets = text_targets[:, 1:].contiguous()
    mask = attention_mask[:, 1:].bool()
    if loss_mask is not None:
        mask = mask & loss_mask[:, 1:].bool()
    audio_logits = audio_logits[:, :, :-1].contiguous()
    audio_targets = audio_targets[:, :, 1:].contiguous()
    B = mask.shape[0]
    denom = mask.float().sum().clamp(min=1.0)

    num_cb = audio_logits.shape[1]
    cb_losses = []
    for c in range(num_cb):
        cl = F.cross_entropy(
            audio_logits[:, c].reshape(-1, audio_logits.size(-1)),
            audio_targets[:, c].reshape(-1),
            reduction="none",
        ).view(B, -1)
        cb_losses.append((cl * mask.float()).sum() / denom)
    audio_loss = torch.stack(cb_losses).mean()
    return {
        "loss": audio_loss,
        "audio_loss": audio_loss.detach(),
        "cb_losses": [cb.detach().item() for cb in cb_losses],
    }


@torch.no_grad()
def generate_from_source(model, mimi, source_codes, num_codebooks, device, max_target_frames=150):
    """Autoregressive generation with hierarchical depth decoder."""
    model.eval()
    src_cb0 = source_codes[0, :].unsqueeze(0).to(device)  # [1, T_src]
    T_src = src_cb0.shape[1]
    generated_cb0 = src_cb0.clone()  # [1, T_src]
    text_ids = torch.full((1, T_src), TinyAyaBackbone.ZERO_PADDING, dtype=torch.long, device=device)

    all_generated = []
    for step in range(max_target_frames):
        mask = torch.ones(1, generated_cb0.shape[1], dtype=torch.long, device=device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            backbone_out = model.backbone(
                text_ids=text_ids,
                audio_codes=generated_cb0[0],
                attention_mask=mask,
            )
            projected = model.projection(backbone_out["hidden_states"])
            ctx = projected[:, -1:, :]
            ctx_expanded = ctx.expand(1, num_codebooks, -1).contiguous()

            depth_input = torch.zeros(1, num_codebooks, dtype=torch.long, device=device)
            frame_tokens = []
            for cb_idx in range(num_codebooks):
                depth_out = model.depth_decoder(
                    input_ids=depth_input,
                    last_hidden_state=ctx_expanded,
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

        if (step + 1) % 50 == 0:
            print(f"    Generated {step + 1}/{max_target_frames} frames")

    gen_codes = torch.stack(all_generated, dim=1)  # [num_cb, T]
    gen_wav = mimi.decode(gen_codes)  # no zero padding!
    model.train()
    return gen_wav.numpy(), gen_codes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/stage2_fixed")
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--num_codebooks", type=int, default=8)
    parser.add_argument("--depth_chunk_size", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=250)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_dir", type=str, default="checkpoints/final")
    parser.add_argument(
        "--infer_wavs", nargs="*", default=[], help="Wav files to translate after training"
    )
    parser.add_argument("--infer_max_frames", type=int, default=120)
    parser.add_argument("--output_dir", type=str, default="inference_outputs")
    args = parser.parse_args()

    device = "cuda"

    # Check disk space (need 1.5x checkpoint size ~18GB)
    import shutil

    free_gb = shutil.disk_usage("/").free / (1024**3)
    print(f"Disk space: {free_gb:.0f}GB free")
    assert free_gb > 20, f"Not enough disk space! Need 20GB, have {free_gb:.0f}GB"

    # Build model
    print("\n=== Building Model ===")
    model = TinyAyaMoshiComposite(num_codebooks=args.num_codebooks)
    model.backbone = apply_lora(model.backbone, r=16)
    register_embedding_grad_mask(model.backbone)
    freeze_depth_internals(model)
    for p in model.projection.parameters():
        p.requires_grad = True
    model = model.to(device)
    model.backbone.gradient_checkpointing_enable()

    # Mimi loaded AFTER training to avoid OOM (model + mimi + optimizer = too much)
    mimi = None

    # Dataset
    print("\n=== Loading Dataset ===")
    dataset = TranslationDataset(args.data_dir, model.backbone.tokenizer, args.max_seq_len)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=InterleavedCollator(),
        num_workers=0,
        pin_memory=True,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(get_param_groups(model), weight_decay=0.01)

    # === TRAIN ===
    print(f"\n{'=' * 60}")
    print(f"Training: {args.max_steps} steps")
    print(f"{'=' * 60}\n")

    model.train()
    step = 0
    data_iter = iter(loader)
    running_loss = 0
    t0 = time.time()

    while step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        text_ids = batch["text_ids"].to(device)
        all_codes = batch["audio_codes"].to(device)
        cb0 = all_codes[:, 0, :]
        mask = batch["attention_mask"].to(device)
        loss_mask = batch.get("loss_mask")
        if loss_mask is not None:
            loss_mask = loss_mask.to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = model(
                text_ids=text_ids,
                audio_codes=cb0,
                attention_mask=mask,
                full_audio_codes=all_codes[:, : args.num_codebooks, :],
                depth_chunk_size=args.depth_chunk_size,
            )
            losses = compute_loss(
                output["text_logits"],
                output["audio_logits"],
                text_ids,
                all_codes[:, : args.num_codebooks, :],
                mask,
                loss_mask,
            )

        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        step += 1
        running_loss += losses["audio_loss"].item()

        if step % args.log_every == 0:
            cb_str = " ".join(f"{cb:.2f}" for cb in losses["cb_losses"])
            print(
                f"step {step:5d} | audio {running_loss / args.log_every:.4f} | "
                f"per_cb [{cb_str}] | {step / (time.time() - t0):.1f} s/s"
            )
            running_loss = 0

    print(f"\nTraining done: {step} steps in {time.time() - t0:.0f}s")

    # === SAVE CHECKPOINT ===
    print("\n=== Saving Checkpoint ===")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    free_gb = shutil.disk_usage("/").free / (1024**3)
    print(f"Disk space before save: {free_gb:.0f}GB")
    assert free_gb > 15, f"Not enough space to save! {free_gb:.0f}GB free"

    torch.save({"step": step, "model_state_dict": model.state_dict()}, save_dir / "checkpoint.pt")
    ckpt_size = os.path.getsize(save_dir / "checkpoint.pt") / (1024**3)
    print(f"Checkpoint saved: {ckpt_size:.1f}GB")

    # Verify checkpoint loads
    test = torch.load(save_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    print(f"Checkpoint verified: step {test['step']}, {len(test['model_state_dict'])} keys")
    del test

    # === INFERENCE ===
    if args.infer_wavs:
        print(f"\n{'=' * 60}")
        print(f"Inference on {len(args.infer_wavs)} recordings")
        print(f"{'=' * 60}\n")

        # Free optimizer memory and load Mimi for decoding
        del optimizer
        torch.cuda.empty_cache()
        print("Loading Mimi for inference...")
        mimi = MimiEncoder(device=device)

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for wav_path in args.infer_wavs:
            print(f"\nTranslating: {wav_path}")
            audio, sr = sf.read(wav_path)
            audio_tensor = torch.from_numpy(audio).float()
            source_codes = mimi.encode(audio_tensor, sr=sr)
            print(
                f"  Encoded: {source_codes.shape[1]} frames ({source_codes.shape[1] / 12.5:.1f}s)"
            )

            gen_wav, gen_codes = generate_from_source(
                model,
                mimi,
                source_codes,
                args.num_codebooks,
                device,
                max_target_frames=args.infer_max_frames,
            )

            stem = Path(wav_path).stem
            sf.write(str(output_dir / f"{stem}_translated.wav"), gen_wav, 24000)

            # Also save source re-encoded
            src_wav = mimi.decode(source_codes[: args.num_codebooks])
            sf.write(str(output_dir / f"{stem}_source.wav"), src_wav.numpy(), 24000)

            print(f"  Saved: {output_dir}/{stem}_translated.wav")

        print(f"\nAll outputs in {output_dir}/")


if __name__ == "__main__":
    main()
