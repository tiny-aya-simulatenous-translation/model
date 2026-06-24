"""Validate a saved checkpoint: load it and verify all tensor shapes + forward pass."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from src.model.composite import TinyAyaMoshiComposite
from src.model.lora_setup import apply_lora
from src.training.checkpointing import load_checkpoint

EXPECTED_SIZES = {
    "projection.pt": 32 * 1024 * 1024,      # ~32MB
    "depth_decoder.pt": 300 * 1024 * 1024,   # ~390MB
    "text_embed.pt": 1500 * 1024 * 1024,     # ~1.7GB
    "audio_heads.pt": 10 * 1024 * 1024,      # ~16MB
    "model_audio_embed.pt": 5 * 1024 * 1024, # ~8MB
}

def validate(ckpt_dir: str, device: str = "cuda"):
    print(f"Validating checkpoint: {ckpt_dir}", flush=True)
    errors = []

    # 1. Check all files exist and have reasonable sizes
    print("\n[1] File presence and sizes:", flush=True)
    required = ["projection.pt", "depth_decoder.pt", "text_embed.pt",
                 "audio_heads.pt", "model_audio_embed.pt", "metadata.json",
                 "peft_adapter/adapter_model.safetensors"]
    for f in required:
        path = os.path.join(ckpt_dir, f)
        if not os.path.exists(path):
            print(f"  FAIL: {f} missing", flush=True)
            errors.append(f"{f} missing")
            continue
        size = os.path.getsize(path)
        basename = os.path.basename(f)
        min_size = EXPECTED_SIZES.get(basename, 1024)
        status = "OK" if size >= min_size else "FAIL (too small)"
        print(f"  {status}: {f} = {size / 1024 / 1024:.1f}MB (min {min_size / 1024 / 1024:.1f}MB)", flush=True)
        if size < min_size:
            errors.append(f"{f} too small: {size}B < {min_size}B")

    if errors:
        print(f"\n*** {len(errors)} ERRORS — checkpoint is corrupted ***", flush=True)
        for e in errors:
            print(f"  - {e}", flush=True)
        return False

    # 2. Load checkpoint into model
    print("\n[2] Loading into model...", flush=True)
    model = TinyAyaMoshiComposite(num_codebooks=8)
    model.backbone = apply_lora(model.backbone, r=16, num_full_ft_layers=0)
    load_checkpoint(model, None, None, ckpt_dir)
    model = model.to(device).to(torch.bfloat16).eval()
    print("  OK: model loaded", flush=True)

    # 3. Verify key tensor shapes
    print("\n[3] Tensor shapes:", flush=True)
    checks = [
        ("projection.weight", model.projection.weight, (4096, 2048)),
        ("audio_heads[0].weight", model.backbone.audio_heads[0].weight, (2048, 2048)),
        ("model_audio_embed.weight", model.backbone.model_audio_embed.weight, (2049, 2048)),
        ("depth_decoder.input_projections", model.depth_decoder.input_projections.weight, (8, 1024, 4096)),
    ]
    for name, param, expected in checks:
        actual = tuple(param.shape)
        status = "OK" if actual == expected else "FAIL"
        print(f"  {status}: {name} = {actual} (expected {expected})", flush=True)
        if actual != expected:
            errors.append(f"{name}: {actual} != {expected}")

    # 4. Forward pass
    print("\n[4] Forward pass...", flush=True)
    T = 50
    text = torch.full((1, T), 262146, dtype=torch.long, device=device)
    user = torch.randint(0, 2048, (1, T), device=device)
    model_audio = torch.full((1, T), 2048, dtype=torch.long, device=device)
    mask = torch.ones(1, T, dtype=torch.long, device=device)
    full_codes = torch.randint(0, 2048, (1, 8, T), dtype=torch.long, device=device)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(text_ids=text, audio_codes=user, model_audio_codes=model_audio,
                    attention_mask=mask, full_audio_codes=full_codes, depth_chunk_size=16)
        text_logits, audio_logits, hidden = out

    print(f"  text_logits:  {text_logits.shape} (expect [1, {T}, 264196])", flush=True)
    print(f"  audio_logits: {audio_logits.shape} (expect [1, 8, {T}, 2048])", flush=True)
    print(f"  hidden:       {hidden.shape} (expect [1, {T}, 2048])", flush=True)

    has_nan = any(torch.isnan(t).any() for t in [text_logits, audio_logits, hidden])
    print(f"  NaN check: {'FAIL' if has_nan else 'OK'}", flush=True)
    if has_nan:
        errors.append("Forward pass produced NaN")

    # 5. Summary
    print(f"\n{'='*60}", flush=True)
    if errors:
        print(f"FAILED: {len(errors)} errors", flush=True)
        for e in errors:
            print(f"  - {e}", flush=True)
        return False
    else:
        print("ALL CHECKS PASSED", flush=True)
        return True


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint_dir")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    ok = validate(args.checkpoint_dir, args.device)
    sys.exit(0 if ok else 1)
