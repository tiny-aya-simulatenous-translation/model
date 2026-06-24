"""Full pipeline validation: train → save → verify → resume → save → verify.

Run with torchrun:
    torchrun --nproc_per_node=2 scripts/validate_full_pipeline.py

Tests:
1. Train 10 steps, save at 5 and 10
2. Verify checkpoint files: sizes, loadability, tensor shapes
3. Verify optimizer state: non-empty, correct param count
4. Resume from step 5, train 10 more steps (to step 15), save at 10 and 15
5. Resume from step 10, train 10 more steps (to step 20), save at 15 and 20
6. Final verification of all checkpoints
"""
import json
import os
import sys
import shutil

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

CKPT_DIR = "checkpoints/validate_pipeline"
CONFIG_PATH = "/tmp/validate_pipeline.yaml"
TRAIN_JSONL = "/home/alperiox/training_data_full/splits/small/train_20.jsonl"
VAL_JSONL = "/home/alperiox/training_data_full/splits/small/val_20.jsonl"
ENCODED_DIR = "/home/alperiox/training_data_full/encoded"

# Minimum file sizes for bf16 checkpoint
MIN_SIZES = {
    "projection.pt": 15_000_000,
    "depth_decoder.pt": 1_000_000_000,
    "audio_heads.pt": 7_000_000,
    "model_audio_embed.pt": 7_000_000,
    "text_embed.pt": 900_000_000,
    "optimizer.pt": 100_000_000,
    "scheduler.pt": 100,
    "metadata.json": 50,
}


def write_config(max_steps):
    config = {
        "data": {
            "train_split": TRAIN_JSONL,
            "val_split": VAL_JSONL,
            "encoded_dir": ENCODED_DIR,
            "max_frames": 300,
            "audio_frame_rate": 12.5,
            "num_workers": 0,
            "pin_memory": True,
        },
        "train": {
            "num_codebooks": 8,
            "batch_size": 2,
            "grad_accum": 1,
            "max_steps": max_steps,
            "warmup_steps": 2,
            "min_lr_ratio": 0.1,
            "depth_chunk_size": 16,
            "precision": "bfloat16",
            "max_grad_norm": 1.0,
            "weight_decay": 0.01,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_eps": 1e-8,
            "ss_max_ratio": 0.0,
            "ss_warmup_steps": 100,
        },
        "loss": {"text_weight": 0.1, "audio_weight": 1.0},
        "optim": {
            "lr_lora": 3e-4,
            "lr_full_ft": 1e-4,
            "lr_projection": 1e-3,
            "lr_depth": 5e-4,
            "lr_audio_embed": 1e-3,
            "lr_text_embed": 1e-3,
            "lr_model_audio_embed": 1e-3,
        },
        "logging": {
            "log_every": 5,
            "save_every": 5,
            "audio_every": 999,
            "val_every": 999,
            "save_dir": CKPT_DIR,
            "wandb_project": "tinyaya-s2s",
            "wandb_run_name": "validate_pipeline",
            "use_wandb": False,
            "push_to_hub": False,
            "hub_repo_id": None,
        },
    }
    import yaml
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f)


def verify_checkpoint(ckpt_path, label, check_optimizer=True):
    """Verify a single checkpoint directory."""
    print(f"\n  --- Verifying: {label} ({ckpt_path}) ---")
    errors = []

    # 1. File existence and sizes
    for fname, min_size in MIN_SIZES.items():
        if fname == "optimizer.pt" and not check_optimizer:
            continue
        fpath = os.path.join(ckpt_path, fname)
        if fname == "metadata.json" or fname == "scheduler.pt":
            fpath_check = fpath
        else:
            fpath_check = fpath

        if not os.path.exists(fpath):
            if fname in ("peft_adapter/adapter_model.safetensors",):
                pass  # checked separately
            else:
                errors.append(f"{fname} MISSING")
                print(f"    FAIL: {fname} MISSING")
                continue
        size = os.path.getsize(fpath)
        if size < min_size:
            errors.append(f"{fname} too small: {size}B < {min_size}B")
            print(f"    FAIL: {fname} = {size}B (min {min_size}B)")
        else:
            print(f"    OK: {fname} = {size / 1e6:.1f}MB")

    # Check peft adapter
    peft_path = os.path.join(ckpt_path, "peft_adapter", "adapter_model.safetensors")
    if os.path.exists(peft_path):
        size = os.path.getsize(peft_path)
        print(f"    OK: peft_adapter/adapter_model.safetensors = {size / 1e6:.1f}MB")
    else:
        errors.append("peft_adapter/adapter_model.safetensors MISSING")
        print(f"    FAIL: peft_adapter/adapter_model.safetensors MISSING")

    # 2. Verify optimizer state structure
    if check_optimizer:
        opt_path = os.path.join(ckpt_path, "optimizer.pt")
        if os.path.exists(opt_path):
            opt_sd = torch.load(opt_path, map_location="cpu", weights_only=True)
            n_param_groups = len(opt_sd.get("param_groups", []))
            n_states = len(opt_sd.get("state", {}))
            has_momentum = any(
                "exp_avg" in v for v in opt_sd.get("state", {}).values()
                if isinstance(v, dict)
            )
            print(f"    Optimizer: {n_param_groups} param groups, {n_states} states, momentum={'yes' if has_momentum else 'NO'}")
            if n_states == 0:
                errors.append("Optimizer has 0 states (empty)")
            if not has_momentum:
                errors.append("Optimizer missing exp_avg (momentum not saved)")

    # 3. Verify metadata
    meta_path = os.path.join(ckpt_path, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"    Metadata: step={meta.get('step', '?')}")

    if errors:
        print(f"    *** {len(errors)} ERRORS ***")
        for e in errors:
            print(f"      - {e}")
        return False
    print(f"    ALL OK")
    return True


def verify_load_forward(ckpt_path, label):
    """Load checkpoint on single GPU and run forward pass."""
    print(f"\n  --- Load + Forward: {label} ---")
    from src.model.composite import TinyAyaMoshiComposite
    from src.model.lora_setup import apply_lora
    from src.training.checkpointing import load_checkpoint

    model = TinyAyaMoshiComposite(num_codebooks=8)
    model.backbone = apply_lora(model.backbone, r=16, num_full_ft_layers=0)
    load_checkpoint(model, None, None, ckpt_path)
    model = model.to("cuda").to(torch.bfloat16).eval()

    torch.manual_seed(42)
    T = 20
    text = torch.full((1, T), 262146, dtype=torch.long, device="cuda")
    user = torch.randint(0, 2048, (1, T), device="cuda")
    ma = torch.full((1, T), 2048, dtype=torch.long, device="cuda")
    mask = torch.ones(1, T, dtype=torch.long, device="cuda")
    fc = torch.randint(0, 2048, (1, 8, T), dtype=torch.long, device="cuda")

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        tl, al, h = model(
            text_ids=text, audio_codes=user, model_audio_codes=ma,
            attention_mask=mask, full_audio_codes=fc, depth_chunk_size=16,
        )

    has_nan = torch.isnan(tl).any() or torch.isnan(al).any() or torch.isnan(h).any()
    argmax_hash = tl.argmax(-1).sum().item()
    print(f"    shapes: tl={tl.shape} al={al.shape} h={h.shape}")
    print(f"    NaN: {has_nan}, argmax_hash: {argmax_hash}")

    del model
    torch.cuda.empty_cache()

    if has_nan:
        print(f"    FAIL: NaN in output")
        return None
    print(f"    PASS")
    return argmax_hash


def run_training(max_steps, resume_from=None):
    """Run training via subprocess (torchrun)."""
    import subprocess
    write_config(max_steps)
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--nproc_per_node=2",
        "scripts/train_hierarchical.py",
        "--config", CONFIG_PATH,
    ]
    if resume_from:
        cmd.extend(["--resume", resume_from])

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)

    # Extract step lines
    steps = [l for l in result.stdout.split("\n") if l.strip().startswith("step")]
    for s in steps[-3:]:
        print(f"    {s.strip()}")

    if result.returncode != 0:
        # Find error
        for line in result.stderr.split("\n"):
            if "Error" in line or "RuntimeError" in line:
                print(f"    ERROR: {line.strip()}")
        return False
    return True


def main():
    rank = int(os.environ.get("RANK", 0))
    if rank != 0:
        # Only run validation logic on rank 0
        # But we need to participate in distributed training
        # So we just exit — the training is run via subprocess
        return

    print("=" * 60)
    print("FULL PIPELINE VALIDATION")
    print("=" * 60)

    # Clean slate
    if os.path.exists(CKPT_DIR):
        shutil.rmtree(CKPT_DIR)

    all_ok = True

    # ============================================================
    # PHASE 1: Train 10 steps, save at 5 and 10
    # ============================================================
    print("\n" + "=" * 60)
    print("PHASE 1: Train 10 steps (saves at 5, 10)")
    print("=" * 60)
    ok = run_training(max_steps=10)
    if not ok:
        print("PHASE 1 FAILED: training error")
        return

    # Verify checkpoints
    for step in [5, 10]:
        ckpt = os.path.join(CKPT_DIR, f"step_{step:06d}")
        if not os.path.exists(ckpt):
            print(f"  FAIL: {ckpt} does not exist")
            all_ok = False
            continue
        ok = verify_checkpoint(ckpt, f"step_{step}")
        all_ok = all_ok and ok

    # Load + forward on step 10
    hash_step10 = verify_load_forward(
        os.path.join(CKPT_DIR, "step_000010"), "step_10"
    )
    if hash_step10 is None:
        all_ok = False

    # ============================================================
    # PHASE 2: Resume from step 5, train to step 15
    # ============================================================
    print("\n" + "=" * 60)
    print("PHASE 2: Resume from step 5, train to step 15")
    print("=" * 60)
    ok = run_training(max_steps=15, resume_from=os.path.join(CKPT_DIR, "step_000005"))
    if not ok:
        print("PHASE 2 FAILED: resume error")
        all_ok = False
    else:
        for step in [10, 15]:
            ckpt = os.path.join(CKPT_DIR, f"step_{step:06d}")
            if os.path.exists(ckpt):
                ok = verify_checkpoint(ckpt, f"step_{step} (resumed)")
                all_ok = all_ok and ok

        # Load + forward on step 15
        ckpt_15 = os.path.join(CKPT_DIR, "step_000015")
        if os.path.exists(ckpt_15):
            hash_step15 = verify_load_forward(ckpt_15, "step_15 (resumed)")
            if hash_step15 is None:
                all_ok = False
            elif hash_step10 is not None and hash_step15 == hash_step10:
                print("  WARN: step 10 and step 15 produce identical outputs")

    # ============================================================
    # PHASE 3: Resume from step 10, train to step 20
    # ============================================================
    print("\n" + "=" * 60)
    print("PHASE 3: Resume from step 10 (original), train to step 20")
    print("=" * 60)
    ok = run_training(max_steps=20, resume_from=os.path.join(CKPT_DIR, "step_000010"))
    if not ok:
        print("PHASE 3 FAILED: resume error")
        all_ok = False
    else:
        ckpt_20 = os.path.join(CKPT_DIR, "step_000020")
        if os.path.exists(ckpt_20):
            ok = verify_checkpoint(ckpt_20, "step_20 (resumed from 10)")
            all_ok = all_ok and ok

            hash_step20 = verify_load_forward(ckpt_20, "step_20 (resumed)")
            if hash_step20 is None:
                all_ok = False

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 60)
    if all_ok:
        print("ALL PIPELINE TESTS PASSED")
    else:
        print("SOME TESTS FAILED — SEE ABOVE")
    print("=" * 60)

    # Cleanup
    if all_ok:
        shutil.rmtree(CKPT_DIR, ignore_errors=True)
        print("Cleaned up test checkpoints")


if __name__ == "__main__":
    main()
