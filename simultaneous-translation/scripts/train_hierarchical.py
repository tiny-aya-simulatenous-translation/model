"""Train TinyAyaMoshiComposite with hierarchical codebook generation.

Dual-mode:
  --dataset_mode memory    (50-pair debug, original TranslationDataset)
  --dataset_mode streaming (scale run, StreamingTranslationDataset)

Reads a YAML config (configs/stage2_scale.yaml) as the source of truth; every
field stays overridable via CLI flags (additive argparse).
"""

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.collator import InterleavedCollator
from src.data.dataset import StreamingTranslationDataset, TranslationDataset
from src.model.backbone import TinyAyaBackbone
from src.model.composite import TinyAyaMoshiComposite
from src.model.lora_setup import apply_lora, register_embedding_grad_mask
from src.training.checkpointing import load_checkpoint, prune_checkpoints, save_checkpoint
from src.training.scheduler import WarmupCosineScheduler
from src.training.translation_loss import compute_hierarchical_translation_loss


# ---------------------------------------------------------------------------
# config loading
# ---------------------------------------------------------------------------

DEFAULTS = {
    "data": {"train_split": None, "val_split": None, "encoded_dir": None,
             "max_frames": 300, "audio_frame_rate": 12.5,
             "num_workers": 4, "pin_memory": True},
    "train": {"num_codebooks": 8, "batch_size": 1, "grad_accum": 1,
              "max_steps": 3000, "warmup_steps": 200, "min_lr_ratio": 0.1,
              "depth_chunk_size": 16, "precision": "bfloat16",
              "max_grad_norm": 1.0, "weight_decay": 0.01,
              "adam_beta1": 0.9, "adam_beta2": 0.999, "adam_eps": 1e-8},
    "loss": {"text_weight": 0.1, "audio_weight": 1.0},
    "optim": {"lr_lora": 1.5e-4, "lr_full_ft": 5e-5, "lr_projection": 5e-4,
              "lr_depth": 2.5e-4, "lr_audio_embed": 5e-4, "lr_text_embed": 5e-4},
    "logging": {"log_every": 20, "save_every": 1000, "audio_every": 1000,
                "val_every": 1000, "save_dir": "checkpoints/stage2_scale",
                "wandb_project": "tinyaya-s2s",
                "wandb_run_name": "stage2_scale",
                "use_wandb": False},
}


def _deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            _deep_update(d[k], v)
        else:
            d[k] = v
    return d


def load_config(path: str | None, overrides: dict) -> dict:
    cfg = json.loads(json.dumps(DEFAULTS))  # deep copy
    if path:
        with open(path) as f:
            _deep_update(cfg, yaml.safe_load(f) or {})
    # apply CLI overrides
    for k, v in overrides.items():
        if v is None:
            continue
        for section in cfg:
            if k in cfg[section]:
                cfg[section][k] = v
                break
        else:
            cfg.setdefault("_cli", {})[k] = v
    return cfg


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def freeze_depth_internals(model):
    frozen, kept = 0, 0
    for name, param in model.depth_decoder.named_parameters():
        if any(k in name for k in ("input_projections", "embed_tokens", "lm_heads")):
            param.requires_grad = True
            kept += param.numel()
        else:
            param.requires_grad = False
            frozen += param.numel()
    print(f"Depth decoder: frozen {frozen/1e6:.0f}M, trainable I/O {kept/1e6:.0f}M")


def get_param_groups(model, optim_cfg):
    groups = {
        "lora": {"params": [], "lr": optim_cfg["lr_lora"]},
        "full_ft": {"params": [], "lr": optim_cfg["lr_full_ft"]},
        "projection": {"params": [], "lr": optim_cfg["lr_projection"]},
        "depth": {"params": [], "lr": optim_cfg["lr_depth"]},
        "audio_embed": {"params": [], "lr": optim_cfg["lr_audio_embed"]},
        "text_embed": {"params": [], "lr": optim_cfg["lr_text_embed"]},
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
    result = [g | {"name": n} for n, g in groups.items() if g["params"]]
    print("\n=== Parameter Groups ===")
    for g in result:
        n = sum(p.numel() for p in g["params"])
        print(f"  {g['name']}: {len(g['params'])} tensors, {n/1e6:.1f}M, lr={g['lr']}")
    return result


# ---------------------------------------------------------------------------
# audio demo generation (codebook-0 AR + hierarchical depth)
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_audio_sample(model, dataset, mimi_encoder, device, num_codebooks,
                          sample_idx=0, max_target_frames=80):
    model.eval()
    sample = dataset[sample_idx]
    src_codes_all = sample["audio_codes"]
    src_len = sample["source_length"]
    tgt_len = sample["target_length"]

    src_cb0 = src_codes_all[0, :src_len].unsqueeze(0).to(device)
    generated_cb0 = src_cb0.clone()
    text_ids = torch.full((1, src_len), TinyAyaBackbone.ZERO_PADDING,
                          dtype=torch.long, device=device)

    all_generated = []
    for _ in range(max_target_frames):
        mask = torch.ones(1, generated_cb0.shape[1], dtype=torch.long, device=device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            backbone_out = model.backbone(text_ids=text_ids, audio_codes=generated_cb0,
                                          attention_mask=mask)
            projected = model.projection(backbone_out["hidden_states"])
            ctx = projected[:, -1:, :]
            ctx_expanded = ctx.expand(1, num_codebooks, -1).contiguous()

            depth_input = torch.zeros(1, num_codebooks, dtype=torch.long, device=device)
            frame = []
            for cb_idx in range(num_codebooks):
                depth_out = model.depth_decoder(
                    input_ids=depth_input, last_hidden_state=ctx_expanded,
                    use_cache=False, return_dict=True,
                )
                tok = depth_out.logits[0, cb_idx, :].argmax(dim=-1)
                frame.append(tok.cpu())
                if cb_idx + 1 < num_codebooks:
                    depth_input[0, cb_idx + 1] = tok
        next_tokens = torch.stack(frame)
        all_generated.append(next_tokens)
        next_cb0 = next_tokens[0].unsqueeze(0).unsqueeze(0).to(device)
        generated_cb0 = torch.cat([generated_cb0, next_cb0], dim=1)
        text_pad = torch.full((1, 1), TinyAyaBackbone.ZERO_PADDING, dtype=torch.long, device=device)
        text_ids = torch.cat([text_ids, text_pad], dim=1)

    gen_codes = torch.stack(all_generated, dim=1)  # [CB, T]
    gt_cb0 = src_codes_all[0, src_len:src_len + gen_codes.shape[1]]
    cb0_acc = (gen_codes[0] == gt_cb0).float().mean().item() if len(gt_cb0) > 0 else 0.0

    src_full = src_codes_all[:, :src_len]
    tgt_full = src_codes_all[:, src_len:src_len + tgt_len]
    model.train()
    return {
        "source_wav": mimi_encoder.decode(src_full).numpy(),
        "target_gt_wav": mimi_encoder.decode(tgt_full).numpy(),
        "generated_wav": mimi_encoder.decode(gen_codes).numpy(),
        "cb0_accuracy": cb0_acc,
    }


# ---------------------------------------------------------------------------
# validation loop
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_validation(model, val_loader, device, num_codebooks, depth_chunk_size,
                   loss_cfg) -> dict:
    model.eval()
    sums = {"loss": 0.0, "text": 0.0, "audio": 0.0}
    per_cb_sum = torch.zeros(num_codebooks, device=device)
    cb0_correct = 0.0
    cb0_total = 0.0
    n = 0
    for batch in val_loader:
        text_ids = batch["text_ids"].to(device)
        all_codes = batch["audio_codes"].to(device)
        cb0 = all_codes[:, 0, :]
        mask = batch["attention_mask"].to(device)
        loss_mask = batch["loss_mask"].to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = model(text_ids=text_ids, audio_codes=cb0, attention_mask=mask,
                           full_audio_codes=all_codes[:, :num_codebooks, :],
                           depth_chunk_size=depth_chunk_size)
            audio_targets = all_codes[:, :num_codebooks, :]
            losses = compute_hierarchical_translation_loss(
                output["text_logits"], output["audio_logits"],
                text_ids, audio_targets, mask, loss_mask,
                text_weight=loss_cfg["text_weight"],
                audio_weight=loss_cfg["audio_weight"],
            )
        sums["loss"] += losses["loss"].item()
        sums["text"] += losses["text_loss"].item()
        sums["audio"] += losses["audio_loss"].item()
        per_cb_sum += losses["per_codebook_loss"]

        # cb0 teacher-forced acc on target positions (shifted next-token)
        pred = output["audio_logits"][:, 0, :-1].argmax(dim=-1)  # [B, T-1]
        target = all_codes[:, 0, 1:]
        m = loss_mask[:, 1:].bool() & mask[:, 1:].bool()
        if m.any():
            cb0_correct += (pred[m] == target[m]).float().sum().item()
            cb0_total += m.float().sum().item()
        n += 1
    model.train()
    if n == 0:
        return {}
    return {
        "val/loss": sums["loss"] / n,
        "val/text_loss": sums["text"] / n,
        "val/audio_loss": sums["audio"] / n,
        "val/cb0_acc": (cb0_correct / cb0_total) if cb0_total > 0 else 0.0,
        "val/per_codebook_loss": (per_cb_sum / n).detach().cpu().tolist(),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--dataset_mode", choices=["memory", "streaming"], default="streaming")

    # memory-mode data dir (back-compat for 50-pair smoke)
    p.add_argument("--data_dir", type=str, default=None)

    # streaming-mode splits (override)
    p.add_argument("--train_split", type=str, default=None)
    p.add_argument("--val_split", type=str, default=None)
    p.add_argument("--encoded_dir", type=str, default=None)

    # overridable training knobs
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--grad_accum", type=int, default=None)
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--depth_chunk_size", type=int, default=None)
    p.add_argument("--num_codebooks", type=int, default=None)
    p.add_argument("--warmup_steps", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)

    # logging
    p.add_argument("--log_every", type=int, default=None)
    p.add_argument("--save_every", type=int, default=None)
    p.add_argument("--audio_every", type=int, default=None)
    p.add_argument("--val_every", type=int, default=None)
    p.add_argument("--save_dir", type=str, default=None)
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--use_wandb", type=lambda s: s.lower() in ("1", "true", "yes"),
                   default=None)

    p.add_argument("--resume", type=str, default=None, help="checkpoint dir to resume from")
    return p


def main():
    args = build_parser().parse_args()
    overrides = {k: v for k, v in vars(args).items()
                 if k not in ("config", "dataset_mode", "data_dir", "resume")}
    cfg = load_config(args.config, overrides)
    print("\n=== Effective config ===")
    print(json.dumps(cfg, indent=2, default=str))

    device = "cuda"
    num_codebooks = cfg["train"]["num_codebooks"]
    depth_chunk = cfg["train"]["depth_chunk_size"]
    max_frames = cfg["data"]["max_frames"]

    # ---- model
    print("\n=== Building composite model ===")
    model = TinyAyaMoshiComposite(num_codebooks=num_codebooks)
    model.backbone = apply_lora(model.backbone, r=16)
    register_embedding_grad_mask(model.backbone)
    freeze_depth_internals(model)
    for p in model.projection.parameters():
        p.requires_grad = True
    model = model.to(device)
    model.backbone.gradient_checkpointing_enable()

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total {total/1e9:.2f}B, trainable {trainable/1e6:.0f}M "
          f"({100*trainable/total:.1f}%)")

    # ---- data
    print("\n=== Datasets ===")
    collator = InterleavedCollator()
    if args.dataset_mode == "streaming":
        if not cfg["data"]["train_split"]:
            raise ValueError("train_split required in streaming mode")
        train_ds = StreamingTranslationDataset(
            cfg["data"]["train_split"], model.backbone.tokenizer,
            max_frames=max_frames, audio_frame_rate=cfg["data"]["audio_frame_rate"],
            encoded_dir=cfg["data"]["encoded_dir"],
        )
        val_ds = None
        if cfg["data"]["val_split"] and Path(cfg["data"]["val_split"]).exists():
            val_ds = StreamingTranslationDataset(
                cfg["data"]["val_split"], model.backbone.tokenizer,
                max_frames=max_frames, audio_frame_rate=cfg["data"]["audio_frame_rate"],
                encoded_dir=cfg["data"]["encoded_dir"],
            )
    else:
        train_ds = TranslationDataset(args.data_dir, model.backbone.tokenizer,
                                      max_frames=max_frames)
        val_ds = None

    num_workers = cfg["data"]["num_workers"]
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
        collate_fn=collator, num_workers=num_workers,
        pin_memory=cfg["data"]["pin_memory"], persistent_workers=num_workers > 0,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
            collate_fn=collator, num_workers=num_workers,
            pin_memory=cfg["data"]["pin_memory"], persistent_workers=num_workers > 0,
        )

    # ---- mimi decoder for demos
    print("\n=== Loading Mimi for audio monitoring ===")
    from src.data.mimi_encoder import MimiEncoder
    mimi_encoder = MimiEncoder(device=device)

    # ---- optimizer + scheduler
    param_groups = get_param_groups(model, cfg["optim"])
    optimizer = torch.optim.AdamW(
        param_groups, weight_decay=cfg["train"]["weight_decay"],
        betas=(cfg["train"]["adam_beta1"], cfg["train"]["adam_beta2"]),
        eps=cfg["train"]["adam_eps"],
    )
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_steps=cfg["train"]["warmup_steps"],
        total_steps=cfg["train"]["max_steps"],
        min_lr_ratio=cfg["train"]["min_lr_ratio"],
    )

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(model, optimizer, scheduler, args.resume)
        print(f"Resumed at step {start_step}")

    # ---- wandb
    use_wandb = cfg["logging"]["use_wandb"]
    if use_wandb:
        import wandb
        wandb.init(project=cfg["logging"]["wandb_project"],
                   name=cfg["logging"]["wandb_run_name"],
                   config=cfg)

    # ---- training loop
    save_dir = Path(cfg["logging"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    data_iter = iter(train_loader)
    running = {"loss": 0.0, "text": 0.0, "audio": 0.0, "per_cb": torch.zeros(num_codebooks)}
    t0 = time.time()
    t_last = t0
    best_val = float("inf")

    grad_accum = cfg["train"]["grad_accum"]
    step = start_step
    max_steps = cfg["train"]["max_steps"]
    log_every = cfg["logging"]["log_every"]
    save_every = cfg["logging"]["save_every"]
    audio_every = cfg["logging"]["audio_every"]
    val_every = cfg["logging"]["val_every"]
    text_w = cfg["loss"]["text_weight"]
    audio_w = cfg["loss"]["audio_weight"]

    print(f"\n=== Training: {max_steps} steps, accum={grad_accum}, "
          f"batch={cfg['train']['batch_size']} ===")
    model.train()
    optimizer.zero_grad()

    while step < max_steps:
        # grad accumulation micro-steps
        micro_loss_sum = 0.0
        micro_text = 0.0
        micro_audio = 0.0
        micro_per_cb = torch.zeros(num_codebooks)
        for micro in range(grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            text_ids = batch["text_ids"].to(device)
            all_codes = batch["audio_codes"].to(device)
            cb0 = all_codes[:, 0, :]
            mask = batch["attention_mask"].to(device)
            loss_mask = batch["loss_mask"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                output = model(text_ids=text_ids, audio_codes=cb0, attention_mask=mask,
                               full_audio_codes=all_codes[:, :num_codebooks, :],
                               depth_chunk_size=depth_chunk)
                audio_targets = all_codes[:, :num_codebooks, :]
                losses = compute_hierarchical_translation_loss(
                    output["text_logits"], output["audio_logits"],
                    text_ids, audio_targets, mask, loss_mask,
                    text_weight=text_w, audio_weight=audio_w,
                )
            loss = losses["loss"] / grad_accum
            loss.backward()
            micro_loss_sum += losses["loss"].item()
            micro_text += losses["text_loss"].item()
            micro_audio += losses["audio_loss"].item()
            micro_per_cb += losses["per_codebook_loss"].detach().cpu()

        # macro-step
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg["train"]["max_grad_norm"])
        if not torch.isfinite(torch.tensor(micro_loss_sum / grad_accum)):
            print(f"!!! Non-finite loss at step {step}. Aborting.")
            sys.exit(2)

        optimizer.step()
        scheduler.step(step + 1)
        optimizer.zero_grad()
        step += 1

        running["loss"] += micro_loss_sum / grad_accum
        running["text"] += micro_text / grad_accum
        running["audio"] += micro_audio / grad_accum
        running["per_cb"] += micro_per_cb / grad_accum

        # ---- logging
        if step % log_every == 0:
            avg = {k: (v / log_every if k != "per_cb" else (v / log_every).tolist())
                   for k, v in running.items()}
            now = time.time()
            step_time = (now - t_last) / log_every
            t_last = now
            lrs = {f"train/lr_{g['name']}": g['lr'] for g in optimizer.param_groups
                   if 'name' in g}
            peak_gb = torch.cuda.max_memory_allocated() / 1e9
            alloc_gb = torch.cuda.memory_allocated() / 1e9
            print(f"step {step:6d} | loss {avg['loss']:.4f} | "
                  f"text {avg['text']:.4f} audio {avg['audio']:.4f} | "
                  f"grad {grad_norm:.3f} | {step_time:.2f}s/step | "
                  f"peak {peak_gb:.1f}G")
            if use_wandb:
                import wandb
                log = {
                    "train/loss": avg["loss"],
                    "train/text_loss": avg["text"],
                    "train/audio_loss": avg["audio"],
                    "train/grad_norm": grad_norm.item(),
                    "perf/step_time": step_time,
                    "mem/peak_gb": peak_gb,
                    "mem/allocated_gb": alloc_gb,
                    **lrs,
                }
                for i, v in enumerate(avg["per_cb"]):
                    log[f"train/per_codebook_loss_{i}"] = v
                wandb.log(log, step=step)
            running = {"loss": 0.0, "text": 0.0, "audio": 0.0,
                       "per_cb": torch.zeros(num_codebooks)}
            torch.cuda.reset_peak_memory_stats()

        # ---- audio demo
        if audio_every and step % audio_every == 0:
            try:
                r = generate_audio_sample(model, train_ds, mimi_encoder, device,
                                          num_codebooks, sample_idx=0,
                                          max_target_frames=80)
                print(f"  demo cb0_acc={r['cb0_accuracy']*100:.1f}%")
                ad = save_dir / "audio_samples" / f"step_{step:06d}"
                ad.mkdir(parents=True, exist_ok=True)
                sf.write(ad / "source.wav", r["source_wav"], 24000)
                sf.write(ad / "target_gt.wav", r["target_gt_wav"], 24000)
                sf.write(ad / "generated.wav", r["generated_wav"], 24000)
                if use_wandb:
                    import wandb
                    wandb.log({
                        "audio/source": wandb.Audio(r["source_wav"], sample_rate=24000),
                        "audio/target_gt": wandb.Audio(r["target_gt_wav"], sample_rate=24000),
                        "audio/generated": wandb.Audio(r["generated_wav"], sample_rate=24000),
                        "audio/cb0_accuracy_train": r["cb0_accuracy"],
                    }, step=step)
            except Exception as e:
                print(f"  demo failed: {e}")

        # ---- validation
        if val_loader is not None and val_every and step % val_every == 0:
            print(f"  running validation at step {step}...")
            val = run_validation(model, val_loader, device, num_codebooks,
                                 depth_chunk, cfg["loss"])
            print(f"  val/loss={val['val/loss']:.4f} cb0_acc={val['val/cb0_acc']*100:.1f}%")
            if use_wandb:
                import wandb
                log = {k: v for k, v in val.items() if k != "val/per_codebook_loss"}
                for i, v in enumerate(val["val/per_codebook_loss"]):
                    log[f"val/per_codebook_loss_{i}"] = v
                wandb.log(log, step=step)
            if val["val/loss"] < best_val:
                best_val = val["val/loss"]
                best_dir = save_dir / "best_by_val"
                if best_dir.exists():
                    import shutil
                    shutil.rmtree(best_dir)
                save_checkpoint(model, optimizer, scheduler, step, str(best_dir),
                                extra_state={"best_val_loss": best_val,
                                             "config": cfg})
                print(f"  * new best val — saved to {best_dir}")

        # ---- periodic save + prune
        if save_every and step % save_every == 0:
            d = save_dir / f"step_{step:06d}"
            save_checkpoint(model, optimizer, scheduler, step, str(d),
                            extra_state={"config": cfg})
            prune_checkpoints(str(save_dir), keep_last=5, keep_best="best_by_val")

    # ---- final save
    d = save_dir / f"step_{step:06d}"
    save_checkpoint(model, optimizer, scheduler, step, str(d),
                    extra_state={"config": cfg, "final": True})
    print(f"\nTraining complete: {step} steps in {(time.time()-t0)/60:.1f} min")
    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
