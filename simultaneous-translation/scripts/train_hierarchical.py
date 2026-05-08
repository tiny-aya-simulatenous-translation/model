"""Train TinyAyaMoshiComposite with hierarchical codebook generation.

WHY THIS EXISTS
---------------
This is the canonical Stage 2 training entry point. It runs in three
mutually exclusive modes:

* **GPU single-process** -- the script just runs (no ``torchrun``).
* **GPU multi-process** -- launch with ``torchrun`` for DDP. The
  ``backend.init_distributed()`` call detects this via env vars.
* **TPU SPMD** -- one Python process per host (each host drives all
  4 chips). The ``backend.init_distributed()`` call sets up the XLA
  ``Mesh`` and the SPMD partitioner.

GPU vs TPU at a glance
----------------------
* On GPU we move data to ``device`` and call ``model(...)``. PyTorch
  schedules the kernels and CUDA streams hide latency.
* On TPU we *also* call ``model(...)`` -- but the first time XLA
  *traces* the whole forward+backward, *compiles* it to TPU machine
  code, and only then *executes* it. The compiled binary is reused
  for every subsequent step, so the first step is slow (minutes) and
  the rest are fast.
* On TPU we additionally call ``backend.mark_sharding(...)`` on the
  inputs so the SPMD partitioner knows the batch dimension is
  sharded across chips. GPU analogue: ``DistributedSampler`` (but
  here SPMD does it without per-rank Python state).

Config precedence
-----------------
YAML (``--config``) provides the defaults; CLI flags override; the
``DEFAULTS`` dict at the top of the file is the ultimate fallback.
Two new keys plumb the scan_layers / grad-checkpoint feature flags:

* ``train.use_scan_layers`` (bool) -- swap layer stacks for the
  ``scan_utils`` proxy. See PLAN.md Phase 1.
* ``train.xla_grad_checkpoint`` (bool) -- per-layer
  ``torch.utils.checkpoint``. See PLAN.md Phase 2.
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
from torch.utils.data.distributed import DistributedSampler


def _patch_attention_mask_for_bf16() -> None:
    """Replace HF attention mask `torch.finfo(dtype).min` with safe -1e4.

    pytorch/xla #4152: HF transformers builds attention masks using
    `torch.finfo(fp32).min = -3.4e38`; when downcast to bf16 this becomes
    `-inf`, then `(1 - mask) * -inf = NaN`. Replacing the constant with
    -1e4 (well within bf16 representable range, still < softmax cutoff)
    avoids the overflow without changing model semantics.

    Idempotent. Must be called before any model loads.
    """
    try:
        from transformers.modeling_attn_mask_utils import AttentionMaskConverter
    except ImportError:
        return

    original_to_4d = AttentionMaskConverter.to_4d
    original_make_causal = AttentionMaskConverter._make_causal_mask

    SAFE_MIN = -1e4

    def patched_to_4d(self, attention_mask_2d, query_length, dtype, key_value_length=None):
        out = original_to_4d(self, attention_mask_2d, query_length, dtype, key_value_length)
        return out.clamp(min=SAFE_MIN)

    @staticmethod
    def patched_make_causal(input_ids_shape, dtype, device, past_key_values_length=0, sliding_window=None):
        out = original_make_causal(
            input_ids_shape, dtype, device,
            past_key_values_length=past_key_values_length,
            sliding_window=sliding_window,
        )
        return out.clamp(min=SAFE_MIN)

    AttentionMaskConverter.to_4d = patched_to_4d
    AttentionMaskConverter._make_causal_mask = patched_make_causal
    print("[bf16-mask-patch] AttentionMaskConverter patched (clamp >= -1e4)", flush=True)


_patch_attention_mask_for_bf16()


sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backend import get_backend
from src.data.collator import InterleavedCollator
from src.data.dataset import StreamingTranslationDataset, TranslationDataset
from src.model.backbone import TinyAyaBackbone
from src.model.composite import TinyAyaMoshiComposite
from src.model.lora_setup import apply_lora
from src.training.checkpointing import (
    prune_checkpoints,
    push_checkpoint_to_hub,
    save_checkpoint,
)
from src.training.scheduler import WarmupCosineScheduler
from src.training.translation_loss import compute_hierarchical_translation_loss

# ---------------------------------------------------------------------------
# config loading
# ---------------------------------------------------------------------------

DEFAULTS = {
    "backend": "auto",
    "data": {
        "train_split": None,
        "val_split": None,
        "encoded_dir": None,
        "max_frames": 300,
        "audio_frame_rate": 12.5,
        "num_workers": 4,
        "pin_memory": True,
    },
    "train": {
        "num_codebooks": 8,
        "batch_size": 1,
        "grad_accum": 1,
        "max_steps": 3000,
        "warmup_steps": 200,
        "min_lr_ratio": 0.1,
        "depth_chunk_size": 16,
        "precision": "bfloat16",
        "max_grad_norm": 1.0,
        "weight_decay": 0.01,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_eps": 1e-8,
        # TPU-only flags. Both safe to leave True on GPU: the
        # scan proxy falls back to a plain loop, and the grad
        # checkpoint shim falls back to a direct call.
        "use_scan_layers": False,
        "xla_grad_checkpoint": False,
    },
    "loss": {"text_weight": 0.1, "audio_weight": 1.0},
    "optim": {
        "lr_lora": 1.5e-4,
        "lr_full_ft": 5e-5,
        "lr_projection": 5e-4,
        "lr_depth": 2.5e-4,
        "lr_audio_embed": 5e-4,
        "lr_text_embed": 5e-4,
    },
    "logging": {
        "log_every": 20,
        "save_every": 1000,
        "audio_every": 1000,
        "val_every": 1000,
        "save_dir": "checkpoints/stage2_scale",
        "wandb_project": "tinyaya-s2s",
        "wandb_run_name": "stage2_scale",
        "use_wandb": False,
        "push_to_hub": False,
        "hub_repo_id": None,
    },
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
        if k in cfg and not isinstance(cfg[k], dict):
            cfg[k] = v
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
    print(f"Depth decoder: frozen {frozen / 1e6:.0f}M, trainable I/O {kept / 1e6:.0f}M")


def get_param_groups(model, optim_cfg):
    groups = {
        "lora": {"params": [], "lr": optim_cfg["lr_lora"]},
        "full_ft": {"params": [], "lr": optim_cfg["lr_full_ft"]},
        "projection": {"params": [], "lr": optim_cfg["lr_projection"]},
        "depth": {"params": [], "lr": optim_cfg["lr_depth"]},
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
        elif "lora_" in name or "lora_embedding" in name:
            groups["lora"]["params"].append(param)
        elif any(f"layers.{i}." in name for i in range(34, 36)):
            groups["full_ft"]["params"].append(param)
        else:
            groups["lora"]["params"].append(param)
    result = [g | {"name": n} for n, g in groups.items() if g["params"]]
    print("\n=== Parameter Groups ===")
    for g in result:
        n = sum(p.numel() for p in g["params"])
        print(f"  {g['name']}: {len(g['params'])} tensors, {n / 1e6:.1f}M, lr={g['lr']}")
    return result


# ---------------------------------------------------------------------------
# audio demo generation (codebook-0 AR + hierarchical depth)
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_audio_sample(
    model,
    dataset,
    mimi_encoder,
    device,
    num_codebooks,
    sample_idx=0,
    max_target_frames=80,
    backend=None,
):
    model.eval()
    sample = dataset[sample_idx]
    src_codes_all = sample["audio_codes"]
    src_len = sample["source_length"]
    tgt_len = sample["target_length"]

    src_cb0 = src_codes_all[0, :src_len].unsqueeze(0).to(device)
    generated_cb0 = src_cb0.clone()
    text_ids = torch.full(
        (1, src_len), TinyAyaBackbone.ZERO_PADDING, dtype=torch.long, device=device
    )

    all_generated = []
    for _ in range(max_target_frames):
        mask = torch.ones(1, generated_cb0.shape[1], dtype=torch.long, device=device)
        autocast = (
            backend.autocast_context(dtype=torch.bfloat16)
            if backend
            else torch.amp.autocast("cuda", dtype=torch.bfloat16)
        )
        with autocast:
            backbone_out = model.backbone(
                text_ids=text_ids, audio_codes=generated_cb0, attention_mask=mask
            )
            projected = model.projection(backbone_out["hidden_states"])
            ctx = projected[:, -1:, :]
            ctx_expanded = ctx.expand(1, num_codebooks, -1).contiguous()

            depth_input = torch.zeros(1, num_codebooks, dtype=torch.long, device=device)
            frame = []
            for cb_idx in range(num_codebooks):
                depth_out = model.depth_decoder(
                    input_ids=depth_input,
                    last_hidden_state=ctx_expanded,
                    use_cache=False,
                    return_dict=True,
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
    gt_cb0 = src_codes_all[0, src_len : src_len + gen_codes.shape[1]]
    cb0_acc = (gen_codes[0] == gt_cb0).float().mean().item() if len(gt_cb0) > 0 else 0.0

    src_full = src_codes_all[:, :src_len]
    tgt_full = src_codes_all[:, src_len : src_len + tgt_len]
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
def run_validation(
    model,
    val_loader,
    device,
    num_codebooks,
    depth_chunk_size,
    loss_cfg,
    *,
    is_ddp=False,
    backend=None,
) -> dict:
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

        autocast = (
            backend.autocast_context(dtype=torch.bfloat16)
            if backend
            else torch.amp.autocast("cuda", dtype=torch.bfloat16)
        )
        with autocast:
            output = model(
                text_ids=text_ids,
                audio_codes=cb0,
                attention_mask=mask,
                full_audio_codes=all_codes[:, :num_codebooks, :],
                depth_chunk_size=depth_chunk_size,
            )
            text_logits, audio_logits, _ = output
            audio_targets = all_codes[:, :num_codebooks, :]
            losses = compute_hierarchical_translation_loss(
                text_logits,
                audio_logits,
                text_ids,
                audio_targets,
                mask,
                loss_mask,
                text_weight=loss_cfg["text_weight"],
                audio_weight=loss_cfg["audio_weight"],
            )
        sums["loss"] += losses["loss"].item()
        sums["text"] += losses["text_loss"].item()
        sums["audio"] += losses["audio_loss"].item()
        per_cb_sum += losses["per_codebook_loss"]

        # cb0 teacher-forced acc on target positions (shifted next-token)
        pred = audio_logits[:, 0, :-1].argmax(dim=-1)  # [B, T-1]
        target = all_codes[:, 0, 1:]
        m = loss_mask[:, 1:].bool() & mask[:, 1:].bool()
        if m.any():
            cb0_correct += (pred[m] == target[m]).float().sum().item()
            cb0_total += m.float().sum().item()
        n += 1

    # all-reduce across ranks so every rank sees identical metrics
    if backend and backend.world_size() > 1:
        for k in sums:
            t = torch.tensor(sums[k], device=device)
            t = backend.reduce_mean(t) * backend.world_size()  # reduce_mean then scale back to sum
            sums[k] = t.item()
        # For per_cb_sum and counts, use the same pattern
        t = torch.tensor([cb0_correct, cb0_total, float(n)], device=device)
        t = backend.reduce_mean(t) * backend.world_size()
        cb0_correct, cb0_total, n = t.tolist()
        per_cb_sum_t = backend.reduce_mean(per_cb_sum) * backend.world_size()
        per_cb_sum = per_cb_sum_t

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
    p.add_argument("--use_wandb", type=lambda s: s.lower() in ("1", "true", "yes"), default=None)

    p.add_argument("--push_to_hub", type=lambda s: s.lower() in ("1", "true", "yes"), default=None)
    p.add_argument("--hub_repo_id", type=str, default=None)

    p.add_argument("--backend", type=str, default=None, choices=["auto", "gpu", "tpu"])
    p.add_argument("--resume", type=str, default=None, help="checkpoint dir to resume from")
    return p


def main():
    args = build_parser().parse_args()
    overrides = {
        k: v
        for k, v in vars(args).items()
        if k not in ("config", "dataset_mode", "data_dir", "resume")
    }
    cfg = load_config(args.config, overrides)

    # ---- distributed init (GPU or TPU)
    backend_type = cfg.get("backend", "auto")
    if backend_type == "auto":
        backend_type = None  # auto-detect
    backend = get_backend(backend_type)
    backend.init_distributed()
    device = backend.get_device()
    is_main = backend.is_main_process()

    # iter 18 lever 4: start the XLA profiler server so external tools
    # (capture_profile.py / TensorBoard) can attach. Rank-0 only --
    # multi-host pods would collide on port 9012. Single-host v6e-8
    # has one process so this is unconditional once is_main passes.
    is_tpu_early = (cfg.get("backend", "auto") == "tpu")
    if is_tpu_early and is_main:
        try:
            import torch_xla.debug.profiler as xp
            xp_port = int(os.environ.get("XLA_PROFILER_PORT", "9012"))
            xp.start_server(xp_port)
            print(f"[profiler] xp.start_server listening on :{xp_port}",
                  flush=True)
        except Exception as e:
            print(f"[profiler] xp.start_server failed: {e}", flush=True)

    torch.manual_seed(42 + int(os.environ.get("LOCAL_RANK", 0)))

    if is_main:
        print("\n=== Effective config ===")
        print(json.dumps(cfg, indent=2, default=str))

    num_codebooks = cfg["train"]["num_codebooks"]
    depth_chunk = cfg["train"]["depth_chunk_size"]
    max_frames = cfg["data"]["max_frames"]

    # ---- model
    if is_main:
        print("\n=== Building composite model ===")
    use_scan = bool(cfg["train"].get("use_scan_layers", False))
    use_xla_ckpt = bool(cfg["train"].get("xla_grad_checkpoint", False))
    model = TinyAyaMoshiComposite(
        num_codebooks=num_codebooks,
        use_scan_layers=use_scan,
        xla_grad_checkpoint=use_xla_ckpt,
    )
    model.backbone = apply_lora(model.backbone, r=16)
    freeze_depth_internals(model)
    for p in model.projection.parameters():
        p.requires_grad = True
    model = model.to(device)

    # Gradient checkpointing strategy:
    #   * GPU: enable HF's per-layer grad checkpoint here, before DDP
    #     wrapping. This is the standard HF Trainer pattern.
    #   * TPU: do NOT call HF's gradient_checkpointing_enable -- in
    #     torch_xla 2.9 it raises from torch._get_device_module("xla").
    #     We use the ``xla_grad_checkpoint`` flag instead, which routes
    #     through scan_utils.
    if cfg.get("backend", "auto") != "tpu":
        model.backbone.gradient_checkpointing_enable()
    model = backend.wrap_model(model)
    if hasattr(backend, "diagnose"):
        backend.diagnose("post-wrap")
    unwrapped = model.module if hasattr(model, "module") else model

    total = sum(p.numel() for p in unwrapped.parameters())
    trainable = sum(p.numel() for p in unwrapped.parameters() if p.requires_grad)
    if is_main:
        print(
            f"Total {total / 1e9:.2f}B, trainable {trainable / 1e6:.0f}M "
            f"({100 * trainable / total:.1f}%)"
        )

    # ---- data
    if is_main:
        print("\n=== Datasets ===")
    # On TPU we MUST pad every batch to the same length, otherwise XLA
    # recompiles per unique sequence-length seen (pytorch/xla #4203 /
    # https://docs.pytorch.org/xla/master/perf/recompilation.html). On
    # GPU/CPU dynamic shapes are fine, so we only force fixed padding
    # when the backend is TPU.
    pad_to = cfg["data"].get("max_frames") if cfg.get("backend", "auto") == "tpu" else None
    collator = InterleavedCollator(pad_to=pad_to)
    if args.dataset_mode == "streaming":
        if not cfg["data"]["train_split"]:
            raise ValueError("train_split required in streaming mode")
        train_ds = StreamingTranslationDataset(
            cfg["data"]["train_split"],
            unwrapped.backbone.tokenizer,
            max_frames=max_frames,
            audio_frame_rate=cfg["data"]["audio_frame_rate"],
            encoded_dir=cfg["data"]["encoded_dir"],
        )
        val_ds = None
        if cfg["data"]["val_split"] and Path(cfg["data"]["val_split"]).exists():
            val_ds = StreamingTranslationDataset(
                cfg["data"]["val_split"],
                unwrapped.backbone.tokenizer,
                max_frames=max_frames,
                audio_frame_rate=cfg["data"]["audio_frame_rate"],
                encoded_dir=cfg["data"]["encoded_dir"],
            )
    else:
        train_ds = TranslationDataset(
            args.data_dir, unwrapped.backbone.tokenizer, max_frames=max_frames
        )
        val_ds = None

    num_workers = cfg["data"]["num_workers"]
    is_tpu = cfg.get("backend", "auto") == "tpu"
    if backend.world_size() > 1 and not is_tpu:
        num_workers = min(num_workers, 6)

    if is_tpu:
        # SPMD is single-process -- no distributed sampler
        train_sampler = None
        val_sampler = None
    elif backend.world_size() > 1:
        train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)
        val_sampler = DistributedSampler(val_ds, shuffle=False, drop_last=False) if val_ds else None
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=cfg["data"]["pin_memory"] and not is_tpu,
        persistent_workers=num_workers > 0 and not is_tpu,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=cfg["train"]["batch_size"],
            shuffle=False,
            sampler=val_sampler,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=cfg["data"]["pin_memory"] and not is_tpu,
            persistent_workers=num_workers > 0 and not is_tpu,
        )

    # ---- mimi decoder for demos
    if is_main:
        print("\n=== Loading Mimi for audio monitoring ===")
    from src.data.mimi_encoder import MimiEncoder

    mimi_device = "cpu" if is_tpu else device
    mimi_encoder = MimiEncoder(device=mimi_device)

    # ---- optimizer + scheduler
    param_groups = get_param_groups(unwrapped, cfg["optim"])
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=cfg["train"]["weight_decay"],
        betas=(cfg["train"]["adam_beta1"], cfg["train"]["adam_beta2"]),
        eps=cfg["train"]["adam_eps"],
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=cfg["train"]["warmup_steps"],
        total_steps=cfg["train"]["max_steps"],
        min_lr_ratio=cfg["train"]["min_lr_ratio"],
    )

    from src.training.checkpointing import find_latest_checkpoint, load_checkpoint_with_backend

    start_step = 0
    resume_dir = args.resume
    if resume_dir == "auto":
        resume_dir = find_latest_checkpoint(cfg["logging"]["save_dir"])
    if resume_dir:
        start_step = load_checkpoint_with_backend(
            unwrapped, optimizer, scheduler, resume_dir, backend
        )
        if is_main:
            print(f"Resumed at step {start_step}")

    # ---- wandb
    # On a multi-host TPU pod (v4-32 = 4 hosts) every host calls this
    # block. Without coordination we get 4 separate runs all named the
    # same, with fragmented loss/system metrics. We use wandb shared-mode
    # (>= 0.19.9): the GLOBAL primary creates the run, broadcasts its
    # run-id via a GCS rendezvous file, and the 3 worker hosts attach
    # to the same run with x_primary=False. Result: ONE wandb dashboard
    # with global metrics from primary + per-host system stats labelled
    # rank_0..rank_3. Pattern: docs.wandb.ai/guides/track/log/
    # distributed-training#track-all-processes-to-a-single-run
    use_wandb = cfg["logging"]["use_wandb"]
    if use_wandb:
        import wandb

        try:
            host_idx = backend.host_index() if hasattr(backend, "host_index") else 0
        except Exception:
            host_idx = 0

        rendezvous_uri = os.environ.get(
            "WANDB_RENDEZVOUS_URI",
            f"gs://tinyaya-stage2-tpu/wandb-rendezvous/{cfg['logging']['wandb_run_name']}.id",
        )

        if is_tpu and not is_main:
            # Worker host: wait for primary to publish run-id, then attach.
            run_id = None
            for attempt in range(60):
                try:
                    import subprocess as _sp
                    out = _sp.run(
                        ["gsutil", "cat", rendezvous_uri],
                        capture_output=True, text=True, timeout=10,
                    )
                    if out.returncode == 0 and out.stdout.strip():
                        run_id = out.stdout.strip()
                        break
                except Exception:
                    pass
                time.sleep(5)

            if run_id:
                wandb.init(
                    project=cfg["logging"]["wandb_project"],
                    id=run_id,
                    settings=wandb.Settings(
                        mode="shared",
                        x_label=f"rank_{host_idx}",
                        x_primary=False,
                        x_update_finish_state=False,
                    ),
                    config=cfg,
                )
                print(f"[wandb] worker host {host_idx} attached to shared run {run_id}",
                      flush=True)
            else:
                print(f"[wandb] worker host {host_idx} failed to find rendezvous; "
                      f"disabling wandb on this host", flush=True)
                use_wandb = False
        elif is_main:
            # Primary host: create the run, publish run-id.
            wandb.init(
                project=cfg["logging"]["wandb_project"],
                name=cfg["logging"]["wandb_run_name"],
                config=cfg,
                settings=wandb.Settings(
                    mode="shared",
                    x_label="rank_0",
                    x_primary=True,
                ) if is_tpu else None,
            )
            if is_tpu:
                run_id = wandb.run.id
                try:
                    import subprocess as _sp, tempfile as _tf
                    with _tf.NamedTemporaryFile("w", suffix=".id", delete=False) as f:
                        f.write(run_id)
                        local_path = f.name
                    _sp.run(["gsutil", "cp", local_path, rendezvous_uri],
                            check=False, capture_output=True, timeout=20)
                    print(f"[wandb] primary published run_id={run_id} to {rendezvous_uri}",
                          flush=True)
                except Exception as e:
                    print(f"[wandb] primary failed to publish run_id: {e}", flush=True)
        else:
            # Non-main on single-host (CPU/GPU): no-op.
            use_wandb = False

    # ---- hub push config
    push_to_hub = cfg["logging"]["push_to_hub"] and is_main
    hub_repo_id = cfg["logging"]["hub_repo_id"]
    hub_token = os.environ.get("HF_TOKEN")

    # ---- training loop
    save_dir = Path(cfg["logging"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    if train_sampler is not None:
        train_sampler.set_epoch(0)
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

    if is_main:
        print(
            f"\n=== Training: {max_steps} steps, accum={grad_accum}, "
            f"batch={cfg['train']['batch_size']} ==="
        )
    model.train()
    optimizer.zero_grad()

    # On TPU/SPMD, calling .item() / .cpu() inside the training loop forces
    # an XLA materialise-and-transfer that BLOCKS until the entire lazy
    # graph compiles + executes. With heterogeneous LoRA + full_ft groups
    # and grad_accum=2, this can take 30-60 minutes per step (see
    # pytorch/xla #4203, #8020, troubleshooting "_local_scalar_dense
    # typically indicates CPU context access"). We keep the running
    # accumulators as XLA tensors and materialise once per log_every.
    if is_tpu:
        running_xla = {
            "loss": torch.tensor(0.0, device=device),
            "text": torch.tensor(0.0, device=device),
            "audio": torch.tensor(0.0, device=device),
        }

    # iter 18 lever 4: auto-capture a 30s profiler trace at step 30.
    # Spans steps ~30-43 at 2.36s/step -> covers compile-cache-warm
    # steady state. Trace lands at /tmp/xla_profile/iter18-<ts>/ and
    # the launcher post-train hook gsutil-uploads to GCS.
    profile_logdir = None
    profile_thread = None
    if is_tpu and is_main:
        ts = int(time.time())
        profile_logdir = f"/tmp/xla_profile/iter18-{ts}"
        try:
            os.makedirs(profile_logdir, exist_ok=True)
        except Exception as e:
            print(f"[profiler] mkdir {profile_logdir} failed: {e}", flush=True)
            profile_logdir = None

    while step < max_steps:
        # iter 18 lever 4: kick off a background trace capture once we
        # are past the first few compile cycles.
        if (
            is_tpu and is_main and step == 30
            and profile_thread is None and profile_logdir is not None
        ):
            try:
                import threading as _threading
                import torch_xla.debug.profiler as xp

                def _capture():
                    try:
                        xp.trace(
                            f"localhost:{int(os.environ.get('XLA_PROFILER_PORT', '9012'))}",
                            profile_logdir,
                            duration_ms=30000,
                        )
                        print(f"[profiler] trace captured to {profile_logdir}",
                              flush=True)
                    except Exception as ex:
                        print(f"[profiler] capture failed: {ex}", flush=True)

                profile_thread = _threading.Thread(target=_capture, daemon=True)
                profile_thread.start()
                print(f"[profiler] trace capture started -> {profile_logdir}",
                      flush=True)
            except Exception as e:
                print(f"[profiler] trace start failed: {e}", flush=True)

        # iter 18 lever 4: per-step profiler scope removed.
        # Both xp.StepTrace AND xp.Trace open an XLA "scope" that is
        # asserted-empty by `_xla_step_marker` on every mark_step
        # call. Our step body already issues 2-3 mark_steps (macro
        # step + logging + diagnose), so any wrapping scope triggers
        # `RuntimeError: Expecting scope to be empty but it is ...`.
        # The auto-captured 30s trace at step 30 (via xp.trace into
        # a logdir) still shows the full HLO timeline with mark_step
        # events; we just lose the per-step text labels in TensorBoard.
        _step_trace = None

        # grad accumulation micro-steps
        if is_tpu:
            micro_loss_sum_xla = torch.tensor(0.0, device=device)
            micro_text_xla = torch.tensor(0.0, device=device)
            micro_audio_xla = torch.tensor(0.0, device=device)
        else:
            micro_loss_sum = 0.0
            micro_text = 0.0
            micro_audio = 0.0
        micro_per_cb = torch.zeros(num_codebooks)
        for micro in range(grad_accum):
            sync_ctx = (
                contextlib.nullcontext() if micro == grad_accum - 1 else backend.no_sync(model)
            )
            with sync_ctx:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    if train_sampler is not None:
                        train_sampler.set_epoch(step)
                    data_iter = iter(train_loader)
                    batch = next(data_iter)

                text_ids = batch["text_ids"].to(device)
                all_codes = batch["audio_codes"].to(device)
                cb0 = all_codes[:, 0, :]
                mask = batch["attention_mask"].to(device)
                loss_mask = batch["loss_mask"].to(device)

                # Mark input sharding for TPU SPMD
                if hasattr(backend, "mark_sharding"):
                    backend.mark_sharding(text_ids, ("fsdp", None))
                    backend.mark_sharding(all_codes, ("fsdp", None, None))
                    backend.mark_sharding(mask, ("fsdp", None))
                    backend.mark_sharding(loss_mask, ("fsdp", None))

                with backend.autocast_context(dtype=torch.bfloat16):
                    output = model(
                        text_ids=text_ids,
                        audio_codes=cb0,
                        attention_mask=mask,
                        full_audio_codes=all_codes[:, :num_codebooks, :],
                        depth_chunk_size=depth_chunk,
                    )
                    text_logits, audio_logits, _ = output
                    audio_targets = all_codes[:, :num_codebooks, :]
                    losses = compute_hierarchical_translation_loss(
                        text_logits,
                        audio_logits,
                        text_ids,
                        audio_targets,
                        mask,
                        loss_mask,
                        text_weight=text_w,
                        audio_weight=audio_w,
                    )
                loss = losses["loss"] / grad_accum
                loss.backward()
            if is_tpu:
                micro_loss_sum_xla = micro_loss_sum_xla + losses["loss"].detach()
                micro_text_xla = micro_text_xla + losses["text_loss"].detach()
                micro_audio_xla = micro_audio_xla + losses["audio_loss"].detach()
            else:
                micro_loss_sum += losses["loss"].item()
                micro_text += losses["text_loss"].item()
                micro_audio += losses["audio_loss"].item()
                micro_per_cb += losses["per_codebook_loss"].detach().cpu()

        # macro-step.
        # On FSDPv2 SPMD, torch.nn.utils.clip_grad_norm_ deadlocks at
        # the all-reduce barrier with heterogeneous parameter groups.
        # The official FSDPv2 example
        # (https://github.com/pytorch/xla/blob/master/examples/fsdp/
        # train_decoder_only_fsdp_v2.py) skips gradient clipping
        # entirely. We follow that pattern on TPU and surface the
        # grad_norm only on GPU/CPU. See HF transformers #41881 and
        # pytorch/xla #3424 for the underlying deadlock pattern.
        #
        # iter 18 (lever 6 -- DEFERRED to iter 19+):
        # Single-host v6e-8 SPMD invalidates the cross-host deadlock
        # concern (no inter-host all-reduce; SPMD partitioner emits a
        # single xla::all_reduce HLO op). Only the per-param graph-
        # break storm remains as a re-enable risk. Three known-safe
        # variants for re-enabling on single-host v6e-8:
        #
        #   6a. Fused single-graph clip (preferred):
        #       total_sq = sum((g.float()**2).sum() for g in grads)
        #       coef = max_norm / (total_sq.sqrt() + 1e-6)
        #       coef = torch.where(coef < 1.0, coef, torch.ones_like(coef))
        #       for p in params: p.grad.mul_(coef)
        #       xm.mark_step()
        #     -> 1 fused HLO, ~5-15% throughput cost, populates metric
        #
        #   6b. clip_grad_value_(parameters, max_value):
        #       element-wise clamp_; smallest graph; loses norm metric
        #
        #   6c. vanilla torch.nn.utils.clip_grad_norm_(...,
        #       error_if_nonfinite=False):
        #       per-param graph break (~550/step under FSDPv2); high
        #       recompile risk; not recommended for v6e-8.
        #
        # See .factory/memories.md "lever 6 variants" for full details.
        if is_tpu:
            import torch_xla.core.xla_model as _xm
            _xm.mark_step()
            grad_norm = torch.tensor(0.0)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg["train"]["max_grad_norm"]
            )
            if not torch.isfinite(torch.tensor(micro_loss_sum / grad_accum)):
                print(f"!!! Non-finite loss at step {step}. Aborting.")
                sys.exit(2)

        backend.optimizer_step(optimizer)
        scheduler.step(step + 1)
        optimizer.zero_grad()
        step += 1
        if step in (1, 2, 5, 10, 25, 50):
            if hasattr(backend, "diagnose"):
                backend.diagnose(f"step={step}")

        if is_tpu:
            running_xla["loss"] = running_xla["loss"] + micro_loss_sum_xla / grad_accum
            running_xla["text"] = running_xla["text"] + micro_text_xla / grad_accum
            running_xla["audio"] = running_xla["audio"] + micro_audio_xla / grad_accum
        else:
            running["loss"] += micro_loss_sum / grad_accum
            running["text"] += micro_text / grad_accum
            running["audio"] += micro_audio / grad_accum
            running["per_cb"] += micro_per_cb / grad_accum

        # ---- logging
        if step % log_every == 0:
            if is_tpu:
                # Single materialisation of all losses at log boundary.
                _xm.mark_step()
                avg = {
                    "loss": (running_xla["loss"] / log_every).item(),
                    "text": (running_xla["text"] / log_every).item(),
                    "audio": (running_xla["audio"] / log_every).item(),
                    "per_cb": [0.0] * num_codebooks,
                }
            else:
                avg = {
                    k: (v / log_every if k != "per_cb" else (v / log_every).tolist())
                    for k, v in running.items()
                }
            now = time.time()
            step_time = (now - t_last) / log_every
            t_last = now
            lrs = {f"train/lr_{g['name']}": g["lr"] for g in optimizer.param_groups if "name" in g}
            mem_info = backend.get_memory_info()
            peak_gb = mem_info["max_allocated_gb"] if mem_info else 0
            alloc_gb = mem_info["allocated_gb"] if mem_info else 0
            if is_main:
                print(
                    f"step {step:6d} | loss {avg['loss']:.4f} | "
                    f"text {avg['text']:.4f} audio {avg['audio']:.4f} | "
                    f"grad {grad_norm:.3f} | {step_time:.2f}s/step | "
                    f"peak {peak_gb:.1f}G"
                )
            if use_wandb and is_main:
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
            if is_tpu:
                running_xla = {
                    "loss": torch.tensor(0.0, device=device),
                    "text": torch.tensor(0.0, device=device),
                    "audio": torch.tensor(0.0, device=device),
                }
            else:
                running = {
                    "loss": 0.0, "text": 0.0, "audio": 0.0,
                    "per_cb": torch.zeros(num_codebooks),
                }
                if mem_info:
                    torch.cuda.reset_peak_memory_stats()
            backend.sync()

        # ---- audio demo
        # Skip on TPU: generate_audio_sample contains an autoregressive
        # Python loop with `tok.cpu()` inside an inner per-codebook
        # loop (640 sync points) AND the backbone forward sees a
        # growing-by-1 sequence each iter, which forces a fresh XLA
        # compile per generation step. Empirically that locks the
        # main thread for 2-3 hours per call. The audio demo is a
        # qualitative sanity check, not a training requirement;
        # generate samples post-training on a GPU instead.
        if audio_every and step % audio_every == 0 and is_main and not is_tpu:
            try:
                r = generate_audio_sample(
                    unwrapped,
                    train_ds,
                    mimi_encoder,
                    device,
                    num_codebooks,
                    sample_idx=0,
                    max_target_frames=80,
                    backend=backend,
                )
                print(f"  demo cb0_acc={r['cb0_accuracy'] * 100:.1f}%")
                ad = save_dir / "audio_samples" / f"step_{step:06d}"
                ad.mkdir(parents=True, exist_ok=True)
                sf.write(ad / "source.wav", r["source_wav"], 24000)
                sf.write(ad / "target_gt.wav", r["target_gt_wav"], 24000)
                sf.write(ad / "generated.wav", r["generated_wav"], 24000)
                if use_wandb:
                    import wandb

                    wandb.log(
                        {
                            "audio/source": wandb.Audio(r["source_wav"], sample_rate=24000),
                            "audio/target_gt": wandb.Audio(r["target_gt_wav"], sample_rate=24000),
                            "audio/generated": wandb.Audio(r["generated_wav"], sample_rate=24000),
                            "audio/cb0_accuracy_train": r["cb0_accuracy"],
                        },
                        step=step,
                    )
            except Exception as e:
                print(f"  demo failed: {e}")

        # ---- validation
        # Skip on TPU for the canary: run_validation contains 4 `.item()`
        # calls inside the per-batch loop plus boolean-indexing that
        # produces dynamic shapes -- both trigger XLA cpu_fallback /
        # recompile cascades. Validation works correctly on GPU; for a
        # TPU canary the training loss curve is the signal we need.
        # Re-enable once the validation loop is rewritten with
        # accumulator tensors instead of .item() (mirroring patch 7).
        if val_loader is not None and val_every and step % val_every == 0 and not is_tpu:
            if is_main:
                print(f"  running validation at step {step}...")
            val = run_validation(
                model,
                val_loader,
                device,
                num_codebooks,
                depth_chunk,
                cfg["loss"],
                is_ddp=False,
                backend=backend,
            )
            if is_main:
                print(f"  val/loss={val['val/loss']:.4f} cb0_acc={val['val/cb0_acc'] * 100:.1f}%")
            if use_wandb and is_main:
                import wandb

                log = {k: v for k, v in val.items() if k != "val/per_codebook_loss"}
                for i, v in enumerate(val["val/per_codebook_loss"]):
                    log[f"val/per_codebook_loss_{i}"] = v
                wandb.log(log, step=step)
            if val["val/loss"] < best_val:
                best_val = val["val/loss"]
                best_dir = save_dir / "best_by_val"
                if is_main and best_dir.exists():
                    import shutil

                    shutil.rmtree(best_dir)
                # Patch 16/17: save_checkpoint runs on ALL hosts so the
                # SPMD .cpu() gather can complete; only is_main writes.
                save_checkpoint(
                    unwrapped,
                    optimizer,
                    scheduler,
                    step,
                    str(best_dir),
                    extra_state={"best_val_loss": best_val, "config": cfg},
                    is_main=is_main,
                )
                if is_main:
                    print(f"  * new best val — saved to {best_dir}")
                    if push_to_hub and hub_repo_id:
                        try:
                            push_checkpoint_to_hub(
                                str(best_dir),
                                hub_repo_id,
                                commit_message=f"best val {best_val:.4f} @ step {step}",
                                token=hub_token,
                            )
                        except Exception as e:
                            print(f"  hub push failed: {e}")
            backend.barrier()

        # ---- periodic save + prune
        if save_every and step % save_every == 0:
            d = save_dir / f"step_{step:06d}"
            # Patch 16/17: ALL hosts enter save_checkpoint to participate
            # in the SPMD .cpu() gather; only host-0 actually writes.
            save_checkpoint(
                unwrapped,
                optimizer,
                scheduler,
                step,
                str(d),
                extra_state={"config": cfg},
                is_main=is_main,
            )
            if is_main:
                prune_checkpoints(str(save_dir), keep_last=5, keep_best="best_by_val")
                if push_to_hub and hub_repo_id:
                    try:
                        push_checkpoint_to_hub(
                            str(d), hub_repo_id, commit_message=f"step {step}", token=hub_token
                        )
                    except Exception as e:
                        print(f"  hub push failed: {e}")
            backend.barrier()

    # ---- final save (multi-host SPMD-safe; see patch 16/17)
    # patch 18b: respect save_every=0 escape hatch -- skip the final save
    # if periodic saves were disabled. Iter 9+10 deadlocked all 4 hosts
    # at save_checkpoint() despite patch 16/17 moving the .cpu() gather
    # outside the is_main gate; root-cause is a deeper XLA<->PEFT
    # interaction (likely safetensors serialization triggering an XLA
    # collective inside PEFT.save_pretrained). For the canary we want
    # to first validate that the training loop itself reaches step 200,
    # so this gate stays even at the final-save site.
    if save_every:
        d = save_dir / f"step_{step:06d}"
        save_checkpoint(
            unwrapped,
            optimizer,
            scheduler,
            step,
            str(d),
            extra_state={"config": cfg, "final": True},
            is_main=is_main,
        )

    # patch 19: end-of-training canonical save (HF transformers issue
    # #36004 fix). Opt-in via cfg.train.final_canonical_save. This
    # bypasses the save_every gate -- with save_every=0 (patch 18a),
    # this is the only chance to land a checkpoint in GCS. Destructive:
    # model.to("cpu") loses FSDPv2 sharding metadata, so this only runs
    # once at the very end of training. ALL hosts must reach this call;
    # the SPMD gather inside .to("cpu") deadlocks if any host skips it.
    if cfg["train"].get("final_canonical_save", False):
        from src.training.checkpointing import save_checkpoint_canonical_final

        d_final = save_dir / f"step_{step:06d}_final"
        if is_main:
            print(f"\n[patch 19] entering canonical final save -> {d_final}")
        save_checkpoint_canonical_final(unwrapped, str(d_final), is_main=is_main)
        if is_main:
            print("[patch 19] canonical final save complete")
    if is_main:
        print(f"\nTraining complete: {step} steps in {(time.time() - t0) / 60:.1f} min")
        if save_every and push_to_hub and hub_repo_id:
            try:
                push_checkpoint_to_hub(
                    str(d), hub_repo_id, commit_message=f"final step {step}", token=hub_token
                )
            except Exception as e:
                print(f"  hub push failed: {e}")
    if use_wandb and is_main:
        import wandb

        wandb.finish()

    backend.barrier()


if __name__ == "__main__":
    main()
