"""Save/load mixed PEFT + full checkpoints for TinyAyaMoshiComposite.

Port of tinyaya-moshi-backbone/src/moshi_backbone/training/checkpointing.py with
one fix: load uses `set_peft_model_state_dict` instead of `PeftModel.from_pretrained`
to avoid re-freezing the last two full-FT layers.
"""

import json
import os
from pathlib import Path

import torch


def save_checkpoint(model, optimizer, scheduler, step: int, save_dir: str,
                    extra_state: dict | None = None):
    os.makedirs(save_dir, exist_ok=True)

    peft_dir = os.path.join(save_dir, "peft_adapter")
    model.backbone.model.save_pretrained(peft_dir)

    torch.save(model.projection.state_dict(), os.path.join(save_dir, "projection.pt"))
    torch.save(model.depth_decoder.state_dict(), os.path.join(save_dir, "depth_decoder.pt"))

    # Text head is part of backbone.model (lm_head in PEFT), already in peft save.
    # Separate text_embed is part of backbone state dict.
    torch.save(model.backbone.text_embed.state_dict(),
               os.path.join(save_dir, "text_embed.pt"))
    torch.save(model.backbone.audio_heads.state_dict(),
               os.path.join(save_dir, "audio_heads.pt"))

    torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        torch.save(scheduler.state_dict(), os.path.join(save_dir, "scheduler.pt"))

    meta = {"step": step}
    if extra_state:
        meta.update(extra_state)
    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


def load_checkpoint(model, optimizer, scheduler, load_dir: str) -> int:
    from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict

    with open(os.path.join(load_dir, "metadata.json")) as f:
        meta = json.load(f)
    step = meta["step"]

    peft_dir = os.path.join(load_dir, "peft_adapter")
    if os.path.isdir(peft_dir):
        sd = load_peft_weights(peft_dir)
        set_peft_model_state_dict(model.backbone.model, sd)

    for fname, mod in [
        ("projection.pt", model.projection),
        ("depth_decoder.pt", model.depth_decoder),
        ("text_embed.pt", model.backbone.text_embed),
        ("audio_heads.pt", model.backbone.audio_heads),
    ]:
        p = os.path.join(load_dir, fname)
        if os.path.exists(p):
            mod.load_state_dict(torch.load(p, map_location="cpu", weights_only=True),
                                strict=False)

    opt_p = os.path.join(load_dir, "optimizer.pt")
    if optimizer is not None and os.path.exists(opt_p):
        optimizer.load_state_dict(torch.load(opt_p, map_location="cpu", weights_only=True))
    sch_p = os.path.join(load_dir, "scheduler.pt")
    if scheduler is not None and os.path.exists(sch_p):
        scheduler.load_state_dict(torch.load(sch_p, map_location="cpu", weights_only=True))

    return step


def prune_checkpoints(save_dir: str, keep_last: int = 5, keep_best: str | None = "best_by_val"):
    """Delete all step_* checkpoints except the last `keep_last` by step, and the best."""
    save_dir = Path(save_dir)
    step_dirs = sorted([p for p in save_dir.glob("step_*") if p.is_dir()],
                       key=lambda p: int(p.name.split("_")[1]))
    keep = set(p.name for p in step_dirs[-keep_last:])
    if keep_best:
        keep.add(keep_best)
    for p in step_dirs:
        if p.name not in keep:
            import shutil
            shutil.rmtree(p, ignore_errors=True)
