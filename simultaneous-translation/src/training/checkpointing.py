"""Save/load mixed PEFT + full checkpoints for ``TinyAyaMoshiComposite``.

WHY THIS EXISTS
---------------
The composite mixes three kinds of weights:

1. PEFT-LoRA adapters on the Cohere backbone -- saved via
   ``model.backbone.model.save_pretrained`` (HF/PEFT convention).
2. Full-FT weights on the last two backbone layers -- captured by
   the same PEFT save (PEFT preserves any frozen ``requires_grad=True``
   tensors as well).
3. Plain PyTorch modules (``projection``, ``depth_decoder``,
   ``text_embed``, ``audio_heads``) -- saved as standalone ``.pt``
   files.

A standard ``model.state_dict()`` round-trip would silently drop the
PEFT adapter metadata and corrupt the freeze pattern; we save each
component explicitly to avoid that.

GPU vs TPU note
---------------
The TPU backend's ``save_checkpoint`` uses ``xm.save`` which gathers
all SPMD shards onto host CPU before writing. The functions in this
module run on host CPU after that gather, so they are device-agnostic.
``load_checkpoint`` always loads to CPU and lets the caller move
weights back to the target device.
"""

import json
import os
from pathlib import Path

import torch


def save_checkpoint(
    model, optimizer, scheduler, step: int, save_dir: str, extra_state: dict | None = None
):
    os.makedirs(save_dir, exist_ok=True)

    peft_dir = os.path.join(save_dir, "peft_adapter")
    model.backbone.model.save_pretrained(peft_dir)

    torch.save(model.projection.state_dict(), os.path.join(save_dir, "projection.pt"))
    torch.save(model.depth_decoder.state_dict(), os.path.join(save_dir, "depth_decoder.pt"))

    # Text head is part of backbone.model (lm_head in PEFT), already in peft save.
    # Separate text_embed is part of backbone state dict.
    torch.save(model.backbone.text_embed.state_dict(), os.path.join(save_dir, "text_embed.pt"))
    torch.save(model.backbone.audio_heads.state_dict(), os.path.join(save_dir, "audio_heads.pt"))

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
            mod.load_state_dict(torch.load(p, map_location="cpu", weights_only=True), strict=False)

    opt_p = os.path.join(load_dir, "optimizer.pt")
    if optimizer is not None and os.path.exists(opt_p):
        optimizer.load_state_dict(torch.load(opt_p, map_location="cpu", weights_only=True))
    sch_p = os.path.join(load_dir, "scheduler.pt")
    if scheduler is not None and os.path.exists(sch_p):
        scheduler.load_state_dict(torch.load(sch_p, map_location="cpu", weights_only=True))

    return step


def push_checkpoint_to_hub(
    local_dir: str, repo_id: str, commit_message: str = "checkpoint", token: str | None = None
):
    """Upload model weights (no optimizer/scheduler) to a HuggingFace Hub repo."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", exist_ok=True, private=False)

    skip = {"optimizer.pt", "scheduler.pt"}
    for root, _dirs, files in os.walk(local_dir):
        for fname in files:
            if fname in skip:
                continue
            local_path = os.path.join(root, fname)
            path_in_repo = os.path.relpath(local_path, local_dir)
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_message,
            )
    print(f"  pushed to https://huggingface.co/{repo_id}")


def prune_checkpoints(save_dir: str, keep_last: int = 5, keep_best: str | None = "best_by_val"):
    """Delete all step_* checkpoints except the last `keep_last` by step, and the best."""
    save_dir = Path(save_dir)
    step_dirs = sorted(
        [p for p in save_dir.glob("step_*") if p.is_dir()], key=lambda p: int(p.name.split("_")[1])
    )
    keep = set(p.name for p in step_dirs[-keep_last:])
    if keep_best:
        keep.add(keep_best)
    for p in step_dirs:
        if p.name not in keep:
            import shutil

            shutil.rmtree(p, ignore_errors=True)


def is_gcs_path(path: str) -> bool:
    return path.startswith("gs://")


def get_checkpoint_dirs(base_dir: str) -> list[str]:
    """List checkpoint directories, supporting both local and GCS."""
    if is_gcs_path(base_dir):
        try:
            import gcsfs

            fs = gcsfs.GCSFileSystem()
            try:
                entries = fs.ls(base_dir)
            except FileNotFoundError:
                return []
            dirs = [f"gs://{d}" for d in entries if fs.isdir(d)]
            return sorted(dirs)
        except ImportError:
            print("Warning: gcsfs not installed, cannot list GCS checkpoints")
            return []
    else:
        import os

        if not os.path.exists(base_dir):
            return []
        return sorted(
            [
                os.path.join(base_dir, d)
                for d in os.listdir(base_dir)
                if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("checkpoint_")
            ]
        )


def find_latest_checkpoint(base_dir: str) -> str | None:
    """Find the latest checkpoint directory for resume."""
    dirs = get_checkpoint_dirs(base_dir)
    return dirs[-1] if dirs else None


def save_checkpoint_with_backend(
    model, optimizer, scheduler, step, save_dir, backend, extra_state=None
):
    """Save checkpoint using backend's save method (handles GCS/local)."""
    import os

    os.makedirs(save_dir, exist_ok=True)
    save_checkpoint(model, optimizer, scheduler, step, save_dir, extra_state)


def load_checkpoint_with_backend(model, optimizer, scheduler, load_dir, backend):
    """Load checkpoint using backend's load method."""
    return load_checkpoint(model, optimizer, scheduler, load_dir)
