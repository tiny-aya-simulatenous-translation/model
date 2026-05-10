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


def _is_xla_tensor(t) -> bool:
    if t is None:
        return False
    dev = getattr(t, "device", None)
    return dev is not None and getattr(dev, "type", None) == "xla"


def _to_cpu_state_dict(module: torch.nn.Module) -> dict:
    return {
        k: v.detach().to("cpu").contiguous() if torch.is_tensor(v) else v
        for k, v in module.state_dict().items()
    }


def _detach_to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().to("cpu").contiguous()
    if isinstance(obj, dict):
        return {k: _detach_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_detach_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_detach_to_cpu(v) for v in obj)
    return obj


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    step: int,
    save_dir: str,
    extra_state: dict | None = None,
    *,
    is_main: bool = True,
):
    """Save a multi-component checkpoint, multi-host SPMD-safe.

    On a multi-host TPU pod, the implicit SPMD gather triggered by
    ``tensor.cpu()`` is a *cross-host collective*. Every host's Python
    process must reach that .cpu() at the same time, otherwise host 0's
    materialization deadlocks waiting for hosts 1..N to contribute their
    chip-local shards. Patch 14 (mark_step + .cpu()) was correct in
    spirit but wrong in placement: the entire body sat behind an
    ``if is_main:`` gate at the call site, so only host 0 ever entered.
    Hosts 1..3 sat at the downstream ``backend.barrier()`` and the
    gather hung forever. (Confirmed empirically: iter 9 reached step
    100, then save_checkpoint hung 4+ min with only the stale iter 7
    README.md file present.)

    The fix (patch 16): split the function into two phases and have ALL
    hosts call it. Phase 1 materializes every state dict to CPU --
    this is the collective and must run on every host. Phase 2 writes
    files -- this runs only on the global primary. Hosts 1..3 return
    after phase 1 and proceed to ``backend.barrier()`` while host 0
    serializes to disk. Pattern mirrors HF transformers PR #27799
    ``_save_tpu`` and issue #36004's recursive_unwrap recipe.
    """
    is_xla = _is_xla_tensor(next(model.parameters(), None))

    if is_xla:
        import torch_xla.core.xla_model as xm

        xm.mark_step()
        xm.wait_device_ops()

    if is_xla:
        peft_state = _to_cpu_state_dict(model.backbone.model)
        proj_state = _to_cpu_state_dict(model.projection)
        depth_state = _to_cpu_state_dict(model.depth_decoder)
        text_state = _to_cpu_state_dict(model.backbone.text_embed)
        audio_state = _to_cpu_state_dict(model.backbone.audio_heads)
        optim_state = _detach_to_cpu(optimizer.state_dict())
        sched_state = (
            _detach_to_cpu(scheduler.state_dict())
            if scheduler is not None and hasattr(scheduler, "state_dict")
            else None
        )
    else:
        peft_state = None  # let save_pretrained build it itself on GPU/CPU
        proj_state = model.projection.state_dict()
        depth_state = model.depth_decoder.state_dict()
        text_state = model.backbone.text_embed.state_dict()
        audio_state = model.backbone.audio_heads.state_dict()
        optim_state = optimizer.state_dict()
        sched_state = (
            scheduler.state_dict()
            if scheduler is not None and hasattr(scheduler, "state_dict")
            else None
        )

    if not is_main:
        return

    os.makedirs(save_dir, exist_ok=True)

    peft_dir = os.path.join(save_dir, "peft_adapter")
    if peft_state is not None:
        model.backbone.model.save_pretrained(peft_dir, state_dict=peft_state)
    else:
        model.backbone.model.save_pretrained(peft_dir)

    torch.save(proj_state, os.path.join(save_dir, "projection.pt"))
    torch.save(depth_state, os.path.join(save_dir, "depth_decoder.pt"))
    torch.save(text_state, os.path.join(save_dir, "text_embed.pt"))
    torch.save(audio_state, os.path.join(save_dir, "audio_heads.pt"))
    torch.save(optim_state, os.path.join(save_dir, "optimizer.pt"))
    if sched_state is not None:
        torch.save(sched_state, os.path.join(save_dir, "scheduler.pt"))

    meta = {"step": step}
    if extra_state:
        meta.update(extra_state)
    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


def save_checkpoint_canonical_final(
    model,
    save_dir: str,
    *,
    is_main: bool = True,
):
    """End-of-training canonical save for FSDPv2-wrapped XLA models.

    DESTRUCTIVE: this function moves the entire wrapped model from the
    XLA device onto host CPU, which destroys FSDPv2 sharding metadata.
    Training cannot continue after this call -- only invoke at the
    final step. All hosts MUST call this function (``model.to("cpu")``
    is an SPMD-wide gather; the gather hangs forever if any host skips
    it). Only the global primary writes files.

    Why this exists
    ---------------
    Iters 9, 10, 11 all reached step 100 then deadlocked all 4 hosts
    at ``futex_wait_queue`` inside ``PEFT.save_pretrained``, even after
    patches 14/16/17 made every host build a CPU state_dict and pass
    it via ``state_dict=peft_state``. Root cause (per HF transformers
    issue #36004, closed Dec 2025): ``save_pretrained`` does not
    support saving models that are still resident on a TPU device --
    it internally re-walks the model's submodules, which on FSDPv2
    triggers XLA collectives that the global state cannot satisfy.
    The canonical fix is to call ``model.to("cpu")`` on the full
    wrapped model (gathering all shards in a single collective) BEFORE
    calling save_pretrained. After that, save_pretrained walks an
    ordinary CPU module with no XLA tensors involved.

    The trade-off is that ``model.to("cpu")`` is destructive on
    FSDPv2: the SPMD partitioner forgets the per-layer mesh
    annotations, and re-running ``backend.wrap_model(model)`` would
    not reproduce the original sharding without a fresh recompile.
    For our canary loop that is acceptable -- we save once at step
    ``max_steps`` and exit.

    GPU vs TPU note
    ---------------
    On non-XLA backends this function delegates to ``save_checkpoint``
    with optimizer/scheduler set to ``None`` (the canary doesn't need
    them in the final artefact).
    """
    is_xla = _is_xla_tensor(next(model.parameters(), None))

    if not is_xla:
        save_checkpoint(
            model,
            optimizer=None,
            scheduler=None,
            step=-1,
            save_dir=save_dir,
            is_main=is_main,
        )
        return

    import torch_xla.core.xla_model as xm

    xm.mark_step()
    xm.wait_device_ops()

    for sub in (
        model.backbone.model,
        model.projection,
        model.depth_decoder,
        model.backbone.text_embed,
        model.backbone.audio_heads,
    ):
        sub.to("cpu")

    xm.rendezvous("post_to_cpu_canonical_final")

    if not is_main:
        return

    for name, p in model.named_parameters():
        assert not _is_xla_tensor(p), (
            f"canonical_final: parameter {name} still on XLA after .to(cpu)"
        )

    is_gcs = save_dir.startswith("gs://") or save_dir.startswith("gs:/")
    if is_gcs:
        import tempfile

        local_dir = tempfile.mkdtemp(prefix="canonical_final_")
        if save_dir.startswith("gs:/") and not save_dir.startswith("gs://"):
            gcs_dest = "gs://" + save_dir[len("gs:/") :]
        else:
            gcs_dest = save_dir
        write_dir = local_dir
        print(
            f"[patch 19] save_dir is GCS ({gcs_dest}); staging to {local_dir}",
            flush=True,
        )
    else:
        write_dir = save_dir

    os.makedirs(write_dir, exist_ok=True)

    peft_dir = os.path.join(write_dir, "peft_adapter")
    model.backbone.model.save_pretrained(peft_dir, safe_serialization=False)

    torch.save(model.projection.state_dict(), os.path.join(write_dir, "projection.pt"))
    torch.save(model.depth_decoder.state_dict(), os.path.join(write_dir, "depth_decoder.pt"))
    torch.save(model.backbone.text_embed.state_dict(), os.path.join(write_dir, "text_embed.pt"))
    torch.save(model.backbone.audio_heads.state_dict(), os.path.join(write_dir, "audio_heads.pt"))

    with open(os.path.join(write_dir, "metadata.json"), "w") as f:
        json.dump({"step": "final", "save_kind": "canonical_final"}, f, indent=2)

    if is_gcs:
        import subprocess

        print(
            f"[patch 19] uploading {write_dir}/* to {gcs_dest}",
            flush=True,
        )
        result = subprocess.run(
            ["gsutil", "-m", "cp", "-r", write_dir + "/.", gcs_dest],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"[patch 19] gsutil stderr: {result.stderr}", flush=True)
            raise RuntimeError(f"gsutil upload failed (rc={result.returncode}): {result.stderr}")
        print(
            f"[patch 19] gsutil upload complete: {gcs_dest}",
            flush=True,
        )
        import shutil

        shutil.rmtree(write_dir, ignore_errors=True)


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
