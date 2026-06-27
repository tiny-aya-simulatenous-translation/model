# TPU Support: Complete Guide to All Changes

## 2026-05-10 update

The launch-time and infrastructure narrative below was written when
this branch first landed (Colab v5e-1, then v4-32 spot canary in
`us-central2-b`). The production path has since pivoted to
**single-host TPU v6e-8 in `europe-west4-a`** (QR
`tinyaya-stage2-spot-v6e8-eu-qr`, optimized production config
`configs/stage2_tpu_v6e_spot_opt_prod5k.yaml`). On v6e-8 there is exactly
ONE Python process (single-host SPMD) driving all 8 chips, so the
multi-host coordination patches (host-index gating, wandb shared-
mode rendezvous, GCS run-id polling) are inert; they remain in the
codebase to support v4-32 (legacy) and v6e-64 (future scale-up).

**Current production result:** iter 24h first completed 5000/5000 steps
on v6e-8 EU spot, W&B run
[`7rrjupc7`](https://wandb.ai/cataluna84/tinyaya-stage2-tpu/runs/7rrjupc7),
final loss 5.3558, training wall 615.9 min, exit status 0. The final
canonical checkpoint uploaded to
`gs://tinyaya-stage2-tpu/checkpoints/stage2-tpu-v6e-spot/step_005000_final/`
(8 objects, 2.37 GiB). `opt-prod5k` then completed 5000/5000 steps
with Phase 1+2+3 optimizations, W&B run
[`kzsijxv5`](https://wandb.ai/cataluna84/tinyaya-stage2-tpu/runs/kzsijxv5),
final loss 5.105, p50 6.14 s/step, p99 6.76 s/step, training wall
562 min, and checkpoint
`gs://tinyaya-stage2-tpu/checkpoints/stage2-tpu-v6e-spot-opt-prod5k/step_005000_final/`.
Phase 4 started with `opt-4-depth32`, W&B run
[`i15igq8d`](https://wandb.ai/cataluna84/tinyaya-stage2-tpu/runs/i15igq8d),
which completed 300/300 steps with exit 0, p50 5.296 s/step, p99
5.725 s/step, and final loss 6.6539.
All sections below referring to "4 hosts", multi-host wandb, v4-32
zones, or canary-only status should be read as historical /
multi-host topology context.

**Branch:** `feat/tpu-support`
**Base repo:** `tinyaya-stage2-scale/simultaneous-translation`
**Verified on:** Colab TPU v5e-1 (forward + backward pass confirmed),
v4-32 spot in `us-central2-b` (iter 7 reached step 100), v6e-8 spot
in `europe-west4-a` (iter 13b reached step 20 + canonical save, iter
17 reached 200-step bf16 canary, iter 24h completed 5000-step
production, `opt-prod5k` completed optimized 5000-step production,
and `opt-4-depth32` passed the Phase 4 300-step depth sweep gate).

---

## Overview

The training pipeline was originally built for GPU with optional DDP (multi-GPU) support. We added TPU support using PyTorch/XLA's SPMD + FSDPv2 while keeping the GPU path intact. The approach is a **backend abstraction** -- the training script doesn't know whether it's running on GPU or TPU; it talks to a backend interface that handles device-specific operations.

**What changed:**
- TPU backend abstraction with SPMD/FSDPv2 strategy selection.
- TPU launch infrastructure (`launch_qr.sh`, `launch_spot.sh`,
  `startup_script.sh`, `ops.sh`, hot redeploy).
- XLA-safe training-loop changes: static shapes, lazy tensor logging,
  TPU-specific optimizer stepping, and fixed grad-accum topology.
- v6e-specific numerics fixes: bf16 attention-mask clamp, disabled
  SDPA mask elision, conservative b=8/grad_accum=4 production config.
- Canonical final checkpoint save for FSDPv2-XLA plus GCS upload.
- External-memory and orchestration docs for self-healing TPU runs.

**What did NOT change:**
- Model architecture (backbone, composite, depth decoder, LoRA setup)
- Data pipeline (dataset, collator, interleaver)
- Loss computation (translation_loss.py)
- Learning rate scheduler (scheduler.py)
- Inference scripts

**Production-specific fixes validated by iter 24h:**
- TPU batches are padded on the host to a static batch dimension
  (`batch_pad_to=batch_size`) while keeping `drop_last=False`, so no
  accepted training row is discarded.
- The training loop tracks micro-batches per epoch and resets the
  DataLoader only between optimizer steps; a 4-way grad-accum macro-step
  never straddles an epoch boundary.
- Static TPU shape assertions fail fast before a new XLA graph can
  compile.
- HF SDPA mask elision is disabled and attention masks are clamped to
  `>= -1e4`, preventing both all-full-batch topology flips and the v6e
  bf16 `-inf` mask NaN.
- Canonical final save stages to a local directory and uploads with
  `gsutil cp -r` when the target is `gs://...`.
- W&B logging defines `global_step` as the x-axis for train/perf/val/
  audio/memory metrics; W&B internal `_step` now counts log events
  without distorting charts.
- Startup can fetch a branch tarball from
  `REPO_TARBALL_GS_URI=gs://...`, avoiding private GitHub clone
  credentials on fresh TPU VMs.

---

## Table of Contents

1. [Backend Abstraction Layer](#1-backend-abstraction-layer)
2. [GPU Backend](#2-gpu-backend)
3. [TPU Backend](#3-tpu-backend)
4. [XLA Compatibility Fixes](#4-xla-compatibility-fixes)
5. [Training Script Changes](#5-training-script-changes)
6. [Checkpointing Changes](#6-checkpointing-changes)
7. [TPU Config](#7-tpu-config)
8. [Testing Results](#8-testing-results)
9. [Known Limitations](#9-known-limitations)

---

## 1. Backend Abstraction Layer

**File:** `src/backend/__init__.py`

The entry point for the backend system. Two functions:

- `detect_backend()` -- checks the `DEVICE_BACKEND` environment variable first (explicit override), then tries to import `torch_xla`. Falls back to `"gpu"` if XLA isn't available.
- `get_backend(backend_type)` -- factory that returns either `GPUBackend` or `TPUBackend`. Uses lazy imports so you don't need `torch_xla` installed to use the GPU path.

```python
# Force a specific backend:
DEVICE_BACKEND=tpu python scripts/train_hierarchical.py ...
DEVICE_BACKEND=gpu python scripts/train_hierarchical.py ...

# Auto-detect (default):
python scripts/train_hierarchical.py ...
```

**File:** `src/backend/base.py`

The abstract base class that both backends implement. Every GPU-specific operation from the original code gets a method here:

| Method | GPU Behavior | TPU Behavior |
|--------|-------------|-------------|
| `init_distributed()` | `init_process_group(backend="nccl")` | `xr.use_spmd()` + create `Mesh` |
| `get_device()` | `torch.device("cuda:0")` | `xm.xla_device()` -> `xla:0` |
| `wrap_model(model)` | `DistributedDataParallel(model)` | `FSDPv2(model, mesh=...)` |
| `optimizer_step(opt)` | `opt.step()` | `xm.optimizer_step(opt)` |
| `autocast_context()` | `torch.amp.autocast("cuda")` | `nullcontext()` (bf16 is native) |
| `no_sync(model)` | `model.no_sync()` | `nullcontext()` (SPMD is single-process) |
| `barrier()` | `dist.barrier()` | `xm.rendezvous("barrier")` |
| `reduce_mean(tensor)` | `dist.all_reduce(SUM) / world_size` | `xm.mesh_reduce(mean)` |
| `save_checkpoint()` | `torch.save()` | `xm.save()` |
| `get_memory_info()` | `torch.cuda.memory_allocated()` | `None` (not available on TPU) |
| `sync()` | no-op (eager execution) | `torch_xla.sync()` (trigger XLA compilation) |

---

## 2. GPU Backend

**File:** `src/backend/gpu_backend.py`

This wraps the existing GPU/DDP behavior into the backend interface. No new functionality -- it's a refactor of what was previously inline in `train_hierarchical.py`.

Key details:
- Uses `LOCAL_RANK` and `WORLD_SIZE` environment variables (set by `torchrun`)
- DDP uses `find_unused_parameters=True` and `broadcast_buffers=False` (required because LoRA leaves some parameters unused)
- Supports CPU fallback: if CUDA isn't available, device falls back to `"cpu"` and autocast uses `torch.amp.autocast("cpu")`

---

## 3. TPU Backend

**File:** `src/backend/tpu_backend.py`

### SPMD + Mesh Setup

SPMD (Single Program Multiple Data) is the TPU-native distributed training approach. Unlike DDP which spawns one process per GPU, SPMD runs a **single process** and the XLA compiler automatically distributes computation across all chips.

```python
xr.use_spmd()  # Enable SPMD mode
mesh = Mesh(
    device_ids=list(range(num_devices)),  # [0, 1, ..., 63]
    mesh_shape=(num_devices,),             # (64,) -- 1D mesh
    axis_names=("fsdp",),                  # Name the axis "fsdp"
)
```

The mesh is 1-dimensional with a single axis called `"fsdp"`. When we later call `mark_sharding(tensor, mesh, ("fsdp", None))`, XLA knows to split the tensor's first dimension (batch) across all 64 chips and replicate the second dimension.

### SPMD Trade-off

> SPMD runs as a single process across all TPU chips. Unlike DDP, there's no per-chip process isolation -- if one chip OOMs or errors, the entire job fails. This is mitigated by: frequent async checkpoints, spot preemption resume, and conservative memory settings. If stability issues arise, consider falling back to multi-process `xmp.spawn` + DDP-style training via `torch_xla.distributed.xla_backend`.

### FSDPv2 Model Wrapping

FSDPv2 (`SpmdFullyShardedDataParallel`) shards model weights across TPU chips. On a 64-chip setup, each chip holds 1/64th of the model weights.

```python
model = FSDPv2(model, mesh=self._mesh, auto_wrap_policy=_wrap_policy)
```

**Critical: auto_wrap_policy.** We discovered that FSDPv2 sharding breaks our embedding table. The backbone uses offset-based indexing (`audio_codes + 262148`) to look up audio tokens in the extended embedding table. When FSDPv2 shards this table, the shard on each chip only has 1/64th of the rows, and the offset exceeds the shard size.

The fix is an `auto_wrap_policy` that tells FSDPv2 which layers to shard:

```python
def _wrap_policy(module, recurse, **kwargs):
    if recurse:
        return True
    # Don't wrap embedding layers -- they use offset-based indexing
    if isinstance(module, Embedding):
        return False
    # Wrap transformer layers (the heavy computation)
    layer_types = (
        "CohereDecoderLayer",  # TinyAya backbone layers
        "MoshiDecoderLayer",   # Depth decoder layers
    )
    return type(module).__name__ in layer_types
```

This shards the 36 transformer layers and 6 depth decoder layers (the heavy params) while keeping all `nn.Embedding` layers replicated on every chip.

**Single-chip handling:** On a single TPU chip (like Colab), FSDPv2 is unnecessary and potentially harmful. The backend detects this and skips wrapping:

```python
if num_devices <= 1:
    print("TPU: single chip, skipping FSDPv2 wrapping")
    return model
```

### Gradient Checkpointing Order

On TPU, gradient checkpointing **must** be enabled before FSDPv2 wrapping. If you enable it after, FSDPv2's recursive module traversal hits an infinite loop. The `wrap_model()` method handles this:

```python
def wrap_model(self, model):
    # MUST be before FSDPv2
    model.backbone.model.gradient_checkpointing_enable()
    model = FSDPv2(model, mesh=self._mesh, auto_wrap_policy=_wrap_policy)
    return model
```

### Input Sharding

The TPU backend has an extra method not in the base class:

```python
def mark_sharding(self, tensor, partition_spec):
    xs.mark_sharding(tensor, self._mesh, partition_spec)
```

Called in the training loop to tell SPMD how to distribute each input tensor:

```python
# text_ids: [B, T] -> shard batch dim, replicate time dim
backend.mark_sharding(text_ids, ("fsdp", None))

# all_codes: [B, CB, T] -> shard batch dim, replicate codebooks and time
backend.mark_sharding(all_codes, ("fsdp", None, None))
```

### Lazy Execution and sync()

PyTorch on GPU executes operations eagerly -- each op runs immediately. PyTorch/XLA builds a **lazy computation graph** and only compiles+executes it when you call `torch_xla.sync()`. This has two implications:

1. **First step is slow:** XLA compiles the entire forward+backward graph on the first training step. Subsequent steps reuse the compiled graph and are fast.

2. **sync() after logging:** When you call `loss.item()` to log metrics, the value isn't materialized yet. `backend.sync()` forces XLA to execute the graph and produce actual numbers. Without it, the graph keeps growing and eventually OOMs.

---

## 4. XLA Compatibility Fixes

Two bugs discovered by actually running on TPU:

### Fix 1: `use_cache=False` in backbone

**File:** `src/model/backbone.py`, line 112

**Problem:** Cohere2 (TinyAya's architecture) has sliding window attention with a KV cache. The cache code does:

```python
self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
```

With `sliding_window=4096` and sequence length `T=10`, this becomes `[:, :, -4095:, :]`. On GPU, PyTorch silently clamps negative slice indices to 0. On XLA, this raises:

```
RuntimeError: Value out of range (expected to be in range of [-10, 9], but got -4095)
```

**Fix:** Add `use_cache=False` to the backbone forward call. During training, we don't need the KV cache -- it's only useful for autoregressive inference. This change is harmless on GPU too.

```python
outputs = self.model(
    inputs_embeds=combined,
    attention_mask=attention_mask,
    output_hidden_states=True,
    return_dict=True,
    use_cache=False,  # XLA/TPU compatibility
)
```

### Fix 2: XLA-compatible gradient checkpointing

**File:** `src/model/composite.py`, lines 17-25

**Problem:** The depth decoder uses `torch.utils.checkpoint.checkpoint()` for memory-efficient chunked forward passes. On XLA, this function internally references `torch.xla` in a way that raises `AttributeError`.

**Fix:** Wrap the checkpoint call in a try/except that falls back to a direct function call on XLA:

```python
def _checkpoint(fn, *args, use_reentrant=False):
    """Gradient checkpoint that works on both GPU (torch) and TPU (XLA)."""
    try:
        return cp.checkpoint(fn, *args, use_reentrant=use_reentrant)
    except AttributeError:
        # Falls back to direct call on XLA
        return fn(*args)
```

This is safe because on TPU, gradient checkpointing for the backbone is already handled at the model level via `model.gradient_checkpointing_enable()`, which is called before FSDPv2 wrapping in the TPU backend.

---

## 5. Training Script Changes

**File:** `scripts/train_hierarchical.py`

The training script had GPU-specific code throughout. Every instance was replaced with a backend call. Here's a mapping of what changed:

### Distributed initialization (was lines 322-330)

```python
# BEFORE:
local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))
is_ddp = world_size > 1
if is_ddp:
    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
device = f"cuda:{local_rank}"
is_main = (not is_ddp) or local_rank == 0

# AFTER:
backend = get_backend(backend_type)
backend.init_distributed()
device = backend.get_device()
is_main = backend.is_main_process()
```

### Model wrapping (was lines 354-359)

```python
# BEFORE:
model.backbone.gradient_checkpointing_enable()
if is_ddp:
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], ...)
unwrapped = model.module if is_ddp else model

# AFTER:
if cfg.get("backend", "auto") != "tpu":
    model.backbone.gradient_checkpointing_enable()  # TPU does this inside wrap_model
model = backend.wrap_model(model)
unwrapped = model.module if hasattr(model, "module") else model
```

### Data loading (was lines 395-412)

```python
# BEFORE:
train_sampler = DistributedSampler(train_ds) if is_ddp else None
train_loader = DataLoader(..., sampler=train_sampler, pin_memory=True)

# AFTER:
if is_tpu:
    train_sampler = None  # SPMD is single-process, no sampler needed
elif backend.world_size() > 1:
    train_sampler = DistributedSampler(train_ds)
else:
    train_sampler = None
train_loader = DataLoader(
    ..., sampler=train_sampler,
    pin_memory=cfg["data"]["pin_memory"] and not is_tpu,  # pin_memory meaningless on TPU
    persistent_workers=num_workers > 0 and not is_tpu,
)
```

### Mimi encoder placement (was line 418)

```python
# BEFORE:
mimi_encoder = MimiEncoder(device=device)

# AFTER:
mimi_device = "cpu" if is_tpu else device  # Mimi not available on TPU
mimi_encoder = MimiEncoder(device=mimi_device)
```

Mimi is used only for decoding audio samples during training monitoring. On TPU, we decode on CPU (~10s per sample, only runs every 1000 steps).

### Gradient accumulation no_sync (was line 487-491)

```python
# BEFORE:
sync_ctx = contextlib.nullcontext() if (not is_ddp or micro == grad_accum - 1) else model.no_sync()

# AFTER:
sync_ctx = contextlib.nullcontext() if micro == grad_accum - 1 else backend.no_sync(model)
```

On GPU DDP, `model.no_sync()` skips the allreduce on non-final accumulation microsteps. On TPU SPMD, `no_sync` returns `nullcontext()` since there's only one process.

### Autocast (was line 507)

```python
# BEFORE:
with torch.amp.autocast("cuda", dtype=torch.bfloat16):

# AFTER:
with backend.autocast_context(dtype=torch.bfloat16):
```

On GPU, this wraps the forward pass in automatic mixed precision. On TPU, this is a no-op (`nullcontext()`) because bf16 is the native dtype -- the model is already in bf16 from initialization.

### Input sharding for TPU (new, after line 505)

```python
# Mark input sharding for TPU SPMD
if hasattr(backend, "mark_sharding"):
    backend.mark_sharding(text_ids, ("fsdp", None))
    backend.mark_sharding(all_codes, ("fsdp", None, None))
    backend.mark_sharding(mask, ("fsdp", None))
    backend.mark_sharding(loss_mask, ("fsdp", None))
```

This tells XLA how to distribute each input tensor across chips. `("fsdp", None)` means: shard dim 0 (batch) across the `fsdp` mesh axis, replicate dim 1 (time). The `hasattr` guard means this only runs on TPU -- the GPU backend doesn't have `mark_sharding`.

### Optimizer step (was line 531)

```python
# BEFORE:
optimizer.step()

# AFTER:
backend.optimizer_step(optimizer)
```

On TPU, `xm.optimizer_step(optimizer)` handles the XLA sync barrier before stepping.

### Memory logging (was lines 550-551)

```python
# BEFORE:
peak_gb = torch.cuda.max_memory_allocated() / 1e9
alloc_gb = torch.cuda.memory_allocated() / 1e9

# AFTER:
mem_info = backend.get_memory_info()
peak_gb = mem_info["max_allocated_gb"] if mem_info else 0
alloc_gb = mem_info["allocated_gb"] if mem_info else 0
```

TPU memory info isn't available through torch_xla, so `get_memory_info()` returns `None` and we log 0.

### sync() after logging (new, line 608)

```python
if mem_info:
    torch.cuda.reset_peak_memory_stats()
backend.sync()  # Force XLA graph execution after reading metric values
```

### Validation all-reduce (was lines 247-255)

```python
# BEFORE:
if is_ddp:
    for k in sums:
        t = torch.tensor(sums[k], device=device)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
        sums[k] = t.item()

# AFTER:
if backend and backend.world_size() > 1:
    for k in sums:
        t = torch.tensor(sums[k], device=device)
        t = backend.reduce_mean(t) * backend.world_size()
        sums[k] = t.item()
```

### Barrier calls (was `if is_ddp: torch.distributed.barrier()`)

```python
# BEFORE:
if is_ddp:
    torch.distributed.barrier()

# AFTER:
backend.barrier()
```

### Resume with auto-detect (was line 434)

```python
# BEFORE:
if args.resume:
    start_step = load_checkpoint(unwrapped, optimizer, scheduler, args.resume)

# AFTER:
resume_dir = args.resume
if resume_dir == "auto":
    resume_dir = find_latest_checkpoint(cfg["logging"]["save_dir"])
if resume_dir:
    start_step = load_checkpoint_with_backend(unwrapped, optimizer, scheduler, resume_dir, backend)
```

The `--resume auto` flag automatically finds the latest checkpoint in the save directory -- useful for resuming after spot preemption without specifying the exact checkpoint path.

---

## 6. Checkpointing Changes

**File:** `src/training/checkpointing.py`

Three new functions appended to the existing file (existing functions untouched):

- `is_gcs_path(path)` -- detects `gs://` paths
- `get_checkpoint_dirs(base_dir)` -- lists checkpoint directories, supports both local filesystem and GCS (via `gcsfs` library)
- `find_latest_checkpoint(base_dir)` -- returns the most recent checkpoint directory, used for `--resume auto`

And two thin wrappers:
- `save_checkpoint_with_backend()` -- delegates to existing `save_checkpoint()` with `os.makedirs` guard
- `load_checkpoint_with_backend()` -- delegates to existing `load_checkpoint()`

The existing `save_checkpoint` / `load_checkpoint` functions already use `map_location="cpu"` for loading, which works on both GPU and TPU.

---

## 7. TPU Config

**Primary files:**

- `configs/stage2_tpu_v6e_spot.yaml` -- validated iter 24h baseline.
- `configs/stage2_tpu_v6e_spot_opt_prod5k.yaml` -- optimized
  5000-step production config.
- `configs/stage2_tpu_v6e_spot_opt_depth32.yaml` -- Phase 4
  depth-chunk candidate.
- `configs/stage2_tpu_v6e_spot_opt_nockpt.yaml` -- Phase 4
  no-activation-checkpoint candidate.

```yaml
backend: tpu

train:
  batch_size: 8
  grad_accum: 4
  compile_warmup_steps: 1
  depth_chunk_size: 16
  xla_grad_checkpoint: true
  # Effective batch = batch_size * grad_accum * num_chips
  # With v6e-8: 8 * 4 * 8 = 256

logging:
  log_every: 10
  save_every: 0        # Canonical end-of-training save only
  keep_last_n: 3
```

The key difference from the GPU config is the batch size calculation. On GPU, effective batch = `batch_size * grad_accum * num_gpus`. On TPU with SPMD, effective batch = `batch_size * grad_accum * num_chips` because SPMD automatically shards the batch across all chips via `mark_sharding`.

---

## 8. Testing Results

Tested on Colab TPU v5e-1 (single chip, 16GB HBM):

| Test | Result |
|------|--------|
| Backend auto-detection | `xla:0` detected correctly |
| Model build (4.63B params, 891M trainable) | OK |
| Move model to TPU device | OK |
| FSDPv2 wrapping (skipped on single chip) | OK |
| Full forward pass (backbone + projection + depth decoder) | OK: `text_logits [1,10,264196]`, `audio_logits [1,8,10,2048]` |
| Loss computation | OK: `loss = 9.12` |
| Backward pass | OK: gradients computed |
| Optimizer step | OOM: expected on single 16GB chip with 5B model |

The OOM on optimizer step is expected -- a 5B parameter model in bf16
is ~9.3GB, plus gradients and optimizer states exceeds 16GB. With
FSDPv2 on 8-64 chips, model parameters are sharded across the TPU mesh;
the v6e-8 production path has now completed multiple 5000-step runs.

Validated Cloud TPU runs:

| Run | Topology | Steps | Result |
|---|---|---:|---|
| `8pse8tzk` | v4-32 spot | 100 | First end-to-end TPU success; loss decreased 9.0273 -> 7.5983. |
| `7rrjupc7` | v6e-8 spot | 5000 | Iter 24h baseline; final loss 5.3558, wall 615.9 min. |
| `kzsijxv5` | v6e-8 spot | 5000 | Optimized `opt-prod5k`; final loss 5.105, p50 6.14 s/step, wall 562 min. |
| `i15igq8d` | v6e-8 spot | 300 | Phase 4 `depth_chunk_size=32` candidate; exit 0, p50 5.296 s/step, p99 5.725 s/step, examples/sec 49.13. |

### Environment variables for Colab TPU

Colab TPUs need these env vars because the GCE metadata endpoint isn't accessible through the SSH tunnel:

```bash
TPU_ACCELERATOR_TYPE=v5litepod-1
TPU_WORKER_HOSTNAMES=localhost
TPU_WORKER_ID=0
CHIPS_PER_HOST_BOUNDS=1,1,1
HOST_BOUNDS=1,1,1
TPU_SKIP_MDS_QUERY=1
DEVICE_BACKEND=tpu
HF_TOKEN=...  # For CohereLabs/tiny-aya-base (gated model)
```

On real Cloud TPU VMs, these are set automatically by the GCE metadata service.

---

## 9. Known Limitations

1. **Large TPU pods remain unvalidated.** Single-host v6e-8 SPMD is
   validated; v6e-64 / v5e-64 multi-host pods are still blocked by
   capacity and external-IP quota constraints.

2. **No GCS async checkpointing yet.** The current production path uses
   synchronous canonical final saves and disables periodic production
   saves (`save_every=0`). Async saves would still be useful for longer
   spot runs.

3. **Phase 4 activation settings are still under test.** Production
   keeps `xla_grad_checkpoint=true` and `depth_chunk_size=16`;
   `depth_chunk_size=32` passed a 300-step gate but still needs HBM
   review before durable promotion, and `xla_grad_checkpoint=false`
   remains untested.

4. **Mimi decoder on CPU for TPU training.** Audio monitoring samples are decoded on CPU (~10s per sample). This is acceptable for periodic monitoring (every 1000 steps) but would be too slow for batch evaluation.

---

## File Change Summary

| File | Change | Lines |
|------|--------|-------|
| `src/backend/__init__.py` | **NEW** -- detection + factory | 30 |
| `src/backend/base.py` | **NEW** -- abstract base class | 71 |
| `src/backend/gpu_backend.py` | **NEW** -- GPU/DDP implementation | 90 |
| `src/backend/tpu_backend.py` | **NEW** -- TPU/SPMD+FSDPv2 | 136 |
| `configs/stage2_tpu*.yaml` | **NEW/MODIFIED** -- baseline, optimized production, and Phase 4 TPU configs | n/a |
| `src/model/backbone.py` | **MODIFIED** -- added `use_cache=False` | +3 |
| `src/model/composite.py` | **MODIFIED** -- XLA-compatible checkpointing | +12 -1 |
| `src/training/checkpointing.py` | **MODIFIED** -- GCS support + find_latest | +43 |
| `scripts/train_hierarchical.py` | **MODIFIED** -- backend abstraction + W&B `global_step` x-axis | +81 -51 |
| `scripts/tpu/startup_script.sh` | **MODIFIED** -- GCS repo tarball startup path | n/a |
| `scripts/tpu/_remote_redeploy.sh` | **MODIFIED** -- uv CPython `libpython` fallback for hot redeploy | n/a |
