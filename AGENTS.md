# AGENTS.md — TinyAya Stage 2 (flattened repo)

Briefing for agents editing this repository. The repo was **flattened**:
training/model/eval/TPU code lives at the root (`src/`, `scripts/`,
`configs/{gpu,tpu}/`, `sweeps/`) — there is no `simultaneous-translation/`
subdir anymore.

## Scope

The **training, model, eval, and TPU launch** code. The composite model
fuses a Cohere2 backbone (LoRA-fine-tuned) with a frozen Moshi depth
decoder. The repo is **dual-backend**: a backend interface
(`src/backend/{base,gpu_backend,tpu_backend}.py`, dispatched by
`get_backend()`) lets the *same* `src/` + `scripts/train_hierarchical.py`
run on GPU or TPU.

## Memory System (read first)

This repo uses the External Memory System. Before any non-trivial task,
read in order: `.claude/PLAN.md` → `.claude/PROGRESS.md` (top) →
`.claude/VERIFY.md` → `.claude/memories.md`. Lifecycle hooks load these
automatically; `Skill("keep-context-fresh")` primes them on demand.

## TPU ↔ GPU separation (the seam — keep it clean)

- **Shared & backend-agnostic:** everything in `src/` (except
  `src/backend/tpu_backend.py`) and `scripts/train_hierarchical.py`.
  Phase 0/1 fixes (loss padding-weight, collator parallel-stream, LoRA
  config, resume) and Phase 2/3 (diag dashboard, `--sweep`) help both
  backends and live in shared code behind flags.
- **TPU-only:** `configs/tpu/` + `scripts/tpu/` + `src/backend/tpu_backend.py`
  (the ONLY place that may `import torch_xla`). All XLA env flags
  (`XLA_NO_SPECIAL_SCALARS`), `mark_sharding`, on-device metric
  accumulation, and compile-warmup stay here.
- **GPU-only:** `configs/gpu/` + (future) `scripts/gpu/`.
- **Rule:** never `import torch_xla` under `src/model/` or `src/data/`.
  CI enforces this (`scripts/ci/check_backend_seam.sh`); scoped configs
  stop the GPU and TPU collaborators from stepping on each other.

## Build & test

```bash
# from the repo root
uv sync                                       # install / update deps
uv run python -m pytest tests/ -v             # if tests dir exists
uv run python -m py_compile $(git ls-files '*.py')  # quick lint
```

## TPU launch (canonical commands)

```bash
# Current production path: v6e-8 spot in europe-west4-a
# (single-host, 8 chips, ONE Python process, 32 GiB HBM/chip).
TRC_PROFILE=v6e-8-eu \
QR_NAME=tinyaya-stage2-spot-v6e8-eu-qr \
NODE_ID=tinyaya-stage2-spot-v6e8-eu \
CONFIG_FILE=configs/tpu/stage2_tpu_v6e_v2.yaml \
TPU_STRATEGY=fsdpv2_lora \
  bash scripts/tpu/launch_spot.sh

# production 15k release run (on the VM, after deploy)
bash scripts/tpu/launch_release.sh configs/tpu/stage2_tpu_v6e_v2.yaml

# hot-redeploy code without recreating the QR
bash scripts/tpu/hot_redeploy.sh

# probe sharding strategy on the live mesh
gcloud compute tpus tpu-vm ssh tinyaya-stage2-spot-v6e8-eu \
    --project=ml-pipelines-315702 --zone=europe-west4-a \
    --worker=0 --command='cd /opt/tinyaya && \
    sudo TPU_STRATEGY=fsdpv2_lora python3 scripts/tpu/probe_strategies.py --strategy=fsdpv2_lora'
```

## TPU sharding strategies (env: `TPU_STRATEGY`)

| Value | Behaviour |
|-------|-----------|
| `replicated` | Every chip holds a full model copy; only data is sharded. **OOMs the 5.17B model on v5e.** Useful for small models. |
| `fsdpv2_lora` | Shards layers that contain trainable params (LoRA-bearing CohereDecoderLayer); replicates frozen MoshiDecoderLayer. **Default for canary + production.** |
| `fsdpv2` | Shards every transformer layer including frozen ones. Tightest memory but highest comm cost. |
| `auto` | Lets the backend pick (currently == `fsdpv2_lora`). |

The strategy is selected inside
`src/backend/tpu_backend.py::wrap_model`. See
`.claude/memories.md` for empirical compile / step / HBM
measurements per strategy.

## Required env vars (TPU runtime)

```bash
export PJRT_DEVICE=TPU                        # auto-set when libtpu present
export XLA_DISABLE_FUNCTIONALIZATION=0        # MUST be 0 (pytorch/xla #8607)
export TPU_STRATEGY=fsdpv2_lora               # see table above
export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH   # libpython
```

`XLA_USE_BF16` and `XLA_DOWNCAST_BF16` are **deprecated** in
torch_xla >= 2.6 and silently no-op. Use the explicit
`model.to(torch.bfloat16)` already wired into `wrap_model`.

## Configs (split per backend)

**GPU** (`configs/gpu/`):
- `stage2_26k_parallel.yaml` — GPU parallel-stream run. Carries the
  Phase 0 `text_padding_weight` fix (ported from TPU).

**TPU** (`configs/tpu/`):
- `stage2_tpu_v6e_v2.yaml` — production 15k v6e-8 release run.
- `stage2_tpu_v6e_valfix_smoke.yaml` — short smoke (inline-val NaN fix,
  parallel stream, dashboard) before committing the slot.
- `stage2_tpu_v6e_proxy.yaml` — short/cheap proxy used by the W&B sweep
  (`sweeps/sweep_stage2.yaml`).

Each TPU config carries the Phase 0–3 knobs: `loss.text_padding_weight`,
the `lora:` block (`r`/`alpha`/`target_modules`/`lora_exclude_top`/
`num_full_ft_layers`), and `logging.{val_on_tpu,diag_metrics}`.

## Per-chip memory budget

### v5litepod-16 (16 GiB / chip)

```
Backbone (10.34 GB) + activations (5-10 GB) + grads + AdamW = OOM under replicated
                                                            = ~7-12 GB under fsdpv2_lora
                                                            = ~6-11 GB under fsdpv2
```

If `diagnose()` reports per-chip HBM > 12 GB, you're heading for OOM
once activations + grads accumulate. Switch strategy or enable
gradient checkpointing.

### v6e-8 / v4-32 (32 GiB / chip)

Both topologies share a 32 GiB HBM/chip budget, so the same per-chip
totals apply to v4-32 (4 hosts x 4 chips = 16 chips) and v6e-8
(1 host x 8 chips). The headroom relative to v5e is roughly 2x: for
the `fsdpv2_lora` strategy, peak per-chip HBM in iter 7 (v4-32) and
iter 13b (v6e-8 EU) sat under 12 GB out of the 32 GB budget. Iter 24h
then completed 5000/5000 production steps on v6e-8 with effective
batch 256 (`batch_size=8`, `grad_accum=4`, 8 chips) and ~6.7-7.0
sec/step after startup compilation. The budget that matters in
practice is activation memory (5-10 GB) + sharded params + grads +
optim state, not HBM ceiling.

## Conventions

- Default to `dataclasses` + YAML configs over kwargs forests.
- New training scripts live under `scripts/`. Keep them runnable
  with `uv run python scripts/<name>.py --help`.
- New TPU launch scripts go under `scripts/tpu/`. They must:
  - Be `bash -n`-clean.
  - Quote all paths.
  - Use `gcloud compute tpus tpu-vm ssh --command='...'`, never
    nested heredocs (use the SCP-companion pattern in
    `_remote_redeploy.sh`).
- Do not write to `/opt/tinyaya/` from a session running on a TPU
  worker — that path is overwritten by `hot_redeploy.sh`.

## TPU code documentation style (mandatory)

Every new or edited Python file under the repo root
follows the conventions below. The explicit goal is that a research
engineer fluent in **PyTorch + GPUs but new to TPU** can read the
file top-to-bottom and understand both *what* the code does and *why
the TPU forces it to look that way*.

### Audience contract

Assume the reader knows: `nn.Module`, autograd, AMP (`torch.cuda.amp`),
DDP, FSDP-on-GPU, gradient checkpointing in concept, and HuggingFace
Trainer-style flows.

Assume the reader does **not** know: PJRT, SPMD partitioner,
`torch_xla.distributed.spmd.Mesh`, FSDPv2 (the SPMD variant),
`scan_layers`, `xm.optimizer_step` vs. `optimizer.step()`,
HBM vs. host RAM, why `XLA_USE_BF16` is deprecated, why
`use_cache=True` breaks XLA tracing, why `xla_device()` is logical
and lazy, or what "lowering" / "tracing" / "HLO" mean.

### File header — `WHY THIS EXISTS`

Every Python file starts with a module docstring whose first section
is titled `WHY THIS EXISTS` and gives a 4–10 line plain-English
description of the module's role and any TPU concept introduced for
the first time in the file. Example:

```python
"""TPU backend with multiple SPMD sharding strategies.

WHY THIS EXISTS
---------------
On GPU we use DDP (one process per GPU, NCCL all-reduce on
backward). On a TPU pod we use **SPMD** — one logical Python
program drives every chip via PJRT, and the XLA partitioner
decides where each tensor lives. This file picks the partitioner's
*sharding strategy* and is the only place that should know about
XLA-specific primitives like `xs.mark_sharding`, `Mesh`, or FSDPv2.
...
"""
```

### GPU-vs-TPU comparison callouts

Whenever a TPU primitive replaces a GPU equivalent, attach a
`# GPU analogue:` comment. Examples already in the codebase:

```python
xm.optimizer_step(optimizer)            # GPU analogue: optimizer.step()
xs.mark_sharding(x, mesh, ("fsdp",))    # GPU analogue: input.cuda(rank) under DDP
model = model.to(torch.bfloat16)        # GPU analogue: torch.cuda.amp.autocast(...)
```

### Function docstrings — NumPy-style with TPU notes

Every public function/method gets a NumPy-style docstring. When the
behavior differs on TPU, add a `Notes` section starting with
`TPU note:`. Example:

```python
def wrap_model(model: nn.Module) -> nn.Module:
    """Wrap `model` with the SPMD strategy chosen via TPU_STRATEGY.

    Args:
        model: The unwrapped composite model. Must be on the XLA
            device already.

    Returns:
        The wrapped model. On a single chip this is a no-op; on a
        pod it is either a replicated mark_sharding'd model, or an
        FSDPv2-wrapped model.

    Notes:
        TPU note: bf16 cast happens *here*, not via env vars. The
        legacy `XLA_USE_BF16=1` was removed in torch_xla 2.6 and
        silently no-ops in 2.9; tensors stay in f32 and the model
        OOMs on v5e (16 GiB / chip). See `.claude/memories.md`.
    """
```

### Inline comments — explain trade-offs, don't restate code

Bad: `# loop over layers`. Good:
```python
# scan_layers compiles ONE layer's HLO and runs it via xla.while; this
# replaces the 36-way unrolled HLO that was costing 25+ min compile
# (see PROGRESS 2026-05-03T14:30:00Z).
```

### Type hints + PEP8

- All new public APIs are fully annotated.
- Run `.venv/bin/python -m ruff format` and `... -m ruff check --fix`
  on every touched file before committing. The repo's ruff config
  (`[tool.ruff]` in `pyproject.toml`) is the source of truth:
  py312 / 100-col / E,F,W,I,B,UP / ignore E501.

### YAML configs

Configs are read by both human researchers and lifecycle hooks. Each
section gets a block comment explaining what the knob does *and* what
changes when running on TPU vs GPU. Cross-link to the relevant memory
entry where useful (`# see .claude/memories.md "Per-chip memory ..."`).

### When you don't have to

Pure utility code with no TPU contact (e.g., text-tokenisation
helpers) only needs the module docstring and NumPy docstrings on
public functions; the GPU-vs-TPU callouts and the `WHY THIS EXISTS`
TPU paragraph are skipped.

### Skill alias

Run `Skill("tpu-doc-style")` to load this convention into a fresh
session's context as a checklist.

## Gotchas (training-specific)

- **XLA compile time blows up with unrolled transformer stacks.**
  36 `CohereDecoderLayer` + 6 `MoshiDecoderLayer` => 25+ minute
  compile. Mitigation: `scan_layers` (open task in `.claude/PLAN.md`).
- **`which uv` is empty under sudo on fresh TPU VMs.** Enumerate
  `/root/.local/bin/uv`, `/usr/local/bin/uv`, `/usr/bin/uv` until
  one resolves.
- **TRC quotas pre-empt.** Spot/preemptible v5e in `europe-west4-b`
  reclaims regularly. Use queued resources + checkpoint every N steps.
- **Mimi audio loading uses `transformers` API**, not the older
  `kyutai/mimi` path; keep the `transformers` pin in `pyproject.toml`.

## Where to log

| Event | Where |
|-------|-------|
| New TPU strategy decision | `.claude/memories.md` (## Architecture decisions) |
| Compile time / HBM measurement | `.claude/memories.md` (## Hardware facts) |
| Failed training run | `.claude/PROGRESS.md` (status: `fail`, kind: `exec`) |
| Successful eval result | `.claude/memories.md` (## Milestones) |

Use `/remember`, `/progress`, or the `#progress` / `#decision` quick-
capture tags.

## Out of scope (this repo)

- Data encoding / generation (separate data-pipeline repo).
- Inference / serving (separate future repo).
- v4-64 path tuning (separate config; not blocking the v6e milestone).
