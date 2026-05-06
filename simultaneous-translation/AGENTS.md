# AGENTS.md — simultaneous-translation

Briefing for agents editing files under
`simultaneous-translation/`. Inherits and overrides the root
`/AGENTS.md`.

## Scope

This subproject contains the **training, model, eval, and TPU launch**
code. The composite model fuses a Cohere backbone (LoRA-fine-tuned)
with a frozen Moshi depth decoder.

## Build & test

```bash
# inside simultaneous-translation/
uv sync                                       # install / update deps
uv run python -m pytest tests/ -v             # if tests dir exists
uv run python -m py_compile $(git ls-files '*.py')  # quick lint
```

## TPU launch (canonical commands)

```bash
# create a queued v5litepod-16 in europe-west4-b
bash scripts/tpu/launch_qr.sh canary

# hot-redeploy code without recreating the QR
bash scripts/tpu/hot_redeploy.sh

# probe sharding strategy on the live mesh
gcloud compute tpus tpu-vm ssh tinyaya-stage2-canary \
    --project=ml-pipelines-315702 --zone=europe-west4-b \
    --worker=0 --command='cd /opt/tinyaya/simultaneous-translation && \
    sudo TPU_STRATEGY=fsdpv2_lora python3 scripts/tpu/probe_strategies.py --strategy=fsdpv2_lora'
```

## TPU sharding strategies (env: `TPU_STRATEGY`)

| Value | Behaviour |
|-------|-----------|
| `replicated` | Every chip holds a full model copy; only data is sharded. **OOMs the 5.17B model on v5e.** Useful for small models. |
| `fsdpv2_lora` | Shards layers that contain trainable params (LoRA-bearing CohereDecoderLayer); replicates frozen MoshiDecoderLayer. **Default for canary.** |
| `fsdpv2` | Shards every transformer layer including frozen ones. Tightest memory but highest comm cost. |
| `auto` | Lets the backend pick (currently == `fsdpv2_lora`). |

The strategy is selected inside
`src/backend/tpu_backend.py::wrap_model`. See
`.factory/memories.md` for empirical compile / step / HBM
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

## Configs

- `configs/stage2_scale.yaml` — full Stage 2 production config.
- `configs/stage2_tpu.yaml` — TPU full run (5000 steps).
- `configs/stage2_tpu_canary.yaml` — short canary with reduced
  `max_frames` (currently 64; restore to 300 once compile is fast).

## Per-chip memory budget (v5litepod-16, 16 GiB / chip)

```
Backbone (10.34 GB) + activations (5-10 GB) + grads + AdamW = OOM under replicated
                                                            = ~7-12 GB under fsdpv2_lora
                                                            = ~6-11 GB under fsdpv2
```

If `diagnose()` reports per-chip HBM > 12 GB, you're heading for OOM
once activations + grads accumulate. Switch strategy or enable
gradient checkpointing.

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

Every new or edited Python file under `simultaneous-translation/`
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
        OOMs on v5e (16 GiB / chip). See `.factory/memories.md`.
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
entry where useful (`# see .factory/memories.md "Per-chip memory ..."`).

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
  compile. Mitigation: `scan_layers` (open task in `.factory/PLAN.md`).
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
| New TPU strategy decision | `.factory/memories.md` (## Architecture decisions) |
| Compile time / HBM measurement | `.factory/memories.md` (## Hardware facts) |
| Failed training run | `.factory/PROGRESS.md` (status: `fail`, kind: `exec`) |
| Successful eval result | `.factory/memories.md` (## Milestones) |

Use `/remember`, `/progress`, or the `#progress` / `#decision` quick-
capture tags.

## Out of subproject scope

- Data encoding (use `phase-3-data-generation-pipeline/` instead).
- Inference / serving (separate future repo).
- v4-64 path tuning (separate config; not blocking the v5e
  milestone).
