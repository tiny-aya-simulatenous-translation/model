# TinyAya Stage 2 — TR↔HI Speech-to-Speech Translation at Scale

> Production-scale Stage 2 training for the TinyAya simultaneous
> Turkish↔Hindi speech-to-speech translation system, running on
> Google Cloud TPU TRC (Research Cloud) hardware.

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch](https://img.shields.io/badge/pytorch--xla-2.9-orange.svg)](https://pytorch.org/xla/release/r2.9/index.html)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](#license)
[![Branch](https://img.shields.io/badge/branch-feat%2Ftpu--support-purple.svg)](https://github.com/)
[![Status](https://img.shields.io/badge/production-v6e--8%205000%20steps%20done-success.svg)](#milestones)

---

## Table of Contents

- [What is this?](#what-is-this)
- [Highlights](#highlights)
- [Architecture](#architecture)
- [Repository layout](#repository-layout)
- [Quick start](#quick-start)
- [Training pipeline](#training-pipeline)
- [TPU operations](#tpu-operations)
- [Self-healing orchestrator](#self-healing-orchestrator)
- [External Memory System](#external-memory-system)
- [Verification](#verification)
- [Configuration](#configuration)
- [Hardware budget](#hardware-budget)
- [Troubleshooting](#troubleshooting)
- [Milestones](#milestones)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Citing](#citing)
- [License](#license)

---

## What is this?

TinyAya Stage 2 trains a 5.17B-parameter composite speech-to-speech
translation model on the **9,212 accepted parallel pairs** of
Turkish↔Hindi audio: 2,440 real FLEURS clips plus 6,772 TTS-augmented
pairs aligned word-by-word with Whisper. The model fuses a
LoRA-fine-tuned **Cohere backbone** (3.36 B params, 36 transformer
layers) with a **frozen Moshi depth decoder** (~617 M params, 6
layers) plus projection heads, encoded against the **Mimi neural audio
codec**.

Trainable surface: ~274 M parameters (~5.3 % of total). The rest is
frozen and sharded by `fsdpv2_lora` across the TPU pod.

This repo contains everything needed to reproduce the run end-to-end
on TRC hardware: data pipeline, training scripts, configs, TPU launch
infrastructure, and a self-healing orchestrator that drives the
canary-to-production loop autonomously with mandatory user check-ins.

---

## Highlights

- **Multi-strategy SPMD backend** — `replicated`, `fsdpv2`,
  `fsdpv2_lora`, `auto`, selectable via `TPU_STRATEGY`. Sharding is
  done by the XLA partitioner; no manual mesh wiring required.
- **Sub-3-minute hot redeploy** — code change → tarball → GCS → SSH
  → tmux restart, without re-creating the queued resource.
- **Cross-host single-pane observability** — on multi-host topologies
  (legacy v4-32, future v6e-64) wandb shared-mode rendezvous via GCS
  attaches all hosts to one run instead of N parallel runs. The
  current single-host v6e-8 production path skips this entirely: ONE Python
  process drives all 8 chips via SPMD, so there is exactly ONE wandb
  run with no rendezvous needed.
- **Self-healing orchestrator** — bounded autonomous iteration with
  mandatory user check-ins at T+15/30/45/60/90 min wall, encoded
  diagnosis-to-patch table, and a circuit-breaker that always
  escalates VM-level corruption to a human.
- **External Memory System** — append-only `PROGRESS.md`,
  `PLAN.md`, `VERIFY.md`, `memories.md` keep AI agents and humans
  on the same page across sessions and compactions.
- **TPU-for-GPU-engineers documentation style** — every Python file
  carries a `WHY THIS EXISTS` paragraph plus `# GPU analogue:`
  callouts, so PyTorch+GPU-fluent readers can ramp on TPU primitives
  without a separate book.

---

## Architecture

```
                      ┌────────────────────────────────────────┐
                      │  TPU v6e-8 (TRC) — 1 host, 8 chips,    │
                      │  32 GiB HBM/chip (legacy v4-32 / next  │
                      │  v6e-64 share the same per-chip HBM)   │
                      └────────────────────────────────────────┘
                                      ▲
                                      │ PJRT (ONE Python proc / pod;
                                      │ single-host SPMD on v6e-8)
                                      │ FSDPv2 SPMD partitioner
                                      │
        ┌─────────────────────────────┴─────────────────────────────┐
        │             Composite TR↔HI translation model              │
        │                                                            │
        │   ┌──────────────────────┐    ┌─────────────────────────┐  │
        │   │  Cohere backbone     │    │  Moshi depth decoder    │  │
        │   │  3.36 B (LoRA-FT)    │    │  617 M (FROZEN)         │  │
        │   │  36 layers           │    │  6 layers               │  │
        │   └──────────────────────┘    └─────────────────────────┘  │
        │                  │                       │                 │
        │                  └─────────┬─────────────┘                 │
        │                            ▼                               │
        │   ┌──────────────────────────────────────────────────────┐ │
        │   │  Projection heads  +  text/audio embed (TRAINABLE)   │ │
        │   └──────────────────────────────────────────────────────┘ │
        └────────────────────────────────────────────────────────────┘
                                      ▲
                                      │ Mimi neural audio codec
                                      │ (12.5 fps frames, 8 codebooks)
                                      │
        ┌─────────────────────────────┴─────────────────────────────┐
        │  Phase 3 data generation pipeline                          │
        │  9,212 TR↔HI pairs (2,440 FLEURS + 6,772 TTS)              │
        │  • encoded_pt.tar.gz       (Mimi tokens, .pt files)        │
        │  • encoded_alignments.tar  (Whisper word timestamps)       │
        └────────────────────────────────────────────────────────────┘
```

Key flows:

- **Train** → `scripts/train_hierarchical.py` reads YAML config,
  builds composite, picks SPMD strategy, runs hierarchical
  text+audio loss with grad_accum, writes checkpoints + wandb run.
- **Eval** → `scripts/eval_stage2.py` runs ASR-BLEU + DNSMOS on the
  best-by-val checkpoint and emits demo wavs.
- **Launch** → `scripts/tpu/launch_qr.sh` + `launch_spot.sh` create
  queued resources; `hot_redeploy.sh` ships code without re-create.

---

## Repository layout

```
tinyaya-stage2-scale/
├── AGENTS.md                                # monorepo agent norms
├── README.md                                # this file
├── CONTRIBUTING.md                          # how to contribute
├── .factory/                                # External Memory System
│   ├── PROGRESS.md  PLAN.md  VERIFY.md  memories.md
│   ├── orchestration/                       # self-healing spec + diagrams
│   ├── skills/  droids/  hooks/  commands/
│   └── settings.json
├── simultaneous-translation/                # training / model / eval
│   ├── AGENTS.md                            # subproject norms
│   ├── configs/                             # YAML configs (canary, prod, spot)
│   ├── scripts/
│   │   ├── train_hierarchical.py            # main training entry-point
│   │   ├── eval_stage2.py                   # ASR-BLEU + DNSMOS
│   │   ├── make_splits.py                   # leak-free 90/10 splits
│   │   └── tpu/
│   │       ├── launch_qr.sh / launch_spot.sh
│   │       ├── hot_redeploy.sh / _remote_redeploy.sh
│   │       ├── startup_script.sh            # TPU-VM boot
│   │       └── probe_strategies.py          # SPMD strategy probe
│   ├── src/
│   │   ├── backend/                         # tpu_backend, gpu_backend
│   │   ├── model/                           # composite, scan_utils, surgery
│   │   ├── data/                            # dataset, collator, mimi_encoder
│   │   └── training/                        # checkpointing, loss, scheduler
│   └── docs/                                # tpu-launch-plan, trc-allocation
├── phase-3-data-generation-pipeline/        # data encode + align
│   ├── AGENTS.md
│   ├── cli.py                               # encode / align subcommands
│   └── src/encoding/                        # whisper_align, mimi
└── _artifacts/                              # runtime scratch (gitignored)
    ├── orch_poll.py                         # background TPU watchdog
    ├── scheduled_checkin.py                 # T+15/30/45/60/90 timers
    └── orch_state.json                      # idempotent orchestrator state
```

---

## Quick start

### Prerequisites

- Python 3.12 (TPU images ship 3.12).
- [`uv`](https://docs.astral.sh/uv/) for dependency management.
- `gcloud` SDK with TPU + GCS access for the live training path.
- A TRC quota grant (see
  [TRC allocation](simultaneous-translation/docs/tpu-trc-allocation.md)).
- Access to the dataset:
  `tiny-aya-translate/fleurs-tr-hi-mimi-encoded` on the HF Hub.

### Local setup

```bash
git clone <repo-url> tinyaya-stage2-scale
cd tinyaya-stage2-scale

# Each subproject has its own uv-managed venv
cd simultaneous-translation && uv sync && cd ..
cd phase-3-data-generation-pipeline && uv sync && cd ..

# Verify the workstation toolchain
bash .factory/skills/verify/SKILL.md   # or: /verify (slash command)
```

### One-command TPU production launch

```bash
# v6e-8 spot in europe-west4-a (current validated production topology)
TRC_PROFILE=v6e-8-eu \
QR_NAME=tinyaya-stage2-spot-v6e8-eu-qr \
NODE_ID=tinyaya-stage2-spot-v6e8-eu \
CONFIG_FILE=configs/stage2_tpu_v6e_spot.yaml \
TPU_STRATEGY=fsdpv2_lora \
PROBE_FIRST=1 \
  bash simultaneous-translation/scripts/tpu/launch_spot.sh
```

The launch script creates the queued resource, attaches the startup
script (which clones from GCS, extracts the encoded dataset, and
starts training under tmux), and watches the run's first hour.

The legacy v4-32 spot path in `us-central2-b`
(`TRC_PROFILE=v4-32-uc2b`,
`CONFIG_FILE=configs/stage2_tpu_canary_v4_spot.yaml`) is still
supported but historical -- it ran iter 1-11 before the v6e-8 path
completed the 5000-step iter 24h production run.

---

## Training pipeline

### 1. Data preparation (workstation; one-time)

```bash
cd phase-3-data-generation-pipeline
PYTHONPATH=. python cli.py --data-dir data encode   # 9,212 .pt files
PYTHONPATH=. python cli.py --data-dir data align    # 18,424 alignment JSONs

python ../simultaneous-translation/scripts/make_splits.py \
    --accepted data/manifests/accepted.jsonl \
    --encoded-dir data/encoded \
    --out-dir data/splits --val-frac 0.10 --seed 42
```

### 2. Training (TPU)

```bash
cd simultaneous-translation
uv run python scripts/train_hierarchical.py \
    --config configs/stage2_tpu_canary_v4_spot.yaml \
    --train_split /mnt/data/splits/train.jsonl \
    --val_split   /mnt/data/splits/val.jsonl \
    --encoded_dir /mnt/data/encoded \
    --use_wandb true
```

On a TPU pod the script auto-discovers PJRT, picks `TPU_STRATEGY` from
env, casts to `bfloat16` inside `wrap_model`, and shards via FSDPv2.
On the validated v6e-8 topology this is one Python process driving
8 chips; legacy v4-32 and future v6e-64 use one process per host.

### 3. Eval (workstation or single-chip TPU)

```bash
uv run python scripts/eval_stage2.py \
    --checkpoint gs://tinyaya-stage2-tpu/checkpoints/<run>/best_by_val \
    --val_split  /path/to/val.jsonl \
    --encoded_dir /path/to/encoded \
    --out_dir eval_outputs/<run>
```

---

## TPU operations

### SPMD strategy matrix (`TPU_STRATEGY` env)

| Value         | Behaviour                                                                                | Use when |
|---------------|------------------------------------------------------------------------------------------|----------|
| `replicated`  | Full model copy per chip; only data sharded.                                             | Tiny stand-in models. **OOMs the 5.17B model on v5e/v4.** |
| `fsdpv2_lora` | Shards layers with trainable params (LoRA-bearing); replicates frozen Moshi.             | **Default for canary + production.** |
| `fsdpv2`      | Shards every transformer layer including frozen ones. Tightest memory; highest comm.     | If `fsdpv2_lora` still hits HBM ceiling. |
| `auto`        | Backend chooses (currently == `fsdpv2_lora`).                                            | Quick iteration. |

### Hot redeploy (no QR re-create, ~3 min)

```bash
cd simultaneous-translation
bash scripts/tpu/hot_redeploy.sh
```

This tarballs the working tree, uploads to
`gs://tinyaya-stage2-tpu/code/`, SSHs to every worker, runs
`_remote_redeploy.sh` (which extracts the tarball, kills the existing
tmux session, restarts training), and confirms new PIDs. On the
current single-host v6e-8 topology the SSH-into-workers loop becomes
a single SSH (one host, one PID, one tmux session); on the legacy
v4-32 / future v6e-64 multi-host topologies it iterates over all
hosts in parallel.

### Probe SPMD strategies on the live mesh

```bash
# v6e-8 EU (current production topology)
gcloud compute tpus tpu-vm ssh tinyaya-stage2-spot-v6e8-eu \
    --project=ml-pipelines-315702 --zone=europe-west4-a \
    --worker=0 --command='cd /opt/tinyaya/simultaneous-translation \
        && sudo TPU_STRATEGY=fsdpv2_lora python3 \
        scripts/tpu/probe_strategies.py --strategy=fsdpv2_lora'
```

### Watch a live run

```bash
# attach to the supervisor tmux session
gcloud compute tpus tpu-vm ssh <node> --project=ml-pipelines-315702 \
    --zone=<zone> --worker=0 -- -t 'sudo tmux attach -t train'
# detach with Ctrl-b d (does NOT kill the run)

# non-interactive scrollback
gcloud compute tpus tpu-vm ssh <node> --project=ml-pipelines-315702 \
    --zone=<zone> --worker=0 -- 'sudo tmux capture-pane -t train -p | tail -200'
```

### TRC quota / fallback tree

| Priority | Profile         | Region            | Notes |
|----------|-----------------|-------------------|-------|
| 0        | spot v6e-8      | europe-west4-a    | **Current production path (single-host, 8 chips, 32 GiB HBM/chip)** |
| 1        | on-demand v4-64 | us-central2-b     | TRC primary |
| 2        | spot v4-32      | us-central2-b     | Legacy / superseded -- ran iter 1-11, currently SUSPENDED |
| 3        | spot v5e-64     | europe-west4-b    | Biggest spot grant |
| 4        | spot v5e-64     | us-central1-a     | US v5e |
| 5        | spot v6e-64     | europe-west4-a    | Newest gen, multi-host scale-up target |
| 6        | spot v6e-64     | us-east1-d        | Newest gen, US |

See [`docs/tpu-trc-allocation.md`](simultaneous-translation/docs/tpu-trc-allocation.md)
for the verbatim TRC welcome email and 90-day window.

---

## Self-healing orchestrator

The orchestrator drives the canary-to-production loop autonomously,
with mandatory user check-ins. It runs as the chat session itself,
delegating to background pollers, custom droids, and skills.

### 4-tier architecture

```
┌────────────────────────────────────────────────────────────┐
│  Tier 1: Orchestrator session (chat)                       │
│           PATCH → DEPLOY → WATCH → CLASSIFY → DECIDE        │
└────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────┐  ┌────────────────────┐  ┌──────────────┐
│ Tier 2:      │  │ Tier 3:            │  │ Tier 4:      │
│ Custom       │  │ Skills             │  │ Background   │
│ droids       │  │ (encoded           │  │ poller       │
│              │  │ playbooks)         │  │              │
│ tpu-watchdog │  │ tpu-orchestrate    │  │ orch_poll.py │
│ tpu-diagnoser│  │ tpu-redeploy       │  │ scheduled_   │
│              │  │                    │  │ checkin.py   │
└──────────────┘  └────────────────────┘  └──────────────┘
```

### State machine

```
[*] → PATCH → DEPLOY → WATCH ──┐
                ▲              │
                │              ▼
            CLASSIFY ◄── stall/crash
                │
                ▼
              DECIDE → PATCH (T0-T2 known fix)
                     → ESCALATE (T3 corruption, T4 unknown)
                     → CHECKIN (T+15/30/45/60/90 wall)
                     → SUCCESS (step ≥ 1, loss decreasing)
```

### Check-in cadence

Every check-in shows a structured snapshot (wandb state, TPU duty
cycle, HBM, host RSS, log tail) plus 4 options:

1. **Continue** — schedule the next check-in.
2. **Abort + diagnose** — kill, dump `met.metrics_report()`, classify.
3. **Adjust + continue** — user provides a patch instruction; redeploy.
4. **Pause** — orchestrator stops; user takes manual control.

See [`.factory/orchestration/SPEC.md`](.factory/orchestration/SPEC.md)
for the full spec, [`playbook/diagnosis-table.md`][dx-table] for the
13-row symptom-to-patch table, and
[`playbook/checkin-protocol.md`][checkin] for the AskUser template.

[dx-table]: .factory/orchestration/playbook/diagnosis-table.md
[checkin]: .factory/orchestration/playbook/checkin-protocol.md

### What it has caught

| Iter | Misdiagnosis caught | Real root cause | Patch |
|------|---------------------|-----------------|-------|
| 1    | "Deadlock at T+71"  | `.item()` cpu_fallback storm — 12-16 min compile per call | Patch 7: XLA-tensor accumulators, `.detach()` |
| 3    | "4 separate wandb runs" | `xm.is_master_ordinal()` is local-to-host, not pod-global | Patch 8: `host_index() == 0 AND is_master_ordinal()` |
| 4-5  | "OOM by 2.4 GB / 41 MB" | Static memory dominates, not activations | Patch 10c: revert `grad_accum` to 2 |
| 6    | "Step 2 reached then stalled" | Per-batch shape variation triggers HLO recompile per step | Patch 11: pad to `max_frames=300` (fixed shape) |

---

## External Memory System

Long-running ML projects accumulate decisions, gotchas, and partial
state that no single chat session can hold. This repo uses
[Tip 10: Keep Context Fresh][tip10] — four append-only files at the
root of `.factory/`:

| File             | What it holds                                       |
|------------------|-----------------------------------------------------|
| `PROGRESS.md`    | Append-only log of edits, decisions, failures, verifications. Most recent at top. |
| `PLAN.md`        | Active goal, Definition of Done, phased task list. |
| `VERIFY.md`      | Bash blocks the `verify` skill executes to prove "done". |
| `memories.md`    | Long-term architecture decisions, gotchas, milestones. Edits marked SUPERSEDED, never deleted. |

Lifecycle hooks (`post_tool_use.py`, `stop.py`, `pre_compact.py`,
`session_end.py`) auto-load these files into context at the start of
each non-trivial task and write back at session end. Manual capture
via `/progress`, `/plan`, `/remember`, `/verify` slash commands or
the `#progress`, `#plan`, `#decision`, `#verify` quick-capture tags
at the start of any chat message.

[tip10]: simultaneous-translation/docs/MEMORY-SYSTEM.md

---

## Verification

The repo enforces a `/verify` gate before any commit. Run it via:

```bash
# slash command (preferred — uses the verify skill)
/verify

# or directly
bash .factory/skills/verify/SKILL.md
```

The gate runs every fenced bash block in
[`.factory/VERIFY.md`](.factory/VERIFY.md) sequentially and reports a
PASS/FAIL table. Sections cover:

- **monorepo** — settings JSON, hooks compile, AGENTS.md tiers
  exist, no merge markers, secret scan.
- **simultaneous-translation** — every `*.py` compiles, every
  `*.yaml` parses, every `*.sh` is `bash -n` clean.
- **phase-3-data-generation-pipeline** — Python compile + CLI help.
- **TPU sharding** (skipped without `PJRT_DEVICE`) — three
  strategy probes on the live mesh.
- **Orchestration artifacts** — spec / diagrams / playbook /
  poller all present and consistent.

The pre-commit hook also runs a secrets regex scrubber over the
staged diff (HF tokens, OpenAI keys, GitHub tokens, AWS access
keys, PEM private keys) — see `.factory/hooks/_lib.py`.

---

## Configuration

Configs live in `simultaneous-translation/configs/`. The ones you
will touch most:

- `stage2_tpu_v6e_spot.yaml` — production v6e-8 spot config
  (CURRENT, validated by iter 24h). `max_steps=5000`, `save_every=0`
  (canonical end-of-training save only), effective batch 256 =
  8 × 4 × 8.
- `stage2_tpu_canary_v4_spot.yaml` — legacy canary on v4-32 spot.
  `max_steps=200`, `save_every=100` (preempt-resilient),
  effective batch 64 = 2 × 2 × 16.
- `stage2_tpu_v4_spot.yaml` — legacy full 5000-step run on v4-32 spot.
  Effective batch 128 = 2 × 4 × 16.

Each config is heavily commented; cross-references to
[`memories.md`](.factory/memories.md) explain *why* each knob is
where it is. Notable knobs:

| Path                          | Default | Notes |
|-------------------------------|---------|-------|
| `train.batch_size`            | 8 (v6e prod) / 2 (legacy canary) | Per-chip batch. v6e/v4 have ~32 GiB/chip; v5e has 16. |
| `train.grad_accum`            | 4 (v6e prod) / 2 (legacy canary) | Effective_batch = batch × accum × num_chips. |
| `train.use_scan_layers`       | false   | Disabled — `_ensure_same_structure` rejects LoRA[0:33]+FullFT[34:35] split (pytorch/xla #8612). |
| `train.xla_grad_checkpoint`   | false   | Disabled — patch 11 (fixed-shape padding) made it unnecessary. |
| `data.max_frames`             | 300     | Patch 11: collator pads every batch to this; XLA compiles one HLO. |
| `logging.save_every`          | 100     | Spot-tuned; on-demand path uses 500. |

Required environment variables (TPU runtime):

```bash
export PJRT_DEVICE=TPU                    # auto-set when libtpu present
export XLA_DISABLE_FUNCTIONALIZATION=0    # MUST be 0 (pytorch/xla #8607)
export TPU_STRATEGY=fsdpv2_lora           # see strategy matrix above
export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH
```

`XLA_USE_BF16` and `XLA_DOWNCAST_BF16` are **deprecated** in
torch_xla ≥ 2.6 and silently no-op. Use the explicit
`model.to(torch.bfloat16)` already wired into `wrap_model`.

---

## Hardware budget

### v6e-8 topology (current production path)

| Property              | Value                |
|-----------------------|----------------------|
| Hosts                 | 1                    |
| Chips per host        | 8                    |
| Total chips           | 8                    |
| Python processes      | 1 (single-host SPMD) |
| HBM per chip          | 32 GiB               |
| Host RAM              | ~96 GiB              |
| Region                | europe-west4-a       |
| Project               | ml-pipelines-315702  |

### v4-32 topology (legacy iter 1-11)

| Property              | Value                |
|-----------------------|----------------------|
| Hosts                 | 4                    |
| Chips per host        | 4                    |
| Total chips           | 16                   |
| Python processes      | 4 (multi-host SPMD)  |
| HBM per chip          | 31.75 GiB            |
| Host RAM              | ~96 GiB              |
| Region                | us-central2-b        |
| Project               | ml-pipelines-315702  |

### Per-chip memory footprint (5.17B model, bf16)

Per-chip HBM is 32 GiB on both v4-32 and v6e-8, so the budget table
below applies to both topologies unchanged.

| Strategy       | Backbone               | Activations | Total       | Verdict |
|----------------|------------------------|-------------|-------------|---------|
| replicated     | 10.34 GB               | 5-10 GB     | 18-24 GB    | OOM on v5e + v4 / v6e |
| fsdpv2_lora    | 0.65 GB sharded + 0.6 GB frozen | 5-10 GB | 7-12 GB | **production default** |
| fsdpv2         | 0.65 GB sharded        | 5-10 GB     | 6-11 GB     | If fsdpv2_lora hits ceiling |

Empirical (iter 7 on v4-32 spot, batch 2 x accum 2 x 16 chips, **legacy v4-32 baseline**):

- Compile wall: ~30 min
- Steady-state: 3.41 sec/step
- Loss step 10 -> 100: 9.0273 -> 7.5983 (decreasing)

Empirical (iter 13b on v6e-8 spot EU, batch 1 x accum 2 x 8 chips, **canonical-save validation**):

- 20 steps + canonical save in 23.3 min wall (run `zd42n7di`)
- fp32 1.60 sec/step steady-state
- 2.4 GB checkpoint written; patch 19 canonical-save validated

Empirical (iter 24h on v6e-8 spot EU, batch 8 x accum 4 x 8 chips, **production**):

- 5000/5000 steps in 615.9 min wall (run
  [`7rrjupc7`](https://wandb.ai/cataluna84/tinyaya-stage2-tpu/runs/7rrjupc7))
- Final loss 5.3558 (`text=10.3176`, `audio=4.3240`)
- Steady-state ~6.7-7.0 sec/step after startup compilation
- Final canonical checkpoint uploaded to
  `gs://tinyaya-stage2-tpu/checkpoints/stage2-tpu-v6e-spot/step_005000_final/`
  (8 objects, 2.37 GiB)
- No NaN, OOM, RESOURCE_EXHAUSTED, fatal, traceback, bus-error, or
  kernel-panic signals

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `Failed to deserialize executable: UNIMPLEMENTED` | XLA persistent cache broken on v4 + torch_xla 2.9 | Remove `XLA_PERSISTENT_CACHE_PATH` from `startup_script.sh` (pytorch/xla #8930, #9094) |
| `ValueError: Layer N has mismatched keys` | `scan_layers` can't handle heterogeneous LoRA[0:33]+FullFT[34:35] | `use_scan_layers: false` in YAML (pytorch/xla #8612) |
| `AssertionError: FakeTensor + aten.index_select` | `is_layer_pure=True` + position-embedding gather | Drop `is_layer_pure=True` from scan_utils call site (PyTorch #105485) |
| `RESOURCE_EXHAUSTED` / OOM / exit 137 | Per-chip HBM exceeded | Halve `batch_size` or `grad_accum` in YAML; consider `xla_grad_checkpoint=true` |
| TPU duty=0 % AND no `step=` for 30 min AND HBM > 50 % | Healthy XLA compile, not stall | Wait. py-spy native stack will show `libtpu.so` frames if true. |
| TPU duty=0 % AND py-spy shows `cpu_fallback` / `_local_scalar_dense` | The `.item()` storm | Apply patch 7 — `.detach()` + XLA-tensor accumulators |
| Python PID at 0% CPU but parent `uv run` PID alive | `uv run` parent sleeps; real PID is the child | `ps -e --forest -o pid,pcpu,etime,comm,args` |
| 4 separate wandb runs across 4 hosts | `is_master_ordinal()` is host-local | Apply patch 8 (`host_index==0 AND is_master_ordinal`) |
| Step 2 reached, then sec/step unbounded | Per-batch shape variation → HLO recompile per step | Apply patch 11 (collator pads to `max_frames`) |
| Spot pre-emption | TRC quota reclaim | Already mitigated by `save_every=100` + `WANDB_RESUME=allow` + tmux supervisor restart loop |
| `gcloud ssh ... Connection refused` for ≥ 3 polls | VM-level corruption | **Tier 3** — escalate, never auto-recreate the QR |
| v6e bf16 NaN at step 1 | pytorch/xla #4152 (HF mask `torch.finfo(fp32).min` -> -inf in bf16) and v6e libtpu numerics #8591 / #8778 | Apply patch 20b -- `_patch_attention_mask_for_bf16` monkey-patch clamps mask values >= -1e4 (called at top of `train_hierarchical.py`); fall back to `precision: float32` if NaN persists |
| NaN or OOM at step 258/259 | Grad-accum macro-step straddles a DataLoader epoch reset, producing a new XLA graph topology | Iter 24h fix -- pad TPU tail batches to `batch_size`, keep `drop_last=False`, and reset epochs only between optimizer steps |
| Step 1/2 show 1000s+ wall time | Optimizer-state graph compiles after first visible step | Expected in iter 24h; no late recompiles after step 2. Future cleanup should pre-warm before visible `step 1` |
| Canonical save writes to `gs:/checkpoints/...` instead of GCS | `torch.save` does not understand `gs://` URIs | Apply patch 20a -- `save_checkpoint_canonical_final` runs `gsutil cp -r` post-save to upload to the actual GCS prefix |

For deeper diagnosis, attach py-spy to the *real* python PID (not the
`uv run` parent) and inspect both Python and native frames:

```bash
sudo /tmp/py_spy-0.4.2.data/scripts/py-spy dump --pid <REAL_PID> --native
```

---

## Milestones

### 2026-05-10 — First 5000-step TPU production run (iter 24h)

- **Run:** [`7rrjupc7`](https://wandb.ai/cataluna84/tinyaya-stage2-tpu/runs/7rrjupc7)
- **Hardware:** TPU v6e-8 spot, europe-west4-a, single host x 8 chips
- **Steps:** 5000/5000
- **Wall time:** 615.9 min
- **Final loss:** 5.3558 (`text=10.3176`, `audio=4.3240`)
- **Checkpoint:** `gs://tinyaya-stage2-tpu/checkpoints/stage2-tpu-v6e-spot/step_005000_final/`
  (8 objects, 2.37 GiB)
- **Stability:** no NaN/OOM/fatal signals; no late recompiles after
  the startup cluster around steps 1-2

This validates the iter 24h step-259 topology fix: TPU tail batches
are padded to a static batch dimension and epoch resets happen only
between optimizer steps, not inside the 4-way grad-accum graph.

### 2026-05-08 — Patch 19 canonical save validated on v6e-8 (iter 13b)

- **Run:** `zd42n7di`
- **Hardware:** TPU v6e-8 spot, europe-west4-a, single host x 8 chips
- **Steps:** 20 training steps + canonical end-of-training save
- **Wall time:** 23.3 min total
- **Throughput:** ~1.60 sec/step steady-state (fp32)
- **Checkpoint:** 2.4 GB written via `save_checkpoint_canonical_final`
  (the `model.to("cpu")` -> `save_pretrained` flow that defeats the
  iters 9-11 FSDPv2-XLA save deadlock)
- **Topology note:** single-host SPMD means ONE Python process drives
  all 8 chips; no cross-host rendezvous, no host-index gating, no
  shared-mode wandb umbrella.

This unblocked iter 14 (patch 20a GCS upload + patch 20b bf16 mask
monkey-patch) -- the next graduation step toward Phase 5.

### 2026-05-06 — First end-to-end TPU success (iter 7, legacy v4-32)

- **Run:** [`8pse8tzk`](https://wandb.ai/cataluna84/tinyaya-stage2-tpu/runs/8pse8tzk)
- **Hardware:** TPU v4-32 spot, us-central2-b, 4 hosts × 4 chips
- **Loss:** step 10 = 9.0273 → step 100 = 7.5983 (decreasing)
- **Throughput:** 3.41 sec/step steady-state from step 30
- **Compile:** ~30 min wall from deploy to first `loss=` line
- **Patches in flight:** 4-11 (see
  [`memories.md`](.factory/memories.md) "Architecture decisions")

This was the first end-to-end TPU success and unblocked the move to
v6e-8 EU (which removed the multi-host coordination burden once v4
spot capacity ran out).

---

## Roadmap

- [x] Validate patch 20a (GCS upload via `gsutil cp -r`) and patch
  20b (HF `AttentionMaskConverter` clamp to >= -1e4) on v6e-8 EU.
- [x] Run v6e-8 canary to `max_steps=200`; confirm checkpoint reaches
  the actual `gs://tinyaya-stage2-tpu/checkpoints/stage2-tpu-v6e-spot-canary/`
  prefix.
- [x] Phase 5: complete 5000-step production run on v6e-8 EU
  (iter 24h).
- [ ] `eval_stage2.py` ASR-BLEU + DNSMOS on the iter 24h final
  checkpoint.
- [ ] Cleanup: pre-warm optimizer-state compilation before visible
  `step 1`.
- [ ] Scale to v6e-64 multi-host pod once spot capacity allows.
- [ ] Decide on patches 12-13 (skip audio sample + validation on TPU).
- [ ] Re-evaluate `xla_grad_checkpoint=true` to free HBM for
  larger effective batch.

---

## Contributing

Pull requests welcome. Please read
[CONTRIBUTING.md](CONTRIBUTING.md) before opening one — it covers
the External Memory System workflow, branching, commit conventions,
the `/verify` gate, and the TPU-for-GPU-engineers documentation
style we expect on every Python file.

Quick contributor checklist:

1. Branch off `main` as `feat/<short-slug>` (or `fix/`, `docs/`).
2. Run `/verify` locally before committing.
3. Use conventional commits (`feat:`, `fix:`, `docs:`, …).
4. Add the co-author line for AI-pair-programmed commits.
5. Don't push directly to `main`; PRs only.

---

## Citing

If you use TinyAya Stage 2 in a paper, please cite:

```bibtex
@misc{tinyaya2026,
  title  = {TinyAya: Simultaneous Turkish-Hindi Speech Translation
            via Composite LoRA-Backbone + Frozen Depth Decoder},
  author = {Mayank Bhaskar et al.},
  year   = {2026},
  note   = {Repository: tinyaya-stage2-scale}
}
```

---

## License

Released under the **MIT License**. A `LICENSE` file containing the
full MIT text will be added at the same time as the public release;
until then, treat this repository as MIT-licensed by intent.

This project depends on:

- [PyTorch](https://pytorch.org/) and
  [PyTorch/XLA](https://pytorch.org/xla/) — BSD-3-Clause.
- [Cohere model weights](https://cohere.com/) — see Cohere's
  license; we ship LoRA deltas only.
- [Moshi / Mimi](https://github.com/kyutai-labs/moshi) — MIT.
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
  — Apache 2.0.

Dataset: [`tiny-aya-translate/fleurs-tr-hi-mimi-encoded`](https://huggingface.co/datasets)
on the HuggingFace Hub. FLEURS source data is CC BY 4.0 (Google).
