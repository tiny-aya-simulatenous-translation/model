# PLAN — Get TPU training stable on v5litepod-16

> Active goal. Edited automatically by the `update-plan` skill via the
> `Stop` hook. Manual edits via `/plan` (regenerate from current goal)
> or by adding a `#plan <task>` quick-capture line.

## Goal

Run a clean 5000-step Stage 2 training on a TPU v5litepod-16 in
`europe-west4-b` with the composite TR<->HI translation model, using
the optimum SPMD sharding strategy that fits 5.17B params per chip.

## Definition of Done

- [ ] `scan_layers` enabled around backbone + depth-decoder transformer
  blocks; XLA compile completes in under 5 minutes.
- [ ] Explicit gradient checkpointing enabled; per-chip HBM usage
  reported by `diagnose()` is under 12 GB.
- [ ] `canary` config restored to `max_frames=300`,
  `depth_chunk_size=16`.
- [ ] `fsdpv2_lora` strategy runs **at least 50 successful training
  steps** with no OOM and decreasing loss.
- [ ] All commands in `VERIFY.md` pass.
- [ ] First successful checkpoint written to GCS and W&B run logged.
- [ ] 5000-step run completes; final loss + ASR-BLEU recorded in
  `memories.md` as a milestone.

## Tasks

### Phase 1 — Make compile tractable

- [x] Add `scan_layers` wrapper around `CohereDecoderLayer` (backbone,
  36 instances). _(see `src/model/scan_utils.py`,
  `replace_layers_with_scan` swaps the HF `model.layers` ModuleList
  with a `_ScannedLayerStack` proxy.)_
- [x] Add `scan_layers` wrapper around `MoshiDecoderLayer` (depth
  decoder, 6 instances). _(same wrapper, applied in
  `composite.TinyAyaMoshiComposite.__init__` when
  `use_scan_layers=True`.)_
- [x] Update `composite.py` to expose `use_scan_layers` flag.
- [ ] Re-run probe with the real model on `tiny_canary` config; confirm
  compile completes in under 5 minutes. _(requires live TPU access;
  see runbook below.)_

### Phase 2 — Memory headroom

- [x] Add `xla_grad_checkpoint` wrapper around scan units. _(threaded
  through `composite.TinyAyaMoshiComposite(xla_grad_checkpoint=True)`;
  uses `_xla_safe_checkpoint` to dodge the torch 2.9 + torch_xla 2.9
  `_get_device_module("xla")` regression.)_
- [ ] Verify activation memory is under 4 GB per chip via `diagnose()`.
  _(requires live TPU access.)_
- [ ] If still tight, try moving frozen `MoshiDecoderLayer` to bf16
  embeddings only (preserve frozen weights as f32).

### Phase 3 — Strategy selection

- [ ] Re-run `probe_strategies.py` against the real model with
  scan_layers enabled. Capture compile + step time per strategy.
- [ ] Decide between `fsdpv2_lora` and `fsdpv2` based on:
  - per-chip HBM headroom (target: > 3 GB free)
  - step time (lower is better)
  - comm volume (lower is better at LoRA scale)
- [ ] Document the decision in `memories.md` under "TPU strategy
  decisions".

### Phase 4 — Restore full canary fidelity

- [x] Set `max_frames=300` in `configs/stage2_tpu_canary.yaml`.
- [x] Set `depth_chunk_size=16` in `configs/stage2_tpu_canary.yaml`.
- [ ] Re-verify a 5-step canary training run. _(requires live TPU; see
  runbook.)_

### Phase 5 — Full 5000-step run

- [x] Configs ready: `configs/stage2_tpu.yaml` has both
  `train.use_scan_layers: true` and `train.xla_grad_checkpoint: true`.
- [ ] Use `scripts/tpu/launch_qr.sh` to start a fresh queued resource
  with `TPU_STRATEGY=<chosen>` metadata.
- [ ] Monitor via `tmux attach -t train` or
  `tail -f /tmp/train.log` for first hour.
- [ ] Confirm W&B run ID + checkpoint GCS prefix.
- [ ] Run `eval_stage2.py` on the best-by-val checkpoint; record
  ASR-BLEU + DNSMOS.

### Phase 7 — Spot fallback path (TRC v4-32 in us-central2-b)

Triggered 2026-05-05 because the on-demand v4 quota in
`us-central2-b` is currently busy. The TRC welcome email's
recommendation is "fall back to preemptible if/when on-demand is
not available". See `simultaneous-translation/docs/tpu-trc-allocation.md`
for the authoritative quota table.

- [x] Capture the TRC allocation verbatim into the repo:
  `simultaneous-translation/docs/tpu-trc-allocation.md`.
- [x] Mark the stale 5-row table in `docs/tpu-launch-plan.md` §2
  as SUPERSEDED and link to the new doc.
- [x] Log the supersedure in `.factory/memories.md`.
- [x] Add `scripts/tpu/launch_spot.sh` -- a `TRC_PROFILE`-aware
  thin wrapper over `launch_qr.sh` (default profile: `v4-32-uc2b`).
- [x] Add `configs/stage2_tpu_canary_v4_spot.yaml` and
  `configs/stage2_tpu_v4_spot.yaml`: copies of the canary and full
  configs retuned for v4-32 spot (32 GiB/chip, 16 chips, batch 4 *
  grad_accum 2 * 16 = 128 effective; `save_every: 100` for preempt
  resilience).
- [x] Wire `WANDB_RESUME=allow` into `startup_script.sh` when
  `SPOT=1` so a preempt resumes the same wandb run instead of
  forking a new one.
- [ ] Submit the spot QR + run probe -> 5-step -> 50-step canary
  -> 5000-step (requires live TPU).

### Phase 6 — Documentation pass (completed)

- [x] Add the "TPU code documentation style (mandatory)" section to
  `simultaneous-translation/AGENTS.md`.
- [x] Log the documentation-style decision into `.factory/memories.md`.
- [x] Add `.factory/skills/tpu-doc-style/SKILL.md` so future agents pick
  up the convention via `/tpu-doc-style`.
- [x] Apply the convention to every Python file under `src/` and
  `scripts/`: `WHY THIS EXISTS`, NumPy docstrings, GPU-vs-TPU
  callouts.
- [x] `ruff format --check` and `ruff check` both pass cleanly across
  `src/` + `scripts/`; every `*.py` survives `py_compile`; every YAML
  parses; every `*.sh` survives `bash -n`.

## Out of scope

- Multi-host scaling beyond v5litepod-16 (deferred to next milestone)
- Full v4-64 path (separate config exists; not blocking)
- Inference / serving path (separate goal)
