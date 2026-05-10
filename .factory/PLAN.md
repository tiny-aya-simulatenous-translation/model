# PLAN — TPU optimization control plane and experiment phases

> Active goal. Edited automatically by the `update-plan` skill via the
> `Stop` hook. Manual edits via `/plan` or `#plan <task>` quick-capture.

## Goal

Unify the TinyAya Stage 2 TPU optimization workflow under
`.factory/orchestration/`, then use the Exa-informed optimization phases
to improve the validated iter 24h single-host v6e-8 training path while
preserving stability, checkpoint safety, and evaluation quality.

## Status snapshot (2026-05-10)

- **Baseline protected:** iter 24h completed 5000/5000 steps on
  single-host v6e-8 EU spot with exit status 0.
- **W&B:** `v6e-spot-stage2-5k-iter24h`, run `7rrjupc7`:
  https://wandb.ai/cataluna84/tinyaya-stage2-tpu/runs/7rrjupc7.
- **Final training stats:** loss `5.3558` (`text=10.3176`,
  `audio=4.3240`), wall `615.9 min`, final step `6.96s/step`.
- **Checkpoint:** canonical final save at
  `gs://tinyaya-stage2-tpu/checkpoints/stage2-tpu-v6e-spot/step_005000_final/`.
- **Stability:** no NaN/OOM/RESOURCE_EXHAUSTED/fatal/traceback/bus-error/kernel-panic
  and no late recompiles through step 5000.
- **Control-plane update:** `.factory/orchestration/CONTROL_PLANE.md`
  and `.factory/orchestration/TPU_OPTIMIZATION_SPEC.md` are the new
  operational sources for optimization work.

## Definition of Done

- [x] First 5000-step v6e-8 baseline run completed and checkpointed.
- [x] Orchestration folder contains optimization spec, control plane,
  experiment matrix, perf schema, and diagrams.
- [x] Skills, droids, hooks, memories, and VERIFY reference the unified
  control plane consistently.
- [x] Phase 0 baseline metrics from iter 24h are captured in the
  optimization record.
- [ ] At least one low-risk optimization candidate passes a 300-step TPU
  gate without NaN/OOM/late-recompile.
- [ ] Best promoted optimization config completes a 1000-step validation
  pass.
- [ ] Best promoted optimization config completes a 5000-step production
  pass or is explicitly rejected with a recorded reason.
- [ ] `eval_stage2.py` ASR-BLEU + DNSMOS are recorded for the selected
  optimized checkpoint or the iter 24h fallback.
- [x] `.factory/VERIFY.md` passes on the workstation.

## Tasks

### Phase 0 — Measurement harness and baseline

- [x] Extract iter 24h baseline metrics: compile duration, p50/p90/p99
  step time, examples/sec, frame-tokens/sec, HBM peak, compile-cause
  count, and matched-step loss.
- [x] Add opt-in perf metrics and XProf labels without changing default
  training behavior.
- [x] Record the baseline in `PROGRESS.md`; add a `memories.md` entry only
  if a durable decision is made.

### Phase 1 — Low-risk sync/logging optimization

- [ ] Test `log_every=10` against baseline.
- [ ] If stable, test `log_every=25`.
- [ ] Promote the fastest logging cadence that keeps monitoring useful.

### Phase 2 — Compile warmup before visible step 1

- [ ] Add opt-in `train.compile_warmup_steps` for TPU.
- [ ] Implement zero-LR/zero-weight-decay static macro-step warmup.
- [ ] Verify no weight drift, no late compile after step 1, and matched
  300-step loss parity.

### Phase 3 — Batch/grad-accum sweep at fixed effective batch 256

- [ ] Re-run baseline `batch_size=8`, `grad_accum=4` with instrumentation.
- [ ] Test `batch_size=16`, `grad_accum=2` through 20-step and 300-step gates.
- [ ] Test `batch_size=32`, `grad_accum=1` only if `b=16/g=2` passes with
  enough HBM headroom.
- [ ] Promote the lowest safe p50 step time.

### Phase 4 — Activation and depth-chunk sweep

- [ ] Test `xla_grad_checkpoint=false` on the best Phase 3 candidate.
- [ ] Test `depth_chunk_size=32`.
- [ ] Test `depth_chunk_size=64` only if HBM remains safe.
- [ ] Keep iter 24h defaults if larger candidates regress or OOM.

### Phase 5 — Input pipeline and transfer profiling

- [ ] Use XProf to determine whether host/device input gaps exist.
- [ ] Add opt-in `MPDeviceLoader`/prefetch only if profiling shows an
  input-bound run.
- [ ] Sweep `num_workers=4/8` only if host feed is a bottleneck.

### Phase 6 — Static bucketing and padding optimization

- [ ] Quantify padding waste at `max_frames=400`.
- [ ] If material, test static prewarmed buckets such as `200/300/400`.
- [ ] Require macro-step-boundary bucket switches and no surprise late
  compile before promotion.

### Phase 7 — Isolated high-risk experiments

- [ ] Probe `scan_layers` only after resolving heterogeneous LoRA/full-FT
  structure constraints.
- [ ] Keep per-layer `Cohere2DecoderLayer` FSDPv2 wraps and
  `fsdp_barrier_hook` disabled unless isolated probes prove safety.
- [ ] Run flash-attention A/B only after lower-risk phases.
- [ ] Try syncfree/optimizer alternatives only if XProf shows optimizer
  sync dominates.

### Phase 8 — Promotion run and evaluation

- [ ] Run a 1000-step validation pass for the best candidate.
- [ ] Run a 5000-step production pass if the 1000-step pass is stable.
- [ ] Run `eval_stage2.py` and record ASR-BLEU + DNSMOS.
- [ ] Update `memories.md` with the promoted config or rejection reason.

## Out of scope

- Recreating TPU queued resources without explicit user approval.
- Overwriting the iter 24h canonical checkpoint.
- Re-enabling historically unstable barrier/per-layer-wrap paths in the
  production config without isolated proof.
