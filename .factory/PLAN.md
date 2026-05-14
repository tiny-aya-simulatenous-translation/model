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
- **Phase 1 result:** `opt-1-log10` and `opt-1-log25` both completed
  300/300 steps without NaN/OOM/traceback. `log10` finished with p50
  `5.9509s`, p90 `5.9741s`, p99 `6.2421s`; `log25` finished with p50
  `5.9213s`, p90 `6.0600s`, p99 `40.9924s`. `log10` is selected
  for the 1000-step validation gate.
- **Validation status:** `opt-1-log10-1k` started but was interrupted by
  service-initiated spot TPU preemption/maintenance before completing
  1000 steps; partial metrics stayed healthy.
- **Validation result:** the same-zone retry `58k4t99h` completed
  1000/1000 steps with exit status 0 and final checkpoint upload, but
  throughput regressed versus iter 24h (`p50=6.9605s`, examples/sec
  `37.13`), so promotion to 5000 steps is not automatic.
- **Diagnostic result:** `opt-1-log10-hot300` completed 300/300 steps
  via hot redeploy without `PT_XLA_DEBUG_LEVEL=1` and restored the
  prior Phase 1 throughput envelope (`p50=5.94085s`, examples/sec
  `42.9489`). Treat the 1000-step startup-path validation as
  throughput-confounded; run a 1000-step hot-redeploy validation before
  any 5000-step promotion or final rejection.
- **Hot validation result:** `opt-1-log10-hot1k` completed 1000/1000
  steps via hot redeploy with exit status 0 and final checkpoint upload.
  It restored throughput versus iter24h (`p50=5.92593s`, examples/sec
  `43.05558`) and is eligible for a user-approved 5000-step production
  pass.
- **Phase 2 result:** `opt-2-warmup-r1` completed 300/300 steps with
  exit status 0. Compile warmup (sampled trainable-weight sentinel)
  passed cleanly. `p50=5.879s`, `p99=6.158s`, examples/sec `42.55`,
  loss `6.655`. Compile warmup is throughput-neutral versus hot1k
  baseline (p50 0.8% faster, p99 2.6% tighter, examples/sec 1.2%
  slower). The tighter p99 confirms fewer late-recompile outliers.
- **Phase 3 result:** `opt-3-b16g2` 20-step smoke passed (4.21s/step,
  60.8 examples/sec) but 300-step gate failed with NaN at step 130.
  Diagnosis: v6e bf16 reduce-scatter bug (pytorch/xla #8591/#8778)
  triggered by 2x larger per-microbatch gradient tensors at b=16.
  b=32/g=1 is also unsafe. Phase 3 batch sweep closed; b=8/g=4
  remains the only viable batch topology under FSDPv2+bf16 on v6e-8.
- **Production pass result:** `opt-prod5k` (W&B `kzsijxv5`) completed
  5000/5000 steps with exit 0 and canonical checkpoint. Combines
  Phases 1+2+3 optimizations. p50=6.14s, p99=6.76s,
  examples/sec=43.04, final loss=5.105. Wall 562 min. 11.8% faster
  step time vs iter24h baseline, 4.7% lower loss.
  W&B: https://wandb.ai/cataluna84/tinyaya-stage2-tpu/runs/kzsijxv5
  GCS: gs://tinyaya-stage2-tpu/checkpoints/stage2-tpu-v6e-spot-opt-prod5k/step_005000_final/
- **Phase 4 result:** `opt-4-depth32` launched on v6e-8 EU with a
  GCS repo tarball startup path after spot preemption/capacity retries,
  then completed 300/300 steps with exit 0. W&B `i15igq8d`:
  https://wandb.ai/cataluna84/tinyaya-stage2-tpu/runs/i15igq8d.
  Final metrics: p50=5.296s, p99=5.725s, examples/sec=49.13,
  loss=6.6539. No NaN/OOM/fatal log signatures. HBM telemetry review
  is now resolved by smoke `enzsklrh`, which recorded
  `mem/hbm_available=1`, `mem/peak_gb=21.57`, and host RSS
  `56.46 GiB`; depth32 remains below the 29 GiB HBM gate.
- **Phase 4 rejection:** `opt-4-no-ckpt` (W&B `wvgzewlk`) was aborted
  before the 300-step gate. It stalled after compile warmup with TPU
  duty 0%, reached internal `step=1` at `29.08 GiB / 31.25 GiB`, and
  crossed the 29 GiB HBM abort gate. Keep `xla_grad_checkpoint=true`.
- **Phase 4 result:** `opt-4-depth64` (W&B `5mhltpif`) completed
  300/300 steps with exit 0 and final checkpoint upload. Final metrics:
  p50=3.810s, p99=3.954s, examples/sec=67.24, loss=6.6668, and
  HBM peak=26.34 GiB. It is the current fastest Phase 4 candidate and
  remains below the 29 GiB HBM gate.

## Definition of Done

- [x] First 5000-step v6e-8 baseline run completed and checkpointed.
- [x] Orchestration folder contains optimization spec, control plane,
  experiment matrix, perf schema, and diagrams.
- [x] Skills, droids, hooks, memories, and VERIFY reference the unified
  control plane consistently.
- [x] Phase 0 baseline metrics from iter 24h are captured in the
  optimization record.
- [x] At least one low-risk optimization candidate passes a 300-step TPU
  gate without NaN/OOM/late-recompile.
- [x] Best promoted optimization config completes a 1000-step validation
  pass.
- [x] Best promoted optimization config completes a 5000-step production
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

- [x] Test `log_every=10` against baseline.
- [x] If stable, test `log_every=25`.
- [x] Promote the fastest logging cadence that keeps monitoring useful.

### Phase 2 — Compile warmup before visible step 1

- [x] Add opt-in `train.compile_warmup_steps` for TPU.
- [x] Implement zero-LR/zero-weight-decay static macro-step warmup.
- [x] Verify no weight drift, no late compile after step 1, and matched
  300-step loss parity.

### Phase 3 — Batch/grad-accum sweep at fixed effective batch 256

- [x] Re-run baseline `batch_size=8`, `grad_accum=4` with instrumentation.
- [x] Test `batch_size=16`, `grad_accum=2` through 20-step and 300-step gates.
- [x] Test `batch_size=32`, `grad_accum=1` only if `b=16/g=2` passes with
  enough HBM headroom.
- [x] Promote the lowest safe p50 step time.

**Result: b=16/g=2 NaN'd at step 130 (v6e bf16 reduce-scatter bug
pytorch/xla #8591/#8778). b=32/g=1 is also unsafe. Phase 3 closed;
b=8/g=4 remains the only viable batch topology under FSDPv2+bf16
on v6e-8.**

### Phase 4 — Activation and depth-chunk sweep

- [x] Restore HBM instrumentation before further Phase 4 promotion
  (W&B `enzsklrh`, `mem/peak_gb=21.57`, `mem/hbm_available=1`).
- [x] Test `xla_grad_checkpoint=false` on the best Phase 3 candidate
  (rejected: W&B `wvgzewlk`, `29.08 GiB` HBM at internal step 1).
- [x] Test `depth_chunk_size=32` (`opt-4-depth32`, W&B `i15igq8d`, passed 300-step gate).
- [x] Test `depth_chunk_size=64` (`opt-4-depth64`, W&B `5mhltpif`, passed 300-step gate).
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

- [x] Run a 1000-step validation pass for the best candidate.
- [x] Run a hot-redeploy 1000-step log10 validation without
  `PT_XLA_DEBUG_LEVEL=1` before any 5000-step promotion.
- [x] Run a 5000-step production pass if the 1000-step pass is stable.
- [ ] Run `eval_stage2.py` and record ASR-BLEU + DNSMOS.
- [ ] Update `memories.md` with the promoted config or rejection reason.

## Out of scope

- Recreating TPU queued resources without explicit user approval.
- Overwriting the iter 24h canonical checkpoint.
- Re-enabling historically unstable barrier/per-layer-wrap paths in the
  production config without isolated proof.
