# Iter 24h Optimization Baseline

Baseline for `TPU_OPTIMIZATION_SPEC.md` Phase 0. Source is W&B run
`7rrjupc7` plus the iter 24h completion log captured in
`.claude/PROGRESS.md`.

`opt-prod5k` is now the promoted production reference for later phases,
but iter 24h remains the protected fallback and comparison baseline.

## Identity

| Field | Value |
|---|---|
| Run name | `v6e-spot-stage2-5k-iter24h` |
| W&B run | `7rrjupc7` |
| W&B URL | https://wandb.ai/cataluna84/tinyaya-stage2-tpu/runs/7rrjupc7 |
| State | `finished` |
| Topology | single-host v6e-8 EU spot, 8 chips |
| Config | `batch_size=8`, `grad_accum=4`, `max_frames=400`, `depth_chunk_size=16` |
| Effective batch | `256` |
| Frame-tokens per optimizer step | `102400` |
| Final checkpoint | `gs://tinyaya-stage2-tpu/checkpoints/stage2-tpu-v6e-spot/step_005000_final/` |

## Runtime and compile

| Metric | Value |
|---|---:|
| Script-reported training wall | `615.9 min` |
| W&B `_runtime` | `605.37 min` |
| First logged step time | `1145.63 s` |
| XLA compilation causes | `18 total` |
| Late recompiles | `0 after steps 1-2` |

The first logged step includes the main startup compile/execute cost.
The optimization target is steady-state throughput first, then moving
optimizer-state compilation before visible `step 1`.

## Steady-state step-time window

Window: W&B history rows with `_step >= 50`.

| Metric | Value |
|---|---:|
| Samples | `4950` |
| Mean step time | `6.8173 s` |
| p50 step time | `6.7476 s` |
| p90 step time | `6.9801 s` |
| p99 step time | `7.2486 s` |
| Examples/sec at p50 | `37.94` |
| Frame-tokens/sec at p50 | `15175.75` |

Window `_step >= 300` was nearly identical: p50 `6.7488 s`, p90
`6.9814 s`, p99 `7.2502 s`.

## Loss curve checkpoints

| W&B `_step` | Loss | Text loss | Audio loss | Step time |
|---:|---:|---:|---:|---:|
| 1 | `9.0227` | `13.6488` | `7.6578` | `1145.63 s` |
| 10 | `8.9338` | `13.7277` | `7.5611` | `6.95 s` |
| 50 | `7.9641` | `12.4778` | `6.7163` | `6.69 s` |
| 100 | `7.4719` | `10.9685` | `6.3750` | `6.70 s` |
| 300 | `6.7806` | `10.3353` | `5.7470` | `6.72 s` |
| 1000 | `6.1829` | `10.0092` | `5.1819` | `6.94 s` |
| 2500 | `5.3336` | `10.1829` | `4.3153` | `6.76 s` |
| final summary | `5.3558` | `10.3176` | `4.3240` | `6.96 s` |

W&B history's final `_step` is `4999`, while the training log reports
`5000/5000`; use the final W&B summary as the matched final metric.

## Memory data

W&B `mem/peak_gb` and `mem/allocated_gb` are `0` for this run, so iter
24h does not have a trustworthy W&B HBM trace. The TPU backend now has a
`tpu-info` fallback for SPMD memory; Phase 0 instrumentation should use
that path for future candidates.

Historical context from the v6e config: iter 19/20 measured peak HBM
around `22.67 GiB` under `batch_size=16`, grad checkpointing, and
`max_frames=400`; the optimization gate stays conservative at
`<= 29 GiB/chip`.

## Baseline verdict

Use iter 24h as the fallback until a candidate beats steady p50 step
time `6.7476 s` or examples/sec `37.94` without violating compile,
memory, stability, or quality gates.

## Promoted production reference: `opt-prod5k`

| Field | Value |
|---|---|
| Run name | `v6e-spot-stage2-opt-prod5k` |
| W&B run | `kzsijxv5` |
| W&B URL | https://wandb.ai/cataluna84/tinyaya-stage2-tpu/runs/kzsijxv5 |
| Config | `log_every=10`, `compile_warmup_steps=1`, `batch_size=8`, `grad_accum=4`, `depth_chunk_size=16`, `xla_grad_checkpoint=true` |
| Steps | `5000/5000` |
| Wall | `562 min` |
| p50 / p99 step time | `6.14 s` / `6.76 s` |
| Examples/sec | `43.04` |
| Final loss | `5.105` (`text=9.990`, `audio=4.106`) |
| Final checkpoint | `gs://tinyaya-stage2-tpu/checkpoints/stage2-tpu-v6e-spot-opt-prod5k/step_005000_final/` |

Use `opt-prod5k` as the Phase 4 comparison point unless a candidate
needs to prove it also beats the original iter 24h fallback.

Latest Phase 4 comparison: `opt-4-depth32` (W&B `i15igq8d`) completed
300/300 steps with p50 `5.296 s`, p99 `5.725 s`, examples/sec `49.13`,
and final loss `6.6539`. HBM telemetry still needs review before
durable promotion.
