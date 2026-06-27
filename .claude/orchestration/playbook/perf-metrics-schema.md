# TPU Performance Metrics Schema

This schema standardizes metrics emitted to W&B, logs, progress entries,
and optimization reports.

## Required identity fields

| Field | Type | Description |
|---|---|---|
| `opt/candidate_id` | string | Optimization candidate, e.g. `opt-3-b16g2`. |
| `opt/base_run` | string | Baseline run, normally `iter24h`. |
| `opt/phase` | int | Optimization phase number. |
| `opt/config_diff` | string | Compact human-readable config diff. |
| `opt/promotion_gate` | string | Gate currently being tested: `20`, `300`, `1000`, `5000`, `eval`. |
| `global_step` | int | Real optimizer step. Hidden W&B step metric used as the x-axis for `train/*`, `perf/*`, `val/*`, `audio/*`, and `mem/*`. |

## Throughput metrics

| Field | Unit | Formula / source |
|---|---:|---|
| `perf/step_time` | seconds | Existing measured wall time per optimizer step. |
| `perf/p50_step_time` | seconds | Median steady-state step time over a window. |
| `perf/p90_step_time` | seconds | 90th percentile steady-state step time. |
| `perf/p99_step_time` | seconds | 99th percentile steady-state step time. |
| `perf/examples_per_sec` | examples/sec | `effective_batch / step_time`. |
| `perf/frame_tokens_per_sec` | frame-tokens/sec | `effective_batch * max_frames / step_time`. |
| `perf/effective_batch` | examples | `batch_size * grad_accum * chip_count`. |
| `perf/log_interval_sec` | seconds | Wall time covered by one logging interval. |

## Compile and profile metrics

| Field | Unit | Description |
|---|---:|---|
| `xla/compile_cause_count` | count | Total compile causes observed. |
| `xla/compile_cause_delta` | count | New compile causes after warmup. |
| `xla/first_visible_step_compile_s` | seconds | Compile time visible before or around step 1. |
| `xprof/profile_path` | string | Local or GCS profile directory. |
| `xprof/trace_window` | string | Step window, e.g. `10-20`. |

## Memory and host metrics

| Field | Unit | Description |
|---|---:|---|
| `mem/peak_gb` | GiB | Existing peak device allocation metric. |
| `mem/allocated_gb` | GiB | Existing allocated device memory metric. |
| `mem/hbm_peak_gb` | GiB | Per-chip HBM peak from TPU telemetry. |
| `host/rss_gb` | GiB | Python process RSS on TPU host. |
| `host/input_gap_ms` | ms | XProf-derived input/data wait gap if available. |
| `host/device_transfer_ms` | ms | XProf-derived host-to-device transfer time if available. |

## Quality and safety metrics

| Field | Type | Description |
|---|---|---|
| `train/loss` | float | Existing aggregate loss. |
| `train/text_loss` | float | Existing text loss. |
| `train/audio_loss` | float | Existing audio loss. |
| `safety/nonfinite_count` | int | Count of non-finite loss/grad detections. |
| `safety/late_recompile` | bool | True if compile occurs after warmup/precompile. |
| `safety/rollback_reason` | string | Reason the candidate was rolled back. |
| `eval/asr_bleu` | float | Evaluation metric from `eval_stage2.py`. |
| `eval/dnsmos` | float | Evaluation metric from `eval_stage2.py`. |

## Progress summary format

Use this one-line shape in `PROGRESS.md` details:

```text
candidate=<id> base=iter24h run=<wandb-id> steps=<n> verdict=<verdict> p50=<s>s eps=<n> hbm=<n>GiB compile_delta=<n>
```

Current promoted production reference:

```text
candidate=opt-prod5k base=iter24h run=kzsijxv5 steps=5000 verdict=promote p50=6.14s p99=6.76s eps=43.04 hbm=n/a compile_delta=0 checkpoint=gs://tinyaya-stage2-tpu/checkpoints/stage2-tpu-v6e-spot-opt-prod5k/step_005000_final/
```

Latest Phase 4 gate:

```text
candidate=opt-4-depth32 base=opt-prod5k run=i15igq8d steps=300 verdict=pass p50=5.296s p99=5.725s eps=49.13 hbm=n/a compile_delta=n/a
```

## Promotion verdicts

| Verdict | Meaning |
|---|---|
| `promote` | Candidate becomes the new baseline for the next phase. |
| `reject` | Candidate is safe but not better enough to keep. |
| `rollback` | Candidate failed a safety/quality gate; revert immediately. |
| `needs-more-data` | Candidate passed smoke but requires a longer run. |
