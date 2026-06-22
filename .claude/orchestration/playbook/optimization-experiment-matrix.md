# TPU Optimization Experiment Matrix

This matrix is the operational checklist for
`TPU_OPTIMIZATION_SPEC.md`. Run one candidate at a time, promote only
after gates pass, and keep iter 24h as fallback.

## Current status (2026-05-13)

- **Promoted production config:** `opt-prod5k` (W&B
  [`kzsijxv5`](https://wandb.ai/cataluna84/tinyaya-stage2-tpu/runs/kzsijxv5))
  completed 5000/5000 steps with Phase 1 `log_every=10`, Phase 2
  `compile_warmup_steps=1`, and Phase 3 stable topology
  `batch_size=8`, `grad_accum=4`.
- **Rejected:** `opt-1-log25` due p99 outliers; `opt-3-b16g2` due NaN
  at step 130; `opt-3-b32g1` by implication because larger
  per-microbatch gradient tensors are unsafe on v6e bf16 FSDPv2.
- **Passed 300-step gate:** `opt-4-depth32` (W&B
  [`i15igq8d`](https://wandb.ai/cataluna84/tinyaya-stage2-tpu/runs/i15igq8d))
  completed 300/300 steps with exit 0, p50 5.296 s/step, p99
  5.725 s/step, examples/sec 49.13, and final loss 6.6539.
- **Next:** review HBM for `depth32`, compare against `opt-prod5k`,
  then test `opt-4-no-ckpt`; try `opt-4-depth64` only with clear HBM
  headroom.

## Candidate table

| ID | Phase | Hypothesis | Change | Run length | Promote if | Roll back if |
|---|---|---|---|---:|---|---|
| `opt-0-metrics` | 0 | We need better observability before tuning. | Add opt-in perf metrics + XProf labels, default off. | 20 steps | Metrics emit without changing default behavior. | Metrics force syncs or change losses. |
| `opt-1-log10` | 1 | Per-step materialization costs throughput. | `logging.log_every: 10`. | 300 steps | p50 step time improves; W&B still useful. | Loss visibility too sparse or late errors hidden. |
| `opt-1-log25` | 1 | Less frequent materialization helps more. | `logging.log_every: 25`. | 300 steps | Improves over log10 without monitoring loss. | Monitoring gap too large. |
| `opt-2-warmup` | 2 | Visible step 1 should be steady-state. | Add zero-LR compile warmup macro-step. | 300 steps | No weight drift; no post-step-1 compile; loss parity. | Weight/state drift, late compile, or loss regression. |
| `opt-3-b16g2` | 3 | Fewer micro-steps improves throughput. | `batch_size: 16`, `grad_accum: 2`. | 20 -> 300 steps | HBM <= ceiling, no late recompile, faster p50. | NaN/OOM/late compile/loss regression. |
| `opt-3-b32g1` | 3 | One micro-step per optimizer step maximizes utilization. | `batch_size: 32`, `grad_accum: 1`. | 20 -> 300 steps | Beats b16/g2 with HBM headroom. | Any safety gate failure. |
| `opt-4-no-ckpt` | 4 | Activation checkpoint recompute may dominate. | `xla_grad_checkpoint: false`. | 20 -> 300 steps | Faster with safe HBM and loss parity. | HBM exceeds ceiling or OOM. |
| `opt-4-depth32` | 4 | Larger depth chunks reduce loop overhead. | `depth_chunk_size: 32`. | 20 -> 300 steps | Faster with safe HBM. | OOM or worse p50. |
| `opt-4-depth64` | 4 | Even larger chunks may improve utilization. | `depth_chunk_size: 64`. | 20 -> 300 steps | Faster than depth32 with safe HBM. | OOM or worse p50. |
| `opt-5-mpdl` | 5 | Host-to-device feed may be input-bound. | Opt-in `MPDeviceLoader`/prefetch path. | 300 steps | XProf idle/transfer gaps shrink. | Shape churn or no throughput gain. |
| `opt-5-workers8` | 5 | More host workers may feed TPU better. | `num_workers: 8`. | 300 steps | Input gap shrinks; RSS safe. | Host RSS/IO contention or no gain. |
| `opt-6-buckets` | 6 | Static buckets can reduce padding waste. | Prewarmed `200/300/400` buckets. | 300 -> 1000 steps | No surprise compile; less wall-clock. | Late compile or bucket switch topology bug. |
| `opt-7-scan` | 7 | `scan_layers` may reduce compile/runtime overhead. | Isolated scan probe after structure fix. | synthetic -> 300 steps | Structure/purity and stability pass. | Mismatched keys, FakeTensor crash, NaN, or no gain. |
| `opt-7-flash` | 7 | Flash attention may improve runtime. | Flash attention A/B. | 300 steps | Faster with identical numerics. | NaN or loss regression. |
| `opt-7-syncfree` | 7 | Optimizer sync may dominate. | Isolated syncfree/optimizer alternative. | 300 steps | Optimizer cost drops and loss parity holds. | Any numerical mismatch. |
| `opt-8-promote` | 8 | Best candidate improves production. | Full selected config. | 1000 -> 5000 steps | Beats iter 24h wall/throughput with eval parity. | Any global stop condition. |

## Standard run ladder

Use the shortest ladder that can prove the candidate:

1. **Static smoke:** compile + 20 steps.
2. **Failure-boundary pass:** 300 steps.
3. **Promotion pass:** 1000 steps.
4. **Production pass:** 5000 steps + final checkpoint + eval.

Candidates in Phases 1-5 may skip directly from 300 to 1000 only after
all lower-risk candidates in that phase have passed. Phase 7 candidates
must start isolated and never bypass the 300-step gate.

## Required record fields

Every candidate result should be summarized in `PROGRESS.md`:

| Field | Example |
|---|---|
| candidate | `opt-3-b16g2` |
| base | `iter24h` |
| config diff | `batch_size 8->16, grad_accum 4->2` |
| run ID | W&B run id |
| run length | `300 steps` |
| p50 step time | `5.2s` |
| examples/sec | `49.2` |
| HBM peak | `27.4 GiB` |
| compile delta | `0 after warmup` |
| verdict | `promote`, `reject`, `rollback`, or `needs-more-data` |

Promoted configs and durable gotchas also get one compact entry in
`memories.md`.
