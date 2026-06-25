---
name: tpu-orchestrate
description: Master TPU run-control and optimization playbook for single-host v6e-8 production runs. Encodes self-healing diagnosis/recovery, T0-T4 escalation tiers, check-in cadence, and the optimization-mode gates from .claude/orchestration/TPU_OPTIMIZATION_SPEC.md. Pairs with tpu-watchdog and tpu-diagnoser droids.
---

# TPU production-run self-healing orchestration

You are the orchestrator for the TinyAya Stage 2 production training
run on **single-host v6e-8** (`tinyaya-stage2-spot-v6e8-eu` in
`europe-west4-a`, QR `tinyaya-stage2-spot-v6e8-eu-qr`).

Current milestone: iter 24h completed the first 5000-step baseline run
(W&B `7rrjupc7`, final checkpoint
`gs://tinyaya-stage2-tpu/checkpoints/stage2-tpu-v6e-spot/step_005000_final/`)
and `opt-prod5k` completed the optimized 5000-step production run
(W&B `kzsijxv5`, final checkpoint
`gs://tinyaya-stage2-tpu/checkpoints/stage2-tpu-v6e-spot-opt-prod5k/step_005000_final/`).
Phase 4 is now active; `opt-4-depth32` completed its 300-step gate as
W&B `i15igq8d` with exit 0, p50 5.296 s/step, p99 5.725 s/step, and
49.13 examples/sec. Use this skill for remaining Phase 4 candidates,
evaluation-adjacent reruns, or scale-up attempts, not to re-run iter
24h or `opt-prod5k` unless explicitly asked.

For throughput work, run this skill in **optimization mode**. Read
`.claude/orchestration/CONTROL_PLANE.md` first, then
`.claude/orchestration/TPU_OPTIMIZATION_SPEC.md` and
`.claude/orchestration/playbook/optimization-experiment-matrix.md`.
Optimization mode preserves iter 24h as the fallback baseline and
promotes candidates only after throughput, stability, memory, compile,
and quality gates pass.

Single-host topology: ONE Python process drives all 8 chips via SPMD.
There is exactly one tmux session, one PID, one wandb run; no
cross-host rendezvous, no host-index gating, no shared-mode wandb
umbrella. The watchdog needs to verify a single PID + a single
wandb run-id.

Drive the run to a known-good first compile + decreasing loss with
**bounded autonomous iteration**, calling the user only at scheduled
check-ins or on circuit-breaker trips. Full design at
`.claude/orchestration/SPEC.md`.

Unified control-plane map:
`.claude/orchestration/CONTROL_PLANE.md`.

Optimization plan:
`.claude/orchestration/TPU_OPTIMIZATION_SPEC.md`.

## Loop you run

```
PATCH -> DEPLOY -> WATCH (poll every 5-10 min)
                     |
                     +--> on T+15/30/45/60/90: AskUser checkin (4 options)
                     +--> on stall/crash: tpu-diagnoser -> classify -> patch or escalate
                     +--> on success (step>=1 + loss decreasing): pause for next phase
```

## Mode selection

Choose the mode from the user's goal:

| Mode | When | Primary spec |
|---|---|---|
| `self-healing` | supervising a live run, retrying a crash, or deploying a known patch | `.claude/orchestration/SPEC.md` |
| `optimization` | improving step time, examples/sec, compile warmup, input pipeline, batch shape, or TPU cost | `.claude/orchestration/TPU_OPTIMIZATION_SPEC.md` |

Optimization mode uses the same WATCH -> CLASSIFY -> DECIDE discipline,
but success means a candidate passes its promotion gate, not merely
`step>=1`.

## Tools you must use

- **tpu-watchdog droid** (Task tool): structured JSON snapshot of
  wandb + gcloud + ps state. In optimization mode, also capture any
  available step-time, examples/sec, profile path, and compile-delta
  fields. Call every 5-10 min wall.
- **tpu-diagnoser droid** (Task tool): regex-classifies last K log
  lines into a known root cause + recommended patches. In optimization
  mode, also classify regressions such as late recompile, input-bound
  behavior, warmup drift, throughput regression, and loss regression.
  Call only on stall/crash/regression verdicts.
- **tpu-redeploy skill** (`/tpu-redeploy`): the encoded redeploy
  procedure (rsync + tmux restart). Call after every PATCH.
- **AskUser**: the 4-option check-in dialog. Mandatory at the cadence.
- **update-plan/update-progress**: keep phase state and candidate run
  summaries in the correct memory files per `CONTROL_PLANE.md`.

## Mandatory: announce the wandb URL on every new run

Whenever a NEW wandb run is created (new `run_id` appears in the
tmux log via `wandb: View run at https://wandb.ai/...`), surface the
run URL to the user on its own line as soon as you notice it. A run
is "new" if its run_id is not the one you already announced for the
current iteration.

What to share, in this order:
1. Full wandb run URL: `https://wandb.ai/<entity>/<project>/runs/<run_id>`
2. Run name (e.g. `v6e-spot-stage2-5k-iter24`) and run_id
3. One-line note about what is unique about this iteration (config
   knobs that differ from the previous iter)

Do this even if the run is still in compile / before the first
`step=` line, so the user can open the dashboard while waiting.

## Diagnosis -> Recovery table (the playbook)

Match priority: top-to-bottom. First regex hit wins.

| # | Symptom (regex on tmux log) | Patch | Tier |
|---|---|---|---|
| 1 | `RESOURCE_EXHAUSTED` OR OOM OR `exit code 137` at consistent step >= 100 | Inspect HLO temp; suspect cache_all_gather ring buffers (iter 24 fix). Halve batch_size as fallback | T2 |
| 2 | TPU duty=0 + HBM>50% + wall>30 min + no `step=` | Kill, dump `met.metrics_report()`, ensure XLA cache unset, add `python -u` | T2 |
| 3 | `gcloud ssh.*Connection refused` | **ESCALATE -- never auto-recreate QR** | T3 |
| 4 | `kernel panic` OR `Bus error` | **ESCALATE -- never auto-recreate QR** | T3 |
| 5 | Worker PID unreachable for 3+ consecutive polls | **ESCALATE -- never auto-recreate QR** | T3 |
| 6 | Same `classification` as previous iteration | **ESCALATE** (doom loop) | T4 |
| 7 | User selects "Abort+Diag" or "Pause" at check-in | ESCALATE / pause | T4 |
| 8 | `compilation_cause_count` rising AND no error AND elapsed < 30 min | Recommend "continue" at check-in | T0 |
| 9 | `compilation_cause_count` rising AND no error AND elapsed > 60 min | Recommend "abort+diag"; offer `XLA_IR_DEBUG=1` next iter | T2 |

Full table with sources: `.claude/orchestration/playbook/diagnosis-table.md`.

## Tiers (T0-T4)

- **T0 Continue** (auto): all signals nominal
- **T1 Inject prompt** (auto): soft drift, no error -- not used in v1
- **T2 Hot redeploy with patch** (auto): rows 1-2, 8-9
- **T3 Recreate QR** (**NEVER AUTO** -- ALWAYS ESCALATE): rows 3-5
- **T4 Pause for human** (n/a): rows 6-7 + budget warnings

Full definitions: `.claude/orchestration/playbook/tier-definitions.md`.

## Check-in protocol

At each scheduled check-in (T+15/30/45/60/90 min wall), STOP the loop
and present the structured snapshot via AskUser with these 4 options:

1. **Continue** -- schedule next check-in
2. **Abort + diagnose** -- kill process, run met.metrics_report, escalate
3. **Adjust + continue** -- user provides patch instruction; you redeploy
4. **Pause loop** -- orchestrator stops; user takes manual control

Snapshot format + full per-option behavior:
`.claude/orchestration/playbook/checkin-protocol.md`.

## Auto-escalation conditions (circuit breakers)

You MUST escalate (regardless of the proposed tier) on:

1. Same classification twice consecutively
2. Tier 3 detected (LOCKED)
3. Past T+90 min wall (default cadence cap)
4. User selects Abort+Diag or Pause
5. Token budget warning from main session
6. wandb `state=crashed` AND no actionable diagnosis from diagnoser
7. Optimization candidate hits NaN/OOM/late-recompile/warmup-drift/loss
   regression or exceeds the HBM ceiling from
   `TPU_OPTIMIZATION_SPEC.md`

## State persistence

Read/write `_artifacts/orch_state.json` to track:

```json
{
  "deploy_t0_ts": "...",
  "last_checkin_min": 30,
  "last_checkin_action": "continue",
  "iteration": 24,
  "last_classification": "...",
  "consecutive_same_classification": 0,
  "checkins_done": [15, 30],
  "next_checkin_min": 45
}
```

This makes check-ins idempotent across orchestrator turns.

## Verification (Definition of Done)

- [ ] Watchdog reports `verdict=success` for 2+ consecutive polls
- [ ] wandb shows `step >= 1` AND >= 3 `loss=` lines AND loss decreasing
- [ ] Past the historical step-258 OOM threshold without RESOURCE_EXHAUSTED
- [ ] Final canonical save writes to GCS at end of run
- [ ] PROGRESS.md entry written via `/update-progress`

Optimization-mode Definition of Done:

- [ ] Candidate has a row in the optimization experiment matrix or a
  documented one-off rationale
- [ ] Candidate has W&B/log metrics for p50 step time, examples/sec,
  HBM peak, and compile delta
- [ ] Candidate passes its 20/300/1000/5000-step gate without NaN, OOM,
  late recompile, or loss regression
- [ ] Promotion/rejection is recorded in `PROGRESS.md`
- [ ] Durable promotion or gotcha is recorded in `memories.md`
