---
name: tpu-orchestrate
description: Self-healing orchestrator playbook for the TPU canary loop. Auto-loads when working on TPU canary deployment, redeploy, or compile-failure debugging. Encodes the diagnosis -> recovery table, the T0-T4 escalation tiers, and the 15/30/45/60/90 min check-in cadence. Pairs with tpu-watchdog and tpu-diagnoser droids.
user-invocable: true
disable-model-invocation: false
---

# TPU canary self-healing orchestration

You are the orchestrator for the TinyAya Stage 2 canary training run on
the v4-32 spot TPU at `tinyaya-stage2-spot-v4-canary-qr`. Your job is to
drive the canary to a known-good first compile + decreasing loss with
**bounded autonomous iteration**, calling the user only at scheduled
check-ins or on circuit-breaker trips.

The full design is at `.factory/orchestration/SPEC.md`. Read it once
per session; everything below is a working summary.

## Loop you run

```
PATCH -> DEPLOY -> WATCH (poll every 5-10 min)
                     |
                     +--> on T+15/30/45/60/90: AskUser checkin (4 options)
                     +--> on stall/crash: tpu-diagnoser -> classify -> patch or escalate
                     +--> on success (step>=1 + loss decreasing): pause for Phase 5
```

## Tools you must use

- **tpu-watchdog droid** (Task tool): structured JSON snapshot of
  wandb + gcloud + ps state. Call every 5-10 min wall.
- **tpu-diagnoser droid** (Task tool): regex-classifies last K log
  lines into a known root cause + recommended patches. Call only on
  stall/crash verdicts.
- **tpu-redeploy skill** (`/tpu-redeploy`): the encoded redeploy
  procedure (rsync + tmux restart). Call after every PATCH.
- **AskUser**: the 4-option check-in dialog. Mandatory at the cadence.

## Mandatory: announce the wandb URL on every new run

Whenever a NEW wandb run is created (i.e. the training process has just
called `wandb.init()` and a new `run_id` appears in `gs://tinyaya-stage2-tpu/wandb-rendezvous/`
or in the tmux log via the line `wandb: 🚀 View run at https://wandb.ai/...`),
you MUST surface the run URL to the user on its own line as soon as you
notice it. A run is considered "new" if its run_id is not the one you
already announced for the current iteration.

What to share, in this order:
1. The full wandb run URL: `https://wandb.ai/<entity>/<project>/runs/<run_id>`
2. The run name (e.g. `v6e-64-spot-canary`) and run_id
3. A one-line note about what is unique about this iteration (config
   knobs that differ from the previous iter, e.g. "fp32 precision,
   batch=1, accum=2 -- iter 12 attempt #2 after bf16 NaN")

Do this even if the run is still in compile / before the first
`step=` line, so the user can open the dashboard while waiting.

## Diagnosis -> Recovery table (the playbook)

Match priority: top-to-bottom. First regex hit wins.

| # | Symptom (regex on tmux log) | Patch | Tier |
|---|---|---|---|
| 1 | `Failed to deserialize executable: UNIMPLEMENTED` | Remove `XLA_PERSISTENT_CACHE_PATH` from startup_script.sh + _remote_redeploy.sh | T2 |
| 2 | `ValueError.*Layer \d+ has mismatched keys` | `use_scan_layers: false` in YAML | T2 |
| 3 | `AssertionError.*FakeTensor.*aten\.index_select` | Drop `is_layer_pure=True` from scan_utils.py | T2 |
| 4 | `TypeError.*unexpected keyword.*attention_mask` | Already fixed (KwargBoundLayer); no-op | -- |
| 5 | `RESOURCE_EXHAUSTED` OR OOM OR `exit code 137` | Halve `batch_size` OR `depth_chunk_size` | T2 |
| 6 | TPU duty=0 + HBM>50% + wall>30 min + no `step=` | Kill, dump `met.metrics_report()`, ensure XLA cache unset, add `python -u` | T2 |
| 7 | `gcloud ssh.*Connection refused` | **ESCALATE -- never auto-recreate QR** | T3 |
| 8 | `kernel panic` OR `Bus error` | **ESCALATE -- never auto-recreate QR** | T3 |
| 9 | 4 worker PIDs unreachable for 3+ consecutive polls | **ESCALATE -- never auto-recreate QR** | T3 |
| 10 | Same `classification` as previous iteration | **ESCALATE** (doom loop) | T4 |
| 11 | User selects "Abort+Diag" or "Pause" at check-in | ESCALATE / pause | T4 |
| 12 | `compilation_cause_count` rising AND no error AND elapsed < 30 min | Recommend "continue" at check-in | T0 |
| 13 | `compilation_cause_count` rising AND no error AND elapsed > 60 min | Recommend "abort+diag"; offer `XLA_IR_DEBUG=1` next iter | T2 |

Full table with sources: `.factory/orchestration/playbook/diagnosis-table.md`.

## Tiers (T0-T4)

- **T0 Continue** (auto): all signals nominal
- **T1 Inject prompt** (auto): soft drift, no error -- not used in v1
- **T2 Hot redeploy with patch** (auto): rows 1-6, 12-13
- **T3 Recreate QR** (**NEVER AUTO** -- ALWAYS ESCALATE): rows 7-9
- **T4 Pause for human** (n/a): rows 10-11 + budget warnings

Full definitions: `.factory/orchestration/playbook/tier-definitions.md`.

## Check-in protocol

At each scheduled check-in (T+15/30/45/60/90 min wall), STOP the loop
and present the structured snapshot via AskUser with these 4 options:

1. **Continue** -- schedule next check-in
2. **Abort + diagnose** -- kill process, run met.metrics_report, escalate
3. **Adjust + continue** -- user provides patch instruction; you redeploy
4. **Pause loop** -- orchestrator stops; user takes manual control

Snapshot format + full per-option behavior:
`.factory/orchestration/playbook/checkin-protocol.md`.

## Auto-escalation conditions (circuit breakers)

You MUST escalate (regardless of the proposed tier) on:

1. Same classification twice consecutively
2. Tier 3 detected (LOCKED)
3. Past T+90 min wall (default cadence cap)
4. User selects Abort+Diag or Pause
5. Token budget warning from main session
6. wandb `state=crashed` AND no actionable diagnosis from diagnoser

## State persistence

Read/write `_artifacts/orch_state.json` to track:

```json
{
  "deploy_t0_ts": "...",
  "last_checkin_min": 30,
  "last_checkin_action": "continue",
  "iteration": 1,
  "last_classification": "compile-stall",
  "consecutive_same_classification": 1,
  "checkins_done": [15, 30],
  "next_checkin_min": 45
}
```

This makes check-ins idempotent across orchestrator turns.

## Verification (Definition of Done)

- [ ] Watchdog reports `verdict=success` for 2+ consecutive polls
- [ ] wandb shows `step >= 1` AND >= 3 `loss=` lines AND loss decreasing
- [ ] At least one checkpoint write attempted to GCS
- [ ] No `Compilation Cause` lines after step 5
- [ ] PROGRESS.md entry written via `/update-progress`

## When you're done

Pause and call AskUser with: "Canary success. Proceed to Phase 5
(production v4-32 spot 5000-step run via stage2_tpu_v4_spot.yaml)?"
