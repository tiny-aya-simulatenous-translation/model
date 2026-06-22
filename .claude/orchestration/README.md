# `.claude/orchestration/`

Source-of-truth design artifacts for the **TPU production self-healing
orchestrator**. Implementation files live elsewhere (skills, droids,
poller); this folder holds the **spec + diagrams + playbook** that
drive how the orchestrator behaves.

The folder now also owns the TPU optimization control plane: how
optimization phases, skills, hooks, droids, memories, and transient
artifacts work together without duplicating state.

## 2026-05-13 status

The orchestrator has now driven the v6e-8 path through two full
5000-step production runs:

- iter 24h baseline: W&B
  [`7rrjupc7`](https://wandb.ai/cataluna84/tinyaya-stage2-tpu/runs/7rrjupc7),
  final checkpoint
  `gs://tinyaya-stage2-tpu/checkpoints/stage2-tpu-v6e-spot/step_005000_final/`.
- `opt-prod5k`: W&B
  [`kzsijxv5`](https://wandb.ai/cataluna84/tinyaya-stage2-tpu/runs/kzsijxv5),
  final loss 5.105, p50 6.14 s/step, p99 6.76 s/step, final
  checkpoint
  `gs://tinyaya-stage2-tpu/checkpoints/stage2-tpu-v6e-spot-opt-prod5k/step_005000_final/`.

Phase 4 activation/depth-chunk work has started on the same
single-host v6e-8 EU profile. `opt-4-depth32` (W&B
[`i15igq8d`](https://wandb.ai/cataluna84/tinyaya-stage2-tpu/runs/i15igq8d))
completed its 300-step gate with exit 0, p50 5.296 s/step, p99
5.725 s/step, and 49.13 examples/sec. The playbook remains active for
the remaining Phase 4 candidates, evaluation follow-ups, and v6e-64
scale-up attempts.

The active optimization plan is tracked in
`TPU_OPTIMIZATION_SPEC.md`; cross-file responsibilities are defined in
`CONTROL_PLANE.md`.

## Layout

```
.claude/orchestration/
|-- CONTROL_PLANE.md                   # Skill/hook/droid/memory ownership map
|-- TPU_OPTIMIZATION_SPEC.md           # Exa-informed TPU optimization phases
|-- SPEC.md                            # Self-healing run-control spec
|-- README.md                          # This file
|-- diagrams/
|   |-- 01-architecture.mmd            # 4-tier architecture (Mermaid flowchart)
|   |-- 02-state-machine.mmd           # PATCH-DEPLOY-WATCH-CLASSIFY-DECIDE
|   |-- 03-sequence.mmd                # "How the loop runs in practice" v2
|   |-- 04-checkin-cadence.mmd         # T+15/30/45/60/90 timeline
|   |-- 05-tier3-escalation.mmd        # VM corruption flow
|   |-- 06-optimization-flow.mmd       # Optimization phase progression
|   |-- 07-control-plane.mmd           # Skills/hooks/agents/memory graph
|   |-- 08-memory-lifecycle.mmd        # Artifacts -> progress -> memories
|   `-- render.sh                      # mmdc helper to produce SVG/PNG
`-- playbook/
    |-- diagnosis-table.md             # 13-entry symptom -> patch table
    |-- tier-definitions.md            # T0/T1/T2/T3/T4 escalation tiers
    |-- checkin-protocol.md            # 4-option AskUser template
    |-- baseline-iter24h.md            # Phase 0 W&B/log baseline
    |-- optimization-experiment-matrix.md
    `-- perf-metrics-schema.md
```

## Implementation files (NOT here -- in their natural Factory locations)

| Artifact | Location | Purpose |
|---|---|---|
| Skill: `tpu-orchestrate` | `.claude/skills/tpu-orchestrate/SKILL.md` | Auto-loaded TPU iteration playbook |
| Skill: `tpu-redeploy` | `.claude/skills/tpu-redeploy/SKILL.md` | Encoded redeploy procedure |
| Droid: `tpu-watchdog` | `.claude/agents/tpu-watchdog.md` | Subagent: read poller log -> JSON snapshot |
| Droid: `tpu-diagnoser` | `.claude/agents/tpu-diagnoser.md` | Subagent: regex classify logs -> JSON diagnosis |
| Poller | `_artifacts/orch_poll.py` | Background: every 60s wandb + gcloud + ps -> JSONL |
| Check-in helper | `_artifacts/scheduled_checkin.py` | Detects T+15/30/45/60/90 thresholds |
| Trigger state | `_artifacts/orch_state.json` | Idempotent check-in tracker |

## How to render diagrams

```bash
cd .claude/orchestration/diagrams
bash render.sh                    # render all .mmd -> svg/ and png/
bash render.sh 03-sequence.mmd    # render one
```

Requires `mmdc` (mermaid-cli). Falls back to `npx @mermaid-js/mermaid-cli`.

## Read order for a new contributor

1. `CONTROL_PLANE.md` (source-of-truth boundaries)
2. `TPU_OPTIMIZATION_SPEC.md` (active optimization phases)
3. `playbook/baseline-iter24h.md` (protected baseline metrics)
4. `playbook/optimization-experiment-matrix.md` (candidate table)
5. `playbook/perf-metrics-schema.md` (metrics and promotion fields)
6. `SPEC.md` (self-healing runtime loop)
7. `diagrams/03-sequence.mmd` (the runtime sequence)
8. `playbook/diagnosis-table.md` (failure -> patch playbook)
9. The implementation files in their natural locations

## Versioning

This folder is checked in. Edits should bump the version note in
`SPEC.md` (top of file) and update affected diagrams in lockstep.
