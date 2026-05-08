# TPU Canary Self-Healing Orchestrator -- SPEC

## 2026-05-08 update banner

The original SPEC below describes the v4-32 spot canary in
`us-central2-b` (4 hosts, 4 Python processes). Since 2026-05-08 the
active canary has pivoted to **single-host TPU v6e-8 spot in
`europe-west4-a`** (QR `tinyaya-stage2-spot-v6e8-eu-qr`, ONE Python
process driving 8 chips via SPMD). All references to "4 worker
PIDs", multi-host wandb shared-mode rendezvous, and the
`gcloud ssh --worker=all` fan-out should be read with the topology
qualifier: on v6e-8 the watchdog inspects ONE worker PID, ONE tmux
session, ONE wandb run; on legacy v4-32 / future v6e-64 it inspects
4 / 8 (one per host). The state machine, check-in cadence, and
diagnosis table are topology-agnostic and remain the source of
truth.

**Version:** v2 (2026-05-06)
**Status:** Approved (current canary is single-host v6e-8 EU since
2026-05-08)
**Branch:** `feat/tpu-support`
**Implementation:** Option B (Skills + Custom Droids hybrid)

---

## 1. Goal

Drive the v4-32 spot canary to a known-good first compile + decreasing
loss with **bounded autonomous iteration** (legacy; current canary is
single-host v6e-8 EU since 2026-05-08). Fix-and-redeploy on
classified errors without re-asking; pause for human only on
(a) success, (b) check-in milestone, or (c) circuit-breaker trip.

## 2. Decisions locked

| Decision | Value | Why |
|---|---|---|
| Folder | `.factory/orchestration/` | Memory-system adjacent |
| Implementation | Option B | Best autonomy/complexity ratio |
| Iteration cap | OPEN (no hard cap) | Replaced by check-ins |
| Wall-time cap | OPEN | Compile time genuinely varies |
| First-step deadline | OPEN | Sharding distribution takes time on FSDPv2-LoRA SPMD |
| Check-in cadence | **15 / 30 / 45 / 60 / 90** min wall | Mandatory `AskUser` |
| Tier 3 (VM corruption) | **Always escalate** (LOCKED) | New QR costs >> human review |

## 3. Architecture (4 tiers)

See `diagrams/01-architecture.mmd`.

```mermaid
flowchart TD
    subgraph T1["Tier 1: Orchestrator (this session)"]
        SM[State Machine]
    end
    subgraph T2["Tier 2: Custom Droids"]
        WD[tpu-watchdog]
        DG[tpu-diagnoser]
    end
    subgraph T3["Tier 3: Skills"]
        SO[tpu-orchestrate]
        SR[tpu-redeploy]
    end
    subgraph T4["Tier 4: Background"]
        BG[orch_poll.py]
        CK[scheduled_checkin.py]
    end
    SM -.invokes.-> WD
    SM -.invokes.-> DG
    SM -.loads.-> SO
    SM -.loads.-> SR
    SM -.reads.-> CK
    BG -.writes.-> Log[orch_poll.log]
    WD -.reads.-> Log
    CK -.reads.-> Log
```

## 4. State machine

See `diagrams/02-state-machine.mmd`.

```mermaid
stateDiagram-v2
    [*] --> PATCH
    PATCH --> DEPLOY
    DEPLOY --> WATCH
    WATCH --> WATCH: heartbeat ok
    WATCH --> CHECKIN: T+15/30/45/60/90
    WATCH --> SUCCESS: step>=1 + loss decreasing
    WATCH --> CLASSIFY: stall or crash
    CHECKIN --> WATCH: user continues
    CHECKIN --> ESCALATE: user aborts
    CHECKIN --> PATCH: user adjusts
    CLASSIFY --> DECIDE
    DECIDE --> PATCH: known fix + not repeat
    DECIDE --> ESCALATE: unknown OR repeat OR T3
    SUCCESS --> [*]
    ESCALATE --> [*]
```

## 5. How the loop runs in practice (concrete sequence)

See `diagrams/03-sequence.mmd`.

```mermaid
sequenceDiagram
    participant U as User
    participant O as Me (orch)
    participant P as orch_poll.py (bg)
    participant W as tpu-watchdog
    participant D as tpu-diagnoser
    participant T as TPU (4 workers)

    U->>O: go
    O->>O: PATCH: apply file edits
    O->>T: DEPLOY: bash _remote_redeploy.sh
    O->>P: launch fireAndForget
    P-->>P: every 60s -> JSONL

    loop every 5-10 min wall
      O->>W: Task("watchdog")
      W-->>O: JSON {state, step, verdict, elapsed_min}

      alt elapsed_min in 15,30,45,60,90 (checkin)
        O->>U: AskUser snapshot + 4 options
        alt user picks Continue
          O->>O: schedule next checkin
        else user picks Abort+Diag
          O->>D: Task("diagnoser")
          D-->>O: classification + patches
          O-->>U: ESCALATE with diagnosis
        else user picks Adjust
          O->>O: PATCH user edits
          O->>T: DEPLOY again
        else user picks Pause
          O-->>U: orchestrator stops
        end

      else verdict=ok and step<1
        O->>O: continue WATCH

      else verdict=stalled or crashed
        O->>D: Task("diagnoser")
        D-->>O: JSON {classification, patches, tier}
        alt tier <= T2 and not repeat
          O->>O: PATCH (apply recommended)
          O->>T: DEPLOY again
        else tier == T3
          O-->>U: ESCALATE VM corruption (no auto-recreate)
        else tier == T4 or repeat
          O-->>U: ESCALATE diagnosis
        end

      else verdict=success
        O-->>U: PAUSE for Phase 5 confirmation
      end
    end
```

## 6. Check-in cadence

See `diagrams/04-checkin-cadence.mmd`.

```mermaid
flowchart LR
    T0[T=0 deploy]
    T0 --> T5[T+5 poll]
    T5 --> T10[T+10 poll]
    T10 --> CK15[T+15 AskUser]
    CK15 --> T25[T+25 poll]
    T25 --> CK30[T+30 AskUser]
    CK30 --> CK45[T+45 AskUser]
    CK45 --> CK60[T+60 AskUser]
    CK60 --> CK90[T+90 AskUser]
    CK90 --> PAUSE[indefinite pause]
```

Each AskUser shows you a structured snapshot (see
`playbook/checkin-protocol.md`) with 4 options:
**Continue / Abort+Diag / Adjust / Pause**.

## 7. Tier 3 (VM corruption) -- locked

See `diagrams/05-tier3-escalation.mmd`.

```mermaid
flowchart TD
    T3[Tier 3 detected:<br/>SSH refused,<br/>kernel panic,<br/>Bus error,<br/>4 PIDs unreachable 3+ polls]
    T3 --> CAPTURE[Capture diagnostic snapshot]
    CAPTURE --> ASK[AskUser: recreate QR?]
    ASK -->|never auto| HUMAN[ALWAYS escalate first]
    HUMAN --> WAIT[Wait for explicit user decision]
```

Topology note: the diagram retains the "4 PIDs unreachable" wording
because it is the diagnostic regex value for the legacy v4-32
multi-host case. On the current single-host v6e-8 EU canary there is
ONE Python PID, so the equivalent is "1 PID unreachable 3+ polls"
(or, equivalently, the single tmux session is gone). On future
v6e-64 multi-host pods it would be 8 PIDs.

## 8. Diagnosis table (excerpt; full table in `playbook/diagnosis-table.md`)

| # | Symptom | Patch | Tier |
|---|---|---|---|
| 1 | `Failed to deserialize executable: UNIMPLEMENTED` | Remove `XLA_PERSISTENT_CACHE_PATH` | T2 |
| 2 | `ValueError.*Layer N has mismatched keys` | `use_scan_layers: false` | T2 |
| 3 | `AssertionError.*FakeTensor.*aten.index_select` | Drop `is_layer_pure=True` | T2 |
| 5 | OOM / exit 137 | Halve batch_size or depth_chunk_size | T2 |
| 6 | TPU duty=0 + HBM>50 + no step in 30 min | Kill + dump met.metrics_report() | T2 |
| 7-9 | SSH refused / kernel panic / N PIDs dead (N=1 on v6e-8, 4 on legacy v4-32, 8 on v6e-64) | ESCALATE (no auto-recreate) | **T3** |
| 10 | Same error twice | ESCALATE | T4 |

## 9. File patches (Phase 1.5)

| File | Change |
|---|---|
| `simultaneous-translation/scripts/tpu/startup_script.sh` | Remove `XLA_PERSISTENT_CACHE_PATH`, drop `while true` loop, add `python -u` |
| `simultaneous-translation/scripts/tpu/_remote_redeploy.sh` | Same XLA cache + loop removals |
| `simultaneous-translation/src/model/scan_utils.py` | Drop `is_layer_pure=True` from scan_layers call site |

YAML configs already done in prior commit (`use_scan_layers: false` everywhere).

## 10. Stop conditions

1. `step >= 1` AND `loss` decreasing -> SUCCESS
2. Same classification twice consecutively -> ESCALATE
3. Tier 3 detected -> ESCALATE (always)
4. User picks Abort/Pause -> ESCALATE / pause
5. Past T+90 wall -> indefinite pause until user resumes

## 11. Verification (Definition of Done)

- [ ] `.factory/orchestration/` folder created with all artifacts
- [ ] 4 implementation files (2 skills, 2 droids) in natural locations
- [ ] poller + checkin script in `_artifacts/`
- [ ] 3 file patches applied
- [ ] Hot redeploy succeeded; new PID(s) confirmed on every worker
  (1 worker on v6e-8 single-host, 4 on legacy v4-32, 8 on v6e-64)
- [ ] Watchdog reports `verdict=success` for 2+ consecutive polls
- [ ] wandb shows `step >= 1` AND >= 3 `loss=` lines AND loss decreasing
- [ ] At least one checkpoint write attempted
- [ ] PROGRESS.md entry written

## 12. Research basis

| Pattern | Source |
|---|---|
| Detect -> Diagnose -> Heal -> Verify (4-phase loop) | Claude Lab "Self-Healing AI Agents" 2026, AgentPatterns.ai |
| Failure-Driven Iteration (paste error -> fix root cause -> re-run) | Claude Code Best Practices, GitHub Copilot CLI |
| Diagnosis-mapping table | Zylos "AI Agent Self-Healing" |
| Tiered recovery (least disruptive first) | Erlang OTP supervisors + K8s liveness/readiness probes |
| Circuit breaker on repeat failures | LangChain GTM Self-Heal April 2026 |
| Bounded loop with check-ins (HUMAN task) | Conductor production agent architecture |
| Stuck detection: action-repetition + progress-plateau | K8s probes |
| Custom droids for context isolation | Factory.ai docs/cli/configuration/custom-droids |
| Skills for encoded playbooks | Factory.ai docs/cli/configuration/skills |
| Hot redeploy via rsync (no QR recreate) | GCP Spot TPU docs |
| XLA cache broken | pytorch/xla #8930, #9094 (both OPEN) |
| scan_layers structure check | torch_xla scan_layers.py source `_ensure_same_structure` |
| FakeTensor index_select | PyTorch #105485 |
