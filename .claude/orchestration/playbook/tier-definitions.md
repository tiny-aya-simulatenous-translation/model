# Tier definitions

The 5-tier escalation ladder. Lower tiers are less disruptive and
preferred when the failure mode is well-understood. Tier 3 is locked
to **always escalate** -- never auto-recreate the QR.

| Tier | Action | Trigger | Auto? | Disruption |
|------|--------|---------|-------|------------|
| **T0** | Continue | All signals nominal (heartbeat ok, log progressing, no error) | yes | none |
| **T1** | Inject prompt to running process | Soft drift, no error yet (e.g., recompile-on-every-step warning) | yes | none |
| **T2** | Hot redeploy with patch | Known classified error from `diagnosis-table.md` rows 1-6,12-13 | yes | ~2 min: rsync code + tmux restart on existing VMs |
| **T3** | Recreate QR | VM-level corruption (rows 7-9): SSH refused, kernel panic, multi-host unreachable | **NO -- always escalate first** | ~10-30 min: capacity dance for new spot QR |
| **T4** | Pause for human | Repeat failure (row 10), user-abort (row 11), unknown classification, or budget warning | n/a | indefinite |

## Auto-escalation conditions

The orchestrator MUST escalate (regardless of tier classification) on:

1. Same `classification` twice consecutively (circuit breaker)
2. Tier 3 detected (LOCKED -- no auto-recreate)
3. Past T+90 min wall (default cadence cap)
4. User selects "Abort+Diag" or "Pause" at any check-in
5. Token budget warning emitted by main session
6. wandb shows `state=crashed` AND no actionable diagnosis from diagnoser

## Locked invariants

- **T3 never auto-acts.** A new QR costs another spot acquisition cycle (10-30 min) and may itself fail (capacity / IP quota). The cost asymmetry favours human review.
- **No auto-recreate even on repeated T3 detection.** A second T3 detection is an even stronger reason to stop and let the human investigate.
- **Check-ins are mandatory** at T+15/30/45/60/90 wall, even if all signals look nominal. The user is the safety valve for "compile is genuinely just slow" vs "stuck and burning quota".
