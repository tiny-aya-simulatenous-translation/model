---
name: tpu-redeploy
description: Hot-redeploy the TinyAya Stage 2 training code to the existing ACTIVE TPU QR (no QR recreate). Rsync code to the v6e-8 worker, restart tmux session, capture new PID. Used by the tpu-orchestrate playbook on every PATCH cycle. Idempotent.
---

# Hot redeploy to existing TPU QR

This skill encodes the canonical hot-redeploy procedure for the
TinyAya Stage 2 production run. It is called by `tpu-orchestrate`
after every PATCH and can also be invoked directly via
`/tpu-redeploy`.

It is subordinate to the unified TPU control plane at
`.claude/orchestration/CONTROL_PLANE.md`: this skill deploys an
already-approved patch or experiment, but it does not choose
optimization candidates.

## Topology

Active topology: **single-host TPU v6e-8 in `europe-west4-a`** (QR
`tinyaya-stage2-spot-v6e8-eu-qr`, node
`tinyaya-stage2-spot-v6e8-eu`). One worker, 8 chips, ONE Python
process driving them via SPMD. The SSH-into-workers loop collapses
to a single SSH because there is exactly ONE worker.

## Pre-flight checks

1. Verify QR is ACTIVE:
   ```bash
   gcloud compute tpus queued-resources describe tinyaya-stage2-spot-v6e8-eu-qr \
     --zone=europe-west4-a --format='value(state.state)'
   ```
   Expected: `ACTIVE`. If `PROVISIONING`/`SUSPENDED`/`FAILED`,
   stop and escalate -- the QR may need recreate (Tier 3, ALWAYS escalate).

2. Capture PRE-deploy PID baseline (so we can confirm a new PID after):
   ```bash
   gcloud compute tpus tpu-vm ssh tinyaya-stage2-spot-v6e8-eu \
     --zone=europe-west4-a --worker=0 \
     --command="pgrep -f 'python.*train_hierarchical' || echo none"
   ```
   Save the output to `_artifacts/orch_state.json` field `pre_deploy_pids`.
   On v6e-8 single-host this returns ONE PID (or `none`).

## Deploy

3. Run the hot-redeploy script:
   ```bash
   cd /home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation
   bash scripts/tpu/_remote_redeploy.sh 2>&1 | tee /tmp/redeploy_$(date +%s).log
   ```

   This script:
   - rsyncs modified files to `/opt/tinyaya/` on the worker
   - discovers the uv-managed CPython lib directory and prepends it to
     `LD_LIBRARY_PATH` so `torch_xla/_XLAC.so` can load
     `libpython3.12.so.1.0`
   - kills any running tmux session named `train`
   - SCPs the iter-N launcher + companion scripts to `/tmp/`
   - starts a new tmux session running the launcher
   - returns when the worker reports tmux session active

## Post-flight checks

4. Confirm NEW PID differs from baseline:
   ```bash
   gcloud compute tpus tpu-vm ssh tinyaya-stage2-spot-v6e8-eu \
     --zone=europe-west4-a --worker=0 \
     --command="pgrep -f 'python.*train_hierarchical'"
   ```
   - The worker should print exactly ONE new PID.
   - That PID should differ from the pre-deploy baseline.
   - If the worker reports no PID after 30 s, ESCALATE.

5. Capture deploy timestamp T=0 in state file:
   ```python
   import json, time
   state = json.load(open('_artifacts/orch_state.json'))
   state['deploy_t0_ts'] = time.time()
   state['iteration'] = state.get('iteration', 0) + 1
   state['checkins_done'] = []
   state['next_checkin_min'] = 15
   json.dump(state, open('_artifacts/orch_state.json', 'w'), indent=2)
   ```

## Verification

- [ ] QR state == `ACTIVE`
- [ ] Worker has a NEW Python PID
- [ ] tmux session `train` exists on worker 0
- [ ] `_artifacts/orch_state.json` updated with new T=0

## On failure

| Symptom | Action |
|---|---|
| `gcloud ssh ... Connection refused` | **ESCALATE** (Tier 3 -- VM corruption) |
| `ImportError: libpython3.12.so.1.0` | Check the `_remote_redeploy.sh` uv CPython lib fallback and `LD_LIBRARY_PATH` export |
| `_remote_redeploy.sh` exits non-zero | Capture stderr, show user, escalate |
| Worker has no PID after 30 s | ESCALATE; partial deploy is unsafe |
| QR state != ACTIVE | ESCALATE; need fresh QR (Tier 3) |

## Hand-off

After successful deploy, hand control back to `tpu-orchestrate` skill,
which will:
1. Launch `_artifacts/orch_poll.py` background poller
2. Wait for first watchdog read at T+5
3. Begin the check-in cadence
