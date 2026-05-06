---
name: tpu-redeploy
description: Hot-redeploy the TinyAya Stage 2 training code to the existing ACTIVE TPU QR (no QR recreate). Rsync code to all 4 workers, restart tmux session, capture new PIDs. Used by the tpu-orchestrate playbook on every PATCH cycle. Idempotent.
user-invocable: true
disable-model-invocation: false
---

# Hot redeploy to existing TPU QR

This skill encodes the canonical hot-redeploy procedure for the
TinyAya Stage 2 canary. It is called by `tpu-orchestrate` after every
PATCH and can also be invoked directly via `/tpu-redeploy`.

## Pre-flight checks

1. Verify QR is ACTIVE:
   ```bash
   gcloud compute tpus queued-resources describe tinyaya-stage2-spot-v4-canary-qr \
     --zone=us-central2-b --format='value(state.state)'
   ```
   Expected output: `ACTIVE`. If `PROVISIONING`/`SUSPENDED`/`FAILED`,
   stop and escalate -- the QR may need recreate (Tier 3, ALWAYS escalate).

2. Capture PRE-deploy PID baseline (so we can confirm new PIDs after):
   ```bash
   gcloud compute tpus tpu-vm ssh tinyaya-stage2-spot-v4-canary \
     --zone=us-central2-b --worker=all \
     --command="pgrep -f 'python.*train_hierarchical' || echo none"
   ```
   Save the output to `_artifacts/orch_state.json` field `pre_deploy_pids`.

## Deploy

3. Run the hot-redeploy script:
   ```bash
   cd /home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation
   bash scripts/tpu/_remote_redeploy.sh 2>&1 | tee /tmp/redeploy_$(date +%s).log
   ```

   This script:
   - rsync's modified files (`scan_utils.py`, `startup_script.sh`,
     `stage2_tpu_canary*.yaml`) to `/opt/tinyaya/` on all 4 workers
   - kills any running tmux session named `training`
   - starts a new tmux session running the training command
   - returns when all 4 workers report tmux session active

## Post-flight checks

4. Confirm NEW PIDs differ from baseline:
   ```bash
   gcloud compute tpus tpu-vm ssh tinyaya-stage2-spot-v4-canary \
     --zone=us-central2-b --worker=all \
     --command="pgrep -f 'python.*train_hierarchical'"
   ```
   - All 4 workers should print a PID
   - Each PID should differ from the pre-deploy baseline
   - If any worker reports no PID after 30s, ESCALATE

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
- [ ] All 4 workers have a NEW python PID
- [ ] tmux session `training` exists on worker 0
- [ ] `_artifacts/orch_state.json` updated with new T=0

## On failure

| Symptom | Action |
|---|---|
| `gcloud ssh ... Connection refused` | **ESCALATE** (Tier 3 -- VM corruption) |
| `_remote_redeploy.sh` exits non-zero | Capture stderr, show user, escalate |
| One worker has no PID after 30s | ESCALATE; partial deploy is unsafe |
| QR state != ACTIVE | ESCALATE; need fresh QR (Tier 3) |

## Hand-off

After successful deploy, hand control back to `tpu-orchestrate` skill,
which will:
1. Launch `_artifacts/orch_poll.py` background poller
2. Wait for first watchdog read at T+5
3. Begin the check-in cadence
