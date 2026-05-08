---
name: tpu-redeploy
description: Hot-redeploy the TinyAya Stage 2 training code to the existing ACTIVE TPU QR (no QR recreate). Rsync code to every worker, restart tmux session, capture new PIDs. On v6e-8 single-host this is a single SSH; on v4-32 / v6e-64 multi-host it iterates over all hosts. Used by the tpu-orchestrate playbook on every PATCH cycle. Idempotent.
user-invocable: true
disable-model-invocation: false
---

# Hot redeploy to existing TPU QR

This skill encodes the canonical hot-redeploy procedure for the
TinyAya Stage 2 canary. It is called by `tpu-orchestrate` after every
PATCH and can also be invoked directly via `/tpu-redeploy`.

## Topology

The current canary runs on **single-host TPU v6e-8 in
`europe-west4-a`** (QR `tinyaya-stage2-spot-v6e8-eu-qr`, node
`tinyaya-stage2-spot-v6e8-eu`). On v6e-8 the SSH-into-workers loop
collapses to a single SSH because there is exactly ONE worker (one
host, 8 chips, ONE Python process driving them via SPMD).

The legacy v4-32 spot canary in `us-central2-b` (QR
`tinyaya-stage2-spot-v4-canary-qr`, node
`tinyaya-stage2-spot-v4-canary`) is multi-host: 4 workers, 4 Python
processes, 4 PIDs to track. Use `--worker=all` for that path. The
future v6e-64 multi-host pod is 8 workers, 8 Python processes, 8
PIDs.

The pre-flight / deploy / post-flight steps below are written for
the active v6e-8 EU topology; the v4-32 invocations are preserved
in commented form for the legacy path.

## Pre-flight checks

1. Verify QR is ACTIVE:
   ```bash
   gcloud compute tpus queued-resources describe tinyaya-stage2-spot-v6e8-eu-qr \
     --zone=europe-west4-a --format='value(state.state)'
   ```
   Expected output: `ACTIVE`. If `PROVISIONING`/`SUSPENDED`/`FAILED`,
   stop and escalate -- the QR may need recreate (Tier 3, ALWAYS escalate).

   Legacy v4-32 invocation (commented for reference):
   ```bash
   # gcloud compute tpus queued-resources describe tinyaya-stage2-spot-v4-canary-qr \
   #   --zone=us-central2-b --format='value(state.state)'
   ```

2. Capture PRE-deploy PID baseline (so we can confirm new PID(s) after):
   ```bash
   gcloud compute tpus tpu-vm ssh tinyaya-stage2-spot-v6e8-eu \
     --zone=europe-west4-a --worker=0 \
     --command="pgrep -f 'python.*train_hierarchical' || echo none"
   ```
   Save the output to `_artifacts/orch_state.json` field `pre_deploy_pids`.
   On v6e-8 single-host this returns ONE PID; on legacy v4-32 multi-host
   the same command with `--worker=all` returns 4 PIDs.

## Deploy

3. Run the hot-redeploy script:
   ```bash
   cd /home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation
   bash scripts/tpu/_remote_redeploy.sh 2>&1 | tee /tmp/redeploy_$(date +%s).log
   ```

   This script:
   - rsync's modified files (`scan_utils.py`, `startup_script.sh`,
     `stage2_tpu_canary*.yaml`, `train_hierarchical.py`,
     `checkpointing.py`) to `/opt/tinyaya/` on every worker
     (1 worker on v6e-8 single-host, 4 on legacy v4-32, 8 on v6e-64)
   - kills any running tmux session named `training`
   - starts a new tmux session running the training command
   - returns when every worker reports tmux session active

## Post-flight checks

4. Confirm NEW PID(s) differ from baseline:
   ```bash
   gcloud compute tpus tpu-vm ssh tinyaya-stage2-spot-v6e8-eu \
     --zone=europe-west4-a --worker=0 \
     --command="pgrep -f 'python.*train_hierarchical'"
   ```
   - The single v6e-8 worker should print exactly ONE new PID
   - That PID should differ from the pre-deploy baseline
   - If the worker reports no PID after 30s, ESCALATE

   Legacy v4-32 invocation (commented for reference):
   ```bash
   # gcloud compute tpus tpu-vm ssh tinyaya-stage2-spot-v4-canary \
   #   --zone=us-central2-b --worker=all \
   #   --command="pgrep -f 'python.*train_hierarchical'"
   ```
   - All 4 workers should print a PID; each must differ from baseline.

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
- [ ] Every worker has a NEW python PID (1 worker on v6e-8 single-
  host, 4 on legacy v4-32, 8 on future v6e-64)
- [ ] tmux session `training` exists on worker 0
- [ ] `_artifacts/orch_state.json` updated with new T=0

## On failure

| Symptom | Action |
|---|---|
| `gcloud ssh ... Connection refused` | **ESCALATE** (Tier 3 -- VM corruption) |
| `_remote_redeploy.sh` exits non-zero | Capture stderr, show user, escalate |
| Any worker has no PID after 30s (single worker on v6e-8, any of 4 on v4-32, any of 8 on v6e-64) | ESCALATE; partial deploy is unsafe |
| QR state != ACTIVE | ESCALATE; need fresh QR (Tier 3) |

## Hand-off

After successful deploy, hand control back to `tpu-orchestrate` skill,
which will:
1. Launch `_artifacts/orch_poll.py` background poller
2. Wait for first watchdog read at T+5
3. Begin the check-in cadence
