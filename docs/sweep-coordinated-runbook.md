# Coordinated multi-host sweep runbook (v6e-16)

The v0.3-scale capacity sweep runs on the single **v6e-16** (4 hosts → one 16-chip
SPMD mesh). A plain per-host `wandb agent` **desyncs** (each host pulls a different
trial), so trials are driven by a **coordinator** that broadcasts identical args to
all hosts via a shared GCS control file. Everything runs **detached in tmux** on the
VM — the workstation can be powered off.

## Pieces
- `scripts/tpu/sweep_coordinator.py` — host-0 driver. `--stage grid` enumerates a
  sweep YAML's `parameters`; `--stage bayes` pulls suggestions from a W&B sweep
  controller. Writes `current_trial.json`, waits for all hosts' done-markers, reads
  the trial's `val/composite` from W&B, records a completion marker (resumable
  across a coordinator reboot), advances. `--dry-run` prints the plan only.
- `scripts/tpu/sweep_host_loop.sh` — runs on ALL hosts; polls the control file and
  runs `train_hierarchical.py --config <proxy> --sweep <args>` (identical args →
  the 16-chip mesh forms). Writes `done/trial_<N>_host_<hostname>`.
- `scripts/tpu/launch_sweep_coordinated.sh` — packages code → GCS, refreshes all
  hosts, starts host loops (`--worker=all`, tmux `sweephost`) + coordinator
  (`--worker=0`, tmux `sweepcoord`).
- Proxy config `configs/tpu/stage2_tpu_v6e16_scale_proxy.yaml` (global batch 256,
  1500-step horizon, de-regularized, grad-clip on, checkpoints namespaced by run id).

## Run
Baseline throughput ≈ **5.14 s/step** → ~2.1 h per 1500-step arm; recompiles only on
rank/structure change (lr changes reuse the graph).

**Stage 1 — structural grid (3 trials, no `wandb sweep` needed):**
```bash
NODE_ID=tinyaya-v6e16-batch-ew4 ZONE=europe-west4-a NAME=scale-grid STAGE=grid \
  bash scripts/tpu/launch_sweep_coordinated.sh
```
Pick the best `target_modules` structure by `val/composite` (beat the baseline
`5.197` / `cb0_acc 19.4%`). Edit `sweeps/sweep_stage2_scale_bayes.yaml`'s
`target_modules` (+ `lora_exclude_top`) to that winner.

**Stage 2 — Bayesian lr × rank (~6–7 trials):**
```bash
wandb sweep sweeps/sweep_stage2_scale_bayes.yaml          # -> ENTITY/PROJECT/ID
NAME=scale-bayes STAGE=bayes SWEEP_ID=ENTITY/PROJECT/ID \
  bash scripts/tpu/launch_sweep_coordinated.sh
```
> ⚠️ Verify the `wandb.controller` path on the FIRST bayes trial (the controller API
> is version-sensitive). Stage 1 (grid) has no such dependency.

## Monitor / stop
```bash
gcloud compute tpus tpu-vm ssh $NODE_ID --zone $ZONE --worker=0 \
  --command 'tmux capture-pane -pt sweepcoord | tail -40; echo ---; tail -20 /tmp/train.log'
# stop:
gcloud compute tpus tpu-vm ssh $NODE_ID --zone $ZONE --worker=all \
  --command 'tmux kill-session -t sweephost; tmux kill-session -t sweepcoord'
```
Checkpoints land per trial at `gs://tinyaya-stage2-tpu/checkpoints/stage2-scale-sweep/<run_id>/step_*` (all kept).

## Promote the winner
```bash
python scripts/promote_sweep_winner.py --sweep <entity/project/id> \
  --config configs/tpu/stage2_tpu_v6e16_b256.yaml --metric val/composite
```
Then run a medium full-corpus confirmation before the full production run.
