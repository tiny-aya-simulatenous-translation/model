# TPU launch scripts

Operator-side scripts for training Stage 2 on Google Cloud TPU TRC
hardware via the Queued Resource API.

## 2026-05-08 update

The active canary topology is **single-host TPU v6e-8 spot in
`europe-west4-a`** (QR `tinyaya-stage2-spot-v6e8-eu-qr`, node
`tinyaya-stage2-spot-v6e8-eu`, profile shorthand `v6e-8-eu`,
config `configs/stage2_tpu_canary_v6e_spot.yaml`). On v6e-8 there
is one host with 8 chips and ONE Python process driving them via
SPMD. v4-32 spot in `us-central2-b` (4 hosts) is currently
SUSPENDED and treated as a Legacy path (see "Legacy v4 examples"
below). The runbook below has been updated to start with the v6e-8
EU example.

For the design rationale and decision history, see
[`../../docs/tpu-launch-plan.md`](../../docs/tpu-launch-plan.md).

## Files

| File | Where it runs | Purpose |
|---|---|---|
| `setup_gcp.sh` | local workstation, once | enables APIs, creates GCS bucket, seeds Secret Manager from `.env`, grants IAM |
| `launch_qr.sh` | local workstation | submits the Queued Resource for `v4-64` on-demand (full or canary, via `CONFIG_FILE`) |
| `launch_canary.sh` | local workstation | wrapper that calls `launch_qr.sh` with canary defaults (200 steps, separate QR + ckpt prefix) |
| `startup_script.sh` | every TPU host at boot | installs uv + Python 3.12.13, syncs `uv.lock`, fetches secrets, downloads data, reads `config-file` from VM metadata, starts training in a tmux restart loop |
| `ops.sh` | local workstation | `status`, `tail-logs`, `attach`, `ssh`, `pull-best`, `delete` |

## Configuration

All four local scripts (`setup_gcp.sh`, `launch_qr.sh`, `launch_canary.sh`,
`ops.sh`) auto-source `<repo-root>/.env` if present.

The repo's `.env` already defines these — override per-invocation by
prefixing the command with `VAR=value bash ...`.

| Var | Default | Used by |
|---|---|---|
| `PROJECT_ID` | `ml-pipelines-315702` | all |
| `REGION` | `us-central2` | `setup_gcp.sh` (bucket location) |
| `ZONE` | `us-central2-b` | `launch_qr.sh`, `ops.sh` |
| `ACCEL_TYPE` | `v4-64` | `launch_qr.sh` |
| `RUNTIME` | `tpu-ubuntu2204-base` | `launch_qr.sh` |
| `QR_NAME` | `tinyaya-stage2-qr` | full-run launch / ops |
| `NODE_ID` | `tinyaya-stage2` | full-run launch / ops |
| `CANARY_QR_NAME` | `tinyaya-stage2-canary-qr` | `launch_canary.sh` default |
| `CANARY_NODE_ID` | `tinyaya-stage2-canary` | `launch_canary.sh` default |
| `BUCKET` | `tinyaya-stage2-tpu` | `setup_gcp.sh`, `ops.sh pull-best` |
| `CKPT_PREFIX` | `checkpoints/stage2-tpu` | `ops.sh pull-best` |
| `CANARY_CKPT_PREFIX` | `checkpoints/canary` | (informational) |
| `CONFIG_FILE` | `configs/stage2_tpu.yaml` (full) / `configs/stage2_tpu_canary.yaml` (canary) | `launch_qr.sh`, `launch_canary.sh` |
| `SECRET_HF` | `hf-token` (script default) | `setup_gcp.sh`, `startup_script.sh` — only override if you renamed the secret in GCP |
| `SECRET_WANDB` | `wandb-api-key` (script default) | `setup_gcp.sh`, `startup_script.sh` — only override if you renamed the secret in GCP |
| `HF_TOKEN` | (from `.env`, required by `setup_gcp.sh`) | seeded into Secret Manager |
| `WANDB_API_KEY` | (from `.env`, optional) | seeded into Secret Manager |

`.env` is `.gitignored` and never leaves the workstation. Secret values
(`HF_TOKEN`, `WANDB_API_KEY`) are seeded into GCP Secret Manager once by
`setup_gcp.sh`; from then on, VMs fetch them from Secret Manager at boot
and the `.env` is no longer needed by the cloud side.

## End-to-end runbook (v6e-8 EU spot canary)

From the repo root, on your local workstation:

```bash
# 0. one-time uv install (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 1. generate / refresh the lockfile
cd simultaneous-translation
uv lock
uv lock --check
cd ..

# 2. one-time gcloud auth (interactive browser)
gcloud auth login
gcloud auth application-default login
# (PROJECT_ID is read from .env, no need to `gcloud config set project`)

# 3. seed GCP (bucket, secrets, IAM)
bash simultaneous-translation/scripts/tpu/setup_gcp.sh

# 4. CANARY: launch single-host v6e-8 spot in europe-west4-a
#    (canonical command for the active topology)
TRC_PROFILE=v6e-8-eu \
QR_NAME=tinyaya-stage2-spot-v6e8-eu-qr \
NODE_ID=tinyaya-stage2-spot-v6e8-eu \
CONFIG_FILE=configs/stage2_tpu_canary_v6e_spot.yaml \
TPU_STRATEGY=fsdpv2_lora \
PROBE_FIRST=1 \
  bash simultaneous-translation/scripts/tpu/launch_spot.sh

# 5. observe canary
QR_NAME=tinyaya-stage2-spot-v6e8-eu-qr \
NODE_ID=tinyaya-stage2-spot-v6e8-eu \
ZONE=europe-west4-a \
  bash simultaneous-translation/scripts/tpu/ops.sh status
QR_NAME=tinyaya-stage2-spot-v6e8-eu-qr \
NODE_ID=tinyaya-stage2-spot-v6e8-eu \
ZONE=europe-west4-a \
  bash simultaneous-translation/scripts/tpu/ops.sh tail-logs

# 6. canary teardown when validation is complete
QR_NAME=tinyaya-stage2-spot-v6e8-eu-qr \
NODE_ID=tinyaya-stage2-spot-v6e8-eu \
ZONE=europe-west4-a \
  bash simultaneous-translation/scripts/tpu/ops.sh delete

# 7-11: see "Legacy v4 examples" below for the historical
# v4-32 / v4-64 full-run runbook (multi-host).
```

## Legacy v4 examples

The historical runbook for the v4-32 / v4-64 path in `us-central2-b`
is preserved here for reference. v4-32 spot is currently SUSPENDED
(no spot capacity); v4-64 on-demand requires TRC quota that is not
currently available.

```bash
# CANARY (legacy v4-32 / v4-64) — 200 steps, separate QR + ckpt prefix
bash simultaneous-translation/scripts/tpu/launch_canary.sh

# observe canary (uses CANARY_QR_NAME / CANARY_NODE_ID from .env)
QR_NAME=$CANARY_QR_NAME NODE_ID=$CANARY_NODE_ID \
  bash simultaneous-translation/scripts/tpu/ops.sh status
QR_NAME=$CANARY_QR_NAME NODE_ID=$CANARY_NODE_ID \
  bash simultaneous-translation/scripts/tpu/ops.sh tail-logs

# canary teardown when 200 steps complete
QR_NAME=$CANARY_QR_NAME NODE_ID=$CANARY_NODE_ID \
  bash simultaneous-translation/scripts/tpu/ops.sh delete

# FULL run (legacy) — 5,000 steps
bash simultaneous-translation/scripts/tpu/launch_qr.sh

# observe full run
bash simultaneous-translation/scripts/tpu/ops.sh status
bash simultaneous-translation/scripts/tpu/ops.sh tail-logs
bash simultaneous-translation/scripts/tpu/ops.sh attach     # tmux on worker 0
bash simultaneous-translation/scripts/tpu/ops.sh ssh 0      # plain ssh

# when training has converged
bash simultaneous-translation/scripts/tpu/ops.sh pull-best ./_artifacts

# teardown (only when explicitly ready — bills GCE host costs until then)
bash simultaneous-translation/scripts/tpu/ops.sh delete

# commit & push (after the experiment is done)
git add simultaneous-translation/pyproject.toml \
        simultaneous-translation/uv.lock \
        simultaneous-translation/configs/stage2_tpu.yaml \
        simultaneous-translation/configs/stage2_tpu_canary.yaml \
        simultaneous-translation/scripts/tpu/ \
        simultaneous-translation/docs/tpu-launch-plan.md \
        .gitignore
git commit -m "feat: TPU v4-64 launch infra (uv + py3.12.13 + torch_xla 2.9 + canary)"
git push origin feat/tpu-support
```

## Canary success criteria

Before running step 7 (full launch), confirm all 7 are green from the canary:

1. QR transitions `WAITING_FOR_RESOURCES → ACTIVE` within ~10 min
2. `tpu-info` on at least one host reports 4 chips
3. `train.log` shows step counters incrementing on **all 4 hosts**
4. At least one checkpoint lands in `gs://$BUCKET/$CANARY_CKPT_PREFIX/`
5. At least one validation pass logs to W&B
6. `train.log` shows clean exit at step 200 (not OOM, not assertion error)
7. `auto_wrap_policy` matched ≥1 layer (no "wrapping 0 modules" warning)

If any criterion fails, diagnose, patch, and re-run the canary before
proceeding to the full run.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| QR stuck in `WAITING_FOR_RESOURCES` | TRC quota exhausted in zone | check `gcloud compute tpus accelerator-types list --zone=us-central2-b` |
| `auto_wrap_policy` does not match anything | class name mismatch | edit `src/backends/tpu_backend.py` to use `Cohere2DecoderLayer` |
| HF download stalls | rate-limit (4 hosts in parallel) | pre-stage to `gs://$BUCKET/data/` and switch startup script to `gcloud storage rsync` |
| `secretmanager.googleapis.com not enabled` | `setup_gcp.sh` didn't run | run it; it's idempotent |
| `permission denied` on GCS write | TPU SA missing role | re-run `setup_gcp.sh` (it re-applies IAM) |
| `uv: command not found` after boot | astral.sh unreachable | the script falls back to `pip install uv`; check `/tmp/startup.log` |

## What this does NOT do

- Use Multislice (single slice only)
- Handle spot/preemptible (we're on-demand)
- Async checkpointing (synchronous via `xm.save`)
- Auto-eval after training (run `eval_stage2.py` separately on a GPU)
