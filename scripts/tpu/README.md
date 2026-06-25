# TPU launch scripts

Operator-side scripts for training Stage 2 on Google Cloud TPU TRC
hardware via the Queued Resource API.

## 2026-05-10 update

The validated production topology is **single-host TPU v6e-8 spot in
`europe-west4-a`** (QR `tinyaya-stage2-spot-v6e8-eu-qr`, node
`tinyaya-stage2-spot-v6e8-eu`, profile shorthand `v6e-8-eu`,
production config `configs/tpu/stage2_tpu_v6e_v2.yaml`). On v6e-8 there
is one host with 8 chips and ONE Python process driving them via
SPMD. Iter 24h completed 5000/5000 steps on this path:
W&B run [`7rrjupc7`](https://wandb.ai/cataluna84/tinyaya-stage2-tpu/runs/7rrjupc7),
final loss 5.3558, training wall 615.9 min, exit status 0, final
checkpoint
`gs://tinyaya-stage2-tpu/checkpoints/stage2-tpu-v6e-spot/step_005000_final/`
(8 objects, 2.37 GiB). The optimized `opt-prod5k` run
[`kzsijxv5`](https://wandb.ai/cataluna84/tinyaya-stage2-tpu/runs/kzsijxv5)
then completed 5000/5000 steps in 562 min with p50 6.14 s/step,
p99 6.76 s/step, final loss 5.105, and checkpoint
`gs://tinyaya-stage2-tpu/checkpoints/stage2-tpu-v6e-spot-opt-prod5k/step_005000_final/`.
Phase 4 uses the same launch path for candidate configs. The first
candidate, `stage2_tpu_v6e_spot_opt_depth32.yaml`, completed 300/300
steps as W&B
[`i15igq8d`](https://wandb.ai/cataluna84/tinyaya-stage2-tpu/runs/i15igq8d)
with p50 5.296 s/step and p99 5.725 s/step. v4-32 spot in `us-central2-b` (4 hosts) is
legacy (see "Legacy v4 examples" below). The runbook below starts
with the v6e-8 EU production example.

For the design rationale and decision history, see
[`../../docs/tpu-launch-plan.md`](../../docs/tpu-launch-plan.md).

## Files

| File | Where it runs | Purpose |
|---|---|---|
| `setup_gcp.sh` | local workstation, once | enables APIs, creates GCS bucket, seeds Secret Manager from `.env`, grants IAM |
| `launch_qr.sh` | local workstation | submits the Queued Resource for legacy v4 launches (full or canary, via `CONFIG_FILE`) |
| `launch_spot.sh` | local workstation | submits the current v6e-8 EU spot queued resource via `TRC_PROFILE=v6e-8-eu` |
| `launch_canary.sh` | local workstation | wrapper that calls `launch_qr.sh` with canary defaults (200 steps, separate QR + ckpt prefix) |
| `startup_script.sh` | every TPU host at boot | installs uv + Python 3.12.13, fetches a GCS repo tarball when `REPO_TARBALL_GS_URI` is set, syncs `uv.lock`, fetches secrets, downloads data, reads `config-file` from VM metadata, starts training in a tmux restart loop |
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
| `CONFIG_FILE` | `configs/tpu/stage2_tpu_v6e_v2.yaml` (full) / `configs/tpu/stage2_tpu_v6e_v2.yaml` (canary) | `launch_qr.sh`, `launch_canary.sh` |
| `TRC_PROFILE` | unset | `launch_spot.sh`; use `v6e-8-eu` for current production/optimization runs |
| `REPO_TARBALL_GS_URI` | unset | `launch_spot.sh`, `startup_script.sh`; when set, startup fetches code from GCS instead of cloning GitHub |
| `PROBE_FIRST` | `1` | `launch_spot.sh`; set `0` for known-good production/optimization configs |
| `SECRET_HF` | `hf-token` (script default) | `setup_gcp.sh`, `startup_script.sh` — only override if you renamed the secret in GCP |
| `SECRET_WANDB` | `wandb-api-key` (script default) | `setup_gcp.sh`, `startup_script.sh` — only override if you renamed the secret in GCP |
| `HF_TOKEN` | (from `.env`, required by `setup_gcp.sh`) | seeded into Secret Manager |
| `WANDB_API_KEY` | (from `.env`, optional) | seeded into Secret Manager |

`.env` is `.gitignored` and never leaves the workstation. Secret values
(`HF_TOKEN`, `WANDB_API_KEY`) are seeded into GCP Secret Manager once by
`setup_gcp.sh`; from then on, VMs fetch them from Secret Manager at boot
and the `.env` is no longer needed by the cloud side.

## End-to-end runbook (v6e-8 EU spot production)

From the repo root, on your local workstation:

```bash
# 0. one-time uv install (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 1. generate / refresh the lockfile
cd .
uv lock
uv lock --check
cd ..

# 2. one-time gcloud auth (interactive browser)
gcloud auth login
gcloud auth application-default login
# (PROJECT_ID is read from .env, no need to `gcloud config set project`)

# 3. seed GCP (bucket, secrets, IAM)
bash scripts/tpu/setup_gcp.sh

# 4. package the current branch for private-repo-safe TPU startup
tar --exclude-vcs --exclude='./_artifacts' --exclude='./wandb' \
  -czf /tmp/tinyaya-phase4.tar.gz .
gsutil cp /tmp/tinyaya-phase4.tar.gz \
  gs://tinyaya-stage2-tpu/code/phase4-depth32-$(date -u +%Y%m%dT%H%M%SZ).tar.gz

# 5. launch single-host v6e-8 spot in europe-west4-a
#    (canonical command for the optimized production topology)
TRC_PROFILE=v6e-8-eu \
QR_NAME=tinyaya-stage2-spot-v6e8-eu-qr \
NODE_ID=tinyaya-stage2-spot-v6e8-eu \
CONFIG_FILE=configs/tpu/stage2_tpu_v6e_v2.yaml \
REPO_TARBALL_GS_URI=gs://tinyaya-stage2-tpu/code/<repo-tarball>.tar.gz \
TPU_STRATEGY=fsdpv2_lora \
PROBE_FIRST=0 \
  bash scripts/tpu/launch_spot.sh

# 6. observe run
QR_NAME=tinyaya-stage2-spot-v6e8-eu-qr \
NODE_ID=tinyaya-stage2-spot-v6e8-eu \
ZONE=europe-west4-a \
  bash scripts/tpu/ops.sh status
QR_NAME=tinyaya-stage2-spot-v6e8-eu-qr \
NODE_ID=tinyaya-stage2-spot-v6e8-eu \
ZONE=europe-west4-a \
  bash scripts/tpu/ops.sh tail-logs

# 7. teardown when validation/checkpoint review is complete
QR_NAME=tinyaya-stage2-spot-v6e8-eu-qr \
NODE_ID=tinyaya-stage2-spot-v6e8-eu \
ZONE=europe-west4-a \
  bash scripts/tpu/ops.sh delete

# 8-12: see "Legacy v4 examples" below for the historical
# v4-32 / v4-64 full-run runbook (multi-host).
```

## Legacy v4 examples

The historical runbook for the v4-32 / v4-64 path in `us-central2-b`
is preserved here for reference. v4-32 spot is currently SUSPENDED
(no spot capacity); v4-64 on-demand requires TRC quota that is not
currently available.

```bash
# CANARY (legacy v4-32 / v4-64) — 200 steps, separate QR + ckpt prefix
bash scripts/tpu/launch_canary.sh

# observe canary (uses CANARY_QR_NAME / CANARY_NODE_ID from .env)
QR_NAME=$CANARY_QR_NAME NODE_ID=$CANARY_NODE_ID \
  bash scripts/tpu/ops.sh status
QR_NAME=$CANARY_QR_NAME NODE_ID=$CANARY_NODE_ID \
  bash scripts/tpu/ops.sh tail-logs

# canary teardown when 200 steps complete
QR_NAME=$CANARY_QR_NAME NODE_ID=$CANARY_NODE_ID \
  bash scripts/tpu/ops.sh delete

# FULL run (legacy) — 5,000 steps
bash scripts/tpu/launch_qr.sh

# observe full run
bash scripts/tpu/ops.sh status
bash scripts/tpu/ops.sh tail-logs
bash scripts/tpu/ops.sh attach     # tmux on worker 0
bash scripts/tpu/ops.sh ssh 0      # plain ssh

# when training has converged
bash scripts/tpu/ops.sh pull-best ./_artifacts

# teardown (only when explicitly ready — bills GCE host costs until then)
bash scripts/tpu/ops.sh delete

# commit & push (after the experiment is done)
git add pyproject.toml \
        uv.lock \
        configs/tpu/stage2_tpu_v6e_v2.yaml \
        configs/tpu/stage2_tpu_v6e_v2.yaml \
        scripts/tpu/ \
        docs/tpu-launch-plan.md \
        .gitignore
git commit -m "feat: TPU v4-64 launch infra (uv + py3.12.13 + torch_xla 2.9 + canary)"
git push origin feat/tpu-support
```

## Production success criteria

For the v6e-8 production path, confirm all 8 are green before deleting
the QR or promoting the checkpoint downstream:

1. QR transitions `WAITING_FOR_RESOURCES → ACTIVE` within ~10 min
2. `tpu-info` on worker 0 reports 8 chips
3. W&B run URL is announced and matches the selected config
   (`v6e-spot-stage2-opt-prod5k`, `v6e-spot-stage2-opt4-*`, etc.)
4. `/tmp/train.log` reaches step 300 without NaN/OOM/FATAL signals
5. `/tmp/train.log` reaches step 5000/5000
6. `training exited with status 0`
7. Canonical final save uploads to
   the config's `gs://tinyaya-stage2-tpu/checkpoints/.../step_005000_final/`
8. GCS lists the final checkpoint files (`metadata.json`,
   `text_embed.pt`, `depth_decoder.pt`, `projection.pt`,
   `audio_heads.pt`, and `peft_adapter/*`)

Iter 24h and `opt-prod5k` satisfied all 8 production criteria.
`opt-4-depth32` satisfied the 300-step Phase 4 variant with exit
status 0 and no NaN/OOM/FATAL signatures.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| QR stuck in `WAITING_FOR_RESOURCES` | TRC quota exhausted in zone | check `gcloud compute tpus accelerator-types list --zone=us-central2-b` |
| `auto_wrap_policy` does not match anything | class name mismatch | edit `src/backends/tpu_backend.py` to use `Cohere2DecoderLayer` |
| HF download stalls | rate-limit (4 hosts in parallel) | pre-stage to `gs://$BUCKET/data/` and switch startup script to `gcloud storage rsync` |
| `secretmanager.googleapis.com not enabled` | `setup_gcp.sh` didn't run | run it; it's idempotent |
| `permission denied` on GCS write | TPU SA missing role | re-run `setup_gcp.sh` (it re-applies IAM) |
| `uv: command not found` after boot | astral.sh unreachable | the script falls back to `pip install uv`; check `/tmp/startup.log` |
| Private GitHub clone fails | Fresh TPU VM has no GitHub credentials | Upload a repo tarball to GCS and pass `REPO_TARBALL_GS_URI=gs://...` |
| `ImportError: libpython3.12.so.1.0` | torch_xla cannot find uv-managed CPython's shared library | Use the current `_remote_redeploy.sh` / startup libpython fallback |
| W&B charts show 0..499 for a 5000-step run | W&B internal `_step` counted log events, not training steps | Current training code logs `global_step` and defines it as the x-axis |

## What this does NOT do

- Use Multislice (single slice only)
- Use multi-host TPU for the current v6e-8 path (single host only)
- Async checkpointing (synchronous via `xm.save`)
- Auto-eval after training (run `eval_stage2.py` separately on a GPU)
