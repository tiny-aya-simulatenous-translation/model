#!/usr/bin/env bash
# Launch the v0.3 regularization sweep (sweeps/sweep_stage2_v3.yaml) as N
# INDEPENDENT per-host trials -- one `wandb agent` per TPU worker, each running a
# single-host v6e-8 SPMD program on its 8 LOCAL chips.
#
# WHY THIS EXISTS
# ---------------
# The sweep is embarrassingly parallel (each trial is an independent short run),
# so the right accelerator is *width*, not a bigger SPMD mesh. On a v6e-64
# (8 hosts x 8 chips) this starts 8 agents at once -> ~8x sweep throughput. Each
# agent pulls its own trials from the shared W&B sweep and runs the proxy config
# (configs/tpu/stage2_tpu_v6e_v3_proxy.yaml -- single-host v6e-8 topology).
#
# ONE-TIME: create the sweep on your workstation and grab the id:
#   wandb sweep sweeps/sweep_stage2_v3.yaml
#   # -> "Run sweep agent with: wandb agent ENTITY/PROJECT/SWEEP_ID"
#
# THEN:
#   SWEEP_ID=ENTITY/PROJECT/SWEEP_ID \
#   NODE_ID=tinyaya-stage2-spot-v6e64-eu ZONE=europe-west4-a \
#       bash scripts/tpu/launch_sweep.sh
#
# ⚠️ MULTI-HOST ISOLATION CAVEAT (validate once on the real v6e-64):
#   A v6e-64 is ONE ICI-connected slice. This script assumes each host's agent
#   initialises as a SINGLE-host program over its 8 LOCAL chips (independent
#   trials). Confirm on the first trial that a worker reports `global=8 local=8`
#   (NOT 64) in its log -- grep `[tpu_backend][post-wrap] global=`. If instead it
#   tries to form a 64-chip mesh (`global=64`), the slice is NOT subdividing;
#   fall back to the ROBUST path below.
#
# ROBUST FALLBACK (no isolation assumption): provision N separate v6e-8 SPOT
# slices and run this script once per slice with WORKER=0:
#   for i in 1 2 3 4; do
#     TRC_PROFILE=v6e-8-eu QR_NAME=tinyaya-sweep-$i NODE_ID=tinyaya-sweep-$i \
#       bash scripts/tpu/launch_spot.sh   # (no config-file auto-train; see note)
#     SWEEP_ID=... NODE_ID=tinyaya-sweep-$i WORKER=0 bash scripts/tpu/launch_sweep.sh
#   done
# Each v6e-8 is unambiguously single-host -> 4-8 independent trials, no caveat.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=./_lib.sh
source "$SCRIPT_DIR/_lib.sh"
load_env_file "$REPO_ROOT/.env"

PROJECT_ID="${PROJECT_ID:-ml-pipelines-315702}"
ZONE="${ZONE:-europe-west4-a}"
NODE_ID="${NODE_ID:-tinyaya-stage2-spot-v6e64-eu}"
WORKER="${WORKER:-all}"          # "all" = one agent per host; "0" = single host
REPO_DIR="${REPO_DIR:-/opt/tinyaya}"
SECRET_HF="${SECRET_HF:-hf-token}"
SECRET_WANDB="${SECRET_WANDB:-wandb-api-key}"
TPU_STRATEGY="${TPU_STRATEGY:-fsdpv2_lora}"

if [ -z "${SWEEP_ID:-}" ]; then
    echo "ERROR: set SWEEP_ID=ENTITY/PROJECT/SWEEP_ID (from 'wandb sweep ...')" >&2
    exit 2
fi

echo "==> launch_sweep: NODE_ID=$NODE_ID worker=$WORKER sweep=$SWEEP_ID"

# Remote command: bring up the SAME root TPU env the redeploy uses, then run one
# wandb agent per host inside a tmux session. The agent execs the sweep command
# (python train_hierarchical.py --config ..._v3_proxy.yaml --sweep ...).
read -r -d '' REMOTE <<REMOTE_EOF || true
set -uo pipefail
sudo bash -c '
  set -uo pipefail
  cd "$REPO_DIR"
  export PATH="/root/.local/bin:\$PATH"
  LIBPYTHON_DIR="\$(dirname "\$(find /root/.local/share/uv/python -name libpython3.12.so.1.0 -type f 2>/dev/null | head -1)")"
  export DEVICE_BACKEND=tpu PJRT_DEVICE=TPU
  export XLA_DISABLE_FUNCTIONALIZATION=0 XLA_NO_SPECIAL_SCALARS=1
  export LIBTPU_INIT_ARGS="--megascale_grpc_enable_xor_tracer=false --xla_tpu_enable_flash_attention=false"
  export LD_LIBRARY_PATH="\$LIBPYTHON_DIR:\${LD_LIBRARY_PATH:-}"
  export TPU_STRATEGY="$TPU_STRATEGY" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
  export HF_TOKEN="\$(gcloud secrets versions access latest --secret=$SECRET_HF 2>/dev/null)"
  export WANDB_API_KEY="\$(gcloud secrets versions access latest --secret=$SECRET_WANDB 2>/dev/null)"
  tmux kill-session -t sweep 2>/dev/null || true
  tmux new-session -d -s sweep "uv run wandb agent $SWEEP_ID > /tmp/sweep_agent.log 2>&1"
  echo "[sweep] agent started on \$(hostname); tail /tmp/sweep_agent.log"
'
REMOTE_EOF

gcloud compute tpus tpu-vm ssh "$NODE_ID" \
    --project="$PROJECT_ID" --zone="$ZONE" \
    --worker="$WORKER" \
    --command="$REMOTE"

echo
echo "==> agents launched. Watch a worker:"
echo "  gcloud compute tpus tpu-vm ssh $NODE_ID --project=$PROJECT_ID --zone=$ZONE --worker=0 --command='tail -f /tmp/sweep_agent.log'"
echo "==> VALIDATE isolation on worker 0 (expect global=8, NOT 64):"
echo "  ... --worker=0 --command='grep -m1 \"post-wrap\\] global=\" /tmp/sweep_agent.log'"
