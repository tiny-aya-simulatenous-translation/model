#!/usr/bin/env bash
# Provision a fleet of N independent v6e-8 SPOT slices, each running a `wandb
# agent` on the pre-staged sweep subset (sweep-mode in startup_script.sh). The
# existing production v6e-8 is NOT touched -- run launch_sweep.sh WORKER=0 on it
# separately if you want it to join the sweep too (8 total = the 64-chip grant).
#
# PREREQUISITES:
#   1. wandb sweep sweeps/sweep_stage2_v3.yaml            -> SWEEP_ID
#   2. bash scripts/tpu/stage_sweep_subset.sh             -> SWEEP_DATA_GS_URI
#
# Usage:
#   SWEEP_ID=ENTITY/PROJECT/SWEEP_ID \
#   SWEEP_DATA_GS_URI=gs://tinyaya-stage2-tpu/data/sweep-subset-200000.tar.gz \
#   N_SLICES=7 \
#       bash scripts/tpu/launch_sweep_fleet.sh
#
# Each slice: tinyaya-sweep-<i> (QR + node). Tear down with:
#   for i in $(seq 1 N); do QR_NAME=tinyaya-sweep-$i NODE_ID=tinyaya-sweep-$i \
#       ZONE=europe-west4-a bash scripts/tpu/ops.sh delete; done

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=./_lib.sh
source "$SCRIPT_DIR/_lib.sh"
load_env_file "$REPO_ROOT/.env"

PROJECT_ID="${PROJECT_ID:-ml-pipelines-315702}"
BUCKET="${BUCKET:-tinyaya-stage2-tpu}"
N_SLICES="${N_SLICES:-7}"
NAME_PREFIX="${NAME_PREFIX:-tinyaya-sweep}"
PROXY_CFG="${PROXY_CFG:-configs/tpu/stage2_tpu_v6e_v3_proxy.yaml}"

: "${SWEEP_ID:?set SWEEP_ID=ENTITY/PROJECT/SWEEP_ID (from 'wandb sweep ...')}"
: "${SWEEP_DATA_GS_URI:?set SWEEP_DATA_GS_URI=gs://... (from stage_sweep_subset.sh)}"

# Package the current branch so each slice's startup pulls code from GCS (private
# repo, no GitHub creds on the VM). Excludes the 2.1G stale subdir + caches.
TARBALL="/tmp/tinyaya-sweep-fleet.tar.gz"
GCS_TARBALL="gs://$BUCKET/code/sweep-fleet-$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
echo "==> packaging repo -> $GCS_TARBALL"
cd "$REPO_ROOT"
tar --exclude-vcs --exclude='./simultaneous-translation' --exclude='./_artifacts' \
    --exclude='./wandb' --exclude='./.venv' --exclude='*.tar.gz' \
    --exclude='__pycache__' --exclude='./eval_results' \
    -czf "$TARBALL" .
gcloud storage cp "$TARBALL" "$GCS_TARBALL" --project="$PROJECT_ID"
rm -f "$TARBALL"

echo "==> launching $N_SLICES sweep slices (profile v6e-8-eu, europe-west4-a)"
for i in $(seq 1 "$N_SLICES"); do
    name="${NAME_PREFIX}-${i}"
    echo "==> [$i/$N_SLICES] $name"
    TRC_PROFILE=v6e-8-eu \
    QR_NAME="${name}-qr" NODE_ID="$name" \
    CONFIG_FILE="$PROXY_CFG" \
    TPU_STRATEGY=fsdpv2_lora \
    PROBE_FIRST=0 \
    SWEEP_ID="$SWEEP_ID" \
    SWEEP_DATA_GS_URI="$SWEEP_DATA_GS_URI" \
    REPO_TARBALL_GS_URI="$GCS_TARBALL" \
        bash "$SCRIPT_DIR/launch_spot.sh"
done

echo
echo "==> fleet submitted. Watch provisioning:"
echo "  for i in \$(seq 1 $N_SLICES); do gcloud compute tpus queued-resources describe ${NAME_PREFIX}-\$i-qr --zone=europe-west4-a --project=$PROJECT_ID --format='value(state.state)'; done"
echo "==> once ACTIVE, each host auto-runs its agent; tail one:"
echo "  gcloud compute tpus tpu-vm ssh ${NAME_PREFIX}-1 --zone=europe-west4-a --project=$PROJECT_ID --worker=0 --command='tail -f /tmp/train.log'"
