#!/bin/bash
# Hot-redeploy: tarball the local repo, push to GCS, fetch on each TPU host,
# and restart the train tmux session WITHOUT re-creating the Queued Resource.
#
# Strategy: SCP the remote helper to /tmp on each worker, then SSH to run it.
# This avoids quoting issues from inlining heredocs in the SSH command body.
#
# Usage:
#   TPU_STRATEGY=replicated bash scripts/tpu/hot_redeploy.sh
#   RUN_PROBE_ONLY=1 bash scripts/tpu/hot_redeploy.sh
#   TPU_STRATEGY=fsdpv2_lora NODE_ID=tinyaya-stage2-canary ZONE=europe-west4-b \
#       bash scripts/tpu/hot_redeploy.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"

# shellcheck source=./_lib.sh
source "$SCRIPT_DIR/_lib.sh"
load_env_file "$ENV_FILE"

PROJECT_ID="${PROJECT_ID:-ml-pipelines-315702}"
ZONE="${ZONE:-europe-west4-b}"
NODE_ID="${NODE_ID:-tinyaya-stage2-canary}"
BUCKET="${BUCKET:-tinyaya-stage2-tpu}"
CONFIG_FILE="${CONFIG_FILE:-configs/stage2_tpu_canary.yaml}"
TPU_STRATEGY="${TPU_STRATEGY:-auto}"
RUN_PROBE_ONLY="${RUN_PROBE_ONLY:-0}"

REPO_DIR="/opt/tinyaya"
TARBALL_NAME="tinyaya-repo-hot.tar.gz"
GCS_URI="gs://$BUCKET/code/$TARBALL_NAME"
LOCAL_TAR="/tmp/$TARBALL_NAME"

echo "==> [1/5] tarballing repo"
cd "$REPO_ROOT"
tar --exclude='.git' --exclude='.venv' --exclude='.env' \
    --exclude='__pycache__' --exclude='*.pyc' \
    --exclude='/mnt/*' --exclude='checkpoints' \
    --exclude='node_modules' --exclude='.pytest_cache' \
    -czf "$LOCAL_TAR" \
    simultaneous-translation README.md
ls -lh "$LOCAL_TAR"

echo "==> [2/5] uploading tarball to $GCS_URI"
gcloud storage cp "$LOCAL_TAR" "$GCS_URI" --project="$PROJECT_ID"
rm -f "$LOCAL_TAR"

echo "==> [3/5] copying remote helper script to all workers"
gcloud compute tpus tpu-vm scp \
    "$SCRIPT_DIR/_remote_redeploy.sh" \
    "$NODE_ID:/tmp/_remote_redeploy.sh" \
    --project="$PROJECT_ID" --zone="$ZONE" --worker=all

echo "==> [4/5] running remote helper (NODE_ID=$NODE_ID strategy=$TPU_STRATEGY probe_only=$RUN_PROBE_ONLY)"
# wandb shared-mode rendezvous: every hot redeploy gets a UNIQUE GCS path
# so worker hosts cannot read a stale run-id left over from a prior
# iteration. Without this suffix, primary writes the new run-id via
# `gsutil cp` AFTER its `wandb.init` returns, but worker hosts begin
# polling the file BEFORE that completes -- they read the previous
# iter's stale run-id and attach to a dead run, while primary attaches
# to the fresh run. Result: 1-of-4 hosts on the new run; 3-of-4 silently
# routed to a zombie. Per-launch suffix kills the race entirely.
LAUNCH_EPOCH="$(date +%s)"
WANDB_RENDEZVOUS_URI="gs://${BUCKET}/wandb-rendezvous/v4-32-spot-canary-${LAUNCH_EPOCH}.id"
echo "==> wandb rendezvous (this launch): $WANDB_RENDEZVOUS_URI"
# Pass config via env vars in the SSH command (small string, no quoting issue).
ENV_PREFIX="export GCS_URI='$GCS_URI' REPO_DIR='$REPO_DIR' TPU_STRATEGY='$TPU_STRATEGY' CONFIG_FILE='$CONFIG_FILE' RUN_PROBE_ONLY='$RUN_PROBE_ONLY' WANDB_RENDEZVOUS_URI='$WANDB_RENDEZVOUS_URI';"
gcloud compute tpus tpu-vm ssh "$NODE_ID" \
    --project="$PROJECT_ID" --zone="$ZONE" \
    --worker=all \
    --command="$ENV_PREFIX bash /tmp/_remote_redeploy.sh"

echo "==> [5/5] redeploy complete"
echo
echo "Tail logs:"
echo "  gcloud compute tpus tpu-vm ssh $NODE_ID --project=$PROJECT_ID --zone=$ZONE --worker=0 --command='sudo tail -n 200 -f /tmp/train.log'"
echo "Probe results:"
echo "  gcloud compute tpus tpu-vm ssh $NODE_ID --project=$PROJECT_ID --zone=$ZONE --worker=0 --command='sudo cat /tmp/probe-results.json'"
