#!/bin/bash
# Daily ops helper for the TinyAya Stage 2 TPU run.
#
# Usage:
#   bash scripts/tpu/ops.sh status
#   bash scripts/tpu/ops.sh tail-logs
#   bash scripts/tpu/ops.sh attach
#   bash scripts/tpu/ops.sh ssh
#   bash scripts/tpu/ops.sh pull-best [LOCAL_DIR]
#   bash scripts/tpu/ops.sh delete
#
# Configuration precedence (highest first):
#   1. shell env vars (e.g. `PROJECT_ID=foo bash ops.sh status`)
#   2. <repo-root>/.env  (auto-sourced)
#   3. defaults below

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"

# shellcheck source=./_lib.sh
source "$SCRIPT_DIR/_lib.sh"
load_env_file "$ENV_FILE"

PROJECT_ID="${PROJECT_ID:-ml-pipelines-315702}"
ZONE="${ZONE:-us-central2-b}"
QR_NAME="${QR_NAME:-tinyaya-stage2-qr}"
NODE_ID="${NODE_ID:-tinyaya-stage2}"
BUCKET="${BUCKET:-tinyaya-stage2-tpu}"
CKPT_PREFIX="${CKPT_PREFIX:-checkpoints/stage2-tpu}"

usage() {
    grep '^#' "$0" | sed 's/^# \{0,1\}//' | head -n 12
    exit 1
}

cmd="${1:-}"
shift || true

case "$cmd" in

    status)
        echo "==> Queued Resource"
        gcloud compute tpus queued-resources describe "$QR_NAME" \
            --project="$PROJECT_ID" --zone="$ZONE" || true
        echo
        echo "==> TPU node (if provisioned)"
        gcloud compute tpus tpu-vm describe "$NODE_ID" \
            --project="$PROJECT_ID" --zone="$ZONE" 2>/dev/null || \
            echo "    (not yet provisioned)"
        ;;

    tail-logs)
        echo "==> tailing /tmp/train.log on all workers"
        gcloud compute tpus tpu-vm ssh "$NODE_ID" \
            --project="$PROJECT_ID" --zone="$ZONE" \
            --worker=all \
            --command="tail -n 80 -f /tmp/train.log"
        ;;

    attach)
        worker="${1:-0}"
        echo "==> attaching tmux 'train' session on worker $worker (Ctrl-b d to detach)"
        gcloud compute tpus tpu-vm ssh "$NODE_ID" \
            --project="$PROJECT_ID" --zone="$ZONE" \
            --worker="$worker" \
            -- -t "tmux attach -t train"
        ;;

    ssh)
        worker="${1:-0}"
        gcloud compute tpus tpu-vm ssh "$NODE_ID" \
            --project="$PROJECT_ID" --zone="$ZONE" \
            --worker="$worker"
        ;;

    pull-best)
        dst="${1:-./_artifacts}"
        mkdir -p "$dst"
        echo "==> pulling best/latest checkpoints from gs://$BUCKET/$CKPT_PREFIX -> $dst"
        gcloud storage rsync -r \
            "gs://$BUCKET/$CKPT_PREFIX/" \
            "$dst/"
        echo "==> done"
        ls -lh "$dst"
        ;;

    delete)
        echo "==> deleting Queued Resource $QR_NAME (this stops billing for the slice)"
        # Force delete works even if the underlying node is still attached.
        gcloud compute tpus queued-resources delete "$QR_NAME" \
            --project="$PROJECT_ID" --zone="$ZONE" \
            --force --quiet
        ;;

    *)
        usage
        ;;
esac
