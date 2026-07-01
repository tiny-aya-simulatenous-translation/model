#!/usr/bin/env bash
# Spin up the always-on GCE SUPERVISOR VM for the coordinated sweep.
#
# The VM (e2-small, default compute SA w/ cloud-platform) runs
# scripts/tpu/sweep_supervisor.sh as a Restart=always systemd service, so the
# sweep survives a workstation shutdown AND a spot preemption of the TPU (the
# supervisor re-provisions + relaunches). Idempotent-ish: delete the VM to stop.
#
# Usage:
#   NODE_ID=tinyaya-v6e16-sweep-ew4 ZONE=europe-west4-a NAME=scale-grid \
#       bash scripts/tpu/launch_sweep_supervisor.sh
#   # tear down:  gcloud compute instances delete tinyaya-sweep-supervisor --zone <VM_ZONE>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=./_lib.sh
source "$SCRIPT_DIR/_lib.sh"
load_env_file "$REPO_ROOT/.env"

PROJECT_ID="${PROJECT_ID:-ml-pipelines-315702}"
BUCKET="${BUCKET:-tinyaya-stage2-tpu}"
VM_NAME="${VM_NAME:-tinyaya-sweep-supervisor}"
VM_ZONE="${VM_ZONE:-us-central1-a}"            # cheap always-on zone (any works)
NODE_ID="${NODE_ID:-tinyaya-v6e16-sweep-ew4}"
QR_NAME="${QR_NAME:-${NODE_ID}-qr}"
ZONE="${ZONE:-europe-west4-a}"                 # the TPU's zone
NAME="${NAME:-scale-grid}"
NTFY_TOPIC="${NTFY_TOPIC:-tinyaya-v6e64-e35c80e64dea}"
WANDB_URL="${WANDB_URL:-https://wandb.ai/cataluna84/tinyaya-stage2-tpu}"

# ----- 1. package repo (incl. sweep_supervisor.sh + launch scripts) -> GCS -----
TB=/tmp/tinyaya-supervisor-code.tar.gz
GCS_TB="gs://$BUCKET/code/supervisor-$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
cd "$REPO_ROOT"
tar --exclude='.git' --exclude='.venv' --exclude='.env' --exclude='.claude' \
    --exclude='__pycache__' --exclude='*.pyc' --exclude='.pytest_cache' \
    -czf "$TB" src scripts configs sweeps docs pyproject.toml uv.lock README.md
gcloud storage cp "$TB" "$GCS_TB" --project="$PROJECT_ID"; rm -f "$TB"

# ----- 2. VM startup script -----
STARTUP=$(mktemp)
cat >"$STARTUP" <<STARTUP_EOF
#!/bin/bash
set -uxo pipefail
export HOME=/root
rm -rf /opt/tinyaya-sup && mkdir -p /opt/tinyaya-sup
for i in \$(seq 1 30); do gsutil cp "$GCS_TB" /tmp/code.tar.gz && break || sleep 10; done
tar -xzf /tmp/code.tar.gz -C /opt/tinyaya-sup
cat >/etc/systemd/system/sweepsup.service <<UNIT
[Unit]
Description=TinyAya coordinated-sweep supervisor
After=network-online.target
Wants=network-online.target
[Service]
Environment=PROJECT_ID=$PROJECT_ID ZONE=$ZONE NODE_ID=$NODE_ID QR_NAME=$QR_NAME NAME=$NAME
Environment=BUCKET=$BUCKET REPO_DIR=/opt/tinyaya-sup NTFY_TOPIC=$NTFY_TOPIC WANDB_URL=$WANDB_URL
ExecStart=/bin/bash /opt/tinyaya-sup/scripts/tpu/sweep_supervisor.sh
Restart=always
RestartSec=30
[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable --now sweepsup.service
STARTUP_EOF

# ----- 3. create the VM (default compute SA, cloud-platform scope) -----
echo "==> creating supervisor VM $VM_NAME in $VM_ZONE"
gcloud compute instances create "$VM_NAME" \
    --project="$PROJECT_ID" --zone="$VM_ZONE" \
    --machine-type=e2-small --image-family=debian-12 --image-project=debian-cloud \
    --scopes=cloud-platform \
    --metadata-from-file=startup-script="$STARTUP"
rm -f "$STARTUP"
echo "==> supervisor VM up. Logs:  gcloud compute ssh $VM_NAME --zone $VM_ZONE --command 'sudo journalctl -u sweepsup -n 50 --no-pager'"
echo "    Stop:  gcloud compute instances delete $VM_NAME --zone $VM_ZONE --quiet"
