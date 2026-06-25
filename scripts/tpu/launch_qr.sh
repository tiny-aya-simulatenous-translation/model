#!/bin/bash
# Create the Queued Resource that provisions the TPU slice and runs
# startup_script.sh on every host.
#
# Run from your local workstation (after setup_gcp.sh has succeeded).
#
# Default behaviour: on-demand v4-64 in us-central2-b. The TRC grant
# (see docs/tpu-trc-allocation.md) also includes spot quotas in five
# (zone, type) combinations; for those, prefer launch_spot.sh which
# wraps this script with TRC_PROFILE-aware defaults and SPOT=1.
#
# Configuration precedence (highest first):
#   1. shell env vars (e.g. `PROJECT_ID=foo bash launch_qr.sh`)
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
ACCEL_TYPE="${ACCEL_TYPE:-v4-64}"
RUNTIME="${RUNTIME:-tpu-ubuntu2204-base}"
CONFIG_FILE="${CONFIG_FILE:-configs/tpu/stage2_tpu_v6e_v2.yaml}"
# SPOT=1  -> request preemptible (spot) capacity. Default empty = on-demand.
SPOT="${SPOT:-}"
# INTERNAL_IPS=1 -> create the TPU hosts WITHOUT external IPs. Required when
# the regional `IN_USE_ADDRESSES` quota is too small for the requested host
# count (each v5e/v6e host normally requests one external IP). Empty = default
# (external IPs allocated). On-demand v4 in us-central2-b currently has 8 IP
# headroom which is enough for v4-64 (4 hosts), so leave this off there.
INTERNAL_IPS="${INTERNAL_IPS:-}"
# Optional GCS path to a full-repo tarball, used by startup_script when the
# repo is private and the VM has no GitHub credentials.
REPO_TARBALL_GS_URI="${REPO_TARBALL_GS_URI:-}"
# Sharding strategy: replicated | fsdpv2 | fsdpv2_lora | auto. See
# src/backend/tpu_backend.py for semantics.
TPU_STRATEGY="${TPU_STRATEGY:-auto}"
# When set to 1, runs probe_strategies.py before the real training to
# benchmark each strategy on the live mesh first.
PROBE_FIRST="${PROBE_FIRST:-0}"

STARTUP_SCRIPT="$SCRIPT_DIR/startup_script.sh"

if [ ! -f "$STARTUP_SCRIPT" ]; then
    echo "ERROR: $STARTUP_SCRIPT not found"
    exit 1
fi

extra_flags=()
is_spot=0
if [ -n "$SPOT" ] && [ "$SPOT" != "0" ] && [ "$SPOT" != "false" ]; then
    extra_flags+=(--spot)
    tier="spot (preemptible)"
    is_spot=1
else
    tier="on-demand"
fi

ips_label="external"
if [ -n "$INTERNAL_IPS" ] && [ "$INTERNAL_IPS" != "0" ] && [ "$INTERNAL_IPS" != "false" ]; then
    extra_flags+=(--internal-ips)
    ips_label="internal-only"
fi

metadata_pairs="config-file=$CONFIG_FILE,tpu-strategy=$TPU_STRATEGY,probe-first=$PROBE_FIRST,is-spot=$is_spot"
if [ -n "$REPO_TARBALL_GS_URI" ]; then
    metadata_pairs+=",repo-tarball-gs-uri=$REPO_TARBALL_GS_URI"
fi

echo "==> creating Queued Resource"
echo "    project:        $PROJECT_ID"
echo "    zone:           $ZONE"
echo "    QR name:        $QR_NAME"
echo "    node id:        $NODE_ID"
echo "    accelerator:    $ACCEL_TYPE"
echo "    runtime:        $RUNTIME"
echo "    tier:           $tier"
echo "    ip mode:        $ips_label"
echo "    startup script: $STARTUP_SCRIPT"
echo "    config file:    $CONFIG_FILE"
echo "    repo tarball:   ${REPO_TARBALL_GS_URI:-<unset, will git clone>}"
echo "    tpu strategy:   $TPU_STRATEGY"
echo "    probe first:    $PROBE_FIRST"
echo

gcloud compute tpus queued-resources create "$QR_NAME" \
    --project="$PROJECT_ID" \
    --zone="$ZONE" \
    --node-id="$NODE_ID" \
    --accelerator-type="$ACCEL_TYPE" \
    --runtime-version="$RUNTIME" \
    --metadata-from-file=startup-script="$STARTUP_SCRIPT" \
    --metadata="$metadata_pairs" \
    "${extra_flags[@]}"

echo
echo "==> QR submitted. Watch with:  QR_NAME=$QR_NAME bash scripts/tpu/ops.sh status"
