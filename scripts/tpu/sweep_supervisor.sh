#!/usr/bin/env bash
# Always-on SUPERVISOR for the coordinated v6e-16 sweep.
#
# Runs on a tiny always-on GCE VM (NOT the workstation, NOT the TPU) so the sweep
# survives both a workstation shutdown AND a spot preemption of the TPU. Loop:
#   - QR ACTIVE + sweep coordinator alive        -> heartbeat (periodic ntfy).
#   - QR ACTIVE + coordinator DEAD + not done    -> (re)launch the coordinated
#       sweep (kills any stray smoke first); this covers a spot reboot where the
#       TPU startup relaunched the single smoke instead of the sweep.
#   - QR FAILED / missing                         -> re-provision the v6e-16, wait
#       for ACTIVE, then launch the sweep (grid resumes: completed trials skipped).
#   - coordinator reports "sweep complete"        -> ntfy the user (Stage-2 launch
#       is a human decision: pick the winning structure), then idle.
#
# Env (from VM metadata / defaults): NODE_ID, ZONE, NAME, PROJECT_ID, NTFY_TOPIC,
#   WANDB_URL, HEARTBEAT_EVERY (loops), REPO_DIR.
set -uo pipefail

PROJECT_ID="${PROJECT_ID:-ml-pipelines-315702}"
ZONE="${ZONE:-europe-west4-a}"
NODE_ID="${NODE_ID:-tinyaya-v6e16-sweep-ew4}"
QR_NAME="${QR_NAME:-tinyaya-v6e16-sweep-ew4-qr}"
NAME="${NAME:-scale-grid}"
BUCKET="${BUCKET:-tinyaya-stage2-tpu}"
REPO_DIR="${REPO_DIR:-/opt/tinyaya-sup}"
NTFY_TOPIC="${NTFY_TOPIC:-tinyaya-v6e64-e35c80e64dea}"
WANDB_URL="${WANDB_URL:-https://wandb.ai/cataluna84/tinyaya-stage2-tpu}"
POLL_S="${POLL_S:-180}"
HEARTBEAT_EVERY="${HEARTBEAT_EVERY:-10}"      # heartbeat every N loops (~30 min)
CONTROL_PREFIX="gs://$BUCKET/sweep-control/$NAME"

ntfy() { curl -s -H "Title: TinyAya sweep" -d "$1" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1 || true; }
qr_state() { gcloud compute tpus queued-resources describe "$QR_NAME" --project="$PROJECT_ID" --zone="$ZONE" --format='value(state.state)' 2>/dev/null || echo "MISSING"; }
coord_alive() { gcloud compute tpus tpu-vm ssh "$NODE_ID" --zone="$ZONE" --project="$PROJECT_ID" --worker=0 --command='sudo tmux has-session -t sweepcoord 2>/dev/null && echo YES || echo NO' 2>/dev/null | grep -q YES; }
sweep_done() { gsutil cat "$CONTROL_PREFIX/current_trial.json" 2>/dev/null | grep -q '"stop": true'; }

launch_sweep() {
    ntfy "relaunching coordinated sweep on $NODE_ID"
    # kill any stray single-run smoke, then start host loops + coordinator.
    gcloud compute tpus tpu-vm ssh "$NODE_ID" --zone="$ZONE" --project="$PROJECT_ID" --worker=all \
        --command='sudo tmux kill-session -t train 2>/dev/null; sudo pkill -9 -f "train_hierarchical.py" 2>/dev/null; true' 2>/dev/null || true
    NODE_ID="$NODE_ID" ZONE="$ZONE" NAME="$NAME" STAGE=grid \
        bash "$REPO_DIR/scripts/tpu/launch_sweep_coordinated.sh" 2>&1 | tail -5 || true
}

reprovision() {
    ntfy "v6e-16 gone ($1) -- re-provisioning"
    gcloud compute tpus queued-resources delete "$QR_NAME" --project="$PROJECT_ID" --zone="$ZONE" --force --quiet 2>/dev/null || true
    local tb; tb=$(gsutil ls "gs://$BUCKET/code/sweep-v6e16-*.tar.gz" 2>/dev/null | sort | tail -1)
    TRC_PROFILE=v6e-64-ew4a ACCEL_TYPE=v6e-16 QR_NAME="$QR_NAME" NODE_ID="$NODE_ID" \
        CONFIG_FILE=configs/tpu/stage2_tpu_v6e16_scale_proxy.yaml \
        REPO_TARBALL_GS_URI="$tb" \
        SWEEP_DATA_GS_URI="gs://$BUCKET/data/sweep-subset-200000.tar.gz" \
        TPU_STRATEGY=fsdpv2_lora \
        bash "$REPO_DIR/scripts/tpu/launch_spot.sh" 2>&1 | tail -3 || true
}

ntfy "supervisor started for $NAME on $NODE_ID -- $WANDB_URL"
i=0
while true; do
    i=$((i + 1))
    st=$(qr_state)
    if sweep_done; then
        ntfy "Stage-1 sweep COMPLETE. Decide the winning structure, then launch Stage 2. $WANDB_URL"
        sleep 3600; continue
    fi
    case "$st" in
        ACTIVE|READY)
            if coord_alive; then
                [ $((i % HEARTBEAT_EVERY)) -eq 0 ] && ntfy "alive: sweep running on $NODE_ID. $WANDB_URL"
            else
                ntfy "coordinator down (QR $st) -- relaunching"; launch_sweep
            fi
            ;;
        FAILED|SUSPENDING|MISSING)
            reprovision "$st"
            # wait up to ~15 min for ACTIVE, then launch
            for _ in $(seq 1 30); do sleep 30; [ "$(qr_state)" = "ACTIVE" ] && break; done
            [ "$(qr_state)" = "ACTIVE" ] && { sleep 60; launch_sweep; }
            ;;
        *) : ;;  # WAITING_FOR_RESOURCES / PROVISIONING -> keep waiting
    esac
    sleep "$POLL_S"
done
