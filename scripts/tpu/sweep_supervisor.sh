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
# --- for the coordinator-ONLY relaunch (restart the driver without touching a
#     healthy in-flight trial). Must mirror launch_sweep_coordinated.sh. ---
TPU_REPO_DIR="${TPU_REPO_DIR:-/opt/tinyaya}"
SWEEP_YAML="${SWEEP_YAML:-sweeps/sweep_stage2_scale_grid.yaml}"
WANDB_ENTITY_PROJECT="${WANDB_ENTITY_PROJECT:-cataluna84/tinyaya-stage2-tpu}"
SAVE_DIR="${SAVE_DIR:-gs://$BUCKET/checkpoints/stage2-scale-sweep}"
FINAL_STEP="${FINAL_STEP:-1500}"
NUM_HOSTS="${NUM_HOSTS:-4}"

ntfy() { curl -s -H "Title: TinyAya sweep" -d "$1" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1 || true; }

# Resilient QR state: retry, and only report a DEFINITIVE state. Transient
# describe failures return UNKNOWN (treated as "keep waiting", never destructive)
# so a network blip can't trigger a re-provision of a healthy running TPU.
qr_state() {
    local s
    for _ in 1 2 3; do
        s=$(gcloud compute tpus queued-resources describe "$QR_NAME" --project="$PROJECT_ID" --zone="$ZONE" --format='value(state.state)' 2>/dev/null)
        if [ -n "$s" ]; then echo "$s"; return; fi
        sleep 10
    done
    echo "UNKNOWN"
}

# Resilient coordinator liveness: YES if any of 3 tries sees the session; only
# "not alive" if all 3 SSHes succeeded-and-saw-no-session or failed. Returns 0
# (alive) on YES, 1 otherwise.
coord_alive() {
    for _ in 1 2 3; do
        if gcloud compute tpus tpu-vm ssh "$NODE_ID" --zone="$ZONE" --project="$PROJECT_ID" --worker=0 \
             --command='sudo tmux has-session -t sweepcoord 2>/dev/null && echo YES || echo NO' 2>/dev/null | grep -q YES; then
            return 0
        fi
        sleep 10
    done
    return 1
}
sweep_done() { gsutil cat "$CONTROL_PREFIX/current_trial.json" 2>/dev/null | grep -q '"stop": true'; }

# Resilient training liveness: YES if ANY host has a live train process. Used to
# distinguish "coordinator died but a trial is still training fine" (relaunch the
# coordinator ONLY -- never kill a healthy trial mid-checkpoint) from "nothing is
# running" (full clean relaunch). Killing a progressing trial to restart the
# driver was what crashed the first attempt mid-checkpoint.
training_alive() {
    for _ in 1 2 3; do
        if gcloud compute tpus tpu-vm ssh "$NODE_ID" --zone="$ZONE" --project="$PROJECT_ID" --worker=all \
             --command='sudo pgrep -f "[s]cripts/train_hierarchical.py" >/dev/null 2>&1 && echo YES || echo NO' 2>/dev/null | grep -q YES; then
            return 0
        fi
        sleep 10
    done
    return 1
}

# Restart ONLY the host-0 coordinator tmux (no pkill, no host-loop restart). The
# relaunched coordinator re-enumerates the grid, skips completed trials by their
# durable checkpoints, and -- via DETERMINISTIC run ids -- re-attaches to the
# in-flight trial's metric when the already-running hosts finish it.
relaunch_coordinator_only() {
    ntfy "coordinator down but trial still training on $NODE_ID -- relaunching COORDINATOR ONLY"
    gcloud compute tpus tpu-vm ssh "$NODE_ID" --zone="$ZONE" --project="$PROJECT_ID" --worker=0 \
      --command="sudo -H tmux kill-session -t sweepcoord 2>/dev/null || true; \
        sudo -H tmux new-session -d -s sweepcoord \
        \"cd $TPU_REPO_DIR; export PATH=/root/.local/bin:\\\$PATH; \
          export WANDB_API_KEY=\\\$(gcloud secrets versions access latest --secret=wandb-api-key 2>/dev/null); \
          uv run python -u scripts/tpu/sweep_coordinator.py --stage grid --sweep-yaml '$SWEEP_YAML' \
            --control-prefix '$CONTROL_PREFIX' --wandb-project '$WANDB_ENTITY_PROJECT' \
            --num-hosts $NUM_HOSTS --max-trials 0 --save-dir '$SAVE_DIR' --final-step $FINAL_STEP \
            2>&1 | tee -a /tmp/coord.log\"" 2>/dev/null || true
}

launch_sweep() {
    ntfy "relaunching coordinated sweep on $NODE_ID"
    # kill any stray single-run smoke, then start host loops + coordinator.
    gcloud compute tpus tpu-vm ssh "$NODE_ID" --zone="$ZONE" --project="$PROJECT_ID" --worker=all \
        --command='sudo tmux kill-session -t train 2>/dev/null; sudo pkill -9 -f "[t]rain_hierarchical.py" 2>/dev/null; true' 2>/dev/null || true
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
i=0; down=0; failed=0; coordonly=0
while true; do
    i=$((i + 1))
    if sweep_done; then
        ntfy "Stage-1 sweep COMPLETE. Decide the winning structure, then launch Stage 2. $WANDB_URL"
        sleep 3600; continue
    fi
    st=$(qr_state)
    case "$st" in
        ACTIVE|READY)
            failed=0
            if coord_alive; then
                down=0; coordonly=0
                [ $((i % HEARTBEAT_EVERY)) -eq 0 ] && ntfy "alive: sweep running on $NODE_ID. $WANDB_URL"
            else
                down=$((down + 1))   # require 2 consecutive misses (each already retried 3x)
                if [ "$down" -ge 2 ]; then
                    if training_alive && [ "$coordonly" -lt 2 ]; then
                        # Trial still training -> restart the driver only, don't kill it.
                        coordonly=$((coordonly + 1))
                        relaunch_coordinator_only; down=0
                    else
                        # No live training (or coord-only failed twice => hung trial):
                        # full clean relaunch (kills stray/hung procs, restarts all).
                        ntfy "coordinator down x$down; training dead/hung -- FULL relaunch"
                        launch_sweep; down=0; coordonly=0
                    fi
                fi
            fi
            ;;
        FAILED)
            failed=$((failed + 1))   # confirmed spot reclaim only after 2 consecutive
            if [ "$failed" -ge 2 ]; then
                reprovision "$st"
                for _ in $(seq 1 30); do sleep 30; [ "$(qr_state)" = "ACTIVE" ] && break; done
                [ "$(qr_state)" = "ACTIVE" ] && { sleep 60; launch_sweep; }
                failed=0
            fi
            ;;
        *) down=0; failed=0 ;;  # UNKNOWN/SUSPENDING/WAITING/PROVISIONING -> wait, never destructive
    esac
    sleep "$POLL_S"
done
