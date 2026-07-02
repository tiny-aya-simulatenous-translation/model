#!/usr/bin/env bash
# Per-host sweep runner -- runs on EVERY host of the v6e-16 slice.
#
# Polls the shared GCS control file that sweep_coordinator.py (host-0) publishes,
# and when a new trial INDEX appears runs `train_hierarchical.py --sweep <args>`
# with the SAME args on every host, so the 4 hosts rendezvous into one 16-chip
# SPMD mesh (exactly like the baseline launch). After the trial process exits it
# writes a per-host done-marker the coordinator waits on, then polls for the next
# trial. Exits when the coordinator publishes {"stop": true}.
#
# Env mirrors startup_script.sh's training launch (PJRT/XLA/secrets) so a trial is
# byte-for-byte the same program as a normal run, only with --sweep args injected.
#
# Usage (launched in tmux on all workers by launch_sweep_coordinated.sh):
#   CONTROL_PREFIX=gs://tinyaya-stage2-tpu/sweep-control/scale-grid \
#     bash scripts/tpu/sweep_host_loop.sh
set -uo pipefail

CONTROL_PREFIX="${CONTROL_PREFIX:?set CONTROL_PREFIX=gs://.../sweep-control/<name>}"
REPO_DIR="${REPO_DIR:-/opt/tinyaya}"
CONFIG_FILE="${CONFIG_FILE:-configs/tpu/stage2_tpu_v6e16_scale_proxy.yaml}"
TPU_STRATEGY="${TPU_STRATEGY:-fsdpv2_lora}"
SECRET_HF="${SECRET_HF:-hf-token}"
SECRET_WANDB="${SECRET_WANDB:-wandb-api-key}"
POLL_S="${POLL_S:-20}"
LOG=/tmp/train.log
HOST_ID="$(hostname)"

export HOME="${HOME:-/root}"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
cd "$REPO_DIR"

# torch_xla's _XLAC.so needs libpython3.12.so.1.0, which uv hides in its managed
# python dir (same derivation as startup_script.sh).
LIBPYTHON_DIR="$(dirname "$(find "$HOME/.local/share/uv/python" -name 'libpython3.12.so.1.0' -type f 2>/dev/null | head -1)")"

HF_TOKEN="$(gcloud secrets versions access latest --secret="$SECRET_HF" 2>/dev/null || true)"
WANDB_API_KEY="$(gcloud secrets versions access latest --secret="$SECRET_WANDB" 2>/dev/null || true)"
export HF_TOKEN WANDB_API_KEY

echo "[$(date -Is)] host_loop start host=$HOST_ID control=$CONTROL_PREFIX" | tee -a "$LOG"

last_index=0
while true; do
    trial_json="$(gsutil cat "$CONTROL_PREFIX/current_trial.json" 2>/dev/null || true)"
    if [ -z "$trial_json" ]; then
        sleep "$POLL_S"; continue
    fi

    # Parse the control file (system python3 is present on the TPU VM).
    stop="$(printf '%s' "$trial_json"   | python3 -c 'import json,sys; print(json.load(sys.stdin).get("stop", False))' 2>/dev/null || echo False)"
    index="$(printf '%s' "$trial_json"  | python3 -c 'import json,sys; print(json.load(sys.stdin).get("index", 0))' 2>/dev/null || echo 0)"
    run_id="$(printf '%s' "$trial_json" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("wandb_run_id",""))' 2>/dev/null || echo '')"
    sweep_id="$(printf '%s' "$trial_json"| python3 -c 'import json,sys; print(json.load(sys.stdin).get("sweep_id",""))' 2>/dev/null || echo '')"

    if [ "$stop" = "True" ]; then
        echo "[$(date -Is)] host_loop stop signal -- exiting host=$HOST_ID" | tee -a "$LOG"
        break
    fi
    if [ "$index" -le "$last_index" ] 2>/dev/null; then
        sleep "$POLL_S"; continue
    fi

    # Compact JSON list of --sweep args -> bash array (args have no newlines).
    mapfile -t TRIAL_ARGS < <(printf '%s' "$trial_json" | python3 -c 'import json,sys; [print(a) for a in json.load(sys.stdin)["args"]]')

    echo "[$(date -Is)] === trial $index (run $run_id) host=$HOST_ID ===" | tee -a "$LOG"
    echo "[trial $index] args: ${TRIAL_ARGS[*]}" | tee -a "$LOG"

    ulimit -n 1048576
    # WANDB_RUN_ID: the coordinator's pre-generated id (train honors it -> its run
    # id + checkpoint dir + the metric the coordinator reads all match). Only set
    # WANDB_SWEEP_ID when NON-empty -- wandb rejects an empty-string sweep id
    # ("Sweep ID cannot be empty"), which is the grid case (no server-side sweep).
    export WANDB_RUN_ID="$run_id"
    if [ -n "$sweep_id" ]; then export WANDB_SWEEP_ID="$sweep_id"; else unset WANDB_SWEEP_ID; fi
    DEVICE_BACKEND=tpu PJRT_DEVICE=TPU \
    XLA_USE_BF16=0 XLA_DOWNCAST_BF16=0 XLA_DISABLE_FUNCTIONALIZATION=0 XLA_NO_SPECIAL_SCALARS=1 \
    LIBTPU_INIT_ARGS='--megascale_grpc_enable_xor_tracer=false --xla_tpu_enable_flash_attention=false' \
    PT_XLA_DEBUG_LEVEL=1 \
    TPU_STRATEGY="$TPU_STRATEGY" \
    LD_LIBRARY_PATH="$LIBPYTHON_DIR:${LD_LIBRARY_PATH:-}" \
    PYTHONUNBUFFERED=1 \
    uv run python -u scripts/train_hierarchical.py \
        --config "$CONFIG_FILE" \
        --sweep "${TRIAL_ARGS[@]}" 2>&1 | tee -a "$LOG"
    trial_rc=${PIPESTATUS[0]}
    echo "[$(date -Is)] trial $index exited rc=$trial_rc host=$HOST_ID" | tee -a "$LOG"

    # done-marker (unique per host via hostname); coordinator waits for all hosts.
    : > /tmp/done_marker
    gsutil cp /tmp/done_marker "$CONTROL_PREFIX/done/trial_${index}_host_${HOST_ID}" 2>/dev/null || true
    last_index="$index"
done
echo "[$(date -Is)] host_loop complete host=$HOST_ID" | tee -a "$LOG"
