#!/usr/bin/env bash
# Launch the COORDINATED multi-host sweep on an existing ACTIVE v6e-16 slice.
#
# Unlike launch_sweep.sh (per-host independent `wandb agent`s -- which DESYNC on a
# single multi-host mesh), this runs ONE trial at a time across all 4 hosts:
#   - host-0 runs sweep_coordinator.py (picks trials, writes a GCS control file),
#   - ALL hosts run sweep_host_loop.sh (poll the control file, run the SAME
#     train_hierarchical.py --sweep <args> so the 16-chip mesh forms per trial).
# Everything runs DETACHED in tmux on the VM, so the workstation can be powered
# off; the W&B controller/leaderboard is server-side.
#
# PREREQUISITE: an ACTIVE v6e-16 slice (NODE_ID) whose baseline run has finished
# (the slice is idle). This pushes the current code onto every host first.
#
# Usage:
#   # Stage 1 (structural grid -- no `wandb sweep` needed, YAML is enumerated):
#   NODE_ID=tinyaya-v6e16-batch-ew4 ZONE=europe-west4-a NAME=scale-grid STAGE=grid \
#       bash scripts/tpu/launch_sweep_coordinated.sh
#
#   # Stage 2 (Bayesian lr x rank -- create the sweep first, pass its id):
#   wandb sweep sweeps/sweep_stage2_scale_bayes.yaml      # -> ENTITY/PROJECT/ID
#   NAME=scale-bayes STAGE=bayes SWEEP_ID=ENTITY/PROJECT/ID \
#       bash scripts/tpu/launch_sweep_coordinated.sh
#
# Monitor:  gcloud compute tpus tpu-vm ssh $NODE_ID --zone $ZONE --worker=0 \
#             --command 'tmux capture-pane -pt sweepcoord | tail -40'
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=./_lib.sh
source "$SCRIPT_DIR/_lib.sh"
load_env_file "$REPO_ROOT/.env"

PROJECT_ID="${PROJECT_ID:-ml-pipelines-315702}"
ZONE="${ZONE:-europe-west4-a}"
NODE_ID="${NODE_ID:-tinyaya-v6e16-batch-ew4}"
BUCKET="${BUCKET:-tinyaya-stage2-tpu}"
NAME="${NAME:-scale-grid}"
STAGE="${STAGE:-grid}"                       # grid | bayes
SWEEP_ID="${SWEEP_ID:-}"                      # required for STAGE=bayes
SWEEP_YAML="${SWEEP_YAML:-sweeps/sweep_stage2_scale_grid.yaml}"
CONFIG_FILE="${CONFIG_FILE:-configs/tpu/stage2_tpu_v6e16_scale_proxy.yaml}"
NUM_HOSTS="${NUM_HOSTS:-4}"                   # v6e-16 = 4 hosts
MAX_TRIALS="${MAX_TRIALS:-0}"
WANDB_ENTITY_PROJECT="${WANDB_ENTITY_PROJECT:-cataluna84/tinyaya-stage2-tpu}"  # for metric reads
SAVE_DIR="${SAVE_DIR:-gs://$BUCKET/checkpoints/stage2-scale-sweep}"  # must match proxy logging.save_dir
FINAL_STEP="${FINAL_STEP:-1500}"              # proxy max_steps -> checkpoint-based completion check
REPO_DIR="/opt/tinyaya"
CONTROL_PREFIX="gs://$BUCKET/sweep-control/$NAME"

if [ "$STAGE" = "bayes" ] && [ -z "$SWEEP_ID" ]; then
    echo "ERROR: STAGE=bayes requires SWEEP_ID=ENTITY/PROJECT/ID (from 'wandb sweep ...')" >&2
    exit 2
fi

echo "==> coordinated sweep: NODE_ID=$NODE_ID stage=$STAGE name=$NAME"
echo "    control-prefix: $CONTROL_PREFIX"

# ----- 1. package code -> GCS (same file set as hot_redeploy.sh) -----
TARBALL="/tmp/tinyaya-sweep-code.tar.gz"
GCS_TARBALL="gs://$BUCKET/code/sweep-coord-$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
echo "==> [1/4] packaging code -> $GCS_TARBALL"
cd "$REPO_ROOT"
tar --exclude='.git' --exclude='.venv' --exclude='.env' --exclude='.claude' \
    --exclude='__pycache__' --exclude='*.pyc' --exclude='.pytest_cache' \
    -czf "$TARBALL" \
    src scripts configs sweeps docs pyproject.toml uv.lock README.md
gcloud storage cp "$TARBALL" "$GCS_TARBALL" --project="$PROJECT_ID"
rm -f "$TARBALL"

ssh_all() { gcloud compute tpus tpu-vm ssh "$NODE_ID" --zone="$ZONE" --project="$PROJECT_ID" --worker=all --command="$1"; }
ssh_0()   { gcloud compute tpus tpu-vm ssh "$NODE_ID" --zone="$ZONE" --project="$PROJECT_ID" --worker=0   --command="$1"; }

# ----- 2. refresh code on every host (overlay onto /opt/tinyaya) -----
echo "==> [2/4] fetching code on all hosts"
# NB: /opt/tinyaya + venv + uv + data + secrets all belong to ROOT (the startup
# script runs as root), so every remote action here runs via sudo -H (HOME=/root,
# so uv at /root/.local/bin and LIBPYTHON under /root/.local/share resolve).
ssh_all "set -e; gcloud storage cp '$GCS_TARBALL' /tmp/sweep-code.tar.gz && \
    sudo tar -xzf /tmp/sweep-code.tar.gz -C '$REPO_DIR' && rm -f /tmp/sweep-code.tar.gz && \
    echo \"[\$(hostname)] code refreshed\""

# ----- 3. start the per-host loop on ALL hosts (detached tmux, as ROOT) -----
# Kill BOTH the old sweephost tmux AND any surviving train_hierarchical child
# first: `tmux kill-session` only SIGHUPs the session, leaving a train process
# (which ignores SIGHUP under `uv run`) as a HUNG zombie holding the mesh -- the
# exact desync that let a relaunched coordinator time out on already-past hosts.
echo "==> [3/4] starting host loops (tmux 'sweephost') on all hosts"
ssh_all "sudo -H tmux kill-session -t sweephost 2>/dev/null || true; \
    sudo pkill -9 -f '[s]cripts/train_hierarchical.py' 2>/dev/null || true; \
    sleep 2; \
    sudo -H tmux new-session -d -s sweephost \
    \"CONTROL_PREFIX='$CONTROL_PREFIX' CONFIG_FILE='$CONFIG_FILE' \
      bash '$REPO_DIR/scripts/tpu/sweep_host_loop.sh'\"; \
    sleep 1; sudo tmux ls 2>/dev/null | grep -q sweephost && echo \"[\$(hostname)] sweephost OK\" || echo \"[\$(hostname)] sweephost FAILED\""

# ----- 4. start the coordinator on host-0 (detached tmux) -----
echo "==> [4/4] starting coordinator (tmux 'sweepcoord') on host-0"
if [ "$STAGE" = "grid" ]; then
    COORD_ARGS="--stage grid --sweep-yaml '$SWEEP_YAML'"
    [ -n "$SWEEP_ID" ] && COORD_ARGS="$COORD_ARGS --sweep-id '$SWEEP_ID'"
else
    COORD_ARGS="--stage bayes --sweep-id '$SWEEP_ID'"
fi
ssh_0 "sudo -H tmux kill-session -t sweepcoord 2>/dev/null || true; \
    sudo -H tmux new-session -d -s sweepcoord \
    \"cd '$REPO_DIR'; export PATH=/root/.local/bin:\\\$PATH; \
      export WANDB_API_KEY=\\\$(gcloud secrets versions access latest --secret=wandb-api-key 2>/dev/null); \
      uv run python -u scripts/tpu/sweep_coordinator.py $COORD_ARGS \
        --control-prefix '$CONTROL_PREFIX' --wandb-project '$WANDB_ENTITY_PROJECT' \
        --num-hosts $NUM_HOSTS --max-trials $MAX_TRIALS \
        --save-dir '$SAVE_DIR' --final-step $FINAL_STEP \
        2>&1 | tee -a /tmp/coord.log\"; \
    sleep 1; sudo tmux ls 2>/dev/null | grep -q sweepcoord && echo 'coordinator OK' || echo 'coordinator FAILED'"

cat <<EOF

==> launched. Monitor:
    gcloud compute tpus tpu-vm ssh $NODE_ID --zone $ZONE --worker=0 \\
        --command 'tmux capture-pane -pt sweepcoord | tail -40; echo ---; tail -20 /tmp/train.log'
    Tear down: ssh --worker=all 'tmux kill-session -t sweephost; tmux kill-session -t sweepcoord'
EOF
