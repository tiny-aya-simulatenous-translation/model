#!/bin/bash
# Companion script for hot_redeploy.sh -- runs ON each TPU worker.
# Receives all configuration via env vars and downloaded files.
#
# Required env (exported by parent or passed via --command env):
#   GCS_URI            - tarball URI (gs://bucket/path/file.tar.gz)
#   REPO_DIR           - local repo dir (e.g. /opt/tinyaya)
#   TPU_STRATEGY       - replicated | fsdpv2 | fsdpv2_lora | auto
#   CONFIG_FILE        - relative config (e.g. configs/tpu/stage2_tpu_v6e_v2.yaml)
#   RUN_PROBE_ONLY     - 0/1
set -euo pipefail

GCS_URI="${GCS_URI:?missing}"
REPO_DIR="${REPO_DIR:-/opt/tinyaya}"
TPU_STRATEGY="${TPU_STRATEGY:-auto}"
CONFIG_FILE="${CONFIG_FILE:-configs/tpu/stage2_tpu_v6e_v2.yaml}"
RUN_PROBE_ONLY="${RUN_PROBE_ONLY:-0}"
WANDB_RENDEZVOUS_URI="${WANDB_RENDEZVOUS_URI:-}"
TARBALL_LOCAL="/tmp/tinyaya-repo-hot.tar.gz"

echo "[remote] host=$(hostname) strategy=$TPU_STRATEGY probe_only=$RUN_PROBE_ONLY"

# Resolve uv binary (the original startup_script.sh installs uv as root).
UV_BIN=""
for candidate in /root/.local/bin/uv /root/.cargo/bin/uv /home/$USER/.local/bin/uv /home/$USER/.cargo/bin/uv; do
    if sudo test -x "$candidate"; then
        UV_BIN="$candidate"
        break
    fi
done
if [ -z "$UV_BIN" ]; then
    UV_BIN="$(sudo find /root /home -maxdepth 5 -name uv -type f -executable 2>/dev/null | head -1)"
fi
if [ -z "$UV_BIN" ]; then
    echo "[remote] ERROR: uv binary not found"
    exit 2
fi
echo "[remote] UV_BIN=$UV_BIN"

# 1. Kill running training / tmux / hung python.
sudo tmux kill-session -t train 2>/dev/null || true
sudo pkill -9 -f 'python.*train_hierarchical' 2>/dev/null || true
sudo pkill -9 -f 'python.*probe_strategies' 2>/dev/null || true
sleep 2

# 2. Fetch tarball + extract over the existing repo.
sudo gcloud storage cp "$GCS_URI" "$TARBALL_LOCAL"
sudo mkdir -p "$REPO_DIR"
sudo tar -xzf "$TARBALL_LOCAL" -C "$REPO_DIR" --no-same-owner --overwrite
sudo rm -f "$TARBALL_LOCAL"

# 3. Resolve libpython for LD_LIBRARY_PATH.
# TPU note: uv-installed CPython on v6e images places libpython3.12.so.1.0
# under /root/.local/share/uv/python/..., but sudo find from a non-root
# user may not traverse into /root. Use a broader search with 2>/dev/null
# and fall back to the known cpython-3.12.13 path if find returns empty.
_libpython="$(sudo find /root/.local/share/uv/python -name 'libpython3.12.so.1.0' -type f 2>/dev/null | head -1)"
if [ -z "$_libpython" ]; then
    # Fallback: try the known uv path directly.
    for _dir in \
        /root/.local/share/uv/python/cpython-3.12.*-linux-x86_64-gnu/lib \
        /root/.local/share/uv/python/cpython-3.12.*/linux-x86_64-gnu/lib; do
        if [ -f "$_dir/libpython3.12.so.1.0" ]; then
            _libpython="$_dir/libpython3.12.so.1.0"
            break
        fi
    done
fi
if [ -z "$_libpython" ]; then
    echo "[remote] ERROR: libpython3.12.so.1.0 not found under uv Python roots"
    exit 2
fi
LIBPYTHON_DIR="$(dirname "$_libpython")"
echo "[remote] LIBPYTHON_DIR=$LIBPYTHON_DIR"

# Persistent XLA compile cache: DISABLED.
# pytorch/xla #8930 + #9094 (OPEN as of 2026-05): TPU v4 + torch_xla 2.9
# crashes "Failed to deserialize executable: UNIMPLEMENTED" when the
# cache is enabled. Stay unset until PR #9759 lands in a released wheel.
echo "[remote] XLA persistent cache disabled (pytorch/xla #8930)"

# 4. Pull HF/W&B secrets.
HF_TOKEN="$(gcloud secrets versions access latest --secret=hf-token)"
WANDB_API_KEY="$(gcloud secrets versions access latest --secret=wandb-api-key 2>/dev/null || echo '')"

cd "$REPO_DIR"

if [ "$RUN_PROBE_ONLY" = "1" ]; then
    echo "[remote] running probe_strategies.py"
    # Only worker 0 actually drives the SPMD program; other workers also
    # need to run the same probe binary so PJRT can initialize the mesh.
    sudo PJRT_DEVICE=TPU XLA_DISABLE_FUNCTIONALIZATION=0 \
        LD_LIBRARY_PATH="$LIBPYTHON_DIR:${LD_LIBRARY_PATH:-}" \
        "$UV_BIN" run python scripts/tpu/probe_strategies.py \
            --strategies replicated fsdpv2_lora fsdpv2 \
            --steps 5 --hidden 1024 \
            --out /tmp/probe-results.json 2>&1 | sudo tee /tmp/probe.log
    echo "[remote] probe done -> /tmp/probe-results.json"
    exit 0
fi

# 5. Launch real training under tmux. We write the inner command to a file
#    so the tmux quoting stays simple. The supervisor 'while true' loop is
#    intentionally absent: GCP spot TPU preemption is a VM-level event,
#    not a process-level one, so process supervision hides compile errors
#    as transient failures. Failures bubble up and the orchestrator
#    classifies + redeploys per .factory/orchestration/playbook/.
INNER_SCRIPT="/tmp/train_loop.sh"
sudo tee "$INNER_SCRIPT" >/dev/null <<INNER
#!/bin/bash
set -e
cd "$REPO_DIR"
echo "[\$(date -Is)] launching train_hierarchical.py [strategy=$TPU_STRATEGY]" | tee -a /tmp/train.log
DEVICE_BACKEND=tpu PJRT_DEVICE=TPU \
XLA_DISABLE_FUNCTIONALIZATION=0 \
XLA_NO_SPECIAL_SCALARS=1 \
TPU_STRATEGY=$TPU_STRATEGY \
LD_LIBRARY_PATH="$LIBPYTHON_DIR:\${LD_LIBRARY_PATH:-}" \
HF_TOKEN="$HF_TOKEN" \
WANDB_API_KEY="$WANDB_API_KEY" \
WANDB_RENDEZVOUS_URI="$WANDB_RENDEZVOUS_URI" \
PYTHONUNBUFFERED=1 \
"$UV_BIN" run python -u scripts/train_hierarchical.py \
    --config "$CONFIG_FILE" \
    --resume auto 2>&1 | tee -a /tmp/train.log
echo "[\$(date -Is)] training exited with status \$?" | tee -a /tmp/train.log
INNER
sudo chmod +x "$INNER_SCRIPT"
sudo tmux new-session -d -s train "bash $INNER_SCRIPT"
echo "[remote] tmux session 'train' started, strategy=$TPU_STRATEGY"
