#!/bin/bash
# TPU VM startup script.
# Runs as the default user on every host of the Queued Resource at boot.
# Idempotent: safe to re-run after a host reboot or `gcloud ... ssh ... -- 'sudo reboot'`.

set -euo pipefail
exec > >(tee -a /tmp/startup.log) 2>&1

# Metadata startup-script context can run with HOME unset (root, no login shell).
# Set it explicitly so `set -u` doesn't trip on $HOME later.
export HOME="${HOME:-/root}"
export USER="${USER:-$(id -un)}"

echo "=== [$(date -Is)] startup_script.sh begin on $(hostname) ==="

# ----- repo / branch knobs (override via VM metadata if needed) -----
REPO_URL="${REPO_URL:-https://github.com/tiny-aya-simulatenous-translation/tinyaya-stage2-scale.git}"
REPO_BRANCH="${REPO_BRANCH:-feat/tpu-support}"
REPO_DIR="${REPO_DIR:-/opt/tinyaya}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12.13}"
# v0.3: the SYNTHETIC FLORES/OPUS/conversational corpus (1.24M Mimi-encoded
# pairs). NOTE: this is the non-`fleurs-` repo. v0.1 trained on this; v0.2
# regressed to the `fleurs-` sibling by accident -- see the model cards.
HF_DATASET="${HF_DATASET:-tiny-aya-translate/tr-hi-mimi-encoded}"
DATA_DIR="${DATA_DIR:-/mnt/data}"
SECRET_HF="${SECRET_HF:-hf-token}"
SECRET_WANDB="${SECRET_WANDB:-wandb-api-key}"
TMUX_SESSION="${TMUX_SESSION:-train}"
# Optional: GCS path to a tarball that overlays the cloned repo with locally
# uncommitted files (pyproject.toml, uv.lock, new configs). Useful when the
# launch infra is being iterated on and we don't want to push every change.
OVERLAY_GS_URI="${OVERLAY_GS_URI:-}"

read_meta() {
    local key=$1 default=$2
    curl -fsS -H 'Metadata-Flavor: Google' \
        "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$key" \
        2>/dev/null || echo "$default"
}
# Training config path (relative to repo root) is read from VM
# metadata so canary vs full run share the same image.
CONFIG_FILE="$(read_meta config-file configs/tpu/stage2_tpu_v6e_v2.yaml)"
# Optional GCS overlay path (see OVERLAY_GS_URI section below).
OVERLAY_GS_URI="$(read_meta overlay-gs-uri '')"
# Optional full-repo tarball in GCS. When set, replaces the `git clone` step
# entirely (private repo without GitHub credentials on the VM).
REPO_TARBALL_GS_URI="$(read_meta repo-tarball-gs-uri '')"
# Sharding strategy passed through to TPUBackend. See src/backend/tpu_backend.py
# for the full list (replicated / fsdpv2 / fsdpv2_lora / auto).
TPU_STRATEGY_META="$(read_meta tpu-strategy auto)"
# When set to 1, run scripts/tpu/probe_strategies.py before the real training
# and dump results to /tmp/probe-results.json. Useful for canary runs to pick
# the optimal strategy.
PROBE_FIRST="$(read_meta probe-first 0)"
# launch_spot.sh stamps `is-spot=1` into the QR metadata so this script can
# tell preemptible runs from on-demand. We use it to set WANDB_RESUME=allow
# so a preempt resumes the same wandb run instead of forking a new one.
IS_SPOT="$(read_meta is-spot 0)"
# Phase E sweep fleet: when `sweep-id` is set this host runs a `wandb agent`
# (one independent trial stream) instead of a single-config training run, and
# pulls the small pre-staged sweep subset from `sweep-data-gs-uri` (GCS) instead
# of the full corpus from HF. Both empty for a normal training run.
SWEEP_ID="$(read_meta sweep-id '')"
SWEEP_DATA_GS_URI="$(read_meta sweep-data-gs-uri '')"
echo "[startup] CONFIG_FILE=$CONFIG_FILE"
echo "[startup] SWEEP_ID=${SWEEP_ID:-<unset>} SWEEP_DATA_GS_URI=${SWEEP_DATA_GS_URI:-<unset>}"
echo "[startup] OVERLAY_GS_URI=${OVERLAY_GS_URI:-<unset>}"
echo "[startup] REPO_TARBALL_GS_URI=${REPO_TARBALL_GS_URI:-<unset>}"
echo "[startup] TPU_STRATEGY=$TPU_STRATEGY_META PROBE_FIRST=$PROBE_FIRST IS_SPOT=$IS_SPOT"
if [ "$IS_SPOT" = "1" ]; then
    export WANDB_RESUME=allow
    echo "[startup] spot mode: WANDB_RESUME=allow"
fi

# ----- 1. system deps -----
# Dpkg::Lock::Timeout=600 lets apt wait for unattended-upgrades (which
# routinely holds the dpkg lock right after first boot) instead of failing.
APT_OPTS=(-o Dpkg::Lock::Timeout=600)
sudo DEBIAN_FRONTEND=noninteractive apt-get "${APT_OPTS[@]}" update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get "${APT_OPTS[@]}" install -y -qq \
    tmux git curl ca-certificates build-essential

# ----- 2. uv (with pip fallback if astral.sh is unreachable) -----
if ! command -v uv >/dev/null 2>&1; then
    if ! curl -fLsS https://astral.sh/uv/install.sh -o /tmp/uv-install.sh; then
        echo "[startup] curl install of uv failed, falling back to pip"
        python3 -m pip install --user uv
    else
        sh /tmp/uv-install.sh
    fi
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
uv --version

# ----- 3. fetch repo: tarball-from-GCS (preferred for private repos) or git clone -----
sudo mkdir -p "$(dirname "$REPO_DIR")"
sudo chown "$USER:$USER" "$(dirname "$REPO_DIR")"
if [ -n "$REPO_TARBALL_GS_URI" ]; then
    echo "[startup] fetching repo tarball from $REPO_TARBALL_GS_URI"
    sudo rm -rf "$REPO_DIR"
    sudo mkdir -p "$REPO_DIR"
    sudo chown "$USER:$USER" "$REPO_DIR"
    gcloud storage cp "$REPO_TARBALL_GS_URI" /tmp/repo.tar.gz
    tar -xzf /tmp/repo.tar.gz -C "$REPO_DIR"
    rm -f /tmp/repo.tar.gz
else
    if [ ! -d "$REPO_DIR/.git" ]; then
        git clone "$REPO_URL" "$REPO_DIR"
    fi
    cd "$REPO_DIR"
    git fetch --all --prune
    git checkout "$REPO_BRANCH"
    git pull --ff-only
fi

cd "$REPO_DIR"

# ----- 3.5. optional overlay of locally-modified files from GCS -----
# The infra files (pyproject.toml, uv.lock, configs/stage2_tpu*.yaml) may not
# be in the cloned commit yet. If OVERLAY_GS_URI metadata is set, fetch and
# unpack on top of the cloned repo before `uv sync`.
if [ -n "$OVERLAY_GS_URI" ]; then
    echo "[startup] fetching code overlay from $OVERLAY_GS_URI"
    gcloud storage cp "$OVERLAY_GS_URI" /tmp/overlay.tar.gz
    tar -xzvf /tmp/overlay.tar.gz -C "$REPO_DIR"
    rm -f /tmp/overlay.tar.gz
fi

# ----- 4. python + deps via uv (frozen lockfile) -----
uv python install "$PYTHON_VERSION"
uv sync --frozen --extra eval

# ----- 5. secrets from Secret Manager -----
HF_TOKEN="$(gcloud secrets versions access latest --secret="$SECRET_HF")"
export HF_TOKEN
if WANDB_API_KEY="$(gcloud secrets versions access latest --secret="$SECRET_WANDB" 2>/dev/null)"; then
    export WANDB_API_KEY
fi

# ----- 6. dataset to /mnt/data (resumable) -----
sudo mkdir -p "$DATA_DIR"
sudo chown "$USER:$USER" "$DATA_DIR"

if [ -n "$SWEEP_DATA_GS_URI" ]; then
    # Phase E sweep fleet: pull the SMALL pre-staged subset tarball from GCS
    # (built by scripts/tpu/stage_sweep_subset.sh) instead of the full HF corpus.
    # Avoids 8 parallel hosts rate-limiting HF and 8x extracting ~4M files.
    if [ ! -f "$DATA_DIR/encoded/.unpacked" ]; then
        echo "[startup] sweep mode: fetching subset from $SWEEP_DATA_GS_URI"
        gcloud storage cp "$SWEEP_DATA_GS_URI" /tmp/sweep_subset.tar.gz
        sudo mkdir -p "$DATA_DIR/encoded" "$DATA_DIR/splits"
        sudo chown -R "$USER:$USER" "$DATA_DIR/encoded" "$DATA_DIR/splits"
        # Tarball lays out encoded/ + splits/ at the top level -> extract into $DATA_DIR.
        tar -xzf /tmp/sweep_subset.tar.gz -C "$DATA_DIR"
        n_pt=$(find "$DATA_DIR/encoded" -maxdepth 1 -name '*.pt' | wc -l)
        echo "[startup] sweep subset ready: $n_pt encoded .pt files"
        touch "$DATA_DIR/encoded/.unpacked"
    fi
else
# We deliberately do NOT set HF_HUB_ENABLE_HF_TRANSFER here: hf_transfer isn't
# pinned in the lockfile. (The synthetic repo is ~11.7 GB across 9 batch
# tarballs; the download dominates startup but is a one-time cost per host.)
uv run huggingface-cli download "$HF_DATASET" \
    --repo-type dataset \
    --local-dir "$DATA_DIR"

# The synthetic tr-hi-mimi-encoded repo ships the encoded .pt tensors AND their
# alignment JSONs bundled in mimi_encoded_batch*.tar.gz (each with a top-level
# encoded/ dir), plus splits/{train,val}.jsonl. Extract every batch into
# $DATA_DIR so files land at $DATA_DIR/encoded/<name>.pt -- src/data/dataset.py
# _resolve() matches on the BASENAME, so no --strip-components needed.
# ~1.24M samples => ~4M small files (fits: ~12.5M inodes free on the v6e VM).
# Idempotent: skip extraction once the marker file exists.
if ls "$DATA_DIR"/mimi_encoded_batch*.tar.gz >/dev/null 2>&1 \
        && [ ! -f "$DATA_DIR/encoded/.unpacked" ]; then
    sudo mkdir -p "$DATA_DIR/encoded"
    sudo chown "$USER:$USER" "$DATA_DIR/encoded"
    for tb in "$DATA_DIR"/mimi_encoded_batch*.tar.gz; do
        echo "[startup] extracting $(basename "$tb") -> $DATA_DIR/encoded"
        tar -xzf "$tb" -C "$DATA_DIR"
    done
    n_pt=$(find "$DATA_DIR/encoded" -maxdepth 1 -name '*.pt' | wc -l)
    echo "[startup] extracted $n_pt encoded .pt files"
    touch "$DATA_DIR/encoded/.unpacked"
fi
fi  # end SWEEP_DATA_GS_URI branch

# ----- 7. launch training with auto-restart in tmux -----
# torch_xla's _XLAC.so dynamically links libpython3.12.so.1.0, which uv keeps
# inside its managed-Python directory rather than on the standard linker path.
LIBPYTHON_DIR="$(dirname "$(find "$HOME/.local/share/uv/python" -name 'libpython3.12.so.1.0' -type f 2>/dev/null | head -1)")"
echo "[startup] LIBPYTHON_DIR=$LIBPYTHON_DIR"

# Persistent XLA compile cache: DISABLED.
# pytorch/xla #8930 + #9094 (both OPEN as of 2026-05): TPU v4 + torch_xla
# 2.9 fails with "Failed to deserialize executable: UNIMPLEMENTED" when
# the cache is enabled. The fix in PR #9759 (Mar 2026) is not yet in a
# released wheel. Until then we explicitly do NOT set
# XLA_PERSISTENT_CACHE_PATH; pay the compile cost on every process start.
# (Hot redeploys via _remote_redeploy.sh do NOT restart the python
# process for cosmetic edits, so this only hurts on spot preemption.)
echo "[startup] XLA persistent cache disabled (pytorch/xla #8930)"

tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true
# Optional pre-flight probe of sharding strategies on the live mesh.
if [ "$PROBE_FIRST" = "1" ]; then
    echo "[startup] running probe_strategies.py before training"
    DEVICE_BACKEND=tpu PJRT_DEVICE=TPU \
    XLA_DISABLE_FUNCTIONALIZATION=0 \
    LD_LIBRARY_PATH="$LIBPYTHON_DIR:${LD_LIBRARY_PATH:-}" \
    uv run python scripts/tpu/probe_strategies.py \
        --strategies replicated fsdpv2_lora fsdpv2 \
        --steps 5 --hidden 1024 \
        --out /tmp/probe-results.json \
        2>&1 | tee /tmp/probe.log || echo "[startup] probe failed (non-fatal, continuing)"
fi

if [ -n "$SWEEP_ID" ]; then
    # Phase E sweep fleet: this host runs an independent wandb agent (single-host
    # v6e-8 trials pulled from the shared sweep). The agent's child processes
    # inherit the exported TPU env below.
    echo "[startup] sweep mode: launching wandb agent $SWEEP_ID"
    tmux new-session -d -s "$TMUX_SESSION" "
        set -uo pipefail
        ulimit -n 1048576
        cd '$REPO_DIR'
        echo \"[\$(date -Is)] launching wandb agent $SWEEP_ID\" | tee -a /tmp/train.log
        export DEVICE_BACKEND=tpu PJRT_DEVICE=TPU
        export XLA_USE_BF16=0 XLA_DOWNCAST_BF16=0 XLA_DISABLE_FUNCTIONALIZATION=0 XLA_NO_SPECIAL_SCALARS=1
        export LIBTPU_INIT_ARGS='--megascale_grpc_enable_xor_tracer=false --xla_tpu_enable_flash_attention=false'
        export TPU_STRATEGY='$TPU_STRATEGY_META'
        export LD_LIBRARY_PATH='$LIBPYTHON_DIR:\${LD_LIBRARY_PATH:-}'
        export HF_TOKEN='$HF_TOKEN' WANDB_API_KEY='${WANDB_API_KEY:-}'
        export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
        uv run wandb agent $SWEEP_ID 2>&1 | tee -a /tmp/train.log
        echo \"[\$(date -Is)] wandb agent exited with status \$?\" | tee -a /tmp/train.log
    "
else
    tmux new-session -d -s "$TMUX_SESSION" "
        set -euo pipefail
        ulimit -n 1048576
        cd '$REPO_DIR'
        echo \"[\$(date -Is)] launching train_hierarchical.py\" | tee -a /tmp/train.log
        DEVICE_BACKEND=tpu PJRT_DEVICE=TPU \
        XLA_USE_BF16=0 \
        XLA_DOWNCAST_BF16=0 \
        XLA_DISABLE_FUNCTIONALIZATION=0 \
        XLA_NO_SPECIAL_SCALARS=1 \
        LIBTPU_INIT_ARGS='--megascale_grpc_enable_xor_tracer=false --xla_tpu_enable_flash_attention=false' \
        PT_XLA_DEBUG_LEVEL=1 \
        XLA_PROFILER_PORT=9012 \
        TPU_STRATEGY='$TPU_STRATEGY_META' \
        LD_LIBRARY_PATH='$LIBPYTHON_DIR:\${LD_LIBRARY_PATH:-}' \
        HF_TOKEN='$HF_TOKEN' \
        WANDB_API_KEY='${WANDB_API_KEY:-}' \
        PYTHONUNBUFFERED=1 \
        uv run python -u scripts/train_hierarchical.py \
            --config '$CONFIG_FILE' \
            --resume auto 2>&1 | tee -a /tmp/train.log
        echo \"[\$(date -Is)] training exited with status \$?\" | tee -a /tmp/train.log
    "
fi
# NOTE: 'while true' supervisor loop intentionally removed.
# GCP spot TPU preemption tears down the VM, not the python process,
# so a process-level supervisor cannot recover. The QR's spot lifecycle
# triggers a host reboot, after which this startup_script.sh re-runs
# (idempotent) and re-launches training fresh. Process-level supervision
# was hiding compile-time errors as transient failures and burning quota.

echo "=== [$(date -Is)] startup_script.sh complete on $(hostname) ==="
echo "Tail logs with: tmux attach -t $TMUX_SESSION   OR   tail -f /tmp/train.log"
