#!/bin/bash
# Release launcher for the public 15k-step v6e-8 run.
#
# Differs from the canary/spot launchers by adding the release
# instrumentation the public run needs (audit items #5, #6, #12):
#   * GIT_SHA / GIT_DIRTY injected into the env so train_hierarchical.py
#     records the exact commit in wandb.config (the VM is not a git
#     checkout; the deploy step writes simultaneous-translation/.git_sha).
#   * the tee'd training log is sanitized (secret redaction) and uploaded
#     to GCS at end-of-run.
set -uo pipefail
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
set -a; . /opt/tinyaya/.env; set +a
export TOKENIZERS_PARALLELISM=false

cd /opt/tinyaya/simultaneous-translation

# Reproducibility: commit SHA written at deploy time (VM has no .git).
export GIT_SHA="$(cat .git_sha 2>/dev/null || echo unknown)"
export GIT_DIRTY="$(cat .git_dirty 2>/dev/null || echo unknown)"
# Dataset revision pinned at staging time (optional).
export DATASET_REVISION="${DATASET_REVISION:-$(cat .dataset_revision 2>/dev/null || echo unknown)}"

CONFIG="${1:-configs/stage2_tpu_v6e_v2.yaml}"
LOG=/tmp/train.log
GCS_PREFIX="${GCS_LOG_PREFIX:-gs://tinyaya-stage2-tpu/checkpoints/stage2-tpu-v6e-v2}"

LIBPYTHON_DIR="$(dirname "$(find "$HOME/.local/share/uv/python" -name "libpython3.12.so.1.0" -type f 2>/dev/null | head -1)")"
echo "[release] $(date -Is) start; GIT_SHA=$GIT_SHA dirty=$GIT_DIRTY config=$CONFIG" | tee "$LOG"

# XLA_NO_SPECIAL_SCALARS=1 disables XLA's "assume no NaN/Inf" (special-scalar)
# optimization, which is the suspected cause of the inline-TPU-val NaN: a
# finite forward whose loss scalar materialises non-deterministically NaN
# under NaN-unsafe algebraic rewrites. See docs + pytorch/xla#1665.
DEVICE_BACKEND=tpu PJRT_DEVICE=TPU \
XLA_DISABLE_FUNCTIONALIZATION=0 \
XLA_NO_SPECIAL_SCALARS=1 \
LIBTPU_INIT_ARGS="--megascale_grpc_enable_xor_tracer=false --xla_tpu_enable_flash_attention=false" \
TPU_STRATEGY=fsdpv2_lora \
WANDB_RESUME=allow \
LD_LIBRARY_PATH="$LIBPYTHON_DIR:${LD_LIBRARY_PATH:-}" \
uv run python -u scripts/train_hierarchical.py \
    --config "$CONFIG" \
    --resume auto 2>&1 | tee -a "$LOG"
STATUS=${PIPESTATUS[0]}
echo "[release] $(date -Is) training exited status $STATUS" | tee -a "$LOG"

# --- sanitize + upload the log to GCS (audit items #5, #12) -------------
SANITIZED=/tmp/train_15k.log
python3 - "$LOG" "$SANITIZED" <<'PY'
import re, sys
raw = open(sys.argv[1]).read()
for pat in (r"hf_[A-Za-z0-9]{20,}", r"\b[0-9a-f]{40}\b",
            r"(?i)(api[_-]?key|token|secret|password|kaggle_key)\s*[=:]\s*\S+",
            r"(?i)bearer\s+[A-Za-z0-9._-]{20,}"):
    raw = re.sub(pat, "***REDACTED***", raw)
raw = raw.replace("/home/cataluna84", "/home/USER")
open(sys.argv[2], "w").write(raw)
print(f"[release] sanitized log -> {sys.argv[2]} ({len(raw)} bytes)")
PY
GSUTIL="$(command -v gsutil || echo /snap/bin/gsutil)"
"$GSUTIL" -q cp "$SANITIZED" "$GCS_PREFIX/train_15k.log" \
  && echo "[release] uploaded log -> $GCS_PREFIX/train_15k.log" \
  || echo "[release] WARN: log upload failed"

exit "$STATUS"
