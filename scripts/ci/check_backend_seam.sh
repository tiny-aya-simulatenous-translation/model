#!/bin/bash
# check_backend_seam.sh — enforce the TPU↔GPU separation seam.
#
# WHY THIS EXISTS
# ---------------
# The repo is dual-backend: shared code in `src/` (and
# `scripts/train_hierarchical.py`) must run on a CPU/GPU box that has no
# `torch_xla` installed. The ONLY module allowed to import `torch_xla` at
# module scope is `src/backend/tpu_backend.py` (+ the backend package glue).
# A *module-level* `import torch_xla` anywhere else drags `libtpu` into the
# import graph and breaks GPU runs and CPU-only unit tests.
#
# Lazy (in-function, indented) `import torch_xla` inside a TPU-only code path
# is allowed — it only fires when actually running on TPU. This check flags
# only column-0 (module-level) imports under the shared trees.
#
# Usage:  bash scripts/ci/check_backend_seam.sh
# Exit 0 = seam holds; exit 1 = a module-level torch_xla import leaked.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Shared trees that must stay XLA-free at import time.
TREES=(src/model src/data src/training src/evaluation)

hits="$(grep -rnE '^(import torch_xla|from torch_xla)' "${TREES[@]}" 2>/dev/null || true)"

if [ -n "$hits" ]; then
  echo "FAIL: module-level torch_xla import outside src/backend/ —" >&2
  echo "      move it into src/backend/tpu_backend.py or make it a lazy" >&2
  echo "      in-function import guarded by the TPU code path." >&2
  echo "$hits" >&2
  exit 1
fi

echo "OK: backend seam holds (no module-level torch_xla under ${TREES[*]})."
