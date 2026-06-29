#!/usr/bin/env bash
# ONE-SHOT: build the ~200K sweep subset and stage it to GCS so the v6e-8 sweep
# fleet can each rsync a small (~2 GB) tarball instead of pulling the full 11.7 GB
# corpus from HF 8x (rate-limits) and extracting ~4M files per host.
#
# Run on a host with HF access + ~25 GB free disk (e.g. the existing v6e-8 VM, or
# any box with the repo + uv). Idempotent-ish: re-running rebuilds + re-uploads.
#
# Usage:
#   N=200000 bash scripts/tpu/stage_sweep_subset.sh
#   # -> uploads gs://tinyaya-stage2-tpu/data/sweep-subset-200000.tar.gz
#   #    (pass that as SWEEP_DATA_GS_URI to launch_sweep_fleet.sh)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

N="${N:-200000}"
HF_DATASET="${HF_DATASET:-tiny-aya-translate/tr-hi-mimi-encoded}"
DATA_DIR="${DATA_DIR:-/mnt/data}"
BUCKET="${BUCKET:-tinyaya-stage2-tpu}"
OUT_GS="${OUT_GS:-gs://$BUCKET/data/sweep-subset-${N}.tar.gz}"
STAGE="$DATA_DIR/sweep_stage"
FILELIST="/tmp/sweep_files_${N}.txt"

cd "$REPO_ROOT"

echo "==> [1/5] fetch splits + batch tarballs from $HF_DATASET (idempotent)"
uv run huggingface-cli download "$HF_DATASET" --repo-type dataset --local-dir "$DATA_DIR"

NVAL="${NVAL:-1000}"
echo "==> [2/5] build the $N-sample train + $NVAL-sample val subsets (+ combined file list)"
uv run python scripts/build_sweep_subset.py \
    --input  "$DATA_DIR/splits/train.jsonl" \
    --output "$DATA_DIR/splits/train_sweep_${N}.jsonl" \
    --n "$N" --filelist "${FILELIST}.train"
# A held-out val subset is REQUIRED -- trials validate (val_every) and the dataset
# torch.load()s val .pt directly; without staging them the trial crashes. The
# repo val.jsonl shares the same schema, so the same sampler works.
uv run python scripts/build_sweep_subset.py \
    --input  "$DATA_DIR/splits/val.jsonl" \
    --output "$DATA_DIR/splits/val_sweep_${NVAL}.jsonl" \
    --n "$NVAL" --filelist "${FILELIST}.val"
cat "${FILELIST}.train" "${FILELIST}.val" > "$FILELIST"

echo "==> [3/5] selective-extract the train+val subset files from each batch tarball"
rm -rf "$STAGE"
mkdir -p "$STAGE/encoded" "$STAGE/splits"
for tb in "$DATA_DIR"/mimi_encoded_batch*.tar.gz; do
    echo "    scanning $(basename "$tb")"
    # Members not in THIS tarball (they live in another batch) warn + non-zero
    # exit; the matching ones still extract. Suppress + continue.
    tar -xzf "$tb" -C "$STAGE" -T "$FILELIST" 2>/dev/null || true
done
n_pt=$(find "$STAGE/encoded" -name '*.pt' | wc -l)
echo "    staged $n_pt .pt files (train target $N + val $NVAL)"
if [ "$n_pt" -lt $((N / 2)) ]; then
    echo "ERROR: extracted far fewer files than expected -- check filelist paths" >&2
    exit 1
fi

echo "==> [4/5] place train + val splits, FILTERED to .pt that actually extracted"
# Some split rows reference .pt absent from the published tarballs; the dataset
# torch.load()s directly and would crash on a miss. Keep only rows whose .pt is
# present so BOTH splits are consistent. (NOTE: this corpus ships no text
# alignments, so text loss is 0 -- training is audio-only by design.)
python3 -c "
import json, os
present = {f for f in os.listdir('$STAGE/encoded') if f.endswith('.pt')}
def filt(src, dst, label):
    rows = [json.loads(l) for l in open(src) if l.strip()]
    keep = [r for r in rows if os.path.basename(r['pt_path']) in present]
    open(dst,'w').write(''.join(json.dumps(r)+chr(10) for r in keep))
    print(f'  {label}: {len(rows)} -> {len(keep)} rows')
filt('$DATA_DIR/splits/train_sweep_${N}.jsonl', '$STAGE/splits/train.jsonl', 'train.jsonl')
filt('$DATA_DIR/splits/val_sweep_${NVAL}.jsonl', '$STAGE/splits/val.jsonl', 'val.jsonl')
print('  present .pt:', len(present))
"

echo "==> [5/5] pack + upload to $OUT_GS"
tar -czf /tmp/sweep_subset.tar.gz -C "$STAGE" encoded splits
gcloud storage cp /tmp/sweep_subset.tar.gz "$OUT_GS"
rm -f /tmp/sweep_subset.tar.gz
echo
echo "==> done. SWEEP_DATA_GS_URI=$OUT_GS"
