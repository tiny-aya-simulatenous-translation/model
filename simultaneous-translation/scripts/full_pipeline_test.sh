#!/bin/bash
# Full pipeline validation: train → save → load → resume → save → load
# Must pass ALL steps before launching real training.
set -e

cd ~/tinyaya-stage2-scale/simultaneous-translation
export PYTHONUNBUFFERED=1

CKPT_DIR="checkpoints/pipeline_test"
CONFIG="/tmp/pipeline_test.yaml"

# Minimum file sizes (bytes) for bf16 checkpoint files
MIN_PROJECTION=15000000    # ~16MB
MIN_DEPTH=1000000000       # ~1.3GB
MIN_AUDIO_HEADS=7000000    # ~8MB
MIN_MODEL_AUDIO=7000000    # ~8MB
MIN_TEXT_EMBED=900000000   # ~1GB

check_sizes() {
    local dir=$1
    local label=$2
    echo ""
    echo "=== Checking checkpoint sizes: $label ==="
    local fail=0
    for pair in "projection.pt:$MIN_PROJECTION" "depth_decoder.pt:$MIN_DEPTH" "audio_heads.pt:$MIN_AUDIO_HEADS" "model_audio_embed.pt:$MIN_MODEL_AUDIO" "text_embed.pt:$MIN_TEXT_EMBED"; do
        fname="${pair%%:*}"
        minsize="${pair##*:}"
        fpath="$dir/$fname"
        if [ ! -f "$fpath" ]; then
            echo "  FAIL: $fname MISSING"
            fail=1
            continue
        fi
        actual=$(stat -c%s "$fpath" 2>/dev/null || stat -f%z "$fpath" 2>/dev/null)
        if [ "$actual" -lt "$minsize" ]; then
            echo "  FAIL: $fname = ${actual}B (min ${minsize}B)"
            fail=1
        else
            echo "  OK: $fname = $(echo "scale=1; $actual/1048576" | bc)MB"
        fi
    done
    if [ "$fail" -eq 1 ]; then
        echo "*** CHECKPOINT CORRUPTED ***"
        exit 1
    fi
    echo "  All sizes OK"
}

load_and_forward() {
    local ckpt_dir=$1
    local label=$2
    local output_file=$3
    echo ""
    echo "=== Load + forward: $label ==="
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python -c "
import torch, sys
sys.path.insert(0, '.')
from src.model.composite import TinyAyaMoshiComposite
from src.model.lora_setup import apply_lora
from src.training.checkpointing import load_checkpoint

model = TinyAyaMoshiComposite(num_codebooks=8)
model.backbone = apply_lora(model.backbone, r=16, num_full_ft_layers=0)
load_checkpoint(model, None, None, '$ckpt_dir')
model = model.to('cuda').to(torch.bfloat16).eval()

torch.manual_seed(42)
T = 30
text = torch.full((1, T), 262146, dtype=torch.long, device='cuda')
user = torch.randint(0, 2048, (1, T), device='cuda')
ma = torch.full((1, T), 2048, dtype=torch.long, device='cuda')
mask = torch.ones(1, T, dtype=torch.long, device='cuda')
fc = torch.randint(0, 2048, (1, 8, T), dtype=torch.long, device='cuda')

with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
    tl, al, h = model(text_ids=text, audio_codes=user, model_audio_codes=ma,
                       attention_mask=mask, full_audio_codes=fc, depth_chunk_size=16)

has_nan = torch.isnan(tl).any() or torch.isnan(al).any() or torch.isnan(h).any()
print(f'shapes: tl={tl.shape} al={al.shape} h={h.shape}')
print(f'nan: {has_nan.item()}')
print(f'proj_shape: {model.projection.weight.shape}')

# Save argmax for comparison
torch.save({'tl': tl.argmax(-1).cpu(), 'al': al.argmax(-1).cpu()}, '$output_file')

if has_nan:
    print('FAIL: NaN in output')
    sys.exit(1)
print('PASS: forward OK')
" 2>&1 | tail -6
}

echo "########################################"
echo "# FULL PIPELINE VALIDATION"
echo "########################################"

# Clean slate
rm -rf "$CKPT_DIR"

# Write test config
cat > "$CONFIG" << 'YAML'
data:
  train_split: /home/alperiox/training_data_full/splits/small/train_20.jsonl
  val_split: /home/alperiox/training_data_full/splits/small/val_20.jsonl
  encoded_dir: /home/alperiox/training_data_full/encoded
  max_frames: 300
  audio_frame_rate: 12.5
  num_workers: 0
  pin_memory: true
train:
  num_codebooks: 8
  batch_size: 2
  grad_accum: 1
  max_steps: 50
  warmup_steps: 5
  min_lr_ratio: 0.1
  depth_chunk_size: 16
  precision: bfloat16
  max_grad_norm: 1.0
  weight_decay: 0.01
  adam_beta1: 0.9
  adam_beta2: 0.999
  adam_eps: 1.0e-8
  ss_max_ratio: 0.0
  ss_warmup_steps: 100
loss:
  text_weight: 0.1
  audio_weight: 1.0
optim:
  lr_lora: 3.0e-4
  lr_full_ft: 1.0e-4
  lr_projection: 1.0e-3
  lr_depth: 5.0e-4
  lr_audio_embed: 1.0e-3
  lr_text_embed: 1.0e-3
  lr_model_audio_embed: 1.0e-3
logging:
  log_every: 10
  save_every: 25
  audio_every: 100
  val_every: 25
  save_dir: checkpoints/pipeline_test
  wandb_project: tinyaya-s2s
  wandb_run_name: pipeline_test
  use_wandb: false
  push_to_hub: false
  hub_repo_id: null
YAML

# ============================================================
# PHASE 1: Train 50 steps on 2 GPUs
# ============================================================
echo ""
echo "=== PHASE 1: Training 50 steps on 2 GPUs ==="
.venv/bin/torchrun --nproc_per_node=2 scripts/train_hierarchical.py --config "$CONFIG" 2>&1 | \
    grep -E '^step|val/|best|complete|Training|Error|Traceback'

echo ""
echo "=== PHASE 1 checkpoint dirs ==="
ls -d "$CKPT_DIR"/*/ 2>/dev/null

# ============================================================
# PHASE 2: Verify checkpoint sizes
# ============================================================
check_sizes "$CKPT_DIR/step_000025" "step_000025"
check_sizes "$CKPT_DIR/step_000050" "step_000050"

# ============================================================
# PHASE 3: Load step 50 checkpoint + forward pass
# ============================================================
load_and_forward "$CKPT_DIR/step_000050" "step_000050" "/tmp/out_step50.pt"

# ============================================================
# PHASE 4: Resume from step 50, train to step 100
# ============================================================
echo ""
echo "=== PHASE 4: Resume training from step 50, run to step 100 ==="
# Update max_steps to 100 for resume
sed 's/max_steps: 50/max_steps: 100/' "$CONFIG" > /tmp/pipeline_resume.yaml

.venv/bin/torchrun --nproc_per_node=2 scripts/train_hierarchical.py \
    --config /tmp/pipeline_resume.yaml \
    --resume "$CKPT_DIR/step_000050" 2>&1 | \
    grep -E '^step|val/|best|complete|Training|Resumed|Error|Traceback|size mismatch'

# ============================================================
# PHASE 5: Verify new checkpoints
# ============================================================
echo ""
echo "=== PHASE 5 checkpoint dirs ==="
ls -d "$CKPT_DIR"/*/ 2>/dev/null

if [ -d "$CKPT_DIR/step_000075" ]; then
    check_sizes "$CKPT_DIR/step_000075" "step_000075 (resumed)"
else
    echo "  WARN: step_000075 not found (resume may have started from 0)"
fi

if [ -d "$CKPT_DIR/step_000100" ]; then
    check_sizes "$CKPT_DIR/step_000100" "step_000100 (resumed)"
fi

# ============================================================
# PHASE 6: Load latest checkpoint + forward pass
# ============================================================
LATEST=$(ls -d "$CKPT_DIR"/step_* 2>/dev/null | sort | tail -1)
if [ -n "$LATEST" ]; then
    load_and_forward "$LATEST" "$(basename $LATEST)" "/tmp/out_latest.pt"
fi

# ============================================================
# PHASE 7: Compare step 50 vs latest — should differ
# ============================================================
echo ""
echo "=== PHASE 7: Compare outputs ==="
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -c "
import torch
a = torch.load('/tmp/out_step50.pt')
b = torch.load('/tmp/out_latest.pt')
text_same = (a['tl'] == b['tl']).float().mean().item()
audio_same = (a['al'] == b['al']).float().mean().item()
print(f'Text argmax same:  {text_same*100:.1f}%')
print(f'Audio argmax same: {audio_same*100:.1f}%')
if text_same < 1.0 or audio_same < 1.0:
    print('PASS: outputs differ (training continued)')
else:
    print('WARN: outputs identical (training may not have resumed correctly)')
"

echo ""
echo "########################################"
echo "# ALL PIPELINE TESTS COMPLETE"
echo "########################################"

# Cleanup
rm -rf "$CKPT_DIR"
