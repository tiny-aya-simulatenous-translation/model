#!/bin/bash
# Full pipeline validation with optimizer state verification.
# Tests: train → save → verify (incl optimizer) → resume → verify → resume → verify
set -e

cd ~/tinyaya-stage2-scale/simultaneous-translation
export PYTHONUNBUFFERED=1

CKPT_DIR="checkpoints/validate_pipeline"
CONFIG="/tmp/validate_pipeline.yaml"

# Minimum file sizes (bytes) for bf16 checkpoint files
MIN_PROJ=15000000
MIN_DEPTH=1000000000
MIN_AUDIO=7000000
MIN_MAE=7000000
MIN_TEXT=900000000
MIN_OPT=100000000

check_file() {
    local fpath=$1
    local min_size=$2
    local label=$3
    if [ ! -f "$fpath" ]; then
        echo "    FAIL: $label MISSING"
        return 1
    fi
    local actual
    actual=$(stat -c%s "$fpath" 2>/dev/null || stat -f%z "$fpath" 2>/dev/null)
    if [ "$actual" -lt "$min_size" ]; then
        echo "    FAIL: $label = ${actual}B (min ${min_size}B)"
        return 1
    fi
    echo "    OK: $label = $((actual / 1048576))MB"
    return 0
}

verify_checkpoint() {
    local dir=$1
    local label=$2
    local fail=0
    echo ""
    echo "  --- Verifying: $label ---"
    check_file "$dir/projection.pt" $MIN_PROJ "projection.pt" || fail=1
    check_file "$dir/depth_decoder.pt" $MIN_DEPTH "depth_decoder.pt" || fail=1
    check_file "$dir/audio_heads.pt" $MIN_AUDIO "audio_heads.pt" || fail=1
    check_file "$dir/model_audio_embed.pt" $MIN_MAE "model_audio_embed.pt" || fail=1
    check_file "$dir/text_embed.pt" $MIN_TEXT "text_embed.pt" || fail=1
    check_file "$dir/optimizer.pt" $MIN_OPT "optimizer.pt" || fail=1
    check_file "$dir/peft_adapter/adapter_model.safetensors" 1000 "peft_adapter" || fail=1

    # Verify optimizer state has momentum
    if [ -f "$dir/optimizer.pt" ]; then
        .venv/bin/python -c "
import torch, sys
sd = torch.load('$dir/optimizer.pt', map_location='cpu', weights_only=True)
n_groups = len(sd.get('param_groups', []))
n_states = len(sd.get('state', {}))
has_mom = any('exp_avg' in v for v in sd.get('state', {}).values() if isinstance(v, dict))
print(f'    Optimizer: {n_groups} groups, {n_states} states, momentum={\"yes\" if has_mom else \"NO\"}')
if n_states == 0 or not has_mom:
    sys.exit(1)
"
        if [ $? -ne 0 ]; then
            echo "    FAIL: optimizer state corrupt or missing momentum"
            fail=1
        fi
    fi

    # Verify metadata
    if [ -f "$dir/metadata.json" ]; then
        step=$(.venv/bin/python -c "import json; print(json.load(open('$dir/metadata.json'))['step'])")
        echo "    Metadata: step=$step"
    fi

    if [ "$fail" -eq 1 ]; then
        echo "    *** CHECKPOINT VERIFICATION FAILED ***"
        return 1
    fi
    echo "    ALL OK"
    return 0
}

load_and_forward() {
    local ckpt=$1
    local label=$2
    local out_file=$3
    echo ""
    echo "  --- Load + Forward: $label ---"
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python -c "
import torch, sys
sys.path.insert(0, '.')
from src.model.composite import TinyAyaMoshiComposite
from src.model.lora_setup import apply_lora
from src.training.checkpointing import load_checkpoint

model = TinyAyaMoshiComposite(num_codebooks=8)
model.backbone = apply_lora(model.backbone, r=16, num_full_ft_layers=0)
load_checkpoint(model, None, None, '$ckpt')
model = model.to('cuda').to(torch.bfloat16).eval()

torch.manual_seed(42)
T = 20
text = torch.full((1, T), 262146, dtype=torch.long, device='cuda')
user = torch.randint(0, 2048, (1, T), device='cuda')
ma = torch.full((1, T), 2048, dtype=torch.long, device='cuda')
mask = torch.ones(1, T, dtype=torch.long, device='cuda')
fc = torch.randint(0, 2048, (1, 8, T), dtype=torch.long, device='cuda')

with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
    tl, al, h = model(text_ids=text, audio_codes=user, model_audio_codes=ma,
                       attention_mask=mask, full_audio_codes=fc, depth_chunk_size=16)

has_nan = torch.isnan(tl).any() or torch.isnan(al).any() or torch.isnan(h).any()
h_val = tl.argmax(-1).sum().item()
print(f'    shapes: tl={tl.shape} al={al.shape} h={h.shape}')
print(f'    NaN: {has_nan.item()}, hash: {h_val}')
torch.save({'hash': h_val}, '$out_file')
if has_nan:
    sys.exit(1)
print('    PASS')
" 2>&1 | grep -E '    |Error|Traceback'
}

echo "########################################"
echo "# FULL PIPELINE VALIDATION"
echo "########################################"

# Clean
rm -rf "$CKPT_DIR"

# Write config: 10 steps, save every 5
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
  max_steps: 10
  warmup_steps: 2
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
  log_every: 5
  save_every: 5
  audio_every: 999
  val_every: 999
  save_dir: checkpoints/validate_pipeline
  wandb_project: tinyaya-s2s
  wandb_run_name: validate_pipeline
  use_wandb: false
  push_to_hub: false
  hub_repo_id: null
YAML

# ============================================================
# PHASE 1: Train 10 steps
# ============================================================
echo ""
echo "========== PHASE 1: Train 10 steps (saves at 5, 10) =========="
.venv/bin/torchrun --nproc_per_node=2 scripts/train_hierarchical.py --config "$CONFIG" 2>&1 | \
    grep -E '^step|complete|Training|Error|Traceback'

echo ""
echo "Checkpoints created:"
ls -d "$CKPT_DIR"/*/ 2>/dev/null

verify_checkpoint "$CKPT_DIR/step_000005" "step_005"
PHASE1_A=$?
verify_checkpoint "$CKPT_DIR/step_000010" "step_010"
PHASE1_B=$?
load_and_forward "$CKPT_DIR/step_000010" "step_010" "/tmp/hash_step10.pt"

# ============================================================
# PHASE 2: Resume from step 5, train to 15
# ============================================================
echo ""
echo "========== PHASE 2: Resume from step 5 → step 15 =========="
sed 's/max_steps: 10/max_steps: 15/' "$CONFIG" > /tmp/validate_resume.yaml
.venv/bin/torchrun --nproc_per_node=2 scripts/train_hierarchical.py \
    --config /tmp/validate_resume.yaml \
    --resume "$CKPT_DIR/step_000005" 2>&1 | \
    grep -E '^step|Loaded|Resuming|complete|Training|Error|Traceback|size mismatch'

echo ""
echo "Checkpoints after resume:"
ls -d "$CKPT_DIR"/*/ 2>/dev/null

# Verify the new checkpoints
if [ -d "$CKPT_DIR/step_000010" ]; then
    verify_checkpoint "$CKPT_DIR/step_000010" "step_010 (after resume from 5)"
fi
if [ -d "$CKPT_DIR/step_000015" ]; then
    verify_checkpoint "$CKPT_DIR/step_000015" "step_015 (after resume from 5)"
    load_and_forward "$CKPT_DIR/step_000015" "step_015" "/tmp/hash_step15.pt"
fi

# ============================================================
# PHASE 3: Resume from step 10 (original), train to 20
# ============================================================
echo ""
echo "========== PHASE 3: Resume from step 10 → step 20 =========="
sed 's/max_steps: 10/max_steps: 20/' "$CONFIG" > /tmp/validate_resume2.yaml
.venv/bin/torchrun --nproc_per_node=2 scripts/train_hierarchical.py \
    --config /tmp/validate_resume2.yaml \
    --resume "$CKPT_DIR/step_000010" 2>&1 | \
    grep -E '^step|Loaded|Resuming|complete|Training|Error|Traceback|size mismatch'

echo ""
echo "Checkpoints after second resume:"
ls -d "$CKPT_DIR"/*/ 2>/dev/null

if [ -d "$CKPT_DIR/step_000020" ]; then
    verify_checkpoint "$CKPT_DIR/step_000020" "step_020 (after resume from 10)"
    load_and_forward "$CKPT_DIR/step_000020" "step_020" "/tmp/hash_step20.pt"
fi

# ============================================================
# PHASE 4: Compare outputs — should all differ
# ============================================================
echo ""
echo "========== PHASE 4: Compare outputs =========="
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -c "
import torch
files = {
    'step10': '/tmp/hash_step10.pt',
    'step15': '/tmp/hash_step15.pt',
    'step20': '/tmp/hash_step20.pt',
}
hashes = {}
for name, path in files.items():
    try:
        hashes[name] = torch.load(path)['hash']
        print(f'  {name}: hash={hashes[name]}')
    except:
        print(f'  {name}: not available')

if len(hashes) >= 2:
    vals = list(hashes.values())
    all_same = all(v == vals[0] for v in vals)
    if all_same:
        print('  WARN: all outputs identical')
    else:
        print('  PASS: outputs differ (training continued correctly)')
"

# ============================================================
echo ""
echo "########################################"
echo "# FULL PIPELINE VALIDATION COMPLETE"
echo "########################################"

# Cleanup
rm -rf "$CKPT_DIR"
echo "Cleaned up"
