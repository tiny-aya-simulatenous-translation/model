"""Test a full training step (forward + loss + backward + optimizer) on TPU."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Patch checkpoint for XLA compatibility
import torch.utils.checkpoint as cp
_orig_ckpt = cp.checkpoint
def _noop_checkpoint(fn, *args, use_reentrant=False, **kwargs):
    return fn(*args)
cp.checkpoint = _noop_checkpoint

import torch
from src.backend import get_backend
from src.model.composite import TinyAyaMoshiComposite
from src.model.lora_setup import apply_lora
from src.training.translation_loss import compute_hierarchical_translation_loss

print("1-init backend", flush=True)
backend = get_backend()
backend.init_distributed()
device = backend.get_device()
print(f"2-Device: {device}", flush=True)

print("3-building model", flush=True)
model = TinyAyaMoshiComposite(num_codebooks=8)
model.backbone = apply_lora(model.backbone, r=16)

# Freeze depth internals (same as training script)
for name, param in model.depth_decoder.named_parameters():
    if any(k in name for k in ("input_projections", "embed_tokens", "lm_heads")):
        param.requires_grad = True
    else:
        param.requires_grad = False

model = model.to(device)
model = backend.wrap_model(model)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"4-Trainable: {trainable/1e6:.0f}M", flush=True)

# Optimizer
param_groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": 1e-4}]
optimizer = torch.optim.AdamW(param_groups)

# Dummy batch
B, T, CB = 1, 10, 8
text_ids = torch.zeros(B, T, dtype=torch.long, device=device)
audio_codes = torch.zeros(B, T, dtype=torch.long, device=device)
mask = torch.ones(B, T, dtype=torch.long, device=device)
loss_mask = torch.ones(B, T, dtype=torch.long, device=device)
full_codes = torch.zeros(B, CB, T, dtype=torch.long, device=device)

print("5-forward pass", flush=True)
model.train()
optimizer.zero_grad()
out = model(
    text_ids=text_ids,
    audio_codes=audio_codes,
    attention_mask=mask,
    full_audio_codes=full_codes,
)
tl = out["text_logits"]
al = out["audio_logits"]
print(f"6-output shapes: text={tl.shape}, audio={al.shape}", flush=True)

print("7-computing loss", flush=True)
losses = compute_hierarchical_translation_loss(
    tl, al,
    text_ids, full_codes, mask, loss_mask,
    text_weight=0.1, audio_weight=1.0,
)
loss_val = losses["loss"]
print(f"8-loss: {loss_val.item():.4f}", flush=True)

print("9-backward", flush=True)
loss_val.backward()
print("10-backward OK", flush=True)

print("11-grad norm", flush=True)
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
print(f"12-grad_norm: {grad_norm}", flush=True)

print("13-optimizer step", flush=True)
backend.optimizer_step(optimizer)
optimizer.zero_grad()
print("14-optimizer step OK", flush=True)

backend.sync()
print("15-FULL TRAINING STEP ON TPU OK!", flush=True)
