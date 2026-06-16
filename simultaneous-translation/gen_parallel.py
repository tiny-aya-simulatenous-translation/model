"""Generate audio with parallel two-stream Moshi architecture + codebook delay."""
import sys, torch
sys.path.insert(0, ".")

from src.data.dataset import StreamingTranslationDataset, undo_codebook_delay, SILENCE_TOKEN
from src.model.composite import TinyAyaMoshiComposite
from src.model.lora_setup import apply_lora
from src.training.checkpointing import load_checkpoint
from src.data.mimi_encoder import MimiEncoder
from transformers import AutoTokenizer
import soundfile as sf, os

CKPT = "checkpoints/overfit_parallel/step_000300"
SPLIT = "/home/alperiox/training_data_full/splits/small/train_20.jsonl"
ENCODED = "/home/alperiox/training_data_full/encoded"
OUT_DIR = "/home/alperiox/audio_parallel_v2"

tokenizer = AutoTokenizer.from_pretrained("CohereLabs/tiny-aya-base", trust_remote_code=True)
model = TinyAyaMoshiComposite(num_codebooks=8)
model.backbone = apply_lora(model.backbone, r=16, num_full_ft_layers=0)
load_checkpoint(model, None, None, CKPT)
model = model.to("cuda").to(torch.bfloat16)
model.eval()

ds = StreamingTranslationDataset(SPLIT, tokenizer, max_frames=300, encoded_dir=ENCODED)
mimi = MimiEncoder(device="cuda")
device = "cuda"

sample = ds[0]
user_codes = sample["user_audio_codes"]    # [8, T] — source then silence (no delay)
model_codes = sample["model_audio_codes"]  # [8, T] — silence then target (WITH delay)
src_len = sample["source_length"]
tgt_len = sample["target_length"]
text_ids = sample["text_ids"]
T = text_ids.shape[0]

# Also get raw audio_codes for ground truth source/target decoding
raw_audio = sample["audio_codes"]  # [8, T] — [src | tgt] concatenated, no delay

print(f"Sample: src={src_len} tgt={tgt_len} total={T}", flush=True)

os.makedirs(f"{OUT_DIR}/teacher_forced", exist_ok=True)
os.makedirs(f"{OUT_DIR}/moshi_style", exist_ok=True)

# === Decode source and ground truth target ===
print("\nDecoding source and ground truth...", flush=True)
src_codes_raw = raw_audio[:, :src_len]
tgt_codes_raw = raw_audio[:, src_len:]
sf.write(f"{OUT_DIR}/source.wav", mimi.decode(src_codes_raw).numpy(), 24000)
sf.write(f"{OUT_DIR}/target_gt.wav", mimi.decode(tgt_codes_raw).numpy(), 24000)
print(f"  Source: {src_codes_raw.shape}, Target GT: {tgt_codes_raw.shape}", flush=True)

# === 1. Teacher-forced ===
print("\n=== Teacher-forced decode ===", flush=True)
user_cb0 = user_codes[0, :].unsqueeze(0).to(device)
model_cb0 = model_codes[0, :].unsqueeze(0).to(device)
text = text_ids.unsqueeze(0).to(device)
mask = torch.ones(1, T, dtype=torch.long, device=device)
full_model = model_codes.unsqueeze(0).to(device)

with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    output = model(
        text_ids=text, audio_codes=user_cb0, model_audio_codes=model_cb0,
        attention_mask=mask, full_audio_codes=full_model[:, :8, :], depth_chunk_size=16,
    )
    text_logits, audio_logits, hidden = output

# Predictions for target positions (next-token: logits at t predict t+1)
pred_codes_delayed = audio_logits[0, :, src_len-1:-1, :].argmax(dim=-1)  # [8, tgt_len]
gt_codes_delayed = model_codes[:, src_len:src_len + pred_codes_delayed.shape[1]]

for cb in range(8):
    acc = (pred_codes_delayed[cb].cpu() == gt_codes_delayed[cb]).float().mean().item()
    print(f"  CB{cb}: {acc*100:.1f}%", flush=True)
print(f"  Overall: {(pred_codes_delayed.cpu() == gt_codes_delayed).float().mean().item()*100:.1f}%", flush=True)

# Next-token predictions on delayed targets already produce un-delayed values.
# (pred at t for CB_k is the original CB_k at frame t, not the delayed value)
# Just clamp any SILENCE_TOKEN padding to valid Mimi range.
pred_codes_clean = pred_codes_delayed.cpu().clamp(max=2047)
sf.write(f"{OUT_DIR}/teacher_forced/predicted.wav", mimi.decode(pred_codes_clean.to(device)).numpy(), 24000)
print("Saved teacher_forced/", flush=True)

# === 2. Autoregressive (parallel two-stream) ===
print("\n=== Autoregressive decode (two-stream, temp=0.8, top-p=0.9) ===", flush=True)
TEMP = 0.8

# User stream: full (pre-known)
user_stream = user_codes[0, :].unsqueeze(0).to(device)  # [1, T]

# Model stream: silence initially, fill in as we generate
model_stream = torch.full((1, T), SILENCE_TOKEN, dtype=torch.long, device=device)

# Text: full sequence (pre-known)
ar_text = text_ids.unsqueeze(0).to(device)

# Store all 8 codebooks for generated frames
gen_all_codes = torch.full((8, T), SILENCE_TOKEN, dtype=torch.long, device=device)

for t in range(src_len, T):
    ar_mask = torch.ones(1, t + 1, dtype=torch.long, device=device)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        backbone_out = model.backbone(
            text_ids=ar_text[:, :t+1],
            audio_codes=user_stream[:, :t+1],
            model_audio_codes=model_stream[:, :t+1],
            attention_mask=ar_mask,
        )
        hidden = backbone_out["hidden_states"]

        # CB0 from backbone
        cb0_logits = model.backbone.audio_heads[0](hidden[:, -1:, :]).squeeze()
        cb0_probs = torch.softmax(cb0_logits.float() / TEMP, dim=-1)
        sorted_p, sorted_i = torch.sort(cb0_probs, descending=True)
        cum = torch.cumsum(sorted_p, dim=-1)
        mask_p = cum - sorted_p > 0.9
        sorted_p[mask_p] = 0.0
        sorted_p = sorted_p / sorted_p.sum()
        cb0_tok = sorted_i[torch.multinomial(sorted_p, 1)].squeeze(-1)

        # Update model stream with generated cb0
        model_stream[0, t] = cb0_tok
        gen_all_codes[0, t] = cb0_tok

        # CB1-7 from depth decoder
        projected = model.projection(hidden[:, -1:, :])
        ctx_expanded = projected.expand(1, model.num_codebooks, -1).contiguous()
        depth_input = torch.zeros(1, model.num_codebooks, dtype=torch.long, device=device)
        depth_input[0, 1] = cb0_tok

        depth_out = model.depth_decoder(
            input_ids=depth_input, last_hidden_state=ctx_expanded,
            use_cache=False, return_dict=True,
        )

        for cb_idx in range(7):
            logits = depth_out.logits[0, cb_idx + 1, :].float()
            probs = torch.softmax(logits / TEMP, dim=-1)
            sorted_p2, sorted_i2 = torch.sort(probs, descending=True)
            cum2 = torch.cumsum(sorted_p2, dim=-1)
            mask_p2 = cum2 - sorted_p2 > 0.9
            sorted_p2[mask_p2] = 0.0
            sorted_p2 = sorted_p2 / sorted_p2.sum()
            tok = sorted_i2[torch.multinomial(sorted_p2, 1)].squeeze(-1)
            gen_all_codes[cb_idx + 1, t] = tok
            if cb_idx + 2 < model.num_codebooks:
                depth_input[0, cb_idx + 2] = tok

    if (t - src_len) % 20 == 0:
        print(f"  Frame {t-src_len}/{tgt_len}", flush=True)

# AR loop generates each frame's codebooks directly (not delayed).
# The depth decoder's teacher-forced input during AR uses the model_stream
# which accumulates CB0 tokens, but the depth decoder outputs are direct
# predictions for the current frame's CB1-7.
gen_target = gen_all_codes[:, src_len:T]

valid_count = (gen_target[0] != SILENCE_TOKEN).sum().item()
print(f"\nGenerated: {gen_target.shape[1]} frames, {valid_count} non-silence CB0", flush=True)
print(f"CB0 unique tokens: {len(set(gen_target[0].cpu().tolist()))}", flush=True)

# Clamp SILENCE_TOKEN to valid Mimi range
gen_clean = gen_target.cpu().clamp(max=2047)
sf.write(f"{OUT_DIR}/moshi_style/generated.wav", mimi.decode(gen_clean.to(device)).numpy(), 24000)
print(f"Saved moshi_style/", flush=True)
print("Done!", flush=True)
