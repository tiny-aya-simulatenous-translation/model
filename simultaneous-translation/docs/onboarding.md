# Onboarding Guide: TinyAya Simultaneous Translation

This document explains everything you need to understand the training pipeline for TR<->HI speech-to-speech translation. It covers the architecture, data flow, code structure, and how each component works — with file paths and line references throughout.

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [Architecture Overview](#2-architecture-overview)
3. [Repository Structure](#3-repository-structure)
4. [Data Pipeline](#4-data-pipeline)
5. [Model Architecture](#5-model-architecture)
6. [Training Pipeline](#6-training-pipeline)
7. [Inference Pipeline](#7-inference-pipeline)
8. [Configuration System](#8-configuration-system)
9. [Running the Code](#9-running-the-code)
10. [Key Constants and Magic Numbers](#10-key-constants-and-magic-numbers)

---

## 1. What This Project Does

This project trains a model that translates **spoken Turkish to spoken Hindi** (and vice versa) — directly, without intermediate text. The model:

1. Takes Turkish audio as input
2. Encodes it into discrete tokens using a neural audio codec (Mimi)
3. Processes the tokens through a language model backbone (TinyAya, 8B params)
4. Generates Hindi audio tokens using a depth decoder (from Moshiko)
5. Decodes the tokens back to Hindi audio using Mimi

The key insight is that speech translation becomes a **sequence-to-sequence problem in token space** — just like text translation, but with audio tokens instead of word tokens.

---

## 2. Architecture Overview

```
Source Audio (Turkish)
    |
    v
Mimi Codec Encode (frozen, 24kHz, 12.5 Hz frame rate)
    |
    v
Audio tokens [8 codebooks, T_src] --- only codebook 0 goes to backbone
    |
    v
+-- text_embed(text_ids) ----+
|                            |  SUM (Moshi-style interleaving)
+-- audio_embed(cb0_codes) --+
    |
    v
TinyAya Backbone (8B Cohere2, LoRA r=16 on layers 0-33, full FT on 34-35)
    |
    v
Hidden states [B, T, 2048]
    |
    +----> LM Head -> text_logits [B, T, 264196]
    |
    v
Projection Linear(2048, 4096)
    |
    v
Moshi Depth Decoder (6 layers, from Moshiko pretrained weights)
    |  autoregressive: CB0 -> CB1 -> ... -> CB7
    v
Audio logits [B, 8, T, 2048]
    |
    v  (argmax during inference)
Target audio tokens [8, T_tgt]
    |
    v
Mimi Codec Decode
    |
    v
Target Audio (Hindi)
```

**Why this architecture?**

- **TinyAya** is a multilingual text LM with native Turkish and Hindi support. We repurpose it for audio by extending its embedding table with audio tokens.
- **Moshi depth decoder** generates codebooks hierarchically (CB0->CB1->...->CB7) — each codebook is conditioned on all previous ones. This produces much cleaner audio than predicting codebooks independently.
- **Mimi codec** is frozen — we don't fine-tune it. It just converts between audio waveforms and discrete tokens. Both the original research (Hibiki, J-Moshi, etc.) and our own experiments showed that fine-tuning codecs degrades quality.

---

## 3. Repository Structure

```
simultaneous-translation/
|
+-- src/
|   +-- backend/                    # GPU/TPU abstraction layer
|   |   +-- __init__.py             # detect_backend(), get_backend()
|   |   +-- base.py                 # BackendBase ABC (14 abstract methods)
|   |   +-- gpu_backend.py          # GPU/DDP implementation
|   |   +-- tpu_backend.py          # TPU/SPMD+FSDPv2 implementation
|   |
|   +-- model/
|   |   +-- backbone.py             # TinyAyaBackbone: text+audio embedding, Cohere2 transformer
|   |   +-- composite.py            # TinyAyaMoshiComposite: backbone + projection + depth decoder
|   |   +-- depth_decoder.py        # Creates MoshiDepthDecoder from extracted weights
|   |   +-- surgery.py              # Extracts depth decoder weights from Moshiko checkpoint
|   |   +-- lora_setup.py           # LoRA application, parameter groups, embedding adapters
|   |
|   +-- data/
|   |   +-- dataset.py              # Dataset classes (InterleavedAudio, Translation, Streaming)
|   |   +-- collator.py             # Batches variable-length samples with padding
|   |   +-- interleaver.py          # Maps Whisper timestamps to text tokens at audio frame positions
|   |   +-- mimi_encoder.py         # Mimi codec wrapper (encode wav -> tokens, decode tokens -> wav)
|   |
|   +-- training/
|       +-- translation_loss.py     # Hierarchical loss: text CE + per-codebook audio CE
|       +-- scheduler.py            # Warmup cosine LR scheduler
|       +-- checkpointing.py        # Save/load PEFT + projection + depth decoder checkpoints
|       +-- loss.py                 # Older loss function (used by Stage 1, kept for compat)
|       +-- trainer.py              # Generic trainer class (used by Stage 1, kept for compat)
|
+-- scripts/
|   +-- train_hierarchical.py       # Main training script (Stage 2, ~670 lines)
|   +-- make_splits.py              # Create train/val splits keyed on sentence_id
|   +-- infer_only.py               # Inference: translate a WAV file
|   +-- prepare_data.py             # Download FLEURS + Whisper annotate + Mimi encode
|   +-- upload_encoded_dataset.py   # Push encoded data to HuggingFace
|   +-- generate_demos.py           # Generate showcase samples from FLEURS
|   +-- eval_stage2.py              # Evaluation: ASR-BLEU + DNSMOS on generated audio
|   +-- test_tpu_training_step.py   # TPU verification script
|
+-- configs/
|   +-- stage2_scale.yaml           # GPU training config
|   +-- stage2_tpu.yaml             # TPU training config
|
+-- docs/
    +-- tpu-changes.md              # Detailed TPU adaptation guide
    +-- onboarding.md               # This file
```

---

## 4. Data Pipeline

The pipeline goes: **raw audio -> Mimi encode -> Whisper align -> train/val split -> DataLoader**.

### 4.1 Encoding Audio with Mimi

**File:** `src/data/mimi_encoder.py`

Mimi converts audio waveforms into discrete tokens. It produces **32 codebooks** at **12.5 Hz** (one frame per 80ms). We only use the first 8 codebooks — Mimi's quantizer dropout during pretraining ensures these 8 carry enough information.

```python
# src/data/mimi_encoder.py:20-49
encoder = MimiEncoder(device="cuda")       # loads kyutai/mimi
codes = encoder.encode(audio, sr=16000)    # -> [32, T] tensor of ints (0..2047)
audio = encoder.decode(codes[:8])          # -> [num_samples] waveform at 24kHz
```

The encode method (line 20) resamples to 24kHz if needed, using `scipy.signal.resample_poly` (avoids torchaudio CUDA issues). The decode method (line 52) handles partial codebooks — you can pass only 8 of 32 and Mimi reconstructs correctly.

**Output:** `.pt` files containing `{"source_audio_codes": [32, T_src], "target_audio_codes": [32, T_tgt], ...}`

### 4.2 Text-Audio Alignment (Interleaving)

**File:** `src/data/interleaver.py`

This is the Moshi-style "inner monologue". At each audio frame, we place a text token that tells the model what word is being spoken at that moment.

```
Frame:     0    1    2    3    4    5    6    7    8    9
Audio:    [a0] [a1] [a2] [a3] [a4] [a5] [a6] [a7] [a8] [a9]
Text:     "mer" "ha" "ba"  PAD  PAD "dun" "ya"  PAD  PAD  PAD
                                    ^word start
```

The Interleaver (line 29) takes Whisper word-level timestamps and maps them to audio frames:

1. **Tokenize** each word individually (`_tokenize()`, line 55)
2. **Filter** words with no duration (`_keep_those_with_duration()`, line 64)
3. **Build token stream** (`build_token_stream()`, line 67): for each frame, determine which text token appears based on timestamps

**Special tokens** (lines 20-23):
| Token ID | Name | Meaning |
|----------|------|---------|
| 262144 | TEXT_PADDING | Frame between words (no word active) |
| 262145 | END_OF_TEXT_PADDING | Marks the end of a word's tokens |
| 262146 | ZERO_PADDING | No text at all (silence, or target in translation) |
| 262147 | IN_WORD_PADDING | Multi-frame words (continuation frames) |

**Whisper alignment format** (loaded by `load_alignments()`, line 134):
```json
{"alignments": [["merhaba", [0.0, 0.5], "SPEAKER"], ["dunya", [0.8, 1.2], "SPEAKER"]]}
```

### 4.3 Creating Train/Val Splits

**File:** `scripts/make_splits.py`

The critical constraint: **no sentence ID leakage between splits**. Since we have multiple TTS variants of the same FLEURS sentence (Kokoro x4, XTTSv2 x2, Chatterbox x2), all variants of sentence #42 must be in the same split. Otherwise the model memorizes specific sentences from the training set and appears to generalize on validation, when it's actually just recognizing familiar content.

```bash
python scripts/make_splits.py \
    --accepted data/manifests/accepted.jsonl \
    --encoded-dir data/encoded/ \
    --out-dir data/splits/ \
    --val-frac 0.10
```

The script (line 74-81):
1. Extracts unique `sentence_id` from each `pair_id` (regex: `fleurs_(\d+)`)
2. Shuffles sentence IDs with a fixed seed (42)
3. Assigns first 10% of IDs to validation, rest to training
4. All TTS variants of a given sentence follow the sentence

**Output:** `train.jsonl` and `val.jsonl`, each row containing:
```json
{
  "pt_path": "fleurs_42_kokoro_hf_alpha_trhi.pt",
  "src_align_path": "fleurs_42_trhi.src.alignments.json",
  "tgt_align_path": "fleurs_42_trhi.tgt.alignments.json",
  "direction": "tr->hi",
  "sentence_id": 42,
  "source_type": "fleurs_tts",
  "tts_model": "kokoro",
  "tts_voice": "hf_alpha"
}
```

### 4.4 Dataset Classes

**File:** `src/data/dataset.py`

There are three dataset classes (in order of development):

1. **InterleavedAudioDataset** (line 13) — Stage 1 monolingual. Not used in Stage 2.

2. **TranslationDataset** (line 74) — Original Stage 2. Loads from a directory of `.pt` files. Used for 50-pair overfitting tests (`--dataset_mode memory`).

3. **StreamingTranslationDataset** (line 159) — Current Stage 2. Reads from JSONL split files created by `make_splits.py`. This is what you'll use (`--dataset_mode streaming`).

**StreamingTranslationDataset details:**

Constructor (line 159): takes `jsonl_path`, `tokenizer`, `max_frames` (default 300), `audio_frame_rate` (12.5), `encoded_dir`.

`__getitem__` (line 208):
1. Load `.pt` file containing `source_audio_codes` and `target_audio_codes` (each `[32, T]`)
2. Load alignment JSONs for source and target
3. **Truncate** if total frames exceed `max_frames` (line 232-244):
   - Prioritize preserving target (that's what we're training to generate)
   - But always keep at least half the frames for source (model needs context)
4. **Concatenate** source and target codes: `[32, T_src + T_tgt]`
5. **Build text IDs** using the Interleaver:
   - Source portion: actual text alignment (from Whisper timestamps)
   - Target portion: actual text alignment (from Whisper timestamps)
6. **Create loss mask**: `[T_total]` where 0 for source frames, 1 for target frames
7. Return dict with `audio_codes`, `text_ids`, `loss_mask`, `num_frames`, `source_length`, `target_length`

### 4.5 Collation (Batching)

**File:** `src/data/collator.py`

**InterleavedCollator** (line 9) pads variable-length samples to the maximum length in each batch:

- `audio_codes`: pad with 0 -> `[B, CB, T_max]`
- `text_ids`: pad with ZERO_PADDING (262146) -> `[B, T_max]`
- `attention_mask`: 1 for real, 0 for padding -> `[B, T_max]`
- `loss_mask`: preserved from dataset, padded with 0 -> `[B, T_max]`

### 4.6 Full Data Flow

```
FLEURS wav files
    |  scripts/prepare_data.py (Whisper + Mimi)
    v
.pt files (audio codes) + .json files (word alignments)
    |  scripts/make_splits.py
    v
train.jsonl + val.jsonl (split manifest)
    |  StreamingTranslationDataset.__getitem__()
    v
{audio_codes: [32, T], text_ids: [T], loss_mask: [T]}
    |  InterleavedCollator.__call__()
    v
{audio_codes: [B, 32, T_max], text_ids: [B, T_max], attention_mask: [B, T_max], loss_mask: [B, T_max]}
    |  train_hierarchical.py training loop
    v
Forward pass -> loss -> backward -> optimizer step
```

---

## 5. Model Architecture

### 5.1 TinyAya Backbone

**File:** `src/model/backbone.py`

The backbone is **CohereLabs/tiny-aya-base** — a 8B parameter multilingual Cohere2 LM loaded in bf16.

**Embedding table extension** (lines 46-58):
The original vocab has 262,144 text tokens. We extend it to 264,196:
```
[0..262143]     = original text tokens
[262144..262147] = 4 special tokens (padding variants)
[262148..264195] = 2,048 audio tokens (Mimi codes for codebook 0)
```

The audio token offset (262148) is added to raw audio codes before embedding lookup:
```python
# backbone.py:95
audio_token_ids = audio_codes + self.audio_token_offset  # 0..2047 -> 262148..264195
audio_embeds = self.get_input_embeddings()(audio_token_ids)
```

**Dual embedding (Moshi-style)** (lines 60-69):
The model sums two embeddings at each frame:
```python
# backbone.py:102
combined = audio_embeds + text_embeds  # [B, T, 2048]
```
- `audio_embeds` come from the backbone's extended embedding table
- `text_embeds` come from a separate `nn.Embedding(262148, 2048)` initialized from the backbone's own weights

This is the "inner monologue" from Moshi — the model processes both what it hears (audio) and what's being said (text) at each frame.

**Forward pass** (line 81):
```
Input: text_ids [B, T], audio_codes [B, T], attention_mask [B, T]
  -> audio_embed + text_embed = combined [B, T, 2048]
  -> backbone transformer (36 Cohere2 layers) = hidden_states [B, T, 2048]
  -> lm_head = text_logits [B, T, 264196]
  -> audio_heads = audio_logits [B, 32, T, 2048]
```

### 5.2 LoRA and Parameter Groups

**File:** `src/model/lora_setup.py`

We don't fine-tune all 8B parameters. LoRA adapts the model with ~26% trainable params:

**apply_lora()** (line 31):
- **Layers 0-33**: LoRA r=16 on `q_proj` and `v_proj` (low-rank adapters)
- **Layers 34-35**: Full fine-tuning (all parameters trainable)
- **text_embed**: Wrapped with `LoRAEmbedding` (line 8) — frozen base + trainable low-rank adapter
- **audio_heads**: Always trainable (they predict audio tokens)
- **embed_tokens**: LoRA adapter on the extended embedding table

**Parameter groups** (`get_param_groups()`, line 81):
Different learning rates for different components:
```
lora adapters:      lr = 1.5e-4  (most params, stable updates)
full FT layers:     lr = 5e-5    (last 2 layers, small LR to avoid catastrophic forgetting)
projection:         lr = 5e-4    (new layer, can learn faster)
depth decoder I/O:  lr = 2.5e-4  (partially pretrained from Moshiko)
text/audio embed:   lr = 5e-4    (embedding adapters)
```

### 5.3 Composite Model

**File:** `src/model/composite.py`

**TinyAyaMoshiComposite** (line 32) assembles three components:

1. **Backbone** — TinyAyaBackbone (handles text + audio embedding and transformer forward)
2. **Projection** — `nn.Linear(2048, 4096)` mapping backbone hidden states to depth decoder input space. Xavier init. (`surgery.py:30-35`)
3. **Depth Decoder** — MoshiDepthDecoder from Moshiko pretrained weights

**Forward pass** (line 66):
```python
# Step 1: Backbone
backbone_out = self.backbone(text_ids, audio_codes, attention_mask)
hidden_states = backbone_out["hidden_states"]   # [B, T, 2048]
text_logits = backbone_out["text_logits"]        # [B, T, 264196]

# Step 2: Project
projected = self.projection(hidden_states)       # [B, T, 4096]

# Step 3: Depth decoder (chunked for memory)
for t_start in range(0, T, depth_chunk_size):    # default chunk=16
    ctx_chunk = projected[:, t_start:t_end, :]
    # Reshape for depth decoder: [B*chunk, num_codebooks, 4096]
    # Teacher-force: position 1..N-1 gets ground truth codebooks 0..N-2
    chunk_logits = _checkpoint(depth_forward, depth_input, ctx_expanded)

audio_logits = cat(chunks) -> [B, 8, T, 2048]
```

The depth decoder is **chunked** (line 108) to save memory — processing 16 timesteps at a time with gradient checkpointing.

### 5.4 Depth Decoder (from Moshiko)

**Files:** `src/model/surgery.py`, `src/model/depth_decoder.py`

The depth decoder generates audio codebooks **hierarchically**: given codebook 0, it predicts codebook 1, then given codebooks 0+1 it predicts codebook 2, and so on. This is from the Moshi/Hibiki architecture.

**Weight extraction** (`surgery.py:8`):
```python
moshiko = MoshiForConditionalGeneration.from_pretrained("kmhf/hf-moshiko")
state_dict = {name: param for name, param in moshiko.depth_decoder.named_parameters()}
# Returns weights for: input_projections, lm_heads, embed_tokens, decoder layers
```

**Decoder creation** (`depth_decoder.py:50`):
- Config: 6 layers, 16 heads, 1024 hidden, 4096 input, 5632 FFN, 2048 audio vocab, 32K text vocab
- If `num_codebooks` differs from Moshiko's original (8), weights are extended cyclically (`extend_weights()`, line 8)
- Loaded with `strict=False` (some Moshiko weights may not match exactly)

**During training**, depth decoder internals are **frozen** — only the I/O layers are trainable:
```python
# train_hierarchical.py:95-104 (freeze_depth_internals)
for name, param in model.depth_decoder.named_parameters():
    if any(k in name for k in ("input_projections", "embed_tokens", "lm_heads")):
        param.requires_grad = True   # I/O layers: trainable
    else:
        param.requires_grad = False  # Transformer layers: frozen
```

---

## 6. Training Pipeline

### 6.1 Loss Function

**File:** `src/training/translation_loss.py`

**compute_hierarchical_translation_loss()** (line 12):

The loss has two parts:
1. **Text loss** (weight 0.1): Cross-entropy on text_logits vs text_targets. This is auxiliary — it helps the backbone understand the inner monologue but isn't the main objective.
2. **Audio loss** (weight 1.0): Cross-entropy on each of the 8 codebooks independently, then averaged. This is the main objective — predicting the correct audio tokens.

Both use **next-token prediction** (logits shifted by 1):
```python
# translation_loss.py:35-36
text_logits = text_logits[:, :-1]    # predict position t+1 from position t
text_targets = text_targets[:, 1:]
```

**Only target positions contribute to the loss** (line 37):
```python
combined_mask = attention_mask[:, 1:] * loss_mask[:, 1:]
```
Source frames have `loss_mask=0`, so the model is only trained to predict the target language audio. The source is the "prefix" that conditions the generation.

### 6.2 Training Loop

**File:** `scripts/train_hierarchical.py`

The main training loop (line 504-608) follows this structure per step:

```
For each gradient accumulation microstep:
  1. Load batch from DataLoader
  2. Move tensors to device
  3. Mark sharding (TPU only)
  4. Forward pass through composite model
  5. Compute hierarchical translation loss (target-only)
  6. loss / grad_accum -> backward()

After all microsteps:
  7. Clip gradients (max_grad_norm=1.0)
  8. Optimizer step (backend-specific)
  9. Scheduler step (warmup + cosine decay)
  10. Log metrics every N steps
  11. Generate audio demo every N steps
  12. Run validation every N steps
  13. Save checkpoint every N steps
```

**Key details:**

- **Gradient accumulation** (line 486-553): Multiple micro-batches are accumulated before one optimizer step. On GPU with DDP, `model.no_sync()` skips allreduce on non-final micro-batches for efficiency. On TPU, this is a no-op.

- **Audio demo generation** (line 578-597): Every `audio_every` steps, the model generates a sample translation and saves it as WAV. This uses `generate_audio_sample()` (line 143) which runs autoregressive inference: backbone generates CB0, depth decoder fills in CB1-CB7, Mimi decodes to audio. On TPU, Mimi runs on CPU.

- **Validation** (line 599-646): Runs teacher-forced forward pass on the val set, computes loss + CB0 accuracy. Saves "best by val loss" checkpoint.

- **Checkpoint saving** (line 632-646): Saves PEFT adapter + projection + depth decoder + optimizer + scheduler. Prunes old checkpoints (keeps last 5 + best).

### 6.3 Optimizer and Scheduler

**Optimizer:** AdamW with per-group learning rates (see section 5.2), weight decay 0.01, betas (0.9, 0.999).

**Scheduler** (`src/training/scheduler.py`):
`WarmupCosineScheduler` (line 9):
- First `warmup_steps` (default 500): linear ramp from 0 to base LR
- Remaining steps: cosine decay from base LR to `min_lr_ratio * base_LR` (default 0.1)
- Each parameter group has its own base LR, the scheduler applies a shared multiplier

### 6.4 Checkpointing

**File:** `src/training/checkpointing.py`

Checkpoints are **modular** — each component is saved separately:

```
checkpoint_dir/
  peft_adapter/          # LoRA weights (via model.save_pretrained)
  projection.pt          # Projection layer state dict
  depth_decoder.pt       # Depth decoder state dict
  text_embed.pt          # Text embedding adapter
  audio_heads.pt         # Audio prediction heads
  optimizer.pt           # Optimizer state
  scheduler.pt           # LR scheduler state
  metadata.json          # Step number + extra state
```

**Loading** (`load_checkpoint()`, line 43): Uses `set_peft_model_state_dict()` instead of `PeftModel.from_pretrained()` — this is critical because `from_pretrained` would re-freeze the last 2 full-FT layers, undoing our setup.

**Spot preemption support:**
- `find_latest_checkpoint()` (line 139): auto-detects latest checkpoint for `--resume auto`
- Supports both local and GCS paths (`is_gcs_path()`, line 114)
- `prune_checkpoints()` (line 100): keeps last N + best checkpoint, deletes the rest

---

## 7. Inference Pipeline

**File:** `scripts/infer_only.py`

The `generate()` function (line 24) runs autoregressive inference:

```
Input: source_codes [CB, T_src] (from Mimi encoding the source audio)

For each target frame (up to max_frames):
  1. Pass all generated CB0 tokens so far to backbone
  2. Project the last hidden state
  3. Use depth decoder to generate CB0..CB7 for this frame:
     - CB0: argmax of depth_decoder logits at position 0
     - CB1: argmax at position 1, conditioned on CB0
     - CB2: argmax at position 2, conditioned on CB0+CB1
     - ...
  4. Append CB0 token to the running sequence for next step

Output: generated_codes [CB, T_tgt] -> Mimi decode -> audio waveform
```

**Usage:**
```bash
python scripts/infer_only.py \
    --checkpoint checkpoints/best_by_val/ \
    --infer_wavs source_turkish.wav \
    --num_codebooks 8 \
    --max_frames 120
```

---

## 8. Configuration System

**File:** `scripts/train_hierarchical.py`, lines 44-90

The config system uses three layers (later overrides earlier):

1. **DEFAULTS dict** (line 44): hardcoded defaults for every parameter
2. **YAML file** (loaded via `--config`): overrides defaults
3. **CLI flags**: override both YAML and defaults

```python
cfg = DEFAULTS.copy()
cfg.update(yaml_config)    # YAML wins over defaults
cfg.update(cli_overrides)  # CLI wins over everything
```

**Config structure:**
```yaml
backend: auto        # "auto", "gpu", or "tpu"

data:
  train_split: ...   # JSONL path for streaming mode
  val_split: ...
  encoded_dir: ...   # Directory with .pt files
  max_frames: 300    # Max sequence length (12.5 Hz -> 24s)
  audio_frame_rate: 12.5

train:
  num_codebooks: 8
  batch_size: 2
  grad_accum: 2      # Effective batch = batch * accum * num_devices
  max_steps: 5000
  warmup_steps: 500
  depth_chunk_size: 16
  max_grad_norm: 1.0

loss:
  text_weight: 0.1   # Auxiliary (inner monologue)
  audio_weight: 1.0  # Main objective

optim:
  lr_lora: 1.5e-4
  lr_full_ft: 5e-5
  lr_projection: 5e-4
  lr_depth: 2.5e-4
  lr_audio_embed: 5e-4
  lr_text_embed: 5e-4

logging:
  log_every: 20
  save_every: 500
  audio_every: 1000
  val_every: 1000
  save_dir: checkpoints/
  use_wandb: false
```

---

## 9. Running the Code

### 9.1 Data Preparation

```bash
# 1. Encode audio with Mimi (from the data generation pipeline)
python cli.py encode

# 2. Create train/val splits
python scripts/make_splits.py \
    --accepted data/manifests/accepted.jsonl \
    --encoded-dir data/encoded/ \
    --out-dir data/splits/

# 3. Upload to HuggingFace (optional)
python scripts/upload_encoded_dataset.py \
    --encoded-dir data/encoded/ \
    --splits-dir data/splits/ \
    --repo-id tiny-aya-translate/fleurs-tr-hi-mimi-encoded
```

### 9.2 Training

**Single GPU:**
```bash
python scripts/train_hierarchical.py \
    --config configs/stage2_scale.yaml \
    --train_split data/splits/train.jsonl \
    --val_split data/splits/val.jsonl \
    --encoded_dir data/encoded/
```

**Multi-GPU (DDP):**
```bash
torchrun --nproc_per_node=4 scripts/train_hierarchical.py \
    --config configs/stage2_scale.yaml ...
```

**TPU:**
```bash
python scripts/train_hierarchical.py \
    --config configs/stage2_tpu.yaml \
    --backend tpu
```

**50-pair smoke test:**
```bash
python scripts/train_hierarchical.py \
    --dataset_mode memory \
    --data_dir data/stage2_fixed/ \
    --max_steps 100
```

**Resume from checkpoint:**
```bash
python scripts/train_hierarchical.py \
    --config configs/stage2_scale.yaml \
    --resume auto  # finds latest checkpoint
    # or: --resume checkpoints/step_002000/
```

### 9.3 Inference

```bash
python scripts/infer_only.py \
    --checkpoint checkpoints/best_by_val/ \
    --infer_wavs test_turkish.wav test_hindi.wav \
    --output_dir outputs/
```

### 9.4 Monitoring Training

With W&B enabled (`--use_wandb true`), the following metrics are logged:
- `train/loss`, `train/text_loss`, `train/audio_loss` — per-step averages
- `train/per_codebook_loss_0` through `_7` — individual codebook losses
- `train/grad_norm` — gradient magnitude
- `val/loss`, `val/cb0_acc` — validation metrics (every `val_every` steps)
- `audio/source`, `audio/target_gt`, `audio/generated` — audio samples (every `audio_every` steps)
- `mem/peak_gb` — peak GPU memory (GPU only)

---

## 10. Key Constants and Magic Numbers

| Constant | Value | Location | Meaning |
|----------|-------|----------|---------|
| Text vocab size | 262,144 | `backbone.py:46` | TinyAya's original vocabulary |
| Special tokens | 262,144-262,147 | `backbone.py:19-22` | TEXT_PAD, END_TEXT_PAD, ZERO_PAD, IN_WORD_PAD |
| Audio token offset | 262,148 | `backbone.py:50` | First audio token position in extended embedding |
| Audio vocab size | 2,048 | `backbone.py:29` | Mimi codebook entries (0..2047) |
| Total vocab | 264,196 | `backbone.py:51` | 262,144 text + 4 special + 2,048 audio |
| Num codebooks | 8 | `configs/` | Of Mimi's 32, we use 8 (matches Moshi/Hibiki) |
| Frame rate | 12.5 Hz | `mimi_encoder.py` | One audio frame per 80ms |
| Sample rate | 24,000 Hz | `mimi_encoder.py` | Mimi's native sample rate |
| Hidden size | 2,048 | `backbone.py:62` | TinyAya's transformer hidden dimension |
| Depth decoder hidden | 1,024 | `depth_decoder.py:58` | Moshi depth decoder hidden size |
| Depth decoder input | 4,096 | `depth_decoder.py:59` | Projection output -> depth decoder input |
| Num backbone layers | 36 | `lora_setup.py` | Cohere2 layers in TinyAya |
| LoRA rank | 16 | `lora_setup.py:37` | Low-rank adapter dimension |
| LoRA target modules | q_proj, v_proj | `lora_setup.py` | Which attention matrices get LoRA |
| Full FT layers | 34, 35 | `lora_setup.py:43` | Last 2 layers fully fine-tuned |
| Depth decoder layers | 6 | `depth_decoder.py:60` | Moshi depth decoder transformer layers |
| Sliding window | 4,096 | Cohere2 config | Attention window (causes XLA issue -> use_cache=False) |
