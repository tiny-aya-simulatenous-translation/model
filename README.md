# TinyAya — TR↔HI Speech-to-Speech Translation

Moshi-style speech-to-speech translation model for Turkish↔Hindi, built on a LoRA-fine-tuned Cohere2 backbone (3B) with a frozen Moshi depth decoder.

## Architecture

```
User audio stream ──┐
                    ├──▶ Backbone (Cohere2 3B, LoRA) ──▶ CB0 prediction
Model audio stream ─┘         │                              │
Text stream ────────┘         ▼                              ▼
                         Projection (2048→4096)        audio_heads[0]
                              │
                              ▼
                    Depth Decoder (Moshiko 6L, frozen) ──▶ CB1-CB7 predictions
                              │
                              ▼
                    Mimi Codec Decode ──▶ Audio output
```

Key design choices:
- **Parallel two-stream format** — user audio + model audio run simultaneously (Moshi-style), model learns turn-taking via silence tokens
- **Codebook delay pattern** — CB_k shifted right by k frames for causal lookahead
- **CB0 from backbone, CB1-7 from depth decoder** — proper Moshi position mapping
- **Separate model_audio_embed** — dedicated embedding table for the model's speaker stream

## Repository Structure

```
├── configs/                    # Training configs (YAML)
│   └── stage2_26k_parallel.yaml
├── scripts/
│   ├── train_hierarchical.py   # Main training script (GPU, FSDP)
│   ├── gen_parallel.py         # Generate audio (teacher-forced + autoregressive)
│   ├── eval_checkpoint.py      # Eval: ASR transcription + BLEU scoring
│   ├── validate_checkpoint.py  # Verify checkpoint integrity
│   ├── validate_full_pipeline.sh # End-to-end pipeline test
│   └── make_splits.py          # Create train/val splits from encoded data
├── src/
│   ├── model/
│   │   ├── composite.py        # TinyAyaMoshiComposite — full model
│   │   ├── backbone.py         # Cohere2 backbone with audio/text embeddings
│   │   ├── depth_decoder.py    # Moshiko depth decoder extraction
│   │   ├── lora_setup.py       # LoRA application + parameter groups
│   │   └── surgery.py          # Projection layer + weight extraction
│   ├── data/
│   │   ├── dataset.py          # Parallel stream dataset + codebook delay
│   │   ├── collator.py         # Batch collation with padding
│   │   ├── interleaver.py      # Text-audio alignment interleaving
│   │   └── mimi_encoder.py     # Mimi codec encode/decode
│   ├── training/
│   │   ├── checkpointing.py    # FSDP-aware save/load + HF push
│   │   └── translation_loss.py # Hierarchical CE loss (text + per-codebook audio)
│   └── backend/
│       ├── gpu_backend.py      # CUDA + FSDP backend
│       └── base.py             # Backend abstraction
├── pyproject.toml
└── uv.lock
```

## Quick Start

### Prerequisites

- Python 3.12, [`uv`](https://docs.astral.sh/uv/)
- 1-2x NVIDIA GPUs with ≥24GB VRAM (tested on H100 80GB)
- Access to `CohereLabs/tiny-aya-base` (gated model, request access on HF)

### Setup

```bash
git clone https://github.com/tiny-aya-simulatenous-translation/model.git
cd model
uv sync
```

### Training (single GPU)

```bash
uv run python scripts/train_hierarchical.py --config configs/stage2_26k_parallel.yaml
```

### Training (multi-GPU with FSDP)

```bash
torchrun --nproc_per_node=2 scripts/train_hierarchical.py \
    --config configs/stage2_26k_parallel.yaml
```

### Resume from checkpoint

```bash
torchrun --nproc_per_node=2 scripts/train_hierarchical.py \
    --config configs/stage2_26k_parallel.yaml \
    --resume checkpoints/stage2_26k_parallel/step_002750
```

Note: model weights are loaded before FSDP wrapping. Optimizer state is restored via `FSDP.optim_state_dict_to_load()`.

### Evaluation

```bash
python scripts/eval_checkpoint.py \
    --checkpoint checkpoints/stage2_26k_parallel/best_by_val \
    --val_jsonl /path/to/val_500.jsonl \
    --encoded_dir /path/to/encoded \
    --num_samples 20 \
    --output_dir eval_results
```

Runs Whisper ASR on generated audio and computes BLEU against reference translations.

### Generate audio samples

```bash
python scripts/gen_parallel.py
```

Produces teacher-forced and autoregressive audio for listening comparison.

## Training Data

The model trains on Mimi-encoded parallel TR↔HI speech pairs:
- **26K quality-filtered subset** from 840K+ generated pairs
- Each sample: source audio (8 codebooks) + target audio (8 codebooks) + word-level text alignments
- Data format: `.pt` files with `src_codes`, `tgt_codes` + alignment JSONs

Dataset on HuggingFace: `tiny-aya-translate/tr-hi-mimi-encoded`

## Pipeline Validation

Before any full training run, validate the complete pipeline:

```bash
bash scripts/validate_full_pipeline.sh
```

This tests: train → save → verify sizes → load → forward → resume → save → verify → compare outputs.

## FSDP Notes

- Checkpoint save uses `FSDP.summon_full_params()` to gather full tensors
- Checkpoint load happens BEFORE `backend.wrap_model()` (FSDP wrapping)
- Optimizer state saved via `FSDP.full_optim_state_dict()`, loaded via `FSDP.optim_state_dict_to_load()`
- Audio demo generation is skipped in distributed mode (would deadlock FSDP)

## Related Repos

- [`data-pipeline`](https://github.com/tiny-aya-simulatenous-translation/data-pipeline) — TTS generation, deployment, Mimi encoding
- [`sound-quality-check`](https://github.com/tiny-aya-simulatenous-translation/sound-quality-check) — 4-stage audio QC pipeline

## License

Apache 2.0. See [LICENSE](LICENSE). Model weights from Cohere and Moshi carry their own licenses.
