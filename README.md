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
├── configs/                    # Training configs (YAML), split per backend
│   ├── gpu/
│   │   └── stage2_26k_parallel.yaml   # GPU parallel-stream run
│   └── tpu/
│       ├── stage2_tpu_v6e_v2.yaml         # production 15k v6e-8 run
│       ├── stage2_tpu_v6e_valfix_smoke.yaml
│       └── stage2_tpu_v6e_proxy.yaml      # W&B sweep proxy
├── scripts/
│   ├── train_hierarchical.py   # Main training script (SHARED, backend-dispatched)
│   ├── gen_parallel.py         # Generate audio (teacher-forced + autoregressive)
│   ├── eval_checkpoint.py      # Eval: ASR transcription + BLEU scoring
│   ├── validate_checkpoint.py  # Verify checkpoint integrity
│   ├── make_splits.py          # Create train/val splits from encoded data
│   ├── ci/check_backend_seam.sh # CI: forbid module-level torch_xla in shared code
│   └── tpu/                    # TPU run-infra (queued-resource launch, redeploy, ops)
├── sweeps/                     # W&B hyperparameter sweep (proxy-first)
├── src/
│   ├── model/                  # composite, backbone, depth_decoder, lora_setup, surgery
│   ├── data/                   # dataset, collator (parallel stream), interleaver, mimi
│   ├── training/               # checkpointing, translation_loss (padding-weighted text CE)
│   └── backend/
│       ├── base.py             # Backend abstraction (get_backend dispatch)
│       ├── gpu_backend.py      # CUDA + FSDP backend
│       └── tpu_backend.py      # XLA/SPMD backend — ONLY module that imports torch_xla
├── docs/                       # run plans + memory-system docs
├── .claude/                    # External Memory System (PLAN/PROGRESS/VERIFY/memories, hooks, skills)
├── AGENTS.md                   # agent briefing + TPU↔GPU seam rules
├── pyproject.toml
└── uv.lock
```

### Backends (GPU ↔ TPU)

The repo is **dual-backend**. The same `src/` and
`scripts/train_hierarchical.py` run on either GPU or TPU; only **configs**
(`configs/gpu/` vs `configs/tpu/`) and **launchers** (`torchrun` vs
`scripts/tpu/`) differ. `torch_xla` is confined to
`src/backend/tpu_backend.py` (enforced by `scripts/ci/check_backend_seam.sh`),
so GPU runs and CPU-only tests never import it.

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
uv run python scripts/train_hierarchical.py --config configs/gpu/stage2_26k_parallel.yaml
```

### Training (multi-GPU with FSDP)

```bash
torchrun --nproc_per_node=2 scripts/train_hierarchical.py \
    --config configs/gpu/stage2_26k_parallel.yaml
```

### Resume from checkpoint

```bash
torchrun --nproc_per_node=2 scripts/train_hierarchical.py \
    --config configs/gpu/stage2_26k_parallel.yaml \
    --resume checkpoints/stage2_26k_parallel/step_002750
```

Note: model weights are loaded before FSDP wrapping. Optimizer state is restored via `FSDP.optim_state_dict_to_load()`.

### Training (TPU v6e-8, SPMD/FSDPv2)

The TPU path uses the same training script with a TPU config and the
launchers under `scripts/tpu/`. On a provisioned v6e-8 (single host,
8 chips):

```bash
# production 15k release run (resumes automatically)
bash scripts/tpu/launch_release.sh configs/tpu/stage2_tpu_v6e_v2.yaml

# short smoke first (inline-val fix, parallel stream, diag dashboard)
bash scripts/tpu/launch_release.sh configs/tpu/stage2_tpu_v6e_valfix_smoke.yaml
```

`XLA_NO_SPECIAL_SCALARS=1` (set by the launchers) is required — it
disables XLA's "assume no NaN/Inf" rewrites that otherwise corrupt the
inline-validation loss scalar. A proxy-first W&B hyperparameter sweep
lives in `sweeps/` (see `sweeps/README.md`).

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
