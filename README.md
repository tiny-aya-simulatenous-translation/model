# TinyAya Stage 2 — TR↔HI speech-to-speech translation at scale

Code used for the full-scale Stage 2 training run on the 9,212 accepted
Turkish↔Hindi parallel pairs (2,440 real FLEURS + 6,772 TTS).

Upstream repos (not vendored here):

- `simultaneous-translation` — training / model / eval code. Canonical repo.
- `phase-3-data-generation-pipeline` — data pipeline (encode + align stages).

Only the files we **added or modified** for Stage 2 scale are committed:

```
simultaneous-translation/
  src/data/dataset.py                    # + StreamingTranslationDataset
  src/training/scheduler.py              # WarmupCosineScheduler
  src/training/translation_loss.py       # compute_hierarchical_translation_loss
  src/training/checkpointing.py          # save/load + prune
  scripts/train_hierarchical.py          # YAML config, grad accum, val loop, sched, ckpt
  scripts/make_splits.py                 # leak-free 90/10 split on sentence_id
  scripts/eval_stage2.py                 # ASR-BLEU + DNSMOS + demos
  scripts/upload_encoded_dataset.py      # HF dataset push
  configs/stage2_scale.yaml              # Stage 2 config
phase-3-data-generation-pipeline/
  cli.py                                 # + align subcommand
  src/encoding/whisper_align.py          # Whisper word-level timestamping
  src/encoding/mimi.py                   # fixed to transformers Mimi API
```

## Data pipeline

```bash
cd phase-3-data-generation-pipeline
PYTHONPATH=. python cli.py --data-dir data encode     # 9,212 .pt files
PYTHONPATH=. python cli.py --data-dir data align      # 18,424 alignment JSONs
python ../simultaneous-translation/scripts/make_splits.py \
    --accepted data/manifests/accepted.jsonl \
    --encoded-dir data/encoded \
    --out-dir data/splits --val-frac 0.10 --seed 42
```

## Training

```bash
cd simultaneous-translation
python scripts/train_hierarchical.py \
    --config configs/stage2_scale.yaml \
    --train_split ../phase-3-data-generation-pipeline/data/splits/train.jsonl \
    --val_split   ../phase-3-data-generation-pipeline/data/splits/val.jsonl \
    --encoded_dir ../phase-3-data-generation-pipeline/data/encoded \
    --use_wandb true
```

## Eval

```bash
python scripts/eval_stage2.py \
    --checkpoint checkpoints/stage2_scale/best_by_val \
    --val_split ../phase-3-data-generation-pipeline/data/splits/val.jsonl \
    --encoded_dir ../phase-3-data-generation-pipeline/data/encoded \
    --out_dir eval_outputs/stage2_scale
```
