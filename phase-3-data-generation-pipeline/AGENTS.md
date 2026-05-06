# AGENTS.md — phase-3-data-generation-pipeline

Briefing for agents editing files under
`phase-3-data-generation-pipeline/`. Inherits and overrides the root
`/AGENTS.md`.

## Scope

This subproject **encodes raw audio to Mimi tokens and aligns them
to text** for the TR <-> HI parallel corpus that feeds the trainer in
`simultaneous-translation/`. It is the upstream of every training
sample; if encoding or alignment regresses, training silently
degrades.

## Build & run

The pipeline is invoked from the subproject root:

```bash
cd phase-3-data-generation-pipeline
PYTHONPATH=. python cli.py --data-dir data encode    # 9,212 .pt files
PYTHONPATH=. python cli.py --data-dir data align     # 18,424 alignment JSONs
```

There is no separate test runner; correctness is enforced by the
audited `accepted.jsonl` manifest and downstream model loss. Add
unit tests under `tests/` when in doubt.

## Stages

| Stage | Input | Output | Module |
|-------|-------|--------|--------|
| `encode` | `data/raw/<sentence_id>/{tr,hi}.wav` | `data/encoded/<sentence_id>_{tr,hi}.pt` (Mimi tokens) | `src/encoding/mimi.py` |
| `align` | encoded tokens + raw text | `data/alignments/<sentence_id>_{tr,hi}.json` (Whisper word-level timestamps) | `src/encoding/whisper_align.py` |
| (downstream) | both above + `accepted.jsonl` | `data/splits/{train,val}.jsonl` | run from `simultaneous-translation/scripts/make_splits.py` |

## Mimi encoding

Uses `transformers.MimiModel` (the new HF API), not the older
`kyutai/mimi` standalone package. The pin lives in
`pyproject.toml`. Sample rate must match Mimi's expected 24 kHz; we
resample on load, never offline.

## Whisper alignment

Uses `transformers.WhisperForConditionalGeneration` with
`return_token_timestamps=True`. Timestamps are at the **word**
granularity. Confidence below `0.4` causes a sample to be dropped
from `accepted.jsonl` (the manifest, not this subproject).

## File-format conventions

- Encoded tensors: `torch.int64` token tensors, shape `(T, K)`.
- Alignments: a JSON per language per sentence with the schema
  documented at the top of `src/encoding/whisper_align.py`.
- Audio scratch: `*.wav.tmp` files are gitignored; never check in raw
  audio.

## Performance

- `encode` is GPU-bound; a single A100 handles ~12 samples/sec on
  Mimi. CPU fallback is supported but ~50x slower.
- `align` is GPU-bound on Whisper-large-v3; expect ~6 samples/sec.
- Both stages are idempotent — rerun is safe; existing outputs are
  skipped via mtime comparison.

## Gotchas

- **HF Hub auth:** the model downloads on first run and may require
  `huggingface-cli login` once; thereafter cached locally.
- **Sample rate drift:** if you swap an audio source, double-check
  the resample in `mimi.py` matches Mimi's 24 kHz expectation.
- **Whisper hallucinations:** silence or noise-only clips produce
  hallucinated text; these get filtered during alignment scoring.
- **Token tensor dtype:** must stay `int64`; cast bugs into `int32`
  silently corrupt downstream training (Mimi uses 32k vocab so it
  fits, but the trainer assumes int64).

## Conventions

- All new modules go under `src/encoding/` (encoder/aligner stages)
  or `src/audit/` (manifest QA).
- Functions on hot loops should be pure and CPU-bound where possible
  to enable multi-process scaling.
- Logs go to stderr, not stdout, so the CLI can stream JSON results
  on stdout if needed.

## How to integrate with training

Once the pipeline produces all four artifacts (encoded TR + HI,
alignments TR + HI), run the trainer's split tool:

```bash
python ../simultaneous-translation/scripts/make_splits.py \
    --accepted data/manifests/accepted.jsonl \
    --encoded-dir data/encoded \
    --out-dir data/splits --val-frac 0.10 --seed 42
```

That writes `data/splits/{train,val}.jsonl` which the trainer reads.

## Where to log

| Event | Where |
|-------|-------|
| Schema change to encoded `.pt` files | `.factory/memories.md` (## Architecture decisions) |
| New filtering rule for `accepted.jsonl` | `.factory/memories.md` (## Architecture decisions) |
| Pipeline failure on a known sample | `.factory/PROGRESS.md` (status: `fail`, kind: `exec`) |

Use `/remember` or `#decision` for the first two; `/progress` or
`#progress` for the third.

## Out of subproject scope

- Training, eval, TPU launch (use `simultaneous-translation/`).
- TTS data generation (handled in an earlier phase, upstream of
  this repo).
- Manifest curation: `accepted.jsonl` is hand-audited; this
  subproject reads it but does not write it.
