---
license: apache-2.0
base_model: CohereLabs/tiny-aya-base
base_model_relation: adapter
library_name: peft
new_version: tiny-aya-translate/tr-hi-s2st-v0.2
datasets:
  - tiny-aya-translate/tr-hi-mimi-encoded
language:
  - tr
  - hi
tags:
  - speech-to-speech-translation
  - speech-translation
  - simultaneous-translation
  - lora
  - moshi
  - mimi
  - tpu
  - tinyaya
pipeline_tag: audio-to-audio
metrics:
  - bleu
# Eval results go here once the GPU eval (ASR-BLEU + DNSMOS) has run; the
# Hub renders model-index as a metrics widget. Template:
# model-index:
#   - name: tr-hi-s2st-v0.1
#     results:
#       - task: { type: audio-to-audio, name: Speech-to-Speech Translation }
#         dataset: { type: google/fleurs, name: FLEURS (tr/hi) }
#         metrics:
#           - { type: bleu, name: ASR-BLEU (tr->hi), value: TBD }
#           - { type: bleu, name: ASR-BLEU (hi->tr), value: TBD }
---

> **Version:** `v0.1.0` — step-15000 checkpoint (first public release; eval pending).
> Versions are git tags in this repo; load a specific one with
> `revision="v0.1.0"`. See **Version history** at the bottom.
>
> 🔜 **A newer version exists:** [`tr-hi-s2st-v0.2`](https://huggingface.co/tiny-aya-translate/tr-hi-s2st-v0.2)
> fixes the text stream (the Phase-0 padding-loss fix) so the inner-monologue
> actually learns. If you want the later recipe, use v0.2.

# TinyAya Stage 2 — Turkish ↔ Hindi Speech-to-Speech Translation (LoRA)

Stage-2 simultaneous-translation adapter for Turkish↔Hindi **speech-to-speech**
translation, trained on TPU v6e-8. This repo ships the authors' **trained
deltas only** — a LoRA adapter over `CohereLabs/tiny-aya-base` plus the
custom projection / Moshi depth-decoder / audio-head / embedding tensors.

- **Developed by:** [tiny-aya-translate](https://huggingface.co/tiny-aya-translate)
- **Funded by:** Google **TPU Research Cloud (TRC)** — see [Acknowledgements](#acknowledgements)
- **Model type:** parallel two-stream S2ST (Cohere2 + LoRA + custom heads; frozen Moshi depth decoder)
- **Languages:** Turkish (`tr`), Hindi (`hi`)
- **License:** Apache-2.0 *(trained deltas + code only — see license scope below)*
- **Finetuned from:** `CohereLabs/tiny-aya-base` (+ Moshi / Mimi from Kyutai)
- **Newer version:** [`tiny-aya-translate/tr-hi-s2st-v0.2`](https://huggingface.co/tiny-aya-translate/tr-hi-s2st-v0.2)

> ## ⚠️ License scope — read first
> The **`apache-2.0`** license declared above covers **only the trained
> weights in this repo** (the LoRA adapter + custom heads) and the authors'
> code. It does **NOT** relicense the components this model builds on, which
> keep their own licenses (mirrored from `THIRD_PARTY_NOTICES.md`):
> - **`CohereLabs/tiny-aya-base`** base weights → **Cohere model license**
>   (NOT Apache). Not included here; you must obtain it from Cohere and
>   comply with its terms.
> - **Moshi / Mimi** (depth decoder + audio codec) → **MIT**.
> - **FLEURS** source data → **CC BY 4.0**.

## What this is

| | |
|---|---|
| Task | TR↔HI speech-to-speech translation (Moshi inner-monologue, hierarchical codebook decode) |
| Base | `CohereLabs/tiny-aya-base` + Moshi depth decoder + Mimi codec |
| Method | LoRA (+ trained projection/heads/embeds), bf16, FSDPv2 SPMD |
| Hardware | Cloud TPU v6e-8 (single host), via Google TRC |
| Steps | 15,000 (effective batch 256) |
| Data | `tiny-aya-translate/tr-hi-mimi-encoded` (Mimi-encoded FLEURS TR/HI) |

## Training procedure

- **Init**: LoRA (r=16) on the `tiny-aya-base` backbone; Moshi depth decoder
  initialized from `kyutai/moshiko`; projection / per-codebook audio heads /
  audio & text embeddings trained from scratch.
- **Recipe**: effective batch 256 (b=8 × grad-accum 4 × 8 chips), bf16,
  FSDPv2 SPMD, cosine LR (lr_lora 1.5e-4), 500 warmup steps, `max_frames=300`,
  8 codebooks, hierarchical codebook loss (text_weight 0.1, audio_weight 1.0).
- **Compute**: single-host Cloud TPU v6e-8 (spot), ~22 h, 15,000 steps
  (≈ 3.3 epochs over 1,178,302 train pairs).

**Convergence note (honest):** the **audio** loss plateaued by ~step 8,000
(≈1.7 epochs) and did not improve over the final 7k steps. The **text /
inner-monologue** stream did not learn in this run (loss ≈ random), pending
a data-pipeline fix. So this checkpoint reflects the audio-translation
capability of the recipe at convergence, not an under-trained model.
**This is fixed in [v0.2](https://huggingface.co/tiny-aya-translate/tr-hi-s2st-v0.2)**
(the text-padding-weighted loss), where the text stream learns.

## Evaluation

Speech-translation quality is measured with **ASR-BLEU** (Whisper transcribes
the generated target audio, BLEU vs. reference target text) and **DNSMOS**
(naturalness). ASR-BLEU is implemented in `scripts/eval_checkpoint.py`; DNSMOS
is not yet wired up. **Pending for this release** (`v0.1.0`); will be filled in
the YAML `model-index` + below.

| Metric | tr→hi | hi→tr | overall |
|--------|-------|-------|---------|
| ASR-BLEU | _TBD_ | _TBD_ | _TBD_ |
| DNSMOS (ovrl) | _TBD_ | _TBD_ | _TBD_ |

## Intended use & limitations

- **Intended**: research on low-resource speech-to-speech translation and
  simultaneous translation; a Stage-2 checkpoint, not a production system.
- **Limitations**: trained on read-speech (FLEURS) — expect degradation on
  spontaneous/noisy audio; two language directions only; **the text stream did
  not learn in this version** (use v0.2 for text); generation is autoregressive
  and not optimized for latency here.
- **History (transparency)**: the run was a spot v6e-8 (preemptible); an
  earlier run had a checkpoint GCS-path bug (fixed) and three W&B metrics
  (per-codebook loss, grad-norm, HBM) that logged as zero (fixed in this run).

## Bias, risks & limitations

Trained on FLEURS (read speech, limited speakers/domains); quality and fairness
across dialects, accents, code-switching, and spontaneous speech are untested.
Speech translation can mistranslate, omit, or fabricate content — outputs must
not be relied upon for high-stakes communication. As noted above, the text /
inner-monologue stream is not functional in this version.

## Inference quickstart

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "CohereLabs/tiny-aya-base"  # obtain per Cohere license
ADAPTER = "tiny-aya-translate/tr-hi-s2st-v0.1"  # this repo

tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, trust_remote_code=True)
model = PeftModel.from_pretrained(base, ADAPTER)  # loads adapter_model.safetensors

# Then attach the custom heads (projection / depth_decoder / audio_heads /
# text_embed / model_audio_embed *.pt) and the Mimi codec; see the training
# repo for the full composite + generation loop.
```

The full speech→speech pipeline (Mimi encode → backbone+depth-decoder →
Mimi decode) lives in the training repo (`src/model/composite.py`).

## Links

- **Training run (W&B)**: https://wandb.ai/cataluna84/tinyaya-stage2-tpu/runs/b7fr72u5
  — full config, loss curves, system/throughput metrics, and the model
  artifact (`tinyaya-stage2-tr-hi-v6e-v2:v0`).
- **Checkpoints (GCS)**: `gs://tinyaya-stage2-tpu/checkpoints/stage2-tpu-v6e-v2/`
- **Dataset**: https://huggingface.co/datasets/tiny-aya-translate/tr-hi-mimi-encoded
- **Code**: https://github.com/tiny-aya-simulatenous-translation/model
- **Next version**: https://huggingface.co/tiny-aya-translate/tr-hi-s2st-v0.2

## Version history

Versions are **git tags** in this repo (HF convention: one checkpoint per
repo, versioned with tags). Load a specific one with `revision=`:

```python
PeftModel.from_pretrained(base, "tiny-aya-translate/tr-hi-s2st-v0.1", revision="v0.1.0")
```

| Version | Step | Notes |
|---------|------|-------|
| `v0.1.0` | 15,000 | First release. Audio converged ~step 8k; text stream not yet learning; eval pending. |

When a materially different model is trained (new data / architecture), it
goes in a **new repo** — here, [`tr-hi-s2st-v0.2`](https://huggingface.co/tiny-aya-translate/tr-hi-s2st-v0.2)
(the `new_version:` field shows a forward banner on the Hub).

## Acknowledgements

This model was trained on Cloud TPU **v6e-8** hardware generously provided by
**Google's TPU Research Cloud (TRC)** program. We thank the TRC team for
supporting this research. See `NOTICE` and `THIRD_PARTY_NOTICES.md` for
component licenses.

## Citation

```bibtex
@misc{tinyaya_tr_hi_s2st_v0_1,
  title  = {TinyAya: Turkish-Hindi Speech-to-Speech Translation (v0.1)},
  author = {tiny-aya-translate},
  year   = {2026},
  note   = {Cohere2 + frozen Moshi depth decoder, LoRA, trained on Google TRC TPU v6e-8},
  url    = {https://huggingface.co/tiny-aya-translate/tr-hi-s2st-v0.1}
}
```
