---
language:
  - tr
  - hi
license: apache-2.0
library_name: peft
pipeline_tag: audio-to-audio
tags:
  - speech-to-speech-translation
  - simultaneous-translation
  - moshi
  - mimi
  - lora
  - tpu
  - turkish
  - hindi
base_model: CohereLabs/tiny-aya-base
datasets:
  - tiny-aya-translate/tr-hi-mimi-encoded
model-index:
  - name: tr-hi-s2st-v0.3
    results: []   # TODO: ASR-BLEU / chrF / DNSMOS — to be filled after GPU eval
---

# TinyAya — Turkish⇄Hindi Speech-to-Speech Translation (v0.3)

> 🚧 **Training in progress — preliminary card.** Weights and downstream metrics
> are **not yet published**. This card documents the dataset and the corrected
> setup ahead of the run, for transparency. It will be updated with checkpoints,
> the recipe-as-run, and evaluation once training completes.

Moshi-style **simultaneous speech-to-speech translation** for **Turkish ⇄ Hindi**:
a LoRA-fine-tuned **Cohere2** backbone fused with a **frozen Moshi depth decoder**,
operating on **Mimi** audio codes in a parallel two-stream format.

- **Developed by:** [tiny-aya-translate](https://huggingface.co/tiny-aya-translate)
- **Funded by:** Google **TPU Research Cloud (TRC)**
- **Model type:** parallel two-stream S2ST (Cohere2 + LoRA → CB0; frozen Moshi depth decoder → CB1–7)
- **Languages:** Turkish (`tr`), Hindi (`hi`)
- **Previous version:** [`tr-hi-s2st-v0.2`](https://huggingface.co/tiny-aya-translate/tr-hi-s2st-v0.2)

## Dataset (corrected from v0.2)

v0.3 trains on **[`tiny-aya-translate/tr-hi-mimi-encoded`](https://huggingface.co/datasets/tiny-aya-translate/tr-hi-mimi-encoded)**
— the project's **synthetic** pipeline: ~56k parallel text pairs from **FLORES**,
**OPUS-100**, and machine-translated **conversational** datasets, rendered with
multi-voice TTS (kokoro / XTTS-v2 / chatterbox) into ~**1.3M** Mimi-encoded clips
(`conv_*`, `flores_dev_*`, `flores_devtest_*`, `opus_*`). Training uses a
quality-filtered subset (size finalized at run time).

### ⚠️ The honest mistake this fixes

**v0.2 was trained on the wrong dataset.** It used the **`fleurs-`**-prefixed
*sibling* repo — [`fleurs-tr-hi-mimi-encoded`](https://huggingface.co/datasets/tiny-aya-translate/fleurs-tr-hi-mimi-encoded)
(Mimi-encoded **FLEURS** read speech: real FLEURS audio + TTS over FLEURS text) —
because the training launcher's `HF_DATASET` default pointed there. So v0.2's
experimental setup did **not** match the FLORES/OPUS-100/conversational synthetic
corpus described in our write-up. v0.3 repoints the data loader to
`tr-hi-mimi-encoded` so the run and the description agree. We're documenting this
openly rather than silently re-labeling v0.2.

## What else changed from v0.2

Beyond the data-source fix, v0.3 carries codebase corrections and the
anti-overfitting recipe:

- **Parallel-stream collator fix** — v0.2's pre-fix collator dropped the model
  audio stream, so `model_audio_embed` received **zero gradient** (the audio
  stream was effectively untrained). Restored in v0.3 (TPU-verified: grad 0 → nonzero).
- **Regularization** — `lora_dropout`, train-only `label_smoothing`, early
  stopping, lower capacity (`lora_r` 64 → 16), higher `weight_decay`, cosine
  LR to zero — directly targeting v0.2's overfit (val bottomed at step 1,000).
- **Deep-codebook learning** — per-codebook loss weighting + progressive
  coarse→fine unmasking (+ an optional low-LR depth-block unfreeze lever) to
  address CB1–7 collapse (v0.2: CB1–7 ≈ 0.5–3.9% val accuracy).

## Status checklist

| Item | Status |
|---|---|
| Data source repointed to `tr-hi-mimi-encoded` | ✅ |
| TPU env validated (uv, sharding, compile, unmask caching) | ✅ smoke-passed |
| Production training run | ☐ in progress |
| Checkpoints published | ☐ pending |
| ASR-BLEU / chrF / DNSMOS / WER eval | ☐ pending |

## Acknowledgements

Trained on Cloud TPU **v6e-8** provided by **Google's TPU Research Cloud (TRC)**.

## Citation

```bibtex
@misc{tinyaya_tr_hi_s2st_v0_3,
  title  = {TinyAya: Turkish-Hindi Speech-to-Speech Translation (v0.3)},
  author = {tiny-aya-translate},
  year   = {2026},
  note   = {Cohere2 + frozen Moshi depth decoder, LoRA; synthetic FLORES/OPUS/conversational corpus; Google TRC TPU v6e-8},
  url    = {https://huggingface.co/tiny-aya-translate/tr-hi-s2st-v0.3}
}
```
