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
  - tiny-aya-translate/fleurs-tr-hi-mimi-encoded
model-index:
  - name: tr-hi-s2st-v0.2
    results: []   # TODO: ASR-BLEU / chrF / DNSMOS — to be filled after GPU eval
---

# TinyAya — Turkish⇄Hindi Speech-to-Speech Translation (v0.2)

> ⚠️ **Research preview — this checkpoint overfit.** Released for transparency
> and to document the full training trajectory. The recommended weights are
> **`best_by_val` (step 1,000)**, *not* the final step-15,000 checkpoint. See
> [Evaluation & the overfitting story](#evaluation--the-overfitting-story).

Moshi-style **simultaneous speech-to-speech translation** for **Turkish ⇄ Hindi**:
a LoRA-fine-tuned **Cohere2** backbone fused with a **frozen Moshi depth
decoder**, operating on **Mimi** audio codes in a parallel two-stream format.

- **Developed by:** [tiny-aya-translate](https://huggingface.co/tiny-aya-translate)
- **Funded by:** Google **TPU Research Cloud (TRC)** — see [Acknowledgements](#acknowledgements)
- **Model type:** parallel two-stream S2ST (Cohere2 + LoRA → CB0; frozen Moshi depth decoder → CB1–7)
- **Languages:** Turkish (`tr`), Hindi (`hi`)
- **License:** Apache-2.0 *(for this adapter + training code; see the base-model caveat below)*
- **Finetuned from:** `CohereLabs/tiny-aya-base` (+ frozen Moshi / Mimi from Kyutai)

## Provenance: sweep → production run

This recipe was selected by a proxy-first **W&B hyperparameter sweep** (8 Bayesian
+ hyperband trials), then trained to 15k steps on a single **TPU v6e-8**.

- **📊 Sweep:** https://wandb.ai/cataluna84/tinyaya-stage2-tpu/sweeps/9ba8h0ho
- **📈 Training run:** https://wandb.ai/cataluna84/tinyaya-stage2-tpu/runs/t1840nkd
- **Code:** the [training repo](https://github.com/tiny-aya-simulatenous-translation/model) (PR #8)

## Training procedure

| | |
|---|---|
| Hardware | Google **TPU v6e-8** (1 host × 8 chips), SPMD / **FSDPv2-LoRA**, bf16 |
| Duration | **15,000 steps in ~24.9 h** (~5.1 s/step), single continuous run |
| Effective batch | 256 (batch 8 × grad-accum 4 × 8 chips), `max_frames` 300 |
| Stability | **0 non-finite / NaN / loss-spike alerts** across the whole run |
| **Recipe** | `lora.r=64`, `lora.alpha=128`, `lr_lora=4.6e-4`, `lr_depth=1.1e-4`, `text_weight=0.2`, `warmup=500`, `weight_decay=0.01` |
| Data | [`tiny-aya-translate/fleurs-tr-hi-mimi-encoded`](https://huggingface.co/datasets/tiny-aya-translate/fleurs-tr-hi-mimi-encoded) (Mimi-encoded FLEURS TR↔HI parallel speech) |

## Evaluation & the overfitting story

The run was **mechanically flawless but the recipe overfit**: validation loss
bottomed at **step 1,000** and rose monotonically while train loss kept falling.

| step | train loss | val/loss | cb0 val acc |
|-----:|-----------:|---------:|------------:|
| 1000 | 5.475 | **2.859 ← best** | 13.9% |
| 5000 | 3.231 | 3.376 | 14.3% |
| 10000 | 1.887 | 4.060 | 13.8% |
| 15000 | 1.566 | **4.197 (worst)** | 13.8% |

Final per-codebook val accuracy: cb0 13.8%, cb1 3.9%, cb2 1.9%, cb3 1.4%,
cb4 0.8%, cb5 0.8%, cb6 0.6%, cb7 0.5%. The text stream effectively memorized
(train text loss → 0.39); audio is the bottleneck. **Likely cause:** the proxy
sweep optimized short-horizon `val/audio_loss`, which favored high capacity
(`lora_r=64`) that then memorized the train set over 15k steps.

### Downstream metrics (to be measured)

No speech-quality metrics have been computed yet. Planned evaluation, with the
checkpoint to use = **`best_by_val` (step 1,000)**:

| Metric | Measures | Status |
|---|---|---|
| ASR-BLEU (Whisper-Lv3 → SacreBLEU) | translation quality | ☐ TODO |
| ASR-chrF / chrF++ | quality, more ASR-roundtrip-robust than BLEU | ☐ TODO |
| BLASER 2.0 | text-free S2ST quality | ☐ optional |
| COMET / COMET-Kiwi | semantic adequacy | ☐ optional |
| DNSMOS / UTMOS / NISQA | audio naturalness (MOS) | ☐ TODO |
| WER | intelligibility | ☐ TODO |

*Reporting protocol (when filled):* ASR backend + version, decoding (greedy/beam),
seed, text normalization, and number of eval samples will be stated.

## Checkpoints in this repo

`keep_last_n` rotation during the run means only these survived (steps
2,000–12,000 were not retained — fixed for future runs):

| Folder | Step | val/loss | Use |
|---|---|---|---|
| `best_by_val/` | **1,000** | **2.859** | ✅ **recommended** (full resumable state) |
| `checkpoints/step_13000/` | 13,000 | ~4.19 | overfit (trajectory) |
| `checkpoints/step_14000/` | 14,000 | ~4.19 | overfit (trajectory) |
| `checkpoints/step_15000/` | 15,000 | 4.197 | overfit (trajectory) |
| `checkpoints/step_15000_final/` | 15,000 | 4.197 | canonical final (overfit) |

Each checkpoint contains the composite components — `depth_decoder.pt`,
`text_embed.pt`, `audio_heads.pt`, `model_audio_embed.pt`, `projection.pt`,
`metadata.json` — plus the LoRA adapter under `peft_adapter/`
(`adapter_model.safetensors` + `adapter_config.json`). `train_15k.log` is the
full training log.

## Intended uses & limitations

- **Intended:** research, reproduction, and studying the TR⇄HI S2ST + overfitting
  trajectory. A teaching artifact for sweep→run→diagnosis.
- **Out of scope / limitations:** **not production-ready** — overfit, low
  per-codebook accuracy, no human eval, narrow domain (FLEURS read speech). May
  hallucinate or produce low-naturalness audio. Do not deploy for real
  translation without re-training (see below).

## How to use

This is a composite model (custom architecture), not a drop-in
`transformers` pipeline. Load via the training repo's
`src/model/composite.py`: base Cohere2 backbone + the LoRA adapter
(`peft_adapter/`) + the `.pt` components, then decode Mimi codes to audio.
See the [repo README](https://github.com/tiny-aya-simulatenous-translation/model)
for the loading + inference path. Use the **`best_by_val`** folder.

## Recommended next run (fixing the overfit)

Lower capacity + regularization + early stopping: `lora_r` 16–32, higher
`weight_decay`/dropout, stop at ~1–2k steps (or use best-N checkpoint
averaging), and/or more/augmented data.

## Bias, risks & limitations

Trained on FLEURS (read speech, limited speakers/domains); quality and fairness
across dialects, accents, code-switching, and spontaneous speech are untested.
Speech translation can mistranslate, omit, or fabricate content — outputs must
not be relied upon for high-stakes communication.

## License caveat (important)

The **Apache-2.0** license here covers **this LoRA adapter and the training
code**. The model is built on `CohereLabs/tiny-aya-base` and Moshi/Mimi — your
use of the assembled model is **governed by those upstream licenses** (the Aya
family is often released under non-commercial terms). Check the base-model and
Moshi/Mimi licenses before any commercial use.

## Acknowledgements

This model was trained on Cloud TPU **v6e-8** hardware generously provided by
**Google's TPU Research Cloud (TRC)** program. We thank the TRC team for
supporting this research.

## Citation

```bibtex
@misc{tinyaya_tr_hi_s2st_v0_2,
  title  = {TinyAya: Turkish-Hindi Speech-to-Speech Translation (v0.2)},
  author = {tiny-aya-translate},
  year   = {2026},
  note   = {Cohere2 + frozen Moshi depth decoder, LoRA, trained on Google TRC TPU v6e-8},
  url    = {https://huggingface.co/tiny-aya-translate/tr-hi-s2st-v0.2}
}
```
