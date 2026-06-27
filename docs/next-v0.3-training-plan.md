# v0.3 Training Plan — Holistic Anti-Overfitting, Deep-Codebook & Multitask Recipe

**Target PR:** #9 (follows #8). **Goal:** a v0.3 TR⇄HI speech-to-speech model where
**validation improves over training** (no overfit), the **deeper RVQ codebooks
learn**, and **both the audio and text streams stay healthy** — designed from
everything the v0.1 and v0.2 runs taught us.

This is a *training-recipe* plan (evaluation comes later). Solutions are
EXA-deep-researched and cited; see [References](#references).

---

## TL;DR — the holistic view (Phases A–E)

Every problem we hit maps to one of five work-phases. **Phase B (weight
averaging) is the single highest-ROI fix** for our exact failure mode
(validation rose while training fell = a sharp-minimum overshoot).

| Phase | Problem (from v0.1 / v0.2) | Root cause | Solution (technique) | Concrete change | Definition of Done |
|---|---|---|---|---|---|
| **A — Capacity & regularization** | v0.2 **overfit**: val/loss bottomed @ step 1000, rose monotonically (train 1.57 vs val 4.20 @ 15k) | LoRA `r=64` + `lr_lora=4.6e-4` too aggressive → overshot the good region in ~0.2 epochs | ↓ capacity + classic regularization | `r=16` (α=32), `lora_dropout=0.08`, `weight_decay=0.05`, `lr_lora≈1e-4` cosine→0 + warmup ~8%, `label_smoothing=0.1`, grad-clip 1.0 | val/loss flat-or-down through the run; train↔val gap bounded |
| **B — Weight averaging / soup ⭐** | best checkpoint was absurdly **early** (step 1000); single-point selection is fragile | trajectory overshot a flat region; SGD lands near a sharp wall | **EMA** + **SWA/LAWA** checkpoint averaging + **greedy model-soup** across sweep trials | maintain EMA (decay 0.999); average late checkpoints; **release the averaged weights**; soup the existing v0.2 sweep (step 0 quick win) | averaged model **beats** best single checkpoint on val; release = averaged |
| **C — Deep-codebook learning** | cb0 val acc ~14%, **cb1–7 <4%** (collapse) | frozen Moshi depth decoder unadapted to TR/HI; deep RVQ residuals are hard + cross-depth gradient conflict | **progressive coarse→fine unmasking** + per-codebook loss weighting + **staged depth-decoder unfreeze** | unmask deeper codebooks over first 5–15% of steps; moderate per-codebook multipliers; low-LR (×0.1) unfreeze of last 1–3 depth blocks *after* shallow stabilizes; keep delay pattern | cb0 acc up materially; cb1–7 non-trivial post-unmask; no semantic collapse |
| **D — Multitask balance** | v0.1 **text stream collapsed**; v0.2 text learned but audio is the bottleneck | static loss weighting; audio↔text trade-off via `text_weight` | balanced weighting + curriculum/adaptive gating + composite selection | `text_weight=0.3` (opt. curriculum 0.5→0.3 or adaptive gating on `L_val/L_train`); **keep our `text_padding_weight=0.01`** (proven, not Moshi's 0.5); composite early-stop `0.4·text+0.6·audio` | both streams improve on val; neither collapses |
| **E — Sweep redesign, data & run** | v0.1 **audio plateaued** ~step 8k (wasted compute); the v0.2 **sweep picked overfitting capacity** | 600-step proxy rewarded short-horizon capacity; static data → multi-epoch memorization | fix the sweep's metric/horizon + scale data + execute | sweep **regularization** knobs (`r∈{8,16,32}`, dropout, wd) on a **longer** proxy with a **generalization** metric; SpecAugment + synthetic/pseudo-labeled pairs; smoke → sweep → v0.3 run with EMA, early-stopped | sweep picks a *generalizing* recipe; v0.3 val improves; ship averaged weights |

> Infra problems from earlier runs (checkpoint rotation, spot preemption, inline-val
> NaN, bf16 NaN) are already fixed in #8 — this plan builds on that.

---

## Phase A — Capacity & regularization (anti-overfitting core)

**Why:** v0.2's `r=64` + `lr_lora=4.6e-4` drove the model out of the
good-generalization basin within ~0.2 epochs. The literature is unanimous that a
*larger LoRA rank → more overfitting*, and that foundation-model fine-tuning wants
low LR + light regularization because the base weights are already a strong prior.

**Changes:**
- **LoRA rank `r=16`** (down from 64; sweep `{8,16,32}` in Phase E), **`alpha=2r`**.
- **`lora_dropout=0.08`** (0.05–0.12 if overfit persists).
- **`weight_decay=0.05`** (0.01–0.1).
- **`lr_lora≈1e-4`** — between the report's very-safe `2e-5` and v0.2's `4.6e-4`;
  **cosine decay to ~0** with **~8% warmup**. Sweep downward in Phase E.
- **`label_smoothing=0.1`** on the cross-entropy.
- **grad-clip 1.0–2.0**; **early stopping** with patience ~4 val cycles.

**DoD:** validation loss decreases or stays flat (does **not** climb) across the run.

## Phase B — Weight averaging & checkpoint selection ⭐ (the flat-minimum fix)

**Why:** our failure was a *textbook sharp-minimum overshoot* — train kept
improving while val degraded. Averaging weights along the trajectory (or across
runs) provably biases toward **flatter minima that generalize better**, at near-zero
cost. This is the most direct rescue for our exact symptom.

**Changes:**
- **EMA of the trainable weights** (decay 0.999) maintained throughout; evaluate the
  EMA model on val.
- **SWA / LAWA**: average the last *k* (3–10) checkpoints from the late plateau
  (enabled by `keep_last_n=0`, landed in #8).
- **Greedy model-soup**: average the best independently-trained sweep trials.
- **Release the averaged weights**, not the raw last step.
- **🔥 Step-0 quick win:** *soup/average the EXISTING v0.2 sweep (8 trials) + its
  surviving checkpoints* — may recover a better-generalizing model from data we
  already have, with **zero new training**. Do this first as a cheap probe.

**DoD:** the averaged checkpoint beats the best single checkpoint on val; the release
candidate is the averaged model.

## Phase C — Deep-codebook learning (acoustic detail)

**Why:** cb1–7 collapsed. Two real causes: (1) the Moshi depth decoder is **frozen**,
so it can't adapt to TR/HI acoustics; (2) deeper RVQ codebooks model the *residual*
of shallower ones and have **cross-depth gradient conflict**. *Caveat:* deep-RVQ
top-1 accuracy is **inherently low** by construction — judge final quality by audio
reconstruction / DNSMOS (Phase-later eval), not top-1. But cb0 at 14% *is* low and
must rise.

**Changes:**
- **Progressive coarse→fine unmasking** (SoundStorm-style): supervise only the first
  K codebooks early, **gradually unmask deeper ones over the first 5–15% of steps**,
  tied to LR warmup.
- **Moderate per-codebook loss multipliers** (start gentle; the report's
  `[1,2,4,…,128]` over-weights deep books aggressively — treat as a *tunable lever*,
  not a default, or it can starve cb0).
- **Staged depth-decoder unfreeze**: keep frozen early; after shallow codebooks
  stabilize, **low-LR (×0.1) unfreeze of the last 1–3 depth blocks** (Siren-style
  shallow-before-deep alignment) — opt-in lever.
- Keep the **codebook delay pattern** (confirmed helpful by Moshi/MusicGen).

**DoD:** cb0 val acc up materially; cb1–7 non-trivial after the unmask phase; no
collapse of the semantic (cb0) prediction.

## Phase D — Multitask balance (audio ↔ text)

**Why:** v0.1's text collapsed (padding domination — fixed in #8's Phase 0); v0.2's
text learned but we must not let either stream dominate over a long run.

**Changes:**
- **`text_weight=0.3`** (Moshi uses 0.5; v0.2 used 0.2 — split the difference), with
  an optional **curriculum** (anneal 0.5→0.3 over the first 20–30%) or **adaptive
  gating** (if a stream's `L_val/L_train > 1.10–1.15`, bump its weight ×1.05–1.15).
- **Keep `text_padding_weight=0.01`** — our Phase-0 value *proven* to make text learn;
  do **not** adopt Moshi's 0.5 blindly (different padding distribution).
- **Composite early-stop / selection metric:** `0.4·text_val + 0.6·audio_val`
  (single-stream metrics mislead).
- Track **per-stream val CE** and **per-codebook top-1** separately.

**DoD:** both streams improve on val; neither collapses; selection uses the composite.

## Phase E — Sweep redesign, data scaling & the run (execution)

**Why:** the v0.2 sweep optimized 600-step *training* audio loss → it rewarded the
very capacity that overfit at 15k. And the dataset is static, so multi-epoch
training memorizes.

**Changes:**
- **Redesign the sweep:** sweep the **regularization** knobs (`r∈{8,16,32}`,
  `lora_dropout`, `weight_decay`, `lr_lora`) on a **longer proxy horizon**, scored by a
  **generalization metric** (best early-stopped *val*), not 600-step train audio loss.
- **Data scaling/augmentation:** **SpecAugment** (time/freq masks) on audio;
  **synthetic / pseudo-labeled** pairs (how SeamlessM4T / CMU-IWSLT scale low-resource
  S2S).
- **Run order:** smoke (verify all new levers) → regularization sweep → v0.3 run with
  **EMA, early-stopped**, releasing the **averaged** weights.

**DoD:** the sweep selects a generalizing recipe; the v0.3 run's val improves over
training; release the averaged model as `tr-hi-s2st-v0.3`.

---

## Consolidated v0.3 config (starting point)

```yaml
lora:   { r: 16, alpha: 32, dropout: 0.08 }              # Phase A — ↓ from r=64
optim:  { lr_lora: 1.0e-4, lr_depth: 1.0e-4, schedule: cosine_to_0, warmup_pct: 0.08 }
loss:
  text_weight: 0.3                                       # Phase D
  text_padding_weight: 0.01                              # KEEP ours (proven), not Moshi's 0.5
  label_smoothing: 0.1                                   # Phase A
  per_codebook_multipliers: progressive                  # Phase C (start gentle)
  progressive_unmask_fraction: 0.10                      # Phase C
train: { weight_decay: 0.05, grad_clip: 1.0, early_stop_patience: 4 }
depth_decoder: staged_unfreeze                           # Phase C — opt-in lever
ema:   { enabled: true, decay: 0.999 }                   # Phase B ⭐ release averaged weights
augment: { specaugment: true }                           # Phase E
logging: { val_every: 250, keep_last_n: 0 }              # keep_last_n landed in #8
```
**Release candidate = EMA / SWA-averaged (or souped) weights, early-stopped on the composite metric.**

## Implementation checklist (what this PR will add)

- [ ] **A:** `lora_dropout` + cosine-to-0 schedule + `label_smoothing` + grad-clip + early-stop in `train_hierarchical.py` / configs.
- [ ] **B:** EMA weight tracking; an SWA/LAWA + greedy-soup averaging utility (reuses the unlimited checkpoints); release-the-average path. **+ run the step-0 v0.2 soup probe.**
- [ ] **C:** progressive codebook unmasking + per-codebook multipliers in the loss; staged depth-decoder unfreeze lever.
- [ ] **D:** `text_weight` curriculum / adaptive gating; composite val metric; per-stream + per-codebook val logging.
- [ ] **E:** redesigned sweep (regularization knobs, generalization metric, longer proxy); SpecAugment; smoke → sweep → run.
- [ ] Unit tests (torch-free where possible) for the averaging utility, the loss multipliers/unmask schedule, and the composite metric.

## Open decisions (need owner input)

1. **Compute budget / horizon** for the v0.3 run (and whether to early-stop hard at ~2–4k vs. train long with EMA).
2. **Dataset:** stick with the ~1.18M pairs, or invest in **synthetic/augmented** scaling first?
3. **Depth-decoder unfreeze** — opt-in for v0.3, or keep fully frozen and rely on Phases A/B/C-unmask first?
4. **Do the step-0 soup probe** on the existing v0.2 artifacts before committing TPU time?

## References

EXA-deep-researched (`exa-research-pro`) + targeted searches. Key sources:
- **LoRA / overfitting:** [Unsloth LoRA hyperparameters guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide), [Raschka — rank/alpha & overfitting](https://sebastianraschka.com/faq/docs/lora-rank-alpha.html), [HF PEFT methods](https://huggingface.co/blog/samuellimabraz/peft-methods), [overfitting-mitigation survey (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC13053638/).
- **Weight averaging:** [Model Soups (Wortsman 2022)](https://arxiv.org/abs/2203.05482), [SWA (Izmailov 2018)](https://arxiv.org/abs/1803.05407), [SWA for PLM fine-tuning (EMNLP 2022)](https://aclanthology.org/2022.findings-emnlp.363.pdf), [LAWA](https://openreview.net/pdf/ce70672be02be34fa0c7be20323a7e5d165e193c.pdf), [When/Where/Why to Average Weights (ICML 2025)](https://openreview.net/forum?id=JN8O01IZYR), [EMA dynamics](https://arxiv.org/html/2411.18704).
- **RVQ / deep codebooks:** [Moshi paper](https://arxiv.org/html/2410.00037v2), [moshi-finetune config](https://github.com/kyutai-labs/moshi-finetune/blob/main/example/moshi_7B.yaml), [MusicGen](https://arxiv.org/pdf/2306.05284), [SoundStorm](https://research.google/blog/soundstorm-efficient-parallel-audio-generation), [Siren (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.1322.pdf), [Continuous Audio LMs](https://arxiv.org/html/2509.06926v3).
- **Augmentation / S2S scaling:** [SpecAugment](https://arxiv.org/abs/1904.08779), [SeamlessM4T](https://ai.meta.com/blog/seamless-m4t), [CMU IWSLT 2025](https://arxiv.org/html/2506.13143v1).
