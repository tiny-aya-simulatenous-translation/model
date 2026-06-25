# Next 15k-Step Run — Stability Metrics, Capacity Audit & Sweep Plan

Planning doc for the next 22 h v6e-8 run. Grounded in the **actual** model
(pulled live from the run, not the "8B" assumption) plus two exa-research-pro
deep dives (stability metrics `r_01kvvzrnv0...`; capacity/sweeps/thresholds
`r_01kvvzt6p7...`).

---

## 0. TL;DR — priorities in order

1. **Fix the text stream FIRST (data/recipe bug, not capacity).** Text CE sits
   at ≈ `ln(V)` = random. Per research this is the textbook signature of a
   pipeline bug (targets not routed / wrong tokenization / no gradient), *not* a
   capacity ceiling. Do **not** spend the 22 h slot until this is triaged.
2. **Fix the silent capacity bug:** the intended top-2-layer unfreeze (`full_ft`)
   **is not happening** — that param group is empty at runtime.
3. **Add a compact 8-metric stability dashboard** (below) — cheap, on-device,
   catches divergence + diagnoses the plateau on a one-shot run.
4. **Capacity:** audio is bottlenecked by the **depth decoder (80% of trainable
   params)** + a *very thin* backbone adapter (q/v-LoRA only, 7.8M). Plateau ≈
   that small decoder saturating. Bump rank/decoder only after #1–#2.
5. **Sweep cheaply** (proxy LR-range test → ASHA proxy sweep) before committing
   the expensive slot.

---

## 1. Current state (ground truth, from the live run)

Composite **5.17B total, 122M trainable (2.4%)**, backbone **36 layers**.

| Trainable group | Params | % of trainable | LR | What it is |
|---|---|---|---|---|
| `depth` (Moshi decoder I/O incl. codebook heads) | **97.8M** | **80%** | 2.5e-4 | the audio generator |
| `projection` (2048→hidden) | 8.4M | 7% | 5e-4 | frozen-backbone → depth bridge |
| `lora` (q_proj+v_proj+embed only, r=16 α=32) | 7.8M | 6% | 1.5e-4 | backbone adaptation |
| `text_embed` | 4.2M | 3% | 5e-4 | |
| `model_audio_embed` | 4.2M | 3% | 5e-4 | |
| `full_ft` (top-2 layers 34–35) | **0 (EMPTY!)** | — | — | **intended unfreeze not active** |

Loss weights: `text_weight=0.1`, `audio_weight=1.0`. LoRA targets:
`q_proj, v_proj, embed_tokens` only (no k/o_proj, no MLP).

Already logged (keep): `train/{loss,text_loss,audio_loss}`,
`train/per_codebook_loss_{0..7}`, `train/grad_norm` (global),
`train/lr_{group}`, `mem/peak_gb`, `host/rss_gb`, and now-finite
`val/{loss,text_loss,audio_loss,cb0_acc,per_codebook_loss_*}`.
Established pattern: on-device `running_xla` accumulators materialized once per
`log_every` (10). Add new metrics the same way — **no per-step `.item()`**.

---

## 2. Stability dashboard — the 8 metrics to add

Priority-ordered. All cheap, on-device, exported every `log_every`. Alert
thresholds from large-model logbooks (OPT/PaLM/GLM) + the GNS paper.

| # | Metric | Formula | Cheap impl (where) | Alerts on |
|---|---|---|---|---|
| 1 | **Per-group grad norm** | `‖g_G‖₂` per group | extend the existing total-norm loop (`train_hierarchical.py:~1659`) to bucket sum-of-squares by `optimizer.param_groups` name — ~free | which component is hot/dead; **is the text branch even getting gradient?** |
| 2 | **Update-to-weight ratio** | `UWR_G ≈ lr_G·‖g_G‖ / (‖θ_G‖+ε)` | needs `‖θ_G‖` (sum-sq per group, cheap) + #1; **no param snapshot needed** | `UWR≈0` → frozen group (text!); `>1e-3` → oversized updates |
| 3 | **Non-finite guard** | `Σ¬isfinite(loss)+grad` | one `torch.isfinite` on the already-materialized loss/grad-norm | **any** non-zero → bf16 overflow / imminent NaN → halt+rollback |
| 4 | **Loss-spike ratio** | `(Lₜ−EMA)/(EMA+ε)`, EMA ρ=0.995 | EMA scalar on-device; loss already materialized | `>0.10` (+grad spike) → divergence → reduce LR ×0.5 / rollback |
| 5 | **Grad-norm spike ratio** | `‖g‖/(EMA‖g‖+ε)` | EMA of the global grad-norm we already compute | `>3×` → catastrophic spike |
| 6 | **Per-codebook top-1 acc + perplexity** | `acc_b`, `exp(CE_b)` | extend val `cb0_acc` to all 8 codebooks (and a train sample) | a codebook stuck low → audio-quality plateau / dead codebook |
| 7 | **Adam 2nd-moment drift** | `mean√v` per group, ratio vs EMA | read `optimizer.state[p]['exp_avg_sq']` mean per group at log_every | `>2×` or `<0.5×` → optimizer-state drift / β₂ mismatch |
| 8 | **Param norm per group** | `‖θ_G‖₂` | sum-sq per group at log_every (also feeds #2) | runaway growth (divergence) or stuck-zero group |

**Optional / periodic (every ~500–2000 steps, higher cost):**
- **Gradient Noise Scale** (McCandlish) → critical-batch-size sanity (are we
  over/under effective-batch 256?). Needs per-microbatch grad variance — we
  already loop micro-batches, so accumulate `Σg` and `Σ‖g‖²` there.
- **Logit max / activation RMS** (bf16 blow-up) — needs a forward hook on the
  heads; make it a `--diag` toggle.

**Action playbook (encode as thresholds):** non-finite>0 or grad-spike>10×
→ halt + restore last ckpt + LR×0.5 + clip; loss-spike>10% with grad-spike>2×
→ rewind one ckpt; Adam-drift>2× or param-norm runaway → drop group LR.

---

## 3. Capacity audit — is depth/breadth/params enough?

**Verdict: the architecture is *not* the first bottleneck — recipe/data are.**
But there are two real capacity limits once those are fixed.

- **Text stream (priority): NOT capacity.** CE≈`ln(V)` from step 0 = uniform
  prediction = data/routing bug. Triage (minutes): (a) compute text CE on one
  batch — if `=ln(V)`, inspect tokenization/target routing; (b) 100–300-step
  text-only run — if it drops, model is fine, data is wrong; (c) metric #1/#2
  above tells you instantly if the text branch gets gradient. Likely culprits
  here: `text_weight=0.1` is tiny, the lm_head is **frozen** (only embed-LoRA
  trains the text path), and the interleaver fills most frames with
  `TEXT_PADDING` so real text tokens are sparse.
- **Backbone adaptation is very thin.** q/v-LoRA only, r=16, 7.8M (0.15%). For a
  new cross-modal (text→audio) mapping this is the classic low-rank ceiling.
  Order of escalation (research-backed): (1) **r 16→32, α=2r=64**; (2) extend
  LoRA targets to `k_proj,o_proj` (+MLP if needed); (3) **actually enable the
  top-2 (→5) layer unfreeze** — it's currently broken; (4) only then go r→64.
- **Audio = the depth decoder (80% of trainable).** Moshi-default depth decoder
  is ~8 layers/512-hidden; ours is 6 layers/1024. The ~step-8k audio plateau is
  most consistent with this small decoder + thin backbone adapter saturating. If
  per-codebook top-1 (metric #6) shows low codebook utilization / later
  codebooks stuck → widen/deepen: **6→10–12 layers, ffn 5632→ keep, hidden
  1024→ consider**, and/or add per-codebook head capacity. Do this *after* the
  cheaper LoRA/unfreeze knobs.

**How to disambiguate (use the new metrics):** capacity ceiling = audio
plateaus *and* low codebook utilization *and* no gain from LR fixes. Data limit
= more/cleaner data helps, big train/val gap. Optimization = LR/warmup/unfreeze
fix moves it. The dashboard above distinguishes all three.

---

## 4. Hyperparameter sweep plan (proxy-first; don't blind-burn 22 h)

Do the expensive work once, after cheap pruning. Stages:

- **Stage A — LR-range test** (Smith) on a small μP proxy (~40M), 300 steps →
  LR bands. Map to full scale with μTransfer coordinate-check.
- **Stage B — ASHA proxy sweep:** 24 trials × 400 steps, prune bottom 50% at
  ~250 steps. Grid:
  - LoRA `r ∈ {16,32,64}`, `α ∈ {r, 2r}`, dropout `{0, 0.1}`
  - LoRA LR `{1e-4, 3e-4}`; **depth/decoder LR `{2.5e-4, 5e-4}`** (ours may be
    under-LR'd vs the 5e-4 the heads/embeds get)
  - **text/audio weight γ `{0.5, 1.0, 2.0}`** (currently 0.1 — almost certainly
    too low once the text data bug is fixed)
  - warmup `{2%, 5%, 10%}`, weight-decay `{0, 1e-3, 1e-2}`
  - **Abort any trial where text CE ≈ `ln(V)`** — that's the pipeline bug, not a
    bad HP.
- **Stage C — refine** top 4–6 at 1000 steps (warmup/WD/decoder-capacity knobs).
- **Stage D — the 22 h slot:** either 2–3 sequential full-model candidates
  (~7–8 h each, compare ASR-BLEU/DNSMOS) **or** one full run if Stage C has a
  clear winner. Inline `val/loss` (now finite!) + per-codebook acc are the live
  signals; checkpoint every 1000 steps as today.

Short (300–1000-step) proxies predict 15k-step quality well (Spearman >0.8 with
μTransfer); the text-stream regression shows up in <200 steps.

---

## 5. Downstream thresholds (so we know "good enough")

Eval = **ASR-BLEU** (Whisper-transcribe generated audio → BLEU) + **DNSMOS**
(naturalness). For tr↔hi S2ST:

| Band | ASR-BLEU | DNSMOS (1–5) |
|---|---|---|
| Weak (unusable) | < 10 | < 2.0 |
| **Decent (target for v0.x)** | **10–20** | **2.0–3.0** |
| Strong (competitive w/ Seamless/Moshi) | > 20 | > 3.0 |

S2ST is harder than text MT — many published systems sit at BLEU 10–20.
**Data/steps to reach "decent":** ~1M–few-M aligned pairs, **2–4 epochs**.
Our 15k steps @ eff-batch-256 ≈ ~3 epochs over ~1.2M — in the right ballpark for
audio, *if* capacity/recipe are fixed. Reaching "strong" needs both more data
and decoder widening. Add `model-index` to the model card once measured.

---

## 6. Concrete next actions (ordered)

1. **Triage the text stream** (metric #1/#2 + 100-step text-only probe). Raise
   `text_weight` 0.1→~1.0 and consider unfreezing/​LoRA-ing the lm_head.
2. **Fix the `full_ft` group match** so top-2 layers actually train (or delete
   the dead code path and decide intentionally).
3. **Implement the 8-metric dashboard** (§2) in the existing `running_xla`
   pattern; wire the alert thresholds into the existing non-finite/rollback path.
4. **Run the proxy LR-range test + ASHA sweep** (§4) — cheapest path to a good
   recipe; do NOT sweep on the 22 h slot.
5. **Then** decide capacity bumps (LoRA r→32 + targets, decoder 6→10–12) based on
   per-codebook utilization from metric #6.
6. Launch the 22 h run with inline `val_on_tpu: true` (now working) + GPU
   ASR-BLEU/DNSMOS eval after, fill `model-index`.

_Sources: McCandlish GNS (1812.06162), OPT-175B logbook, GLM-130B (2210.02414),
μTransfer (Tensor Programs V), LoRA (2106.09685)/QLoRA (2305.14314), Moshi/Mimi
(kyutai), SeamlessM4T, ASHA/Hyperband (1810.05934), LR-range test (1506.01186),
DNSMOS, COMPASS S2ST benchmark (2606.03241)._
