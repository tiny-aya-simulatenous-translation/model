# PLAN — Next 15k Run: Capacity Fixes, Stability Metrics & W&B Sweeps

Branch: `feat/training-metrics-sweeps` → PR (new).
Strategy detail: `simultaneous-translation/docs/next-15k-run-plan.md`.
Supersedes the merged public-release plan (PR #6; history in PROGRESS.md).
Inline TPU val now works (`val_on_tpu: true`) — use it for the live curve.

## Goal
Make the next 22 h v6e-8 run actually move the needle: fix the two recipe
bugs that block learning, instrument the run so a one-shot can't silently
fail, and find a good recipe cheaply via a W&B sweep before committing the
expensive slot.

## Definition of Done
- [ ] Text stream learns (val/text_loss drops well below ln(V)≈12.5) OR the
      data-pipeline root cause is identified + ticketed.
- [ ] `full_ft` (top-N layer unfreeze) is either active (non-empty group) or
      the dead code is removed by decision.
- [ ] 8-metric stability dashboard live in W&B (on-device, no per-step host
      sync); non-finite/loss-spike guards wired to the existing rollback path.
- [ ] W&B sweep runnable end-to-end (`wandb sweep` → agent → dashboard) on a
      proxy config; winner promotable to the full config.
- [ ] One full run launched with the swept recipe; GPU ASR-BLEU/DNSMOS eval
      filled into the model card `model-index`.

---

## Phase 0 — Triage the text stream  ✅ ROOT-CAUSED + FIXED (TPU confirm pending)
**Root cause (empirically verified on the train split, 2026-06-24):** the
supervised target text is **38% real tokens / 62% special padding tokens**
(IN_WORD 35% + END_OF_TEXT 15% + TEXT_PADDING 11%; 0% zero_padding — so real
HI/TR text IS present, every row). The padding specials (262144–262147) have
**mean-initialised, ~frozen lm_head rows**. Stage-2's hierarchical loss weighted
them EQUALLY with real tokens (uniform `_masked_sum`), so the un-learnable
padding pinned text CE at ~ln(V) and drowned out the real tokens. **It was a
regression** — Stage-1 (`src/training/loss.py`) down-weights padding 100×.
- [x] Confirmed real text present (not missing-alignments) via CPU data probe.
- [x] **Fix:** ported padding-aware weighting into `translation_loss.py`
      (`text_padding_weight=0.01`, `zero_padding_weight=0.0`); plumbed through
      both call sites + configs; numerically verified (uniform 7.61 →
      weighted 2.13, real-token-focused, finite).
- [ ] Confirm on TPU that train/val text CE actually drops (rides next smoke).
- [ ] (sweep) tune `text_weight` (0.1→?) now that padding no longer dominates;
      consider making the new-token lm_head rows trainable for END_OF_TEXT etc.
- **DoD:** ✅ exact defect documented + fix verified at loss level. Live TPU
  drop is the final tick (Phase 4 smoke).

## Phase 1 — Capacity / recipe bug fixes
- [ ] Fix `get_param_groups`: the `full_ft` group (layers 34–35) is EMPTY at
      runtime — the `layers.{34,35}` name match fails. Make top-N unfreeze
      actually populate, or delete the path intentionally.
- [ ] Make LoRA rank/targets configurable from cfg (`lora.r`, `lora.alpha`,
      `lora.target_modules`) so the sweep can vary them (currently hard-coded
      r=16, q/v/embed in `lora_setup.apply_lora`).
- **DoD:** param-group printout shows the intended trainable surface; rank +
  targets driven by config.

## Phase 2 — Stability dashboard (8 metrics, on-device)
Implement in the existing `running_xla` accumulator pattern in
`scripts/train_hierarchical.py`; export every `log_every`; NO per-step
`.item()`. (Formulas + thresholds: docs/next-15k-run-plan.md §2.)
- [ ] 1. Per-group grad norm — extend the existing total-norm loop (~L1659).
- [ ] 2. Update-to-weight ratio `lr_G·‖g_G‖/‖θ_G‖` (no snapshot needed).
- [ ] 3. Non-finite guard on loss+grad → reuse rollback path.
- [ ] 4. Loss-spike ratio (EMA ρ=0.995); 5. grad-norm spike ratio (EMA).
- [ ] 6. Per-codebook top-1 acc + perplexity (extend val `cb0_acc` to all 8).
- [ ] 7. Adam 2nd-moment drift per group (read `exp_avg_sq`).
- [ ] 8. Per-group param norms.
- [ ] Periodic/optional (`--diag`): Gradient Noise Scale, logit/activation RMS.
- [ ] Alerts: non-finite>0 or grad-spike>10× → halt+rollback+LR×0.5;
      loss-spike>10% w/ grad-spike>2× → rewind 1 ckpt. Watch `host/rss_gb`
      (smoke showed 67→124 GB with val on — possible inline-val host leak).
- **DoD:** all 8 visible in W&B on a smoke; alerts fire on injected NaN.

## Phase 3 — W&B hyperparameter sweep
Artifacts in `simultaneous-translation/sweeps/` (scaffolded in this PR):
`sweep_stage2.yaml`, `README.md`.
- [ ] `--sweep` flag in `train_hierarchical.py`: map flat `wandb.config` keys
      → nested cfg overrides (`lr_lora`, `lr_depth`, `lora_r`,
      `lora_alpha_mult`, `text_weight`, `warmup_steps`, `weight_decay`,
      `max_steps`, `val_every`, `val_on_tpu`).
- [ ] `configs/stage2_tpu_v6e_proxy.yaml` — short/cheap proxy config.
- [ ] Verify hyperband early-termination kills weak trials; verify a trial
      with text CE≈ln(V) is auto-rejectable (log a `sweep/text_ok` flag).
- **DoD:** `wandb sweep … && wandb agent …` runs ≥3 proxy trials, dashboard
  ranks them, winner HPs copy cleanly into the prod config.

## Phase 4 — Full run + eval
- [ ] Promote winning recipe → `configs/stage2_tpu_v6e_v2.yaml`.
- [ ] Launch 22 h run (`launch_release.sh`); monitor the dashboard + inline val.
- [ ] GPU ASR-BLEU + DNSMOS (`scripts/eval_release.py`); fill `model-index`
      in `MODEL_CARD.md`; (decision) flip HF repos public.
- **DoD:** model card has real eval numbers; run reproducible from config.

---

## Steps YOU take (manual / decisions)
1. **Approve scope** — confirm the order (Phase 0 text-bug first) and whether
   to raise LoRA rank / unfreeze layers now or let the sweep decide.
2. **Triage call (Phase 0):** is the text stream in scope for this run, or
   ship audio-only and fix text separately?
3. **Run the sweep (Phase 3):** `wandb sweep sweeps/sweep_stage2.yaml`, then
   `wandb agent <id>` on the v6e-8 (sweeps/README.md). Pick the winner.
4. **Greenlight the 22 h run** with the chosen config (spot v6e-8 billing).
5. **Post-run:** review ASR-BLEU/DNSMOS; decide public release + `new_version`.

## Notes / risks
- Single v6e-8 ⇒ sweep trials are SEQUENTIAL and each pays ~18 min compile;
  keep proxy `max_steps` small and lean on hyperband. Consider a tiny CPU/GPU
  proxy for the LR-range test.
- Don't sweep on the 22 h slot. Don't shorten the cosine horizon after launch.
