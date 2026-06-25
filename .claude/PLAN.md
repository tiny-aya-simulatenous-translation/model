# PLAN — Next 15k Run: Capacity Fixes, Stability Metrics & W&B Sweeps

Branch: `feat/training-metrics-sweeps` → PR (new).
Strategy detail: `docs/next-15k-run-plan.md`.
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
- [x] **TPU-CONFIRMED (smoke 2026-06-24):** text CE drops — `text 12.02→10.88`
      in one step (was flat ~13.5–14 = random); `val/loss=8.90` finite.
- [ ] (sweep) tune `text_weight` (0.1→?) now that padding no longer dominates;
      consider making the new-token lm_head rows trainable for END_OF_TEXT etc.
- **DoD:** ✅ exact defect documented + fix verified at loss level. Live TPU
  drop is the final tick (Phase 4 smoke).

## Phase 1 — Capacity / recipe bug fixes  ✅ DONE
- [x] Diagnosed the empty `full_ft`: the unfreeze LOGIC is fine — a VM probe
      showed it lifts trainable 3.62M→159.86M (layers 34–35, +156M). The live
      no-op was downstream/effectively-off, NOT a broken match. Decision: keep
      top-N full-FT OFF by default (each block adds ~78M + AdamW state; v6e-8
      HBM is thin at ~26/31 GB), make it a robust opt-in lever.
- [x] `apply_lora` is now **config-driven** (`lora.r`, `lora.alpha`,
      `lora.target_modules`, `lora.num_full_ft_layers`) — sweep-ready. The
      top-N unfreeze is **module-based + asserts** it took effect (no silent
      no-op). `get_param_groups` full-FT detection is now index-agnostic
      (`.layers.`), so it tracks any `num_full_ft_layers`.
- [x] Added the `lora:` block to prod + smoke configs (default
      `num_full_ft_layers: 0`).
- [x] Decoupled LoRA layer-coverage from full-FT via `lora_exclude_top`
      (default 2 = proven surface). Smoke found LoRA-on-ALL-layers spikes HBM
      ~29.5/31 GB + non-finite forward (fsdpv2_lora wrapping the heavy top
      blocks); default now reproduces the finite 0..33 surface @26.4 GB.
- [x] Fixed a brittle warmup sentinel: exact float-equality on bf16/XLA was
      flaky (non-deterministic read noise) → now tolerance-based (atol 1e-2).
- **DoD:** ✅ rank/targets/exclude_top/full-FT config-driven; **TPU-CONFIRMED
  (smoke 2026-06-24):** `[lora] lora_layers=0..33 exclude_top=2`, finite
  forward, 26.4 GB, val finite. full-FT opt-in asserted.

## Phase 2 — Stability dashboard (8 metrics, on-device)  ✅ DONE (CPU-verified)
Implemented as `_group_grad_diag()` (on-device per-group bundle, materialised
in ONE transfer at log boundaries — no per-step `.item()`) + host-side EMA
derivation. Gated by `logging.diag_metrics` (default on). All emit as `diag/*`
in W&B.
- [x] 1. Per-group grad norm; 8. per-group param norm — `_group_grad_diag`.
- [x] 2. Update-to-weight ratio `lr_G·‖g_G‖/‖θ_G‖` (host-derived, no snapshot).
- [x] 3. Non-finite guard (per-group grad + train loss) → `[ALERT]` prints.
- [x] 4. Loss-spike + 5. grad-spike ratios (host EMA, ρ=0.9; alerts >10% / >3×).
- [x] 6. Per-codebook top-1 acc — extended val `cb0_acc` to all 8
      (`val/per_codebook_acc_*`).
- [x] 7. Adam 2nd-moment (`exp_avg_sq`) mean + drift per group.
- [ ] Deferred (optional): Gradient Noise Scale, logit/activation RMS (a `--diag`
      heavy tier; not needed for the first run).
- [x] Alerts: non-finite grads, loss-spike >10%, grad-spike >3× print `[ALERT]`
      (logging, not auto-rollback — safer for a one-shot). `host/rss_gb` already
      tracked (watch the inline-val host-RSS growth).
- **DoD:** ✅ logic CPU-unit-tested (`_group_grad_diag` keys, UWR, NaN-detect,
  spike). All `diag/*` + `val/per_codebook_acc_*` wired to W&B. Live TPU
  visibility rides the next smoke.

## Phase 3 — W&B hyperparameter sweep  ✅ DONE (code; live run is a user step)
Plumbing via the existing CLI-override path (simpler than reading wandb.config
under the late wandb.init): `wandb agent` passes each swept param as a CLI arg
(`${args}` in `sweep_stage2.yaml`).
- [x] `--sweep` flag + CLI args: `lr_lora`/`lr_depth`→optim, `text_weight`→loss,
      `warmup_steps`/`weight_decay`/`max_steps`→train, `val_every`/`val_on_tpu`
      →logging (all via load_config); `lora_r`/`lora_alpha_mult`→`lora.r`/
      `lora.alpha` (=mult·r) explicit. **Unit-tested** (all 10 map correctly).
- [x] `configs/stage2_tpu_v6e_proxy.yaml` — max_frames 200, max_steps 600,
      cheap val, wandb on.
- [x] `sweep/text_ok` flag (1.0 when train text_loss < 11.5) so the sweep can
      reject text-stuck trials. Hyperband early_terminate already in the yaml.
- [x] README runbook updated: CLI-args mechanism + **export TPU env before
      `wandb agent`** (it inherits `${env}`).
- **DoD (code):** ✅ override mapping unit-tested; sweep yaml + proxy config +
  runbook ready. The live `wandb sweep … && wandb agent …` (≥3 trials, pick
  winner) is a **user step** — needs the v6e-8 + `wandb login` (README §Runbook).

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
