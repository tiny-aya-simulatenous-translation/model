# PLAN — Lock in TPU canary success and graduate to Phase 5

> Active goal. Edited automatically by the `update-plan` skill via the
> `Stop` hook. Manual edits via `/plan` (regenerate from current goal)
> or by adding a `#plan <task>` quick-capture line.

## Goal

Validate the **single-host v6e-8 EU** spot canary at >= 200
successful steps with the patches landed during the iter 1-13 self-
heal loop (now extended with patches 20a/b for v6e-specific bf16
NaN + GCS upload), then graduate to the 5000-step Phase 5 run on
either v6e-8 EU (continued) or v6e-64 multi-host once spot capacity
allows. The legacy v4-32 spot path in `us-central2-b` carried the
canary through iter 1-11 (run `8pse8tzk` reached step 100 on v4-32);
it is preserved as historical baseline only -- v4 spot capacity has
been reclaimed.

## Status snapshot (2026-05-08)

- **Topology pivot:** active canary is now single-host TPU v6e-8 spot
  in `europe-west4-a` (QR `tinyaya-stage2-spot-v6e8-eu-qr`, node
  `tinyaya-stage2-spot-v6e8-eu`). v4-32 spot in `us-central2-b` is
  SUSPENDED (no spot capacity) and now treated as legacy.
- Iter 13b on v6e-8 EU completed **20 steps + canonical
  end-of-training save** in 23.3 min wall (run `zd42n7di`); patch 19
  (`save_checkpoint_canonical_final` via `model.to("cpu")` ->
  `save_pretrained`) validated, 2.4 GB checkpoint written.
- Iter 14 in flight on v6e-8 EU: patch 20a (`gsutil cp -r` GCS upload
  inside `save_checkpoint_canonical_final`) + patch 20b
  (`AttentionMaskConverter` monkey-patch clamping mask values >=
  -1e4, called at top of `train_hierarchical.py`) target the
  remaining bf16 NaN (pytorch/xla #4152) + GCS-prefix-not-uploaded
  bugs.
- Single-host SPMD = ONE Python process drives all 8 chips; no
  multi-host wandb shared-mode rendezvous, no host-index gating, no
  GCS run-id polling. Exactly one tmux session, one PID, one wandb
  run.

## Status snapshot (2026-05-06, historical)

- Iter 7 reached **step 100** with loss decreasing
  (9.0273 -> 7.5983) on TPU v4-32 spot @ us-central2-b.
- All 4 hosts attached to one wandb run via shared-mode rendezvous
  (run `8pse8tzk`).
- Steady-state throughput: **3.41 sec/step** (from step 30 onwards).
- No OOM, no deadlock, no cross-host divergence.
- Patches 4-11 validated; patches 12-13 drafted (skip
  `generate_audio_sample` + `run_validation` on TPU during canary).

## Definition of Done

### Canary (immediate, in flight)

- [x] `fsdpv2_lora` strategy completes XLA compile within ~30 min on
  v4-32 spot (achieved iter 7 at T+30).
- [x] Per-chip HBM stays under 31.75 GiB (achieved at
  `batch_size=2`, `grad_accum=2`).
- [x] First step reaches `loss=` line with all 4 hosts contributing
  (achieved iter 7).
- [x] Single wandb run aggregates all 4 hosts (achieved via patch 9
  shared-mode + rank-0 publish to GCS rendezvous).
- [x] >= 100 successful training steps with monotonically decreasing
  loss (9.0273 -> 7.5983 over steps 10 -> 100).
- [ ] >= 200 successful steps (canary `max_steps=200`); first
  checkpoint written to
  `gs://tinyaya-stage2-tpu/checkpoints/stage2-tpu-spot-canary/`.
- [ ] Patches 12 + 13 either landed and verified, or proven
  unnecessary by iter 7+ profile.
- [ ] All commands in `VERIFY.md` (monorepo + simultaneous-translation
  sections) pass on workstation.

### Phase 5 (next)

- [ ] 5000-step run completes (canary -> full config); final loss
  recorded in `memories.md` Milestones.
- [ ] `eval_stage2.py` ASR-BLEU + DNSMOS recorded against
  best-by-val checkpoint.

## Tasks

### Phase 1 — Make compile tractable (SUPERSEDED by Phase 8)

- [x] Add `scan_layers` wrapper around `CohereDecoderLayer` (backbone,
  36 instances). _(see `src/model/scan_utils.py`,
  `replace_layers_with_scan` swaps the HF `model.layers` ModuleList
  with a `_ScannedLayerStack` proxy.)_
- [x] Add `scan_layers` wrapper around `MoshiDecoderLayer` (depth
  decoder, 6 instances). _(same wrapper, applied in
  `composite.TinyAyaMoshiComposite.__init__` when
  `use_scan_layers=True`.)_
- [x] Update `composite.py` to expose `use_scan_layers` flag.
- [x] **Superseded:** `scan_layers` left **disabled** in canary
  config. `_ensure_same_structure` rejects the heterogeneous
  LoRA[0:33] + FullFT[34:35] split (pytorch/xla #8612). The
  XLA-compile-time problem was solved instead by patches 7 (`.item()`
  removal) + 11 (fixed-shape padding). Compile now lands in ~30 min
  on iter 7.

### Phase 2 — Memory headroom

- [x] Add `xla_grad_checkpoint` wrapper around scan units. _(threaded
  through `composite.TinyAyaMoshiComposite(xla_grad_checkpoint=True)`;
  uses `_xla_safe_checkpoint` to dodge the torch 2.9 + torch_xla 2.9
  `_get_device_module("xla")` regression.)_
- [ ] Verify activation memory is under 4 GB per chip via `diagnose()`.
  _(requires live TPU access.)_
- [ ] If still tight, try moving frozen `MoshiDecoderLayer` to bf16
  embeddings only (preserve frozen weights as f32).

### Phase 3 — Strategy selection

- [ ] Re-run `probe_strategies.py` against the real model with
  scan_layers enabled. Capture compile + step time per strategy.
- [ ] Decide between `fsdpv2_lora` and `fsdpv2` based on:
  - per-chip HBM headroom (target: > 3 GB free)
  - step time (lower is better)
  - comm volume (lower is better at LoRA scale)
- [ ] Document the decision in `memories.md` under "TPU strategy
  decisions".

### Phase 4 — Restore full canary fidelity

- [x] Set `max_frames=300` in `configs/stage2_tpu_canary.yaml`.
- [x] Set `depth_chunk_size=16` in `configs/stage2_tpu_canary.yaml`.
- [ ] Re-verify a 5-step canary training run. _(requires live TPU; see
  runbook.)_

### Phase 5 — Full 5000-step run

- [x] Configs ready: `configs/stage2_tpu.yaml` has both
  `train.use_scan_layers: true` and `train.xla_grad_checkpoint: true`.
- [ ] Use `scripts/tpu/launch_qr.sh` to start a fresh queued resource
  with `TPU_STRATEGY=<chosen>` metadata.
- [ ] Monitor via `tmux attach -t train` or
  `tail -f /tmp/train.log` for first hour.
- [ ] Confirm W&B run ID + checkpoint GCS prefix.
- [ ] Run `eval_stage2.py` on the best-by-val checkpoint; record
  ASR-BLEU + DNSMOS.

### Phase 7 — Spot fallback path (TRC v4-32 in us-central2-b)

Triggered 2026-05-05 because the on-demand v4 quota in
`us-central2-b` is currently busy. The TRC welcome email's
recommendation is "fall back to preemptible if/when on-demand is
not available". See `simultaneous-translation/docs/tpu-trc-allocation.md`
for the authoritative quota table.

- [x] Capture the TRC allocation verbatim into the repo:
  `simultaneous-translation/docs/tpu-trc-allocation.md`.
- [x] Mark the stale 5-row table in `docs/tpu-launch-plan.md` §2
  as SUPERSEDED and link to the new doc.
- [x] Log the supersedure in `.factory/memories.md`.
- [x] Add `scripts/tpu/launch_spot.sh` -- a `TRC_PROFILE`-aware
  thin wrapper over `launch_qr.sh` (default profile: `v4-32-uc2b`).
- [x] Add `configs/stage2_tpu_canary_v4_spot.yaml` and
  `configs/stage2_tpu_v4_spot.yaml`: copies of the canary and full
  configs retuned for v4-32 spot (32 GiB/chip, 16 chips, batch 4 *
  grad_accum 2 * 16 = 128 effective; `save_every: 100` for preempt
  resilience).
- [x] Wire `WANDB_RESUME=allow` into `startup_script.sh` when
  `SPOT=1` so a preempt resumes the same wandb run instead of
  forking a new one.
- [ ] Submit the spot QR + run probe -> 5-step -> 50-step canary
  -> 5000-step (requires live TPU).

### Phase 6 — Documentation pass (completed)

- [x] Add the "TPU code documentation style (mandatory)" section to
  `simultaneous-translation/AGENTS.md`.
- [x] Log the documentation-style decision into `.factory/memories.md`.
- [x] Add `.factory/skills/tpu-doc-style/SKILL.md` so future agents pick
  up the convention via `/tpu-doc-style`.
- [x] Apply the convention to every Python file under `src/` and
  `scripts/`: `WHY THIS EXISTS`, NumPy docstrings, GPU-vs-TPU
  callouts.
- [x] `ruff format --check` and `ruff check` both pass cleanly across
  `src/` + `scripts/`; every `*.py` survives `py_compile`; every YAML
  parses; every `*.sh` survives `bash -n`.

### Phase 8 — Self-healing orchestrator + 11 patches (completed)

Triggered 2026-05-06 after iter 1 misdiagnosed a compile storm as
deadlock. Locked-in artifacts in commit `ee01024`.

- [x] `.factory/orchestration/` SPEC v2 + 5 Mermaid diagrams +
  3 playbook MDs (diagnosis-table, tier-definitions,
  checkin-protocol).
- [x] Custom droids: `tpu-watchdog` (read-only state inspector),
  `tpu-diagnoser` (regex classify -> JSON).
- [x] Skills: `tpu-orchestrate` (entry-point), `tpu-redeploy`
  (sub-3-min hot redeploy without QR re-create).
- [x] Background poller (`_artifacts/orch_poll.py`) + scheduled
  check-in helper (`_artifacts/scheduled_checkin.py`).
- [x] Eight iterations driven autonomously; 5 hot-redeploys; 0
  Tier-3 escalations; 5 mandatory user check-ins at
  T+15/30/45/60/90.
- [x] Patches 4-11 applied; patches 12-13 drafted.
  See `.factory/memories.md` "## Architecture decisions" for each
  patch's rationale and the diagnosis row that motivated it.

### Phase 9 — Validate iter 7 on canary

- [ ] Run iter 7 to canary `max_steps=200`; confirm
  - [ ] sec/step holds at ~3.41 (no late recompile spike)
  - [ ] loss continues to decrease step 100 -> 200
  - [ ] first checkpoint write succeeds to GCS
- [ ] Decide on patches 12-13 inclusion (skip
  `generate_audio_sample` + `run_validation` on TPU). If iter 7's
  first audio + val checkpoint pass without recompile spike,
  patches 12-13 are unnecessary.
- [ ] Update `.factory/memories.md` with iter 7 run-id, loss curve,
  and per-step timing as a milestone entry.
- [x] (SUPERSEDED 2026-05-08) v4-32 spot quota was reclaimed; canary
  pivoted to single-host v6e-8 in `europe-west4-a`. See Phase 10.

### Phase 10 — Iter 14 validation on single-host v6e-8 EU

Active 2026-05-08. Topology pivot: single host, 8 chips, ONE Python
process, no cross-host coordination needed.

- [x] Iter 13b -- 20-step run on v6e-8 EU + canonical save validated
  (run `zd42n7di`, 23.3 min wall, 2.4 GB checkpoint). See
  `memories.md` 2026-05-08 milestones.
- [ ] Iter 14 -- patch 20a (`gsutil cp -r` upload to actual GCS
  prefix inside `save_checkpoint_canonical_final`) lands and the
  checkpoint reaches
  `gs://tinyaya-stage2-tpu/checkpoints/stage2-tpu-v6e-spot-canary/`
  (not the local `gs:/...` path bug).
- [ ] Iter 14 -- patch 20b (`AttentionMaskConverter` monkey-patch
  clamping HF `torch.finfo(fp32).min = -3.4e38` mask values to
  >= -1e4) eliminates bf16 NaN at step 1 (pytorch/xla #4152). Run
  with `precision: bfloat16`; confirm loss is finite.
- [ ] Confirm iter 14 reaches `max_steps=20` (then 200) under
  bf16 with non-NaN loss; record run id + loss curve in
  `memories.md`.
- [ ] Once iter 14 is green, decide between staying on v6e-8 spot
  (single-host, 1.60 sec/step fp32) or graduating to v6e-64 multi-
  host pod (8 hosts) for Phase 5 5000-step run.

## Out of scope

- Multi-host scaling beyond v4-32 (deferred to Phase 5 / v4-64)
- Full v4-64 production path (separate config exists; awaits TRC
  on-demand quota)
- Inference / serving path (separate goal)
- Re-enabling `scan_layers` (blocked by `_ensure_same_structure`
  and pytorch/xla #8612; not on the path to first 5000-step run)
