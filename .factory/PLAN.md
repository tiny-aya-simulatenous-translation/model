# PLAN — Record iter 24h production success and prepare evaluation

> Active goal. Edited automatically by the `update-plan` skill via the
> `Stop` hook. Manual edits via `/plan` (regenerate from current goal)
> or by adding a `#plan <task>` quick-capture line.

## Goal

Record the first successful **5000-step Phase 5 production run** on
single-host v6e-8 EU spot, commit the validated iter 24h code/docs, and
prepare the downstream evaluation pass against the final canonical
checkpoint. The legacy v4-32 spot path in `us-central2-b` carried the
early canary through iter 1-11 (run `8pse8tzk` reached step 100 on
v4-32); it is preserved as historical baseline only.

## Status snapshot (2026-05-10)

- **Production milestone complete:** iter 24h finished
  **5000/5000 steps** on single-host v6e-8 EU spot
  (`tinyaya-stage2-spot-v6e8-eu`) with exit status 0.
- **W&B:** `v6e-spot-stage2-5k-iter24h`, run `7rrjupc7`:
  https://wandb.ai/cataluna84/tinyaya-stage2-tpu/runs/7rrjupc7.
- **Final training stats:** step 5000 loss 5.3558
  (`text=10.3176`, `audio=4.3240`); training wall time 615.9 min.
- **Checkpoint:** canonical final save uploaded to
  `gs://tinyaya-stage2-tpu/checkpoints/stage2-tpu-v6e-spot/step_005000_final/`
  (8 objects, 2.37 GiB: depth decoder, text embeddings, audio heads,
  projection, metadata, and PEFT adapter files).
- **Stability:** no NaN, OOM, RESOURCE_EXHAUSTED, fatal, traceback,
  bus-error, or kernel-panic signals were found in `/tmp/train.log`.
- **Compile behavior:** 18 XLA compilation causes total; 12 before the
  first visible step and 6 around steps 1-2. No late recompiles through
  step 5000. Future cleanup can pre-warm the optimizer-state graph
  before the visible `step 1` line, but it is not blocking correctness.
- **Next milestone:** run `eval_stage2.py` against the final checkpoint
  and record ASR-BLEU + DNSMOS.

## Status snapshot (2026-05-08)

- **Topology pivot (historical):** canary moved to single-host TPU v6e-8 spot
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
- [x] >= 200 successful steps (canary `max_steps=200`); checkpoint
  written to
  `gs://tinyaya-stage2-tpu/checkpoints/stage2-tpu-v6e-spot-canary/step_000200_final/`
  (completed in Phase 11 on v6e-8 EU).
- [x] Patches 12 + 13 either landed and verified, or proven
  unnecessary for the production path. They remain deferred because
  the v6e-8 5000-step run used canonical end-of-training save only.
- [ ] All commands in `VERIFY.md` (monorepo + simultaneous-translation
  sections) pass on workstation.

### Phase 5 (next)

- [x] 5000-step run completes (canary -> full config); final loss
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
- [x] Use `scripts/tpu/launch_qr.sh` to start a fresh queued resource
  with `TPU_STRATEGY=<chosen>` metadata.
- [x] Monitor via `tmux attach -t train` or
  `tail -f /tmp/train.log` for first hour.
- [x] Confirm W&B run ID + checkpoint GCS prefix.
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

### Phase 10 — Iter 14 validation on single-host v6e-8 EU (COMPLETED)

Active 2026-05-08. Topology pivot: single host, 8 chips, ONE Python
process, no cross-host coordination needed.

- [x] Iter 13b -- 20-step run on v6e-8 EU + canonical save validated
  (run `zd42n7di`, 23.3 min wall, 2.4 GB checkpoint). See
  `memories.md` 2026-05-08 milestones.
- [x] Iter 14 -- patch 20a (`gsutil cp -r` upload to actual GCS
  prefix inside `save_checkpoint_canonical_final`) lands; the
  checkpoint reached
  `gs://tinyaya-stage2-tpu/checkpoints/stage2-tpu-v6e-spot-canary/step_000020_final/`
  (2.36 GiB, all 5 components + peft_adapter/).
- [x] Iter 14 -- patch 20b (`AttentionMaskConverter` monkey-patch
  clamping HF `torch.finfo(fp32).min = -3.4e38` mask values to
  >= -1e4) eliminates bf16 NaN at step 1 (pytorch/xla #4152). Run
  `ovttp92v` with `precision: bfloat16` was finite throughout: loss
  10.68 -> 8.59 over 20 steps in 18.6 min wall (1.76 sec/step).
- [x] Decision: stay on v6e-8 spot for the canary (1.76 sec/step in
  bf16 is competitive with v4-32's 3.41 sec/step in fp32) and reserve
  v6e-64 multi-host for the eventual 5000-step Phase 5 run.

### Phase 11 — 200-step bf16 canary on v6e-8 EU (DONE)

Iter 17 closed the phase: batch=8, grad_accum=2, grad_checkpoint=true,
bf16 + patch 20a/b. 200 steps in 31.8 min wall (compile=482s, then
2.36 sec/step), loss 10.33 -> 6.96, no NaN, canonical save uploaded
to GCS at `gs://tinyaya-stage2-tpu/checkpoints/stage2-tpu-v6e-spot-
canary/step_000200_final/` (2.36 GiB, 5 components + metadata.json).
Throughput **5.96x** iter 14 (54.2 vs 9.1 examples/sec).

- [x] Bump `train.max_steps: 20 -> 200`.
- [x] Iter 17 tarball deployed; wandb run `0v4uizbu`.
- [x] step >= 200 with monotonic loss decrease.
- [x] Patch-19 canonical save complete; GCS bucket lists all 5 files.
- [x] Final GCS URL surfaced to user.
- [x] Memory probe (`_artifacts/memory_probe.py`) confirmed real HBM
  ceiling = 31.246 GiB/chip (not 32 GiB marketing); peak ~1.2 GiB
  during 5x4096 matmul warmup; zero `aten::` fallbacks.

### Phase 12 — Iter 18-20 throughput + telemetry (DONE)

- [x] Iter 18: 5 levers (batch=16, max_frames=400, real HBM probe,
  profiler, debug-level=1) committed at `2400ada`.
- [x] Iter 19: 3 fixes + lever 6 (fused clip_grad_norm, variant 6a)
  committed at `7073d77`.
- [x] Iter 20: profile capture finally working end-to-end via
  standalone `_artifacts/capture_xla_profile.py` SCP'd alongside the
  launcher. 4 trace dirs uploaded to
  `gs://tinyaya-stage2-tpu/profiles/iter20-*` (`9131e0f`).

### Phase 13 — Iter 24 5000-step production run (DONE)

Goal: complete the first 5000-step Phase 5 production run on
single-host v6e-8. Iter 21/22/23 all OOM-crashed at step 258 with
89 GiB HLO temp on a 31 GiB chip; root cause was XLA SPMD
``cache_all_gather=True`` (defaulted, openxla/xla #20508) retaining
the all-gather output of every sharded layer's full params from
forward through backward. Iter 24 applies the
``apply_backward_optimization_barrier`` per FSDPv2-wrapped layer
(pytorch/xla #6379 step 5; HF Llama 2 SPMD reference pattern) and
re-enables every previously gated knob.

- [x] Iter 21: rename canary -> production v6e config; start 5000-step
  run (`61b87fb`). OOM at step 258.
- [x] Iter 22: stable grad topology via `set_to_none=False` +
  requires_grad iteration (`7948cbb`). Same OOM at step 258.
- [x] Iter 23: gate lever 6 behind `enable_clip_grad_norm: false`
  (`659ecb9`). Same OOM at step 258.
- [x] Iter 24 fix A -- new helper
  `_apply_fsdpv2_backward_barriers(model)` in
  `scripts/train_hierarchical.py`; called after `backend.wrap_model`
  on the TPU path. Walks `model.modules()`, registers a backward
  hook that fires `xm.optimization_barrier_(grads + grad_input)` on
  every Cohere2DecoderLayer / CohereDecoderLayer / MoshiDecoderLayer
  via class-name match. Logs
  `[fsdpv2] applied backward optimization barrier to 42 layers`.
- [x] Iter 24 fix B (REVERTED in 24b/24c) -- dropping
  `--xla_tpu_enable_flash_attention=false` proved orthogonal to NaN.
  Restored in 24b/24c to match iter 17-23 stable LIBTPU_INIT_ARGS.
- [x] Iter 24 re-enable lever 6 (`enable_clip_grad_norm: true`).
- [x] Build iter 24 tarball + upload + deploy + launch.
- [x] Iter 24a (5ep7z7ab): NaN at step 24 (audio_loss). flash-on.
- [x] Iter 24b (49kjpp6v): bit-identical NaN at step 24. flash-off.
  Confirmed flash-attention is not the trigger.
- [x] Diagnose via exa: pytorch/xla #8591/#8778 (v6e bf16
  reduce-scatter NaN bug, no fix); FSDPv2 has no
  `fp32_reduce_scatter`. Per-layer reduce-scatters from the iter
  24 class-name fix are the trigger; iter 21-23 had a stale class
  name so only ONE outer reduce-scatter -> bf16 stable.
- [x] Iter 24c -- revert tpu_backend.py auto_wrap_policy class-name
  patch (`layer_type_names = ("CohereDecoderLayer",
  "MoshiDecoderLayer")`); keep barrier helper unchanged
  (`register_full_backward_hook` works without FSDPv2 wrapping);
  bump `wandb_run_name -> v6e-spot-stage2-5k-iter24c`.
- [x] Iter 24c: build tarball, upload to
  `gs://tinyaya-stage2-tpu/code/tinyaya-repo-iter24c.tar.gz`,
  deploy + launch.
- [x] Iter 24c: announce wandb URL
  https://wandb.ai/cataluna84/tinyaya-stage2-tpu/runs/83evwy38;
  validate `[fsdpv2] applied backward optimization barrier to 42
  layers` (CONFIRMED at runtime).
- [x] Iter 24c: FAILED at step 32 (audio_loss NaN first); per-layer
  wraps contributed but barrier helper remained a NaN trigger.
- [x] Iter 24d: drop lever 6, keep barrier; FAILED at step 3. Lever 6
  was masking the NaN, not causing it.
- [x] Iter 24e: drop barrier helper and lever 6; CLEAN through step
  258, then historical boundary reproduced.
- [x] Iter 24f: b=8/grad_accum=4; CLEAN through step 258, then step
  259 compiled a 28.3 GB graph and audio_loss NaN'd first.
- [x] Iter 24g patch: freeze the step-259 graph topology via TPU
  `drop_last=True`, static-shape assertions, and disabled HF SDPA
  mask elision; upload
  `gs://tinyaya-stage2-tpu/code/tinyaya-repo-iter24g.tar.gz`.
- [x] Iter 24h patch: pad TPU tail batches instead of `drop_last`,
  and guarantee epoch reset happens only between optimizer steps;
  upload `gs://tinyaya-stage2-tpu/code/tinyaya-repo-iter24h.tar.gz`.
- [x] Launch iter 24h on a usable TPU; announce W&B URL.
  W&B: https://wandb.ai/cataluna84/tinyaya-stage2-tpu/runs/7rrjupc7.
- [x] Iter 24h: pass step 300 with no NaN and no RESOURCE_EXHAUSTED.
- [x] Iter 24h: reach step 5000 with stable finite loss;
  canonical save to
  `gs://tinyaya-stage2-tpu/checkpoints/stage2-tpu-v6e-spot/step_005000_final/`
  succeeds.
- [x] Verify final GCS checkpoint listing: 8 objects, 2.37 GiB.
- [x] Commit iter 24h after live validation.

### Phase 14 — Post-run evaluation and cleanup (next)

- [ ] Run `eval_stage2.py` against
  `gs://tinyaya-stage2-tpu/checkpoints/stage2-tpu-v6e-spot/step_005000_final/`.
- [ ] Record ASR-BLEU, DNSMOS, representative samples, and any quality
  regressions in `memories.md`.
- [ ] Decide whether to run a follow-up quality-tuning pass or move to
  v6e-64 / inference packaging.
- [ ] Cleanup patch: move optimizer-state/warmup compilation before
  visible `step 1` so the first printed training step is steady-state.

## Out of scope

- Multi-host scaling beyond v4-32 (deferred to Phase 5 / v4-64)
- Full v4-64 production path (separate config exists; awaits TRC
  on-demand quota)
- Inference / serving path (separate goal)
- Re-enabling `scan_layers` (blocked by `_ensure_same_structure`
  and pytorch/xla #8612; not on the path to first 5000-step run)
