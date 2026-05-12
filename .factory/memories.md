# memories — long-term project decisions

> Permanent record of architecture decisions, gotchas, and domain
> knowledge for the TinyAya Stage 2 repo. Append via `/remember <text>`
> or by hand. Decisions reversed later are marked SUPERSEDED rather
> than deleted, so the *reason for the change* is preserved.

---

## Project context

- **Name:** TinyAya Stage 2
- **Goal:** Speech-to-speech TR <-> HI translation at scale on TPU TRC
- **Dataset:** 9,212 accepted parallel pairs (2,440 real FLEURS +
  6,772 TTS)
- **Composite model size:** ~5.17B parameters (3.36B backbone Cohere +
  ~617M frozen Moshi depth decoder + LoRA + projections)
- **Trainable params:** ~274M (~5.3% of total) — mix of LoRA, projection
  heads, depth I/O, text embeddings.

---

## Architecture decisions

### 2026-05-12: opt-prod5k 5000-step production pass completed

**Decision:** The combined Phase 1+2+3 production config
(log_every=10, compile_warmup_steps=1, b=8/g=4) completed 5000/5000
steps on v6e-8 spot with exit 0 and canonical save. p50=6.14s,
p99=6.76s, examples/sec=43.04, final loss=5.105. This is 11.8% faster
per step and 4.7% lower loss versus the iter 24h baseline.

**W&B:** `kzsijxv5` (v6e-spot-stage2-opt-prod5k)
**Checkpoint:**
`gs://tinyaya-stage2-tpu/checkpoints/stage2-tpu-v6e-spot-opt-prod5k/step_005000_final/`

**Gotcha:** The TPU was preempted after the run completed (spot VM
reclaimed). The QR will need recreation for future runs.

### 2026-05-12: W&B shared mode ignores wandb.log step=N parameter

**Decision:** `wandb.log(data, step=N)` is silently ignored in W&B
shared mode (used for SPMD TPU). The `_step` counter auto-increments
from 0 regardless, causing charts to show 0..499 instead of training
steps 10..5000 when log_every=10. Fix: use `wandb.define_metric()`
with a custom `global_step` metric as `step_metric` for all
train/perf/val/audio/mem metric groups, and include `"global_step":
step` in every `wandb.log()` data dict instead of passing `step=step`.

**Gotcha:** This affects all future runs. Past runs (iter24h, hot300,
hot1k, warmup-r1, prod5k) retain the broken step counter in W&B.

### 2026-05-11: log_every=10 hot-redeploy validation promoted to production candidate

**Decision:** `log_every=10` is the current Phase 1 production
candidate, but only when launched through the hot-redeploy path that
does not export `PT_XLA_DEBUG_LEVEL=1`. The hot-redeploy 1000-step
validation (`opt-1-log10-hot1k`, W&B `pdhz1f95`) completed 1000/1000
with p50 `5.92593s`, examples/sec `43.05558`, final loss `6.17039`,
and final checkpoint
`gs://tinyaya-stage2-tpu/checkpoints/stage2-tpu-v6e-spot-opt-log10-hot1k/step_001000_final/`.

**Gotcha:** The earlier startup-path 1000-step retry (`58k4t99h`)
completed but regressed to p50 `6.9605s`. A hot-redeploy 300-step
repeat (`opt-1-log10-hot300`, W&B `c2kq10nn`) restored the original
300-step throughput envelope, so treat the startup-path result as
throughput-confounded by fresh-start/debug environment rather than as a
rejection of `log_every=10`.

### 2026-05-11: compile warmup passes 300-step gate with sampled sentinel

**Decision:** `compile_warmup_steps=1` with a sampled trainable-weight
sentinel (8 tensors x 16 values each) is the current Phase 2
candidate. It is throughput-neutral versus the hot-redeploy log10
baseline (p50 0.8% faster, p99 2.6% tighter, examples/sec 1.2% slower)
and eliminates the late-recompile outliers that inflate p99 in
non-warmup runs.

**Gotcha:** The initial full XLA reduction checksum
(`trainable_weight_checksum`) triggered a false-positive weight-drift
assertion (delta 0.08 over 2283.73 in bf16). The sampled sentinel
copies small parameter slices before/after warmup and compares them
element-wise, avoiding the unstable global sum reduction.

### 2026-05-11: batch_size=16/grad_accum=2 rejected due to bf16 reduce-scatter NaN

**Decision:** `batch_size=16` and `grad_accum=2` (effective batch 256)
is rejected for the v6e-8 FSDPv2+bf16 path. The 20-step smoke passed
with excellent throughput (4.21s/step, 60.8 examples/sec) but the
300-step gate NaN'd at step 130 due to the known v6e bf16
reduce-scatter numerics bug (pytorch/xla #8591/#8778). Larger
per-microbatch gradient tensors (2x vs b=8) accumulate enough bf16
rounding error to diverge. `batch_size=32`/`grad_accum=1` would be
even worse and should not be tested.

**Gotcha:** FSDPv2 does not support `fp32_reduce_scatter` (only
FSDPv1 does; pytorch/xla #3588/#8056 remain open). The only proven
stable batch topology on this hardware/precision combo is
`batch_size=8`, `grad_accum=4`. Phase 3 batch sweep is closed.

**Follow-up:** Before changing more knobs, ask the user whether to run a
5000-step production pass for `opt-1-log10-hot1k` or continue to Phase 2
compile-warmup experiments.

### 2026-05-10: Unified TPU optimization control plane

**Decision:** `.factory/orchestration/CONTROL_PLANE.md` is now the
source of truth for how TPU optimization specs, self-healing run
orchestration, skills, droids, hooks, and memory files interact.
`tpu-orchestrate` remains the single master TPU run-control skill and
selects either self-healing mode (`SPEC.md`) or optimization mode
(`TPU_OPTIMIZATION_SPEC.md`).

**Why:** Iter 24h created a stable production baseline, so the next
phase needs controlled throughput experiments without duplicating state
across `PLAN.md`, `PROGRESS.md`, `memories.md`, skills, and playbooks.
The boundary is:
`PLAN.md` = active checklist, `PROGRESS.md` = chronological event log,
`memories.md` = durable decisions/gotchas, `.factory/orchestration/` =
operational specs/playbooks, `_artifacts/` = ephemeral logs/profiles.

**Follow-up:** Future TPU optimization candidates should follow
`TPU_OPTIMIZATION_SPEC.md` and
`playbook/optimization-experiment-matrix.md`; promoted configs or
durable rejected knobs get compact memory entries, while routine run
events stay in `PROGRESS.md`.

### 2026-05-10: Iter 24h -- first 5000-step v6e-8 production run completed

**Decision (validated):** Iter 24h is the first successful Phase 5
production run on single-host v6e-8 spot. The remaining step-259
topology bug was a grad-accum epoch-boundary graph switch: iter 24g's
``drop_last=True`` kept micro-batches full-size, but the train split
has 8,283 rows. At ``batch_size=8`` that produced 1,035 micro-batches;
with ``grad_accum=4``, step 259 still crossed the epoch boundary inside
one XLA-traced macro-step. Iter 24h pads the final TPU batch and only
resets epochs between optimizer steps, so XLA sees the same
four-microbatch graph for the full 5000 steps.

**Patch pattern (SUPERSEDES iter 24g data-loader part):**
1. ``InterleavedCollator`` now supports ``batch_pad_to`` and
   ``expected_num_codebooks``. On TPU, the final short DataLoader batch
   is padded on the host to ``[batch_size, num_codebooks, max_frames]``;
   padded rows get zero attention/loss masks.
2. TPU training uses ``drop_last=False`` again, preserving all examples
   while still presenting a static batch dimension to XLA.
3. The training loop tracks micro-batches consumed in the current epoch
   and resets the iterator only **between** optimizer steps, never inside
   a 4-way grad-accum graph.
4. Iter 24g's SDPA mask-elision patch remains in place.

**Artifact:** ``gs://tinyaya-stage2-tpu/code/tinyaya-repo-iter24h.tar.gz``.

**Live validation:**
- W&B run ``7rrjupc7``:
  https://wandb.ai/cataluna84/tinyaya-stage2-tpu/runs/7rrjupc7.
- Step 5000/5000 reached; final loss 5.3558
  (text 10.3176, audio 4.3240).
- Training completed in 615.9 min and exited status 0.
- No NaN, OOM, RESOURCE_EXHAUSTED, fatal, traceback, bus-error, or
  kernel-panic signals were found.
- Final canonical checkpoint uploaded to
  ``gs://tinyaya-stage2-tpu/checkpoints/stage2-tpu-v6e-spot/step_005000_final/``.
- GCS lists 8 objects / 2.37 GiB:
  ``metadata.json``, ``text_embed.pt``, ``depth_decoder.pt``,
  ``projection.pt``, ``audio_heads.pt``, and PEFT adapter files.
- XLA compiled only at startup: 18 compilation causes total, 12 before
  visible step 1 and 6 around steps 1-2; no late recompiles through
  step 5000.

**Workstation validation:** simultaneous-translation validators pass:
``ruff format --check src/ scripts/``, ``ruff check src/ scripts/``,
byte-compile all tracked Python files, parse all YAML configs, and
``bash -n`` TPU shell scripts + launcher.

**Follow-up:** move optimizer-state/warmup compilation before the
visible ``step 1`` line so step 1 is steady-state. This is a cleanup
only; it did not affect correctness of iter 24h.

### 2026-05-09: Iter 24g -- fix step-259 NaN by freezing XLA data + mask topology

**Decision (SUPERSEDED by iter 24h; TPU preempted before launch):**
Iter 24g targeted the real step-259 graph change discovered by 24f, not
the downstream symptom. Exa/XLA docs confirm that recompilation happens
when input shapes or value-dependent Python branches change; the HF
SDPA helper also calls ``torch.all(attention_mask == 1)`` and may elide
the 4D mask for all-full batches. That creates a different attention
graph on XLA and can expose the same v6e bf16 reduce-scatter NaN.

**Patch pattern:**
1. TPU ``DataLoader(drop_last=True)`` so tail batches cannot change the
   SPMD batch dimension.
2. Static TPU shape assertions before tensors move to XLA:
   ``text/mask/loss == [batch_size, max_frames]`` and
   ``audio == [batch_size, num_codebooks, max_frames]``.
3. Clamp HF attention masks to ``>= -1e4`` as before, and additionally
   disable SDPA mask elision by making
   ``AttentionMaskConverter._ignore_causal_mask_sdpa`` always return
   ``False``. This keeps all-full and padded batches on the same 4D-mask
   graph.
4. Keep conservative ``batch_size=8, grad_accum=4`` from iter 24f until
   the run passes step 300; only then consider restoring b=16/g=2.

**Artifact:** ``gs://tinyaya-stage2-tpu/code/tinyaya-repo-iter24g.tar.gz``.

**Blocker:** QR/node ``tinyaya-stage2-spot-v6e8-eu`` was terminal
``PREEMPTED`` as of upload time. User approved two recreates. The first
replacement reached ACTIVE/READY but preempted before iter 24g could
launch. The second reached compile and created W&B run ``kmvydrq1``,
then preempted before step 1. Per orchestrator policy, do not
auto-recreate QR again without user approval.

### 2026-05-09: Iter 24f -- step-259 recompile is the actual root failure (NaN with b=8, OOM with b=16)

**Decision (NEW failure mode discovered):** Iter 24f (b=8, grad_accum=4,
no barrier, no lever 6 -- "minimal" config) ran 258 steps NaN-free
(loss 10.36 -> 6.86, ~6s/step) and then **NaN at step 259**. A NEW
graph compiles at the step-258 -> 259 boundary with massively different
topology (28.29 GB intermediate tensor, 19-min recompile, hash
``c6ae6ddabdc8d3dccc5c8ee2d36b03ce``). audio_loss NaN'd first
(``text=10.35 audio=nan``) -- exact same fingerprint as iter 24a-d.

**Same boundary, two failure modes:**
- iter 23 / 24e (b=16): the 28 GB graph overflows -> OOM at step 259
- iter 24f (b=8):  the 28 GB graph fits -> bf16 reduce-scatter NaN

**Implication:** OOM at 258 in iter 23/24e was NOT transient ring-buffer
growth. There is a real graph-topology change at step 259 that triggers
both failures. b=8 only delays the symptom, doesn't cure it.

**Hypotheses for the step-259 graph change (untested):**
1. Dataset boundary -- 258 * 8 * 4 = 8,256 samples; dataset has 9,212
   pairs; an epoch reset / loader rebucket happens here.
2. Sequence-length / max_frames variation in this micro-batch.
3. Warmup -> decay schedule crossover at step 200 (less likely).

**Wandb:** 0x4keilj NaN @ 259.

### 2026-05-09: Iter 24e -- BREAKTHROUGH: barrier helper IS the NaN trigger; OOM at 258 was transient

**Decision (numerics fix, SUPERSEDES iter 24c):** Iter 24e proves
that the per-layer ``register_full_backward_hook`` calls inside
``_apply_fsdpv2_backward_barriers`` (added in iter 24) are the SOLE
NaN trigger. Lever 6 (fused fp32 ``clip_grad_norm``) was actually
MASKING the NaN, not causing it.

**Bisection table:**

| iter   | Barrier | Lever6 | Per-layer wraps | b  | Result |
|--------|---------|--------|-----------------|----|--------|
| 17     | 0       | 0      | 0               | 8  | stable but tiny |
| 23     | 0       | 0      | 0               | 16 | NaN-free 257 steps, OOM at 258 |
| 24a/b  | 1       | 1      | 1               | 16 | NaN @ 24 |
| 24c    | 1       | 1      | 0               | 16 | NaN @ 32 |
| 24d    | 1       | **0**  | 0               | 16 | **NaN @ 3** (lever 6 was MASKING) |
| 24e    | **0**   | 0      | 0               | 16 | **PASSED step 258, no NaN, no OOM** |

**Root cause confirmed:** the 42 ``register_full_backward_hook``
calls force XLA to materialize 42 backward-graph boundaries that
don't fuse, exposing per-layer bf16 underflow on v6e
(pytorch/xla #8591). Lever 6's fp32 norm reduction was clearing
some pending bf16 numerical issues at the global-norm step, hence
delaying NaN from step 3 (without lever 6) to step 32 (with).

**OOM at step 258 was TRANSIENT** -- iter 24e bypassed it cleanly.
Possible causes for iter 23's OOM that no longer apply:
1. Stale TPU state from previous run (preempt + restart cleared)
2. Slightly different libtpu cache state
3. Different graph compile path due to barrier hook removal -
   without the hook, XLA may fuse more aggressively, reducing peak
   HBM
4. Lower per-step memory pressure without 42 hook closures

**Production path forward:** keep ``fsdp_barrier_hook: false`` in
the YAML (default) and ``enable_clip_grad_norm: false`` (matches
iter 23 stable). If OOM resurfaces at step ~258 in future runs,
use batch_size=8 (per pytorch/xla #8591 explicit upstream advice)
rather than re-enabling the barrier hook.

**Code changes preserved:**
- ``train_hierarchical.py`` line 651: barrier helper guarded by
  ``cfg["train"].get("fsdp_barrier_hook", False)`` (default False)
- ``configs/stage2_tpu_v6e_spot.yaml``: both
  ``enable_clip_grad_norm: false`` and ``fsdp_barrier_hook: false``
  documented with the bisection rationale
- ``tpu_backend.py``: ``layer_type_names = ("CohereDecoderLayer",
  "MoshiDecoderLayer")`` (Cohere2DecoderLayer DROPPED to keep one
  outer-level FSDPv2 reduce-scatter; matches iter 23 stable)

**Wandb runs:**
- 24a: ``5ep7z7ab`` (NaN @ 24, flash-on)
- 24b: ``49kjpp6v`` (NaN @ 24, flash-off)
- 24c: ``83evwy38`` (NaN @ 32, barrier+lever6, no per-layer)
- 24d: ``d3hrus4k`` (NaN @ 3, barrier+no-lever6)
- 24e: ``c0ybrkz4`` (CLEAN through step 258+; iter 23 reproduction)

---

### 2026-05-08: Iter 24c -- v6e bf16 reduce-scatter NaN bug forces barrier-only OOM mitigation (SUPERSEDED by iter 24e)

> **SUPERSEDED 2026-05-09:** iter 24e proved the barrier helper
> itself is the NaN trigger and that the OOM at step 258 was
> transient. The per-layer FSDPv2 wrap revert is still kept (it
> matches iter 23) but the barrier helper is now disabled by
> default. See iter 24e entry above for the corrected analysis.

**Decision (numerics fix):** Iter 24a/24b BOTH hit a bit-deterministic
NaN at step 24 (audio_loss first, then both losses) regardless of
whether FlashAttention was on (24a) or off (24b). Bisection
confirmed flash-attention is NOT the trigger.

**Root cause:** the iter 24 patch to ``tpu_backend.py`` added
``Cohere2DecoderLayer`` to ``layer_type_names`` in the FSDPv2
auto-wrap policy. Iters 21-23 had a stale class name
(``CohereDecoderLayer``) that NEVER matched the real HF class, so
the 36 backbone layers were unwrapped and there was a single
outer-level reduce-scatter. The class-name fix introduced 36
per-layer FSDPv2 reduce-scatters. On v6e, bf16 reduce-scatter is
known-buggy: pytorch/xla #8591 (open Jan 2025; v6e+bf16+batch>=16
NaN, no fix; XLA team confirmed v6e-specific) and #8778 (open Mar
2025; same NaN signature on v4-8 with larger batch). FSDPv2 has NO
``fp32_reduce_scatter`` flag (only FSDPv1 does -- pytorch/xla
#3588; #8056 tries to add it to FSDPv2 but is open since Sep 2024,
not merged). pytorch/pytorch #106395 confirms ``reduce_dtype=None``
(fp32) instead of bf16 fixes the NaN; HF accelerate #2127 (pacman100)
notes "layer norms, softmax and the output logits are required to
be in FP32 for stable training" with FSDP+bf16.

**Fix (iter 24c):** revert the class-name patch in
``tpu_backend.py`` (``layer_type_names = ("CohereDecoderLayer",
"MoshiDecoderLayer")`` -- drops ``Cohere2DecoderLayer``). The 36
backbone layers are no longer wrapped by FSDPv2 individually; ONE
outer-level reduce-scatter at the composite level matches the iter
17-23 stable bf16 regime. Keep the
``_apply_fsdpv2_backward_barriers`` helper unchanged: its targets
still include ``Cohere2DecoderLayer`` but it uses
``register_full_backward_hook`` (a plain torch hook), which works
regardless of whether the layer is wrapped by FSDPv2. **Verified
at runtime:** ``[fsdpv2] applied backward optimization barrier to
42 layers`` is still logged. Combines iter 21-23 stable numerics +
barrier-only OOM mitigation.

**Why barrier-only without per-layer wraps still works for OOM:**
the cache_all_gather=True ring buffer (openxla/xla #20508) retains
all-gather'd full params from forward through backward at every
FSDPv2 wrap. Without per-layer FSDPv2 wraps, there's just ONE
all-gather buffer at the composite level instead of 36. The
``xm.optimization_barrier_(grads + grad_input)`` at each
Cohere2DecoderLayer's backward still forces XLA to materialise
gradients and free intermediate activation buffers before the
previous layer's backward begins, breaking the CSE chain that hit
the step-258 OOM.

**Other knobs preserved from iter 24:**
- ``enable_clip_grad_norm: true`` (lever 6 fused variant 6a).
- ``--xla_tpu_enable_flash_attention=false`` in LIBTPU_INIT_ARGS
  (matches iter 17-23 stable; iter 24a/24b proved flash-attention
  is not the NaN cause but it wasn't on the critical path either).
- ``wandb_run_name: v6e-spot-stage2-5k-iter24c``,
  profile dir ``/tmp/xla_profile/iter24c-<ts>/``.

**SUPERSEDES** the iter 24 "Also re-enabled" decision below: the
flash-attention re-enable in 24's launcher LIBTPU_INIT_ARGS was
reverted in 24b/24c, and the per-layer FSDPv2 wraps that the
class-name fix introduced were reverted in 24c.

**Where:**
- ``simultaneous-translation/src/backend/tpu_backend.py`` lines
  331-350 (``_wrap_fsdpv2.layer_type_names``).
- ``simultaneous-translation/scripts/train_hierarchical.py`` lines
  118-202 (``_FSDPV2_BARRIER_TARGET_CLASSES`` + helper) UNCHANGED.
- ``configs/stage2_tpu_v6e_spot.yaml`` (``wandb_run_name``).
- ``_artifacts/launch_train_v6e_v2.sh`` (banner + LIBTPU_INIT_ARGS).
- wandb URL: https://wandb.ai/cataluna84/tinyaya-stage2-tpu/runs/83evwy38.

**References:**
- pytorch/xla #8591, #8778 -- v6e bf16 reduce-scatter NaN bug.
- pytorch/xla #3588, #8056 -- fp32_reduce_scatter (FSDPv1 has it,
  FSDPv2 does not).
- pytorch/pytorch #106395 -- reduce_dtype=None fixes Llama FSDP NaN.
- HF accelerate #2127 -- FSDP+bf16 best practice from pacman100.

### 2026-05-08: Iter 24 -- FSDPv2 backward optimization barrier kills the cache_all_gather OOM
**Decision (root-cause fix):** Iter 21/22/23 all OOM-crashed at step
258 with the same 89 GiB HLO temp on a 31 GiB v6e chip,
bit-deterministic across runs even after iter 22 stabilised the
grad topology (`set_to_none=False` + `requires_grad` iteration) and
iter 23 gated lever 6 entirely. The crash signature (5.7x
steady-state HBM at a single recompile boundary) matches the
documented behaviour of XLA SPMD's `cache_all_gather=True`
(defaulted, no programmatic disable; openxla/xla #20508): the
all-gather output of every sharded layer's full params is retained
from forward through backward, accumulating ~28 GiB of ring buffers
across the 36 FSDPv2-wrapped CohereDecoderLayer instances; step 258
is the first batch whose graph hash forces XLA's CSE to stop
eliding equivalent all-gathers, so all 20+ retained buffers
materialise simultaneously and OOM the chip.

**Fix (iter 24):** install per-layer
``register_full_backward_hook`` that calls
``xm.optimization_barrier_(grads + grad_input)`` on every inner
``SpmdFullyShardedDataParallel`` wrap, forcing XLA to materialise
gradients and free the cached all-gather output before the previous
layer's backward begins. This is step 5 of the FSDPv2 SPMD recipe
in pytorch/xla #6379 and the HF Llama 2 / DeepSeek SPMD reference
pattern. Implemented as
``_apply_fsdpv2_backward_barriers(model)`` in
``scripts/train_hierarchical.py``, called after
``backend.wrap_model(model)`` on the TPU path.

**Also re-enabled in iter 24** (now that the OOM is fixed, every
gated knob is restored):
- ``enable_clip_grad_norm: true`` (lever 6, fused variant 6a) in
  ``configs/stage2_tpu_v6e_spot.yaml``.
- Dropped ``--xla_tpu_enable_flash_attention=false`` from
  ``_artifacts/launch_train_v6e_v2.sh`` LIBTPU_INIT_ARGS so XLA's
  FlashAttention TPU kernel runs (sidesteps the torch_xla 2.6+
  SDPA 2.5x SPMD memory regression -- pytorch/xla #8423).
- Bumped ``wandb_run_name -> v6e-spot-stage2-5k-iter24`` so the
  fresh attempt is distinguishable from the iter 21/22/23 crashed
  runs in the wandb UI.

**Why this is the right intervention** (vs the alternatives):
- ``XLA_FLAGS=--xla_spmd_cache_all_gather=false`` is not
  programmatically configurable (openxla/xla #20508 was closed
  without exposing the flag).
- FSDPv1 (``XlaFullyShardedDataParallel``) has explicit
  ``optimization_barrier_in_forward/backward`` flags that fire the
  same ``xm.optimization_barrier_`` calls internally; switching to
  FSDPv1 would re-architect the SPMD path (multi-process vs
  single-process) and is much higher-risk than the additive hook.
- Reducing batch_size, max_frames, or grad_accum would mask the
  symptom but the ring-buffer accumulation would still scale and
  eventually OOM at a deeper step.

**Where:**
- Code: ``simultaneous-translation/scripts/train_hierarchical.py``
  (``_apply_fsdpv2_backward_barriers`` helper + call after
  ``wrap_model``).
- Config: ``simultaneous-translation/configs/stage2_tpu_v6e_spot.yaml``
  (``enable_clip_grad_norm: true``,
  ``wandb_run_name: v6e-spot-stage2-5k-iter24``).
- Launcher: ``_artifacts/launch_train_v6e_v2.sh`` (banner +
  LIBTPU_INIT_ARGS).
- PLAN: ``.factory/PLAN.md`` Phase 13.

**References:**
- openxla/xla #20508 -- ``cache_all_gather`` default + no flag.
- pytorch/xla #6379 step 5 -- FSDPv2 SPMD backward-barrier recipe.
- pytorch/xla #8423 -- SDPA 2.5x SPMD memory regression workaround.
- pytorch/xla #8809 -- closed Mar 2025; same gradient-buffer-lifetime
  symptom under MarkShardingFunction.

### 2026-05-08: Iter 18 -- five optimization levers + lever 6 variants documented
**Decision (perf milestone):** Lift v6e-8 EU canary throughput ~1.7x over
iter 17 (2.36 sec/step, 54.2 examples/sec, ~10x iter 14) while
instrumenting the run for live HBM + profiler telemetry.
**Levers applied** (iter 18):
  * Lever 1: `batch_size: 8 -> 16` (KEEP `xla_grad_checkpoint: true`).
    Effective batch 16 * 2 * 8 = 256 (2x iter 17). HBM est. ~13-15
    GB/chip vs measured 31.246 GiB ceiling -> ~50% util.
  * Lever 2: `max_frames: 300 -> 400`. Per-chip token batch 6400.
  * Lever 3: `tpu_backend.py:get_memory_info` now calls
    `xm.mark_step() + xm.wait_device_ops()` BEFORE
    `xm.get_memory_info(device)`. Without sync the lazy graph hasn't
    materialised on chip and HBM reads as 0 GB. Verified by
    `_artifacts/memory_probe.py`.
  * Lever 4: Profiler instrumentation. `xp.start_server(9012)` at
    rank-0 startup; `xp.StepTrace('train_step', step_num=step)`
    wraps every step body; auto-capture a 30s trace at step 30 ->
    `/tmp/xla_profile/iter18-<ts>/`. Launcher post-train hook
    `gsutil cp -r` to `gs://tinyaya-stage2-tpu/profiles/`.
  * Lever 5: `export PT_XLA_DEBUG_LEVEL=1` in launcher. Iter 17 had
    0 `aten::` fallbacks per `met.metrics_report()`; this acts as a
    regression detector if iter 18 introduces any.
**Lever 6 DEFERRED to iter 19+** (in-code TODO comment block in
`train_hierarchical.py:macro-step` documents three re-enable
variants below). The TPU branch keeps
`grad_norm = torch.tensor(0.0)` for now to keep the iter 18 baseline
free of confounders.
**Reason for deferring lever 6:** even the safest fused variant
costs 5-15% throughput from the global-norm reduction. We want to
isolate the gain from levers 1-5 first, then A/B compare with-clip
vs without-clip on identical config in iter 19.
**Where:** branch `feat/tpu-support`; commit (pending iter 18
validation), wandb run TBD, GCS checkpoint
`gs://tinyaya-stage2-tpu/checkpoints/stage2-tpu-v6e-spot-canary/
step_000200_final/` (overwrites iter 17 artifact on completion).

### 2026-05-08: Lever 6 variants for re-enabling clip_grad_norm_ on FSDPv2 SPMD
**Decision (deferred lever):** Three known-safe ways to re-enable
gradient clipping on single-host v6e-8 SPMD; ranked by safety.
**Why currently disabled:** patch 5/6 set
`grad_norm = torch.tensor(0.0)` on the TPU branch. The original
disable was driven by two distinct issues:
  1. **Cross-host all-reduce deadlock** (HF #41881, pytorch/xla
     #3424) with heterogeneous parameter groups. **Single-host v6e-8
     invalidates this concern** (no inter-host all-reduce; SPMD
     partitioner emits a single `xla::all_reduce` HLO op).
  2. **Per-parameter graph-break storm** -- vanilla
     `torch.nn.utils.clip_grad_norm_` does ~550 graph breaks per
     step under FSDPv2 (`for p in params: norms.append(p.grad.norm())`
     plus `if torch.isfinite(coef).item():` plus
     `for p: p.grad.mul_(coef)`). The recompile cost is real.

**Variant 6a (preferred for single-host v6e-8) -- fused single-graph
clip:**
```python
total_sq = torch.zeros([], device=device)
for g in (p.grad for p in model.parameters() if p.grad is not None):
    total_sq = total_sq + (g.float() ** 2).sum()
total_norm = total_sq.sqrt()
clip_coef = max_norm / (total_norm + 1e-6)
clip_coef = torch.where(clip_coef < 1.0, clip_coef, torch.ones_like(clip_coef))
for p in model.parameters():
    if p.grad is not None:
        p.grad.mul_(clip_coef)
xm.mark_step()
```
Properties: 1 fused HLO; no `.item()`; populates `train/grad_norm`
metric; ~5-15% throughput cost from the global-norm reduction.

**Variant 6b -- element-wise clip_grad_value_:**
`torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)`. Smallest
graph (per-element clamp). Loses the norm metric. Use if the metric
is unimportant and we want zero throughput cost.

**Variant 6c -- vanilla clip_grad_norm_:**
`torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm,
error_if_nonfinite=False)`. NOT recommended on FSDPv2 SPMD --
per-parameter graph break storm causes recompile cascade. Will work
on single-host (no deadlock) but throughput collapses.

**Recommendation:** apply variant 6a in iter 19 only after iter 18
validates levers 1-5. A/B against iter 18 baseline on identical
config to measure the throughput cost cleanly.
**Where:** in-code TODO comment at `train_hierarchical.py:macro-step
(if is_tpu)`; `.factory/PLAN.md` Phase 12.

### 2026-05-06: TPU canary breakthrough -- iter 7 reaches step 100, loss decreasing
**Decision (milestone):** First end-to-end Stage 2 success on TPU.
Run `8pse8tzk` reached step 100 with loss 9.0273 -> 7.5983 (steady
decrease). Steady-state 3.41 sec/step from step 30 onwards. All 4
v4-32 hosts attached to one wandb umbrella via shared-mode rendezvous.
**Reason this is a milestone:** before this run we had not produced a
single `loss=` line on TPU after eight weeks of infrastructure work.
This unblocks Phase 5 (5000-step production run).
**Reproduce:** `TRC_PROFILE=v4-32-uc2b
QR_NAME=tinyaya-stage2-spot-v4-canary-qr
NODE_ID=tinyaya-stage2-spot-v4-canary
CONFIG_FILE=configs/stage2_tpu_canary_v4_spot.yaml
TPU_STRATEGY=fsdpv2_lora bash scripts/tpu/launch_spot.sh`
with patches 4-11 in `feat/tpu-support` HEAD.
**Where:** wandb URL above; loss curve + sec/step in run scalars.

### 2026-05-06: Patch 4 -- strategy-aware optimizer step
**Decision:** `tpu_backend.optimizer_step(optimizer, mode)` branches
on the active SPMD strategy. For `replicated`: `xm.optimizer_step
(optimizer)` (the GPU-AllReduce analogue). For `fsdpv2_lora` /
`fsdpv2`: `optimizer.step()` then `xm.mark_step()`. The FSDPv2 path
already gathers shards inside the wrapper, so calling
`xm.optimizer_step` would do an extra all-reduce on already-reduced
grads.
**Reason:** running `xm.optimizer_step` under FSDPv2 produces a 2x
all-reduce and surprises the pjrt scheduler at large param counts.
The mark_step flush is needed because FSDPv2 lazily defers the
gradient gather; without an explicit mark_step the next iteration's
forward pre-empts the still-running backward.
**Where:** `simultaneous-translation/src/backend/tpu_backend.py`
`optimizer_step`.

### 2026-05-06: Patch 5 + 6 -- skip clip_grad_norm on TPU
**Decision:** call `xm.mark_step()` once before the (optional) clip,
then **skip** `clip_grad_norm_` entirely under FSDPv2 strategies.
**Reason:** `clip_grad_norm_` reduces gradients across params with a
`norm()` then a `mul_(coef)` per param tensor. Under FSDPv2 each
gradient is an XLA tensor sharded across the chip mesh; the per-param
norm forces a graph break (one HLO per param). With 36 backbone
LoRA layers + a few projection heads + AdamW optim states, that's
hundreds of graph breaks per step and tens of minutes of recompile
hell. Skipping is empirically safe at the LoRA scale (we never see
gradient spikes >1e3 in our run), and Phase 5 can re-enable a
sharded-aware clip if needed.
**Where:** `simultaneous-translation/scripts/train_hierarchical.py`
TPU branch of the training step.

### 2026-05-06: Patch 7 -- the .item() cpu_fallback storm
**Decision:** in the TPU inner loop, replace `.item()` with `.detach
()` on the loss; accumulate sums in XLA-tensor accumulators (e.g.
`micro_loss_sum_xla`); materialise a single Python float at the
`log_every` boundary only. This is the FSDPv2-reference example
pattern.
**Reason this is the single biggest fix in the run:** a `.item()`
on an XLA tensor traces into `at::native::item ->
_local_scalar_dense -> cpu_fallback -> XLANativeFunctions::_to_cpu`.
Each call forces the partitioner to materialise the *entire* graph
to host RAM. With `grad_accum=2` we had 4 sibling `.item()` calls
inside the micro-batch loop; iter 1/2 misdiagnosed the resulting
12-16 min per-call compile storm as a deadlock. py-spy native stack
showed `xla::PjRtCApiClient::CompileAndLoad ->
InitializeArgsAndCompile -> XLANativeFunctions::_to_cpu ->
cpu_fallback -> _local_scalar_dense -> at::native::item ->
train_hierarchical.py:696`, an unambiguous fingerprint.
**Where:** `simultaneous-translation/scripts/train_hierarchical.py`
TPU path (`micro_loss_sum_xla`, `running_xla` dict).
**Reference:** pytorch/xla #4203, #8020 (open issues on
`item()` performance under FSDPv2).

### 2026-05-06: Patch 8 -- cross-host is_main_process gate
**Decision:** `tpu_backend.is_main_process` returns
`xr.host_index() == 0 AND xm.is_master_ordinal()`. Added
`tpu_backend.host_index()` for callers that need just the host id.
**Reason:** `xm.is_master_ordinal()` is **local to the host**
(returns True on chip 0 of every host). On a 4-host pod that's 4
"master" processes. Without the host_index check, every host opens
its own wandb run, writes its own checkpoints, and emits its own
log lines -- producing the 4-separate-runs observability gap we hit
on iter 3.
**Where:** `simultaneous-translation/src/backend/tpu_backend.py`
`is_main_process` and `host_index`.

### 2026-05-06: Patch 9 -- wandb shared-mode rendezvous via GCS
**Decision:** rank-0 (host 0, chip 0) creates the wandb run and
writes the run_id to
`gs://tinyaya-stage2-tpu/wandb-rendezvous/<run-name>.id`. Hosts 1-3
poll `gsutil cat` (60 retries, 5 s spacing) to read the run_id, then
attach via `wandb.init(mode="shared", id=<run_id>, x_primary=False,
x_label=f"rank_{host_index}")`. Rank-0 attaches with
`x_primary=True`.
**Reason:** wandb shared-mode (>= 0.19.9) is the canonical way to
roll up multi-host SPMD telemetry into a single run. The TPU image
ships 0.19.11. GCS is a dependency-free way to share the run_id;
all 4 hosts already have GCS access for checkpoints. Alternative
(rank-0-only) loses per-host telemetry.
**Where:** `simultaneous-translation/scripts/train_hierarchical.py`
wandb init block (gated on `is_main_process` for the publish, or on
host_index>0 for the attach).
**Reference:** pytorch/xla #9681 (multi-host SPMD docs), wandb
multi-process / shared-mode docs.

### 2026-05-06: Patch 10 / 10b / 10c -- grad_accum trade-off space
**Decision:** revert `grad_accum: 8 -> 4 -> 2`. Static memory
(params + grads + optim_state + sharded buffers) dominates
activations on FSDPv2-LoRA at the 5.17B scale. Iter 4 (g=8): OOM by
2.41 GB. Iter 5 (g=4): OOM by 41 MB. Iter 6 (g=2): clean compile.
**Reason:** the fused HLO of `g` micro-steps' activations + the
sharded heterogeneous params (LoRA[0:33] + FullFT[34:35]) crosses
v4's 31.75 GiB / chip even at modest `g`. `xla_grad_checkpoint=true`
was an option (recompute activations, ~half memory, 2x backward
compute); not chosen because g=2 is a known-good wiring once patch 7
+ patch 11 are in.
**Effective batch:** batch_size 2 * grad_accum 2 * 16 chips = 64.
Phase 5 production target is 128; will re-evaluate after canary
Phase 9 closes.
**Where:** `simultaneous-translation/configs/stage2_tpu_canary_v4_spot
.yaml` `train.grad_accum`.

### 2026-05-06: Patch 11 -- fixed-shape padding eliminates per-batch recompile
**Decision:** in the TPU collator, pad every batch to
`cfg.data.max_frames` (300 for canary, 300 for full). Shape becomes
fixed across all batches; XLA compiles one HLO and reuses it.
**Reason:** iter 6 reached step 2 then stalled in a per-batch
recompile loop -- each batch's actual frame count differed by 1-15
frames, so the partitioner re-traced a fresh HLO per batch.
sec/step was unbounded because compile dominated. Padding is the
canonical pytorch/xla recompilation-avoidance fix
("bucketization / padding"). Iter 7 reached step 100 in ~10 min
of steady-state runtime after the initial ~30 min compile.
**Trade-off:** wastes some FLOPs on padded frames, but the loss
function masks them so the gradient is correct. Steady-state
cost: ~20% more flops, ~50x faster wall-time vs recompile.
**Where:** `simultaneous-translation/src/data/collator.py` TPU
padding path; `cfg.data.max_frames` is the truth source.
**Reference:** pytorch/xla user-guide -- "Avoiding Recompilations".

### 2026-05-08: Patch 19 -- canonical end-of-training save for FSDPv2-XLA
**Decision (gotcha):** `model.save_pretrained` does NOT support
saving models that are still resident on a TPU device, even when a
pre-built CPU `state_dict` is passed via the `state_dict=` kwarg.
The function internally re-walks the model's submodules
(`named_parameters`, `state_dict()` introspection inside
`get_peft_model_state_dict`), and on FSDPv2 those walks trigger SPMD
collectives that hang forever in our setup because only host 0 ever
reaches the inner save logic.

**Empirical evidence:** iters 9, 10, 11 all reached step 100 with
loss 7.5983 (deterministic), then deadlocked all 4 v4-32 workers at
`futex_wait_queue` inside `save_checkpoint` -> `save_pretrained` ->
safetensors path. Patches 14/16/17 (split materialize/write phases,
all hosts call the function) were necessary but not sufficient.

**Canonical fix (HF transformers issue #36004, closed Dec 2025):**
```python
model.to("cpu")     # SPMD-wide gather; ALL hosts call this
unwrap_model(model).save_pretrained(...)  # CPU model now; safe
```
The `.to("cpu")` collective on a wrapped FSDPv2 model gathers every
shard onto host RAM in a single pass; afterwards `save_pretrained`
walks an ordinary CPU module with no XLA tensors involved.

**Trade-off / scope:** `.to("cpu")` is destructive to FSDPv2
sharding metadata. Training cannot continue after this call; the
SPMD partitioner forgets the per-layer mesh annotations and
re-running `backend.wrap_model(model)` would not reproduce the
sharding without a fresh recompile. So this is **end-of-training
only**, gated on `cfg.train.final_canonical_save`.

**Implementation:**
- new `save_checkpoint_canonical_final(model, save_dir, *, is_main)`
  in `src/training/checkpointing.py`.
- wired into `scripts/train_hierarchical.py` end-of-training block.
- DEFAULT FALSE for iter 12 (validate step 200 reached first); will
  flip to TRUE for iter 13 once we know the loop is healthy.

**Where:**
`simultaneous-translation/src/training/checkpointing.py` ::
`save_checkpoint_canonical_final` (new function near top of file).
`simultaneous-translation/scripts/train_hierarchical.py` end-of-loop
block (just below the patch 18b save_every gate).

### 2026-05-06: Patches 12 + 13 -- defer audio sample + validation on TPU canary (drafted)
**Decision (drafted, not yet validated):** during canary
(`max_steps <= 200`), skip `generate_audio_sample` and
`run_validation` on TPU. They feed the model with non-canonical
shapes (different sequence lengths, different batch dims) and
re-trigger XLA recompiles. Canary is about wiring validation, not
quality; the validation pass adds 10+ min of recompile per call.
**Reason:** iter 7 `audio_every=100` and `val_every=100` will fire
at step 100. If they fire and recompile, sec/step regresses; if they
fire cleanly, we don't need the patches.
**Status:** patches drafted but not pushed; waiting on iter 7 to
hit step 100 to decide.
**Where (drafted):**
`simultaneous-translation/scripts/train_hierarchical.py`
`generate_audio_sample` + `run_validation` call sites; gate on
`backend.kind != "tpu" or cfg.train.max_steps > 200`.

### 2026-05-06: Self-healing orchestrator -- bounded iteration with mandatory check-ins
**Decision:** Adopt a 4-tier orchestration architecture (Orchestrator
session -> custom droids -> skills -> background poller) with
mandatory user check-ins at T+15/30/45/60/90 min wall. The
orchestrator picks among 4 actions per check-in (Continue, Abort+
Diag, Adjust+Continue, Pause). Tier 3 (VM corruption) is locked to
**always escalate**; never auto-recreate a QR.
**Reason:** Without check-ins, a slow XLA compile (which can
legitimately take 30+ min) is indistinguishable from a stall, and
we'd burn quota retrying the same patch. With check-ins the user is
the safety valve. Without bounded iteration, a doom loop could
spend hours on the same misclassification. The 4-phase Detect ->
Diagnose -> Heal -> Verify pattern is from Claude Lab "Self-Healing
AI Agents" (2026); the tier ladder is from Erlang OTP supervisors +
K8s liveness probes.
**Empirical results:** Iter 1 misdiagnosed deadlock; check-in at
T+71 forced the user to ask "do not stop, diagnose it" -- which
broke the loop and led to py-spy + the patch 7 root cause. Without
the check-in, the misclassification would have repeated.
**Where:** `.factory/orchestration/SPEC.md` v2,
`.factory/skills/tpu-orchestrate/SKILL.md`,
`.factory/droids/tpu-watchdog.md`,
`.factory/droids/tpu-diagnoser.md`, `_artifacts/orch_poll.py`,
`_artifacts/scheduled_checkin.py`.

### 2026-05-06: py-spy 0.4.2 binary install for live native-stack diagnostics
**Decision:** Download py-spy 0.4.2 wheel from the GitHub releases
page on the workstation, extract the binary, and SCP it to the TPU
worker `/tmp/py_spy-0.4.2.data/scripts/py-spy`. Use `sudo` (NOPASSWD
verified) to attach to the python training PID.
**Reason:** XLA compile silently consumes hours of CPU with no log
output. Without a native-stack snapshot we cannot distinguish
"healthy compile" (libtpu.so frames) from "stalled `.item()`
storm" (cpu_fallback frames). py-spy attaches without process
restart, captures Python + native frames, and is the only tool that
gave us a definitive root-cause for iter 1.
**The trick that took 30 min:** the actual python training PID is
NOT the `uv run` PID printed by `pgrep python`. `uv run` spawns
python and *sleeps* until the child exits. `ps -e --forest -o pid,
pcpu,etime,comm,args` reveals the real Python at e.g. 514% CPU
under the `uv run` parent.
**Where:** orchestrator notes; binary lives at `/tmp/py_spy-0.4.2
.data/scripts/py-spy` on every worker (idempotent install).

### 2026-05-03: SPMD partitioner crash workaround
**Decision:** Force `XLA_DISABLE_FUNCTIONALIZATION=0` in `tpu_backend.py`.
**Reason:** With value `1` (the torch_xla 2.6 default in some builds),
the SPMD partitioner crashes on multi-output composite models. Fix
documented in pytorch/xla #8607.
**Where:** `simultaneous-translation/src/backend/tpu_backend.py`,
top of `wrap_model`.

### 2026-05-03: Use explicit bf16 cast, not legacy env vars
**Decision:** `model.to(torch.bfloat16)` inside `wrap_model` instead of
`XLA_USE_BF16=1` / `XLA_DOWNCAST_BF16=1`.
**Reason:** Both env vars were removed in `torch_xla>=2.6`. The legacy
path silently no-ops; tensors stay in f32 and the model OOMs.
**Where:** `simultaneous-translation/src/backend/tpu_backend.py`.
**Trade-off:** AdamW keeps moments internally in f32 even when params
are bf16, so optimizer numerics stay clean.

### 2026-05-03: Three SPMD strategies, selectable via env
**Decision:** Backend supports `replicated`, `fsdpv2`, `fsdpv2_lora`,
`auto` selectable via `TPU_STRATEGY` env var.
**Reason:** No single strategy fits all model sizes. Probe matrix
empirically showed:
- `replicated`: full copy per chip, OOM on 5.17B model.
- `fsdpv2_lora`: shards layers with trainable params (LoRA-bearing
  CohereDecoderLayer); replicates frozen MoshiDecoderLayer.
- `fsdpv2`: shards everything; highest comm cost but tightest memory.
**Default for canary:** `fsdpv2_lora`.

### 2026-05-03: GCS bucket is code transport, not sharding
**Decision:** `gs://tinyaya-stage2-tpu/code/` exists purely to ship
code to TPU VMs (private GitHub repo, no GitHub creds on TPUs).
**Not** part of any sharding mechanism, not consulted at runtime.
**Status:** Will be deprecated once TPU VM has its own `git clone`
of the `feat/tpu-support` branch.

### 2026-05-03: Three-tier AGENTS.md hierarchy
**Decision:** Root `AGENTS.md` for monorepo norms;
`simultaneous-translation/AGENTS.md` for TPU/training specifics;
`phase-3-data-generation-pipeline/AGENTS.md` for data pipeline.
**Reason:** Factory.ai's discovery rule is "closest wins, parents
merged". Each subproject has wildly different gotchas, so per-subproject
AGENTS.md prevents one large file becoming a kitchen sink.

### 2026-05-03: Hot redeploy via SCP companion script
**Decision:** `hot_redeploy.sh` uses an SCP'd companion
(`_remote_redeploy.sh`) instead of a heredoc.
**Reason:** Nested-quote SSH heredocs fail in opaque ways (status 255).
SCP + companion is robust and debuggable.
**Where:** `simultaneous-translation/scripts/tpu/`.

### 2026-05-03: Sub-3-min iteration via tarball-on-GCS
**Decision:** Code redeploy = tarball -> GCS -> SCP-trigger restart.
Avoids the 5-15 min queued-resource recreation.
**Reason:** Iteration loop dominated by infra time, not code edits.
Tarball is ~480 KB and uploads in seconds.

### 2026-05-05: Documentation conventions (TPU code for GPU engineers)
**Decision:** Every Python file under `simultaneous-translation/`
must carry a `WHY THIS EXISTS` paragraph in its module docstring,
NumPy-style function docstrings with explicit `TPU note:` blocks
when behaviour diverges from the GPU equivalent, and inline
`# GPU analogue: ...` callouts whenever a TPU primitive replaces a
familiar GPU one. PEP8 is enforced via `ruff` (config in
`pyproject.toml`, `target=py312`, `line-length=100`,
`select=E,F,W,I,B,UP`, `ignore=E501`).
**Reason:** The audience for this codebase is research engineers
fluent in PyTorch + GPUs but new to TPU. PJRT, SPMD, FSDPv2,
`scan_layers`, `xm.optimizer_step`, HBM-vs-host-RAM, and the bf16
cast quirks are all silent traps. Forcing each file to teach the
reader about the trap they're about to step on saves a week of
debugging per onboard.
**Where:**
- Convention: `simultaneous-translation/AGENTS.md`
  ("TPU code documentation style (mandatory)")
- Skill: `.factory/skills/tpu-doc-style/SKILL.md`
- Lint: `.venv/bin/python -m ruff format` and `... -m ruff check`
  in `simultaneous-translation/`.
**Trade-off:** New files take ~30% longer to write because of the
explanatory commentary. Pays for itself the first time a reader
asks "why is `use_cache=False` hardcoded here?".

### 2026-05-05: scan_layers TypeError on torch_xla 2.9 -> manual loop -> 4h+ compile
**Decision:** `_FusedScanLayer.forward` in
`simultaneous-translation/src/model/scan_utils.py` calls
`scan_fn(layers, hidden_states, *args, **kwargs)`, but the actual
`torch_xla.experimental.scan_layers.scan_layers` signature is
`scan_layers(layers, input_data, partition_fn=..., is_layer_pure=...)`
- it does NOT accept arbitrary kwargs. HuggingFace's Cohere2 layer
forward passes `attention_mask`, `position_embeddings`, and
`position_ids` as kwargs every step, so every scan call raises
`TypeError("scan_layers() got an unexpected keyword argument
'attention_mask'")` and we silently fall back to the manual loop. The
manual loop emits 36 unrolled `CohereDecoderLayer` HLO copies plus 6
`MoshiDecoderLayer` copies, which is exactly the slow path the scan
wrapper was built to avoid.
**Empirical impact (observed 2026-05-05 on a v4-32 spot canary):**
forward step 1 took ~20 min wall, then **backward step 1 / 2 took
4h 25min wall before any further progress**. Process did not OOM and
host RAM grew steadily (41 GB -> 86 GB RES across 2.5h) -- it was
genuinely tracing more graph, not stuck. After ~8h of CPU work the
process exited (likely supervisor-side timeout or OOM-kill) and the
tmux while-loop in startup_script.sh restarted it from scratch with
no XLA compile cache reused.
**Fix (drafted, not yet applied):** add a `_KwargBoundLayer(nn.Module)`
that closes over the loop-invariant `args`/`kwargs` and exposes
`forward(carry) -> carry`, then call
`scan_fn([_KwargBoundLayer(L, args, kwargs) for L in layers], hidden_states)`.
The closure tensors become loop-invariant side inputs to the scan
body; `scan_layers._ensure_same_structure` is satisfied because every
wrapper holds the same kwargs and wraps an isomorphic layer.
**Reason this matters:** without the patch every restart pays the
multi-hour compile tax and we never reach a `loss=` line. With the
patch the inner layer body is compiled once and reused for all 36
iterations via `xla.while_loop`, dropping compile to single-digit
minutes per `.factory/PLAN.md` Phase 1/2 expectations.
**Where:** `simultaneous-translation/src/model/scan_utils.py`
`_FusedScanLayer.forward`. Reference for the real signature:
https://github.com/pytorch/xla/blob/master/torch_xla/experimental/scan_layers.py

### 2026-05-05: XLA compile cache must be configured or every restart pays the full compile
**Decision:** The supervisor loop in
`simultaneous-translation/scripts/tpu/startup_script.sh` restarts
training every time the process exits, sleeping 30s between attempts.
That loop is correct for spot preemption recovery but it currently
does NOT export `XLA_PERSISTENT_CACHE_PATH`, so every restart starts
XLA tracing from scratch. After the long backward compile observed on
2026-05-05, the process died and the supervisor restarted it; the new
run repeated the full 4h+ compile rather than reusing cached HLO.
**Fix (proposed):** add `export XLA_PERSISTENT_CACHE_PATH=/var/cache/xla`
(or equivalent path on persistent disk) to the env block before the
tmux supervisor section in `startup_script.sh`. The cache survives
process restarts and makes recovery near-instant once the first
compile lands.
**Reason this matters:** spot preemptions are routine, OOM-kills
under long compile pressure are routine; without a persistent cache
the supervisor pays the compile tax on every cycle and cannot make
forward progress.
**Where:** `simultaneous-translation/scripts/tpu/startup_script.sh`,
between the dataset extraction block and the tmux launch.

### 2026-05-05: tmux session 'train' is the canonical observation interface
**Decision:** `startup_script.sh` launches training inside
`tmux new-session -d -s train ...`, which means each TPU worker has a
running tmux session named `train` that captures the stdout/stderr of
the supervisor + the actual training process. The session is owned by
root (the metadata startup-script always runs as root).
**Operational consequence:** to watch the live run, attach with:
```
gcloud compute tpus tpu-vm ssh <node> --project=ml-pipelines-315702 \
    --zone=<zone> --worker=0 -- -t 'sudo tmux attach -t train'
```
The `-- -t` is required to allocate a TTY. Detach without killing
with `Ctrl-b d`. Read-only attach with `tmux attach -t train -r`.
For non-interactive scrollback dumps, prefer
`sudo tmux capture-pane -t train -p | tail -N`. Multiple clients can
attach concurrently without disturbing the running process.
**Reason:** documenting this so future sessions don't reach for
`journalctl` or invent their own log-tailing mechanism. The tmux
session is already there and survives ssh disconnects.
**Where:** `simultaneous-translation/scripts/tpu/startup_script.sh`
section "launch training with auto-restart in tmux".

### 2026-05-05: HF dataset ships .pt + alignments inside packed/ tarballs
**Decision:** The `tiny-aya-translate/fleurs-tr-hi-mimi-encoded` HF
dataset publishes the encoded tensors and word-alignment JSONs as two
gzipped tarballs under `packed/`:
  - `packed/encoded_pt.tar.gz`
  - `packed/encoded_alignments.tar.gz`
Both archives have a top-level `encoded/` directory inside, so they
must be extracted with `--strip-components=1` into `/mnt/data/encoded/`,
which is the path `configs/*.yaml encoded_dir` points to and the path
`src/data/dataset.py::_resolve` falls back to via `encoded_dir / Path(p).name`.
**Fix applied 2026-05-05:** added an idempotent extraction block in
`startup_script.sh` after the `huggingface-cli download` step. Block
uses `[ ! -f "$DATA_DIR/encoded/.unpacked" ]` as the idempotency
marker so re-runs of the script (after host reboot, after spot
preemption recovery) do not re-extract.
**Reason:** without extraction, the dataset code crashes with
`FileNotFoundError: [Errno 2] No such file or directory:
'/home/claudeuser/ws/.../fleurs_973_hitr.pt'`. The hardcoded absolute
path comes from the JSONL splits files and is meaningless on TPU; the
`_resolve` fallback uses `encoded_dir / pp.name` to find the file by
basename. That fallback only works if the .pt files actually exist
in `encoded_dir`.
**Where:** `simultaneous-translation/scripts/tpu/startup_script.sh`
dataset extraction block.

### 2026-05-05: find_latest_checkpoint must tolerate missing GCS prefix
**Decision:** `simultaneous-translation/src/training/checkpointing.py
::get_checkpoint_dirs` for GCS paths now wraps `fs.ls(base_dir)` in a
`try/except FileNotFoundError -> return []` block. Previously the
local branch checked `os.path.exists` but the GCS branch raised on
the first run when the checkpoint prefix
`gs://tinyaya-stage2-tpu/checkpoints/<run-name>/` did not exist yet.
**Fix applied 2026-05-05.**
**Reason:** the very first canary run cannot have a checkpoint
to resume from; `--resume auto` should mean "resume if exists, else
start fresh" without crashing. Symptom: `gcsfs.core.FileNotFoundError:
b/tinyaya-stage2-tpu/o/checkpoints%2Fstage2-tpu-spot-canary` raised
during `main()` before training begins.
**Where:** `simultaneous-translation/src/training/checkpointing.py`.

### 2026-05-05: v4-32 spot capacity is hour-by-hour
**Decision:** The autonomous fallback policy in
`docs/tpu-capacity-log.md` should treat v4-32 spot in us-central2-b
as a transient first-class option, not a fallback below v4-64
on-demand. Empirical retries on the same day showed:
  - 09:23 UTC: 17+ min queued, no progress -> cancelled.
  - 13:42 UTC: ACTIVE in 3.5 min from same QR submission.
The TRC pool clears on a sub-hour timescale so cancelling and
retrying is a valid strategy as long as we only hold one QR at a
time and respect the 10-min poll timeout per attempt.
**Where:** `simultaneous-translation/docs/tpu-capacity-log.md`
section 2 (observed-wait table) + section 3 (per-profile heuristics).

### 2026-05-05: Regional IP quota gates v5e/v6e provisioning
**Decision:** v5litepod-64 and v6e-64 slices in this project hit
`IN_USE_ADDRESSES limit` on PROVISIONING because every region we
have TPU quota in caps `IN_USE_ADDRESSES` at 8, and these 8-host
slices each request one external IP per host. The fix is to launch
with `INTERNAL_IPS=1` (added to `launch_qr.sh` 2026-05-05), which
requires:
  1. Private Google Access on the `default` subnet for the region
     (`gcloud compute networks subnets update default --region=<R>
     --enable-private-ip-google-access`).
  2. Dataset pre-mirrored to `gs://tinyaya-stage2-tpu/encoded/` so
     the boot path doesn't depend on HF Hub.
**Reason:** observed FAILURE on 2026-05-05 spot v5e-64 in
europe-west4-b -- transitioned to PROVISIONING for ~2 minutes then
SUSPENDING -> FAILED with `IN_USE_ADDRESSES limit. [EID: ...]`.
Quota inspection confirmed regional cap of 8 IPs across
us-central2, us-central1, europe-west4. v4-32 (4 hosts) stays
under, but v4-64 / v5e-64 / v6e-64 are at or above.
**Where:** `simultaneous-translation/scripts/tpu/launch_qr.sh`
(INTERNAL_IPS env var), `simultaneous-translation/docs/tpu-capacity-log.md`
section 7.1, this file.

### 2026-05-05: Autonomous TPU fallback policy
**Decision:** Future sessions must follow a fixed fallback tree when
submitting TPU QRs, recorded in
`simultaneous-translation/docs/tpu-capacity-log.md`. The tree is:
1. on-demand v4-64 (us-central2-b) -- try first per TRC guidance.
2. spot v4-32 (us-central2-b) -- same zone, zero infra change.
3. spot v5e-64 (europe-west4-b) -- biggest spot grant; may have less
   competition in the EU region.
4. spot v5e-64 (us-central1-a) -- US v5e.
5. spot v6e-64 (europe-west4-a / us-east1-d) -- newest gen.
6. All fail -> ask user. Never auto-delete a QR the user queued.
**Rules:** one QR at a time, delete before trying next, 10-min
poll timeout per profile, stop after 3 consecutive failures.
**Reason:** On 2026-05-05 the spot v4-32 waited 17 min with no
progress; the on-demand v4-64 also queued 4+ min. Autonomous
fallback avoids burning user time on capacity misses.
**Where:** `simultaneous-translation/docs/tpu-capacity-log.md`.

### 2026-05-05: Authoritative TRC allocation captured + spot fallback
**Decision:** The verbatim TRC welcome email (sent to
`mayankbhaskar007@gmail.com`, project `ml-pipelines-315702`, 90-day
free trial) is now archived in
`simultaneous-translation/docs/tpu-trc-allocation.md`. The older
5-row TRC table in `docs/tpu-launch-plan.md` §2 was a draft and is
**SUPERSEDED** by that file. The actual grant is:

- 32 spot + 32 on-demand Cloud TPU v4 chips in `us-central2-b`
- 64 spot Cloud TPU v5e chips in `europe-west4-b`
- 64 spot Cloud TPU v5e chips in `us-central1-a`
- 64 spot Cloud TPU v6e chips in `europe-west4-a`
- 64 spot Cloud TPU v6e chips in `us-east1-d`

When the on-demand v4 in `us-central2-b` is busy we fall back to the
spot v4-32 in the SAME zone (TRC profile `v4-32-uc2b`). This keeps
IAM, VPC, runtime image, and SPMD strategy identical -- the only
knob that changes is `--spot`.
**Reason:** Per the TRC email's own guidance: "If you have access to
both on-demand and preemptible quotas, we recommend preferring
on-demand and falling back to preemptible if/when on-demand is not
available." Same-zone spot fallback is the smallest possible blast
radius.
**Where:**
- Doc: `simultaneous-translation/docs/tpu-trc-allocation.md`.
- Launch wrapper: `simultaneous-translation/scripts/tpu/launch_spot.sh`.
- Configs: `simultaneous-translation/configs/stage2_tpu_canary_v4_spot.yaml`,
  `simultaneous-translation/configs/stage2_tpu_v4_spot.yaml`.
**Trade-off:** Spot capacity can be reclaimed at any time. We
mitigate with `save_every: 100` in the spot configs (vs 500 for the
on-demand path) and the existing tmux `--resume auto` restart loop
in `startup_script.sh`. W&B run is configured with
`WANDB_RESUME=allow` so reruns continue the same wandb run instead
of forking a new one.

### 2026-05-05: scan_layers wrapper as a ModuleList proxy
**Decision:** Insert `scan_layers` into HuggingFace transformer
backbones (Cohere2 + Moshi depth decoder) by swapping the model's
`self.layers` (a `nn.ModuleList`) with a `_ScannedLayerStack` proxy.
The proxy implements `__getitem__(slice)` to return a one-element
list whose single element, when called, runs the entire original
stack via `torch_xla.experimental.scan_layers.scan_layers`. HF's
`for layer in self.layers[:N]:` then iterates exactly once.
**Reason:** Avoids re-implementing HF's `Cohere2Model.forward` (~150
lines, version-fragile). The proxy is reversible, idempotent, and
falls back to a manual loop with `torch.utils.checkpoint.checkpoint`
on GPU/CPU or whenever scan_layers raises.
**Where:** `simultaneous-translation/src/model/scan_utils.py`,
called from `composite.py`.
**Trade-off:** HF code that relies on `len(self.layers) == config.
num_hidden_layers` will see the original count (we keep `__len__`
honest), but anything iterating to collect per-layer hidden states
will only see two snapshots (input, output). For training with
`output_hidden_states=False` this is fine; for inference / probing
hooks we leave the proxy disabled.

### 2026-05-08: Canary topology pivots to single-host v6e-8 in europe-west4-a
**Decision:** The canary topology pivots from multi-host
v4-32 spot in `us-central2-b` to **single-host v6e-8 spot in
`europe-west4-a`**. The v6e-8 node is `tinyaya-stage2-spot-v6e8-eu`
(QR `tinyaya-stage2-spot-v6e8-eu-qr`). Single-host means ONE Python
process drives all 8 chips via SPMD: no cross-host rendezvous, no
host-index gating, no GCS run-id polling, no wandb shared-mode
umbrella. Exactly one tmux session, one PID, one wandb run. Patches
8 (host-index gate) and 9 (wandb shared-mode rendezvous) remain in
the codebase but are inert on this topology.
**Reason:** v4 spot capacity in `us-central2-b` was reclaimed by
other TRC users for multiple hours straight, leaving the v4-32
QR SUSPENDED with no recovery. The single-host v6e-8 EU profile
provisions in ~5 min and side-steps the multi-host coordination
burden entirely. Per-chip HBM is 32 GiB on both v4-32 and v6e-8, so
all batch-size and memory tuning carries over unchanged. v6e-64
multi-host is the next scale-up target once spot capacity allows.
**Where:** `tinyaya-stage2-spot-v6e8-eu` in `europe-west4-a`; QR
`tinyaya-stage2-spot-v6e8-eu-qr`; config
`simultaneous-translation/configs/stage2_tpu_v6e_spot.yaml`;
checkpoint prefix
`gs://tinyaya-stage2-tpu/checkpoints/stage2-tpu-v6e-spot/`;
wandb run name `v6e-spot-stage2-5k`.
**Reproduce:** `TRC_PROFILE=v6e-8-eu
QR_NAME=tinyaya-stage2-spot-v6e8-eu-qr
NODE_ID=tinyaya-stage2-spot-v6e8-eu
CONFIG_FILE=configs/stage2_tpu_v6e_spot.yaml
TPU_STRATEGY=fsdpv2_lora bash scripts/tpu/launch_spot.sh`
with patches 4-19 in `feat/tpu-support` HEAD; iter 14 also requires
patch 20a (`gsutil cp -r` upload inside
`save_checkpoint_canonical_final`) and patch 20b
(`AttentionMaskConverter` monkey-patch clamping mask values >= -1e4
to dodge pytorch/xla #4152 bf16 NaN, called at top of
`scripts/train_hierarchical.py`).

---

## Hardware facts

### v5litepod-16 topology

| Property | Value |
|----------|-------|
| Hosts (gcloud workers) | 4 |
| Chips per host | 4 |
| Total chips | 16 |
| HBM per chip | 16 GiB |
| Host RAM | ~96 GiB |
| Region used | europe-west4-b |
| Project | ml-pipelines-315702 |

A `gcloud --worker=N` is a host (VM), not a chip. Each host runs its
own Python process in PJRT mode; that process drives all 4 local chips.

### Per-strategy per-chip footprint (5.17B model, bf16)

| Strategy | Backbone | Activations | Total |
|----------|----------|-------------|-------|
| replicated | 10.34 GB | 5-10 GB | 18-24 GB (OOM) |
| fsdpv2_lora | 0.65 GB sharded + 0.6 GB frozen | 5-10 GB | 7-12 GB |
| fsdpv2 | 0.65 GB sharded | 5-10 GB | 6-11 GB |

---

## Known gotchas

### XLA compile time blows up on unrolled transformer stacks
36 `CohereDecoderLayer` + 6 `MoshiDecoderLayer` unrolled into the HLO
graph causes 25+ minute compile. Mitigation: `scan_layers` (open task
in `PLAN.md`).

### Pre-emption on TRC quota
Spot/preemptible TPUs in `europe-west4-b` get reclaimed regularly.
Mitigation: queued resources, `make_resilient: true` in YAML,
checkpoint every N steps.

### `which uv` empty under sudo
The startup script enumerates `/root/.local/bin/uv` and friends
explicitly. Do not rely on `which` under sudo on a fresh TPU VM.

---

## Glossary

- **SPMD:** Single Program Multiple Data — XLA's data-and-model
  parallelism abstraction.
- **FSDPv2:** PyTorch/XLA's SPMD-based Fully Sharded Data Parallel
  (v2 vs the older v1 wrapper).
- **HBM:** High-Bandwidth Memory on the TPU chip; distinct from host
  CPU RAM.
- **QR:** Queued Resource (gcloud TPU primitive that waits for capacity).
- **Mimi:** Kyutai's neural audio codec used for speech tokenisation.
- **canary:** A small-data, short-step config used to verify the full
  pipeline before paying for a multi-day run.

---

## Milestones (completed)

### 2026-05-10: First 5000-step TPU production run completed (iter 24h)

- **What:** First full Phase 5 Stage 2 production training run to
  complete on TPU.
- **Hardware:** single-host TPU v6e-8 spot, europe-west4-a. 1 host x
  8 chips, 32 GiB HBM/chip, 1 Python process via SPMD.
- **Run id:** ``7rrjupc7`` (project ``tinyaya-stage2-tpu``), run name
  ``v6e-spot-stage2-5k-iter24h``.
- **W&B:** https://wandb.ai/cataluna84/tinyaya-stage2-tpu/runs/7rrjupc7.
- **Training:** 5000/5000 steps, 615.9 min wall, exit status 0.
- **Final loss:** 5.3558 (text 10.3176, audio 4.3240).
- **Steady-state throughput:** ~6.7-7.0 sec/step after startup compile.
- **Checkpoint:** canonical final save at
  ``gs://tinyaya-stage2-tpu/checkpoints/stage2-tpu-v6e-spot/step_005000_final/``
  with 8 objects / 2.37 GiB.
- **Stability:** no NaN, OOM, RESOURCE_EXHAUSTED, fatal, traceback,
  bus-error, or kernel-panic signals in the training log.
- **Compile:** 18 startup-only XLA compilation causes; no late
  recompiles after steps 1-2 through step 5000.

### 2026-05-06: First end-to-end TPU canary success (iter 7)

- **What:** First run that reached `step=100` with monotonically
  decreasing loss on a TPU pod. Validates the entire SPMD + FSDPv2
  + LoRA + cross-host wandb pipeline that we have been chasing
  since the start of `feat/tpu-support`.
- **Hardware:** v4-32 spot, us-central2-b. 4 hosts x 4 chips = 16
  chips. 31.75 GiB HBM / chip. Effective batch =
  2 (per-chip) * 2 (grad_accum) * 16 (chips) = 64.
- **Run id:** `8pse8tzk` (project `tinyaya-stage2-tpu`).
- **Loss trajectory:** step 10 = 9.0273; step 100 = 7.5983.
- **Throughput:** sec/step = 3.41 (steady-state from step 30).
- **Compile wall:** ~30 min from deploy to first `loss=` line.
- **Patches in flight:** 4-11 (see "Architecture decisions" above).
- **Reproduce:** see the architecture-decision entry of the same
  date for the `launch_spot.sh` invocation.

(Next milestone: evaluation metrics from the iter 24h checkpoint.)

### 2026-05-08: Patch 19 validated -- canonical end-of-training save works on v6e-8
**Decision:** `save_checkpoint_canonical_final()` in `checkpointing.py` does
`model.to("cpu")` per-submodule (gather all FSDPv2 shards to host RAM),
then host-0 writes via `save_pretrained(safe_serialization=False)` plus
`torch.save()` for plain modules. A CPU verification assertion checks
no XLA tensors remain before writing. The function is gated by
`cfg.train.final_canonical_save` (default False).
**Reason:** Iters 9/10/11 deadlocked at `PEFT.save_pretrained` on FSDPv2-XLA
(pytorch/xla #36004: save_pretrained doesn't support TPU-resident models).
The canonical fix is `model.to("cpu")` before save, which gathers all shards
in a single collective. On single-host v6e-8 this works cleanly; on
multi-host v4-32 all hosts must call `.to("cpu")` together (SPMD gather).
**`safe_serialization=False`** is required because safetensors cannot access
XLA storage pointers (HF #29608: `RuntimeError: Attempted to access the
data pointer on an invalid python storage`).
**Known limitation:** If save_dir is a gs:// URI, `torch.save` writes to a
local directory `gs:/...` instead of actual GCS. Need post-save upload.
**Where:** `simultaneous-translation/src/training/checkpointing.py`
`save_checkpoint_canonical_final`; wired into `train_hierarchical.py:1065`.

### 2026-05-08: v6e bf16 NaN -- must use float32 or fix attention mask
**Decision:** v6e-8 + torch_xla 2.9 + bfloat16 produces NaN loss at step 1.
Root cause is two independent bugs:
1. pytorch/xla #4152: HF transformers uses `torch.finfo(fp32).min = -3.4e38`
   in attention masks; when cast to bf16 this becomes `-inf`, then
   `(1-mask)*-inf = NaN`. Fix: either run `model.to(bfloat16)` BEFORE
   mask construction so `torch.finfo(bfloat16).min` is used, or replace
   `torch.finfo(dtype).min` with a safe constant like -10000.
2. pytorch/xla #8591/#8778: v6e-specific NaN at larger batch sizes even
   without the HF mask issue. Appears to be a v6e libtpu numerics bug.
   Workaround: use float32 precision (2x memory, 40x slower per step
   due to no bf16 matmul acceleration on TPU).
**Current status:** iter 12 ran fp32 successfully (200 steps, loss 7.36).
Long-term fix: upcast attention mask computation to fp32 while keeping
rest of model in bf16, or patch the HF model's mask value.
**Where:** `configs/stage2_tpu_v6e_spot.yaml` precision field.

### 2026-05-08: v6e FD limit -- ulimit must be set before libtpu init
**Decision:** v6e-8 with 8 chips + libtpu 0.0.21 opens ~100k file
descriptors during `eventfd_interrupt_async` init. The systemd-managed
startup_script inherited LimitNOFILE=100000 (or lower), causing
`eventfd() = -1: Too many open files` crash before any XLA compile.
Fix: launch training with explicit `ulimit -n 1048576` in the tmux
session, NOT via the systemd startup_script. The dedicated launcher
`/tmp/launch_train_v6e_v2.sh` handles this correctly.
**Where:** TPU host `/tmp/launch_train_v6e_v2.sh` (uploaded per QR).

### 2026-05-08: v6e-8 spot provisioning notes
**Decision:** v6e-8 spot instances in us-east1-d get preempted within
15 min during maintenance events. europe-west4-a has better capacity
(4:44 to ACTIVE). v6e-64 (8-host) fails at PROVISIONING with code 13
"internal error" (16-host alignment failure). v5litepod-8 spot quota
is 4 chips in us-central1-a (TRC promised 64; reality is 4). v4-32
spot QR has been SUSPENDED for hours (no capacity).
**Recommendation:** For production runs, use v4 on-demand (no preemption)
or v6e-8 spot in EU. Avoid v6e-64 spot and v5litepod spot.

### 2026-05-08: Patch 19 validated -- canonical end-of-training save works on v6e-8
**Decision:** `save_checkpoint_canonical_final()` in `checkpointing.py` does
`model.to("cpu")` per-submodule (gather all FSDPv2 shards to host RAM),
then host-0 writes via `save_pretrained(safe_serialization=False)` plus
`torch.save()` for plain modules. A CPU verification assertion checks
no XLA tensors remain before writing. The function is gated by
`cfg.train.final_canonical_save` (default False).
**Reason:** Iters 9/10/11 deadlocked at `PEFT.save_pretrained` on FSDPv2-XLA
(pytorch/xla #36004: save_pretrained doesn't support TPU-resident models).
The canonical fix is `model.to("cpu")` before save, which gathers all shards
in a single collective. On single-host v6e-8 this works cleanly; on
multi-host v4-32 all hosts must call `.to("cpu")` together (SPMD gather).
**safe_serialization=False** is required because safetensors cannot access
XLA storage pointers (HF #29608).
**Known limitation:** If save_dir is a gs:// URI, torch.save writes to a
local directory `gs:/...` instead of actual GCS. Need post-save upload.
**Where:** `simultaneous-translation/src/training/checkpointing.py`
`save_checkpoint_canonical_final`; wired into `train_hierarchical.py:1065`.

### 2026-05-08: v6e bf16 NaN -- must use float32 or fix attention mask
**Decision:** v6e-8 + torch_xla 2.9 + bfloat16 produces NaN loss at step 1.
Root cause is two independent bugs: pytorch/xla #4152 (HF attention mask
torch.finfo(fp32).min becomes -inf in bf16) and pytorch/xla #8591/#8778
(v6e-specific NaN at larger batch sizes). Workaround: float32 precision.
**Where:** `configs/stage2_tpu_v6e_spot.yaml` precision field.

### 2026-05-08: v6e FD limit -- ulimit must be set before libtpu init
**Decision:** v6e-8 + libtpu 0.0.21 opens ~100k FDs during eventfd init.
systemd startup_script LimitNOFILE was too low. Fix: launch with explicit
`ulimit -n 1048576` via /tmp/launch_train_v6e_v2.sh.
