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

(none yet — first milestone will be the 5000-step run completing.)
