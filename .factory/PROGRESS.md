# PROGRESS

Append-only running log of changes, decisions, failures, and next steps.

Auto-managed by `.factory/hooks/post_tool_use.py`,
`.factory/hooks/stop.py`, `.factory/hooks/pre_compact.py`, and
`.factory/hooks/session_end.py`. Quick-capture entries land here when
you start a message with `#progress`. Manual capture via
`/progress <text>`.

Format per entry:

```
## YYYY-MM-DDTHH:MM:SSZ | <branch>@<short-sha> | <status> | <kind>
<one-line summary>

<optional detail block>
```

Status: `info | done | fail | block`
Kind: `edit | exec | decide | plan | verify | session`

The most recent entry is at the top. Older entries beyond 90 days are
moved to `.factory/archive/PROGRESS-YYYY-Qn.md` by the
`archive-progress` skill.

---

## 2026-05-10T02:49:54Z | feat/tpu-support@cbdff89 | done | verify
Repository verification passed after iter 24h documentation and checkpoint-status updates.

Details:
- `.factory/VERIFY.md`: all 19 fenced bash blocks passed.
- TPU sharding probes were skipped because `PJRT_DEVICE` is unset on
  the workstation, as intended.
- Focused simultaneous-translation ruff validators passed:
  `ruff format --check src/ scripts/` and `ruff check src/ scripts/`.
- Verification fixes included excluding `.venv` from the secrets grep
  and lazily importing the Phase 3 `PipelineConfig` so
  `python3 cli.py --help` works even when optional pipeline modules are
  unavailable.

## 2026-05-10T02:41:43Z | feat/tpu-support@cbdff89 | done | exec
Iter 24h completed the first 5000-step v6e-8 production run and uploaded the final canonical checkpoint.

Details:
- W&B run: https://wandb.ai/cataluna84/tinyaya-stage2-tpu/runs/7rrjupc7
  (`v6e-spot-stage2-5k-iter24h`).
- Training reached step 5000/5000 and exited with status 0 at
  2026-05-10T01:47:24Z after 615.9 min.
- Final step line:
  `loss 5.3558 | text 10.3176 audio 4.3240 | 6.96s/step`.
- No NaN, OOM, RESOURCE_EXHAUSTED, fatal, traceback, bus-error, or
  kernel-panic signals were found in `/tmp/train.log`.
- Canonical final save completed and uploaded to
  `gs://tinyaya-stage2-tpu/checkpoints/stage2-tpu-v6e-spot/step_005000_final/`.
- GCS listing shows 8 objects / 2.37 GiB:
  `metadata.json`, `text_embed.pt`, `depth_decoder.pt`,
  `projection.pt`, `audio_heads.pt`, and PEFT adapter files.
- XLA compilation was startup-only: 18 compilation causes total, 12
  before visible step 1 and 6 around steps 1-2; no late recompiles
  through step 5000.

## 2026-05-09T15:21:21Z | feat/tpu-support@cbdff89 | done | edit
Iter 24h closes the remaining step-259 grad-accum epoch-boundary graph gap and uploads a fresh tarball.

Details:
- Exa/XLA evidence reinforced that HLO changes on input-shape or
  value-dependent branch changes; PyTorch/XLA #8616 also highlights that
  gradient accumulation wants a fixed [GA, batch, ...] topology.
- Added TPU host-side batch-axis padding in InterleavedCollator
  (`batch_pad_to`, `expected_num_codebooks`).
- Changed TPU DataLoader back to drop_last=False; tail rows are padded
  with zero attention/loss masks instead of dropped.
- Added macro-step guard so epoch reset happens only between optimizer
  steps, never inside a 4-way grad-accum graph.
- Kept iter24g SDPA mask-elision disable.
- Uploaded gs://tinyaya-stage2-tpu/code/tinyaya-repo-iter24h.tar.gz.
- Validators passed for simultaneous-translation: ruff format/check,
  py_compile all tracked Python, YAML parse, bash -n TPU scripts +
  launcher.

## 2026-05-09T11:32:55Z | feat/tpu-support@cbdff89 | block | exec
Second user-approved v6e-8 retry preempted during iter24g compile; no step was reached.

Details:
- Patched startup_script.sh to include the same stable TPU env as the
  manual launcher (flash disabled, PT_XLA_DEBUG_LEVEL=1) and ulimit
  1048576, then refreshed the iter24g tarball in GCS.
- Recreated v6e-8 spot QR again; startup completed and manual launcher
  started iter24g.
- W&B run was created:
  https://wandb.ai/cataluna84/tinyaya-stage2-tpu/runs/kmvydrq1
- Runtime reached compile (`Graph Hash: 6be815b64f...`) with 0 NaN/OOM
  signals and no loss lines yet.
- TPU was preempted during first compile; QR=SUSPENDED/SERVICE and
  node=PREEMPTED. No iter24g training step reached.

## 2026-05-09T10:45:58Z | feat/tpu-support@cbdff89 | block | exec
Approved v6e-8 QR recreate succeeded, but the replacement spot TPU preempted before iter 24g could launch.

Details:
- Deleted stale preempted QR/node and recreated
  tinyaya-stage2-spot-v6e8-eu-qr with iter24g tarball metadata.
- QR reached ACTIVE/READY, then startup failed once because the tarball
  omitted root README.md required by hatchling.
- Rebuilt and re-uploaded iter24g tarball including README.md.
- Before manual redeploy could complete, node returned to PREEMPTED and
  QR to SUSPENDED (SERVICE).
- No iter24g W&B run was created; no training step ran on the replacement.

## 2026-05-09T10:11:57Z | feat/tpu-support@cbdff89 | block | exec
Iter 24g patch + tarball are ready, but the v6e-8 TPU node is terminal PREEMPTED.

Details:
- Exa/XLA evidence points to a step-259 graph-topology change from dynamic
  batch/mask branches.
- Patched train_hierarchical.py to use TPU drop_last=True, static-shape
  assertions, and disabled HF SDPA attention-mask elision.
- Updated stage2_tpu_v6e_spot.yaml and launcher to iter24g.
- Uploaded gs://tinyaya-stage2-tpu/code/tinyaya-repo-iter24g.tar.gz.
- Focused validators passed: py_compile, YAML parse, bash -n, ruff format
  check, ruff check.
- tpu-watchdog verdict=crashed: node tinyaya-stage2-spot-v6e8-eu is
  PREEMPTED and cannot be used; do not auto-recreate QR without approval.

## 2026-05-09T03:16:14Z | feat/tpu-support@cbdff89 | info | session
SessionEnd (other): 28 item(s) carried forward

Next steps:
- >= 200 successful steps (canary `max_steps=200`); checkpoint
- Patches 12 + 13 either landed and verified, or proven
- All commands in `VERIFY.md` (monorepo + simultaneous-translation
- 5000-step run completes (canary -> full config); final loss
- `eval_stage2.py` ASR-BLEU + DNSMOS recorded against
- Verify activation memory is under 4 GB per chip via `diagnose()`.
- If still tight, try moving frozen `MoshiDecoderLayer` to bf16
- Re-run `probe_strategies.py` against the real model with


## 2026-05-09T03:16:14Z | feat/tpu-support@cbdff89 | info | session
SessionEnd (other): 28 item(s) carried forward

Next steps:
- >= 200 successful steps (canary `max_steps=200`); checkpoint
- Patches 12 + 13 either landed and verified, or proven
- All commands in `VERIFY.md` (monorepo + simultaneous-translation
- 5000-step run completes (canary -> full config); final loss
- `eval_stage2.py` ASR-BLEU + DNSMOS recorded against
- Verify activation memory is under 4 GB per chip via `diagnose()`.
- If still tight, try moving frozen `MoshiDecoderLayer` to bf16
- Re-run `probe_strategies.py` against the real model with


## 2026-05-09T03:13:31Z | feat/tpu-support@cbdff89 | info | session
SessionEnd (other): 28 item(s) carried forward

Next steps:
- >= 200 successful steps (canary `max_steps=200`); checkpoint
- Patches 12 + 13 either landed and verified, or proven
- All commands in `VERIFY.md` (monorepo + simultaneous-translation
- 5000-step run completes (canary -> full config); final loss
- `eval_stage2.py` ASR-BLEU + DNSMOS recorded against
- Verify activation memory is under 4 GB per chip via `diagnose()`.
- If still tight, try moving frozen `MoshiDecoderLayer` to bf16
- Re-run `probe_strategies.py` against the real model with


## 2026-05-09T03:13:26Z | feat/tpu-support@cbdff89 | info | session
PreCompact (auto): 28 unchecked PLAN items

Top open items:
- >= 200 successful steps (canary `max_steps=200`); checkpoint
- Patches 12 + 13 either landed and verified, or proven
- All commands in `VERIFY.md` (monorepo + simultaneous-translation
- 5000-step run completes (canary -> full config); final loss
- `eval_stage2.py` ASR-BLEU + DNSMOS recorded against
- Verify activation memory is under 4 GB per chip via `diagnose()`.
- If still tight, try moving frozen `MoshiDecoderLayer` to bf16
- Re-run `probe_strategies.py` against the real model with
- Decide between `fsdpv2_lora` and `fsdpv2` based on:
- Document the decision in `memories.md` under "TPU strategy


## 2026-05-09T03:12:44Z | feat/tpu-support@cbdff89 | info | session
SessionEnd (other): 28 item(s) carried forward

Next steps:
- >= 200 successful steps (canary `max_steps=200`); checkpoint
- Patches 12 + 13 either landed and verified, or proven
- All commands in `VERIFY.md` (monorepo + simultaneous-translation
- 5000-step run completes (canary -> full config); final loss
- `eval_stage2.py` ASR-BLEU + DNSMOS recorded against
- Verify activation memory is under 4 GB per chip via `diagnose()`.
- If still tight, try moving frozen `MoshiDecoderLayer` to bf16
- Re-run `probe_strategies.py` against the real model with


## 2026-05-09T03:11:50Z | feat/tpu-support@cbdff89 | info | session
SessionEnd (other): 28 item(s) carried forward

Next steps:
- >= 200 successful steps (canary `max_steps=200`); checkpoint
- Patches 12 + 13 either landed and verified, or proven
- All commands in `VERIFY.md` (monorepo + simultaneous-translation
- 5000-step run completes (canary -> full config); final loss
- `eval_stage2.py` ASR-BLEU + DNSMOS recorded against
- Verify activation memory is under 4 GB per chip via `diagnose()`.
- If still tight, try moving frozen `MoshiDecoderLayer` to bf16
- Re-run `probe_strategies.py` against the real model with


## 2026-05-09T03:11:50Z | feat/tpu-support@cbdff89 | info | session
SessionEnd (other): 28 item(s) carried forward

Next steps:
- >= 200 successful steps (canary `max_steps=200`); checkpoint
- Patches 12 + 13 either landed and verified, or proven
- All commands in `VERIFY.md` (monorepo + simultaneous-translation
- 5000-step run completes (canary -> full config); final loss
- `eval_stage2.py` ASR-BLEU + DNSMOS recorded against
- Verify activation memory is under 4 GB per chip via `diagnose()`.
- If still tight, try moving frozen `MoshiDecoderLayer` to bf16
- Re-run `probe_strategies.py` against the real model with


## 2026-05-09T03:11:18Z | feat/tpu-support@cbdff89 | info | session
SessionEnd (other): 28 item(s) carried forward

Next steps:
- >= 200 successful steps (canary `max_steps=200`); checkpoint
- Patches 12 + 13 either landed and verified, or proven
- All commands in `VERIFY.md` (monorepo + simultaneous-translation
- 5000-step run completes (canary -> full config); final loss
- `eval_stage2.py` ASR-BLEU + DNSMOS recorded against
- Verify activation memory is under 4 GB per chip via `diagnose()`.
- If still tight, try moving frozen `MoshiDecoderLayer` to bf16
- Re-run `probe_strategies.py` against the real model with


## 2026-05-09T03:11:18Z | feat/tpu-support@cbdff89 | info | session
SessionEnd (other): 28 item(s) carried forward

Next steps:
- >= 200 successful steps (canary `max_steps=200`); checkpoint
- Patches 12 + 13 either landed and verified, or proven
- All commands in `VERIFY.md` (monorepo + simultaneous-translation
- 5000-step run completes (canary -> full config); final loss
- `eval_stage2.py` ASR-BLEU + DNSMOS recorded against
- Verify activation memory is under 4 GB per chip via `diagnose()`.
- If still tight, try moving frozen `MoshiDecoderLayer` to bf16
- Re-run `probe_strategies.py` against the real model with


## 2026-05-09T03:06:04Z | feat/tpu-support@cbdff89 | info | session
SessionEnd (other): 28 item(s) carried forward

Next steps:
- >= 200 successful steps (canary `max_steps=200`); checkpoint
- Patches 12 + 13 either landed and verified, or proven
- All commands in `VERIFY.md` (monorepo + simultaneous-translation
- 5000-step run completes (canary -> full config); final loss
- `eval_stage2.py` ASR-BLEU + DNSMOS recorded against
- Verify activation memory is under 4 GB per chip via `diagnose()`.
- If still tight, try moving frozen `MoshiDecoderLayer` to bf16
- Re-run `probe_strategies.py` against the real model with


## 2026-05-09T03:06:04Z | feat/tpu-support@cbdff89 | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-09T02:26:14Z | feat/tpu-support@cbdff89 | info | session
SessionEnd (other): 28 item(s) carried forward

Next steps:
- >= 200 successful steps (canary `max_steps=200`); checkpoint
- Patches 12 + 13 either landed and verified, or proven
- All commands in `VERIFY.md` (monorepo + simultaneous-translation
- 5000-step run completes (canary -> full config); final loss
- `eval_stage2.py` ASR-BLEU + DNSMOS recorded against
- Verify activation memory is under 4 GB per chip via `diagnose()`.
- If still tight, try moving frozen `MoshiDecoderLayer` to bf16
- Re-run `probe_strategies.py` against the real model with


## 2026-05-09T02:26:14Z | feat/tpu-support@cbdff89 | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-09T02:00:30Z | feat/tpu-support@cbdff89 | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-09T01:50:32Z | feat/tpu-support@cbdff89 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/_artifacts/launch_train_v6e_v2.sh`


## 2026-05-09T01:50:27Z | feat/tpu-support@cbdff89 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/_artifacts/launch_train_v6e_v2.sh`


## 2026-05-09T01:50:21Z | feat/tpu-support@cbdff89 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/_artifacts/launch_train_v6e_v2.sh`


## 2026-05-09T01:50:11Z | feat/tpu-support@cbdff89 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/_artifacts/launch_train_v6e_v2.sh`


## 2026-05-09T01:50:05Z | feat/tpu-support@cbdff89 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/configs/stage2_tpu_v6e_spot.yaml`


## 2026-05-09T01:50:00Z | feat/tpu-support@cbdff89 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/configs/stage2_tpu_v6e_spot.yaml`


## 2026-05-09T01:35:47Z | feat/tpu-support@cbdff89 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/.factory/memories.md`


## 2026-05-09T00:49:47Z | feat/tpu-support@cbdff89 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/train_hierarchical.py`


## 2026-05-09T00:49:08Z | feat/tpu-support@cbdff89 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/_artifacts/launch_train_v6e_v2.sh`


## 2026-05-09T00:49:04Z | feat/tpu-support@cbdff89 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/_artifacts/launch_train_v6e_v2.sh`


## 2026-05-09T00:48:59Z | feat/tpu-support@cbdff89 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/_artifacts/launch_train_v6e_v2.sh`


## 2026-05-09T00:48:51Z | feat/tpu-support@cbdff89 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/_artifacts/launch_train_v6e_v2.sh`


## 2026-05-09T00:48:32Z | feat/tpu-support@cbdff89 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/_artifacts/launch_train_v6e_v2.sh`


## 2026-05-09T00:48:24Z | feat/tpu-support@cbdff89 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/configs/stage2_tpu_v6e_spot.yaml`


## 2026-05-09T00:48:07Z | feat/tpu-support@cbdff89 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/configs/stage2_tpu_v6e_spot.yaml`


## 2026-05-09T00:48:01Z | feat/tpu-support@cbdff89 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/train_hierarchical.py`


## 2026-05-08T19:04:01Z | feat/tpu-support@cbdff89 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/_artifacts/launch_train_v6e_v2.sh`


## 2026-05-08T19:03:56Z | feat/tpu-support@cbdff89 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/_artifacts/launch_train_v6e_v2.sh`


## 2026-05-08T19:03:51Z | feat/tpu-support@cbdff89 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/_artifacts/launch_train_v6e_v2.sh`


## 2026-05-08T19:03:33Z | feat/tpu-support@cbdff89 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/_artifacts/launch_train_v6e_v2.sh`


## 2026-05-08T19:02:44Z | feat/tpu-support@cbdff89 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/configs/stage2_tpu_v6e_spot.yaml`


## 2026-05-08T19:01:18Z | feat/tpu-support@cbdff89 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/configs/stage2_tpu_v6e_spot.yaml`


## 2026-05-08T18:16:48Z | feat/tpu-support@0edb0c8 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/.factory/PLAN.md`


## 2026-05-08T18:16:09Z | feat/tpu-support@0edb0c8 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/.factory/memories.md`


## 2026-05-08T18:15:13Z | feat/tpu-support@0edb0c8 | done | edit
created `/home/cataluna84/Workspace/tinyaya-stage2-scale/_artifacts/orch_state.json`


## 2026-05-08T17:46:36Z | feat/tpu-support@0edb0c8 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/configs/stage2_tpu_v6e_spot.yaml`


## 2026-05-08T17:46:24Z | feat/tpu-support@0edb0c8 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/_artifacts/launch_train_v6e_v2.sh`


## 2026-05-08T17:46:19Z | feat/tpu-support@0edb0c8 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/_artifacts/launch_train_v6e_v2.sh`


## 2026-05-08T17:46:13Z | feat/tpu-support@0edb0c8 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/_artifacts/launch_train_v6e_v2.sh`


## 2026-05-08T17:46:04Z | feat/tpu-support@0edb0c8 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/_artifacts/launch_train_v6e_v2.sh`


## 2026-05-08T17:45:30Z | feat/tpu-support@0edb0c8 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/configs/stage2_tpu_v6e_spot.yaml`


## 2026-05-08T17:45:20Z | feat/tpu-support@0edb0c8 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/src/backend/tpu_backend.py`


## 2026-05-08T17:38:32Z | feat/tpu-support@0edb0c8 | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-08T17:01:36Z | feat/tpu-support@0edb0c8 | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-08T16:44:17Z | feat/tpu-support@0edb0c8 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/configs/stage2_tpu_v6e_spot.yaml`


## 2026-05-08T16:43:48Z | feat/tpu-support@0edb0c8 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/_artifacts/launch_train_v6e_v2.sh`


## 2026-05-08T16:43:43Z | feat/tpu-support@0edb0c8 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/_artifacts/launch_train_v6e_v2.sh`


## 2026-05-08T16:43:37Z | feat/tpu-support@0edb0c8 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/_artifacts/launch_train_v6e_v2.sh`


## 2026-05-08T16:43:28Z | feat/tpu-support@0edb0c8 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/_artifacts/launch_train_v6e_v2.sh`


## 2026-05-08T16:43:10Z | feat/tpu-support@0edb0c8 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/_artifacts/launch_train_v6e_v2.sh`


## 2026-05-08T15:56:42Z | feat/tpu-support@659ecb9 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/src/backend/tpu_backend.py`


## 2026-05-08T15:56:00Z | feat/tpu-support@659ecb9 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/train_hierarchical.py`


## 2026-05-08T15:51:55Z | feat/tpu-support@659ecb9 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/train_hierarchical.py`


## 2026-05-08T15:40:44Z | feat/tpu-support@659ecb9 | done | edit
created `/home/cataluna84/Workspace/tinyaya-stage2-scale/_artifacts/orch_state.json`


## 2026-05-08T15:40:18Z | feat/tpu-support@659ecb9 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/.factory/memories.md`


## 2026-05-08T15:39:51Z | feat/tpu-support@659ecb9 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/.factory/PLAN.md`


## 2026-05-08T15:38:50Z | feat/tpu-support@659ecb9 | done | edit
created `/home/cataluna84/Workspace/tinyaya-stage2-scale/.factory/skills/tpu-redeploy/SKILL.md`


## 2026-05-08T15:38:19Z | feat/tpu-support@659ecb9 | done | edit
created `/home/cataluna84/Workspace/tinyaya-stage2-scale/.factory/skills/tpu-orchestrate/SKILL.md`


## 2026-05-08T15:37:13Z | feat/tpu-support@659ecb9 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/_artifacts/launch_train_v6e_v2.sh`


## 2026-05-08T15:37:07Z | feat/tpu-support@659ecb9 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/_artifacts/launch_train_v6e_v2.sh`


## 2026-05-08T15:37:01Z | feat/tpu-support@659ecb9 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/_artifacts/launch_train_v6e_v2.sh`


## 2026-05-08T15:36:54Z | feat/tpu-support@659ecb9 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/_artifacts/launch_train_v6e_v2.sh`


## 2026-05-08T15:36:26Z | feat/tpu-support@659ecb9 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/configs/stage2_tpu_v6e_spot.yaml`


## 2026-05-08T15:36:17Z | feat/tpu-support@659ecb9 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/configs/stage2_tpu_v6e_spot.yaml`


## 2026-05-08T15:36:10Z | feat/tpu-support@659ecb9 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/configs/stage2_tpu_v6e_spot.yaml`


## 2026-05-08T15:35:50Z | feat/tpu-support@659ecb9 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/train_hierarchical.py`


## 2026-05-08T15:35:40Z | feat/tpu-support@659ecb9 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/train_hierarchical.py`


## 2026-05-08T15:10:24Z | feat/tpu-support@659ecb9 | info | session
SessionEnd (other): 35 item(s) carried forward

Next steps:
- >= 200 successful steps (canary `max_steps=200`); checkpoint
- Patches 12 + 13 either landed and verified, or proven
- All commands in `VERIFY.md` (monorepo + simultaneous-translation
- 5000-step run completes (canary -> full config); final loss
- `eval_stage2.py` ASR-BLEU + DNSMOS recorded against
- Verify activation memory is under 4 GB per chip via `diagnose()`.
- If still tight, try moving frozen `MoshiDecoderLayer` to bf16
- Re-run `probe_strategies.py` against the real model with


## 2026-05-08T14:25:52Z | feat/tpu-support@659ecb9 | info | session
SessionEnd (other): 35 item(s) carried forward

Next steps:
- >= 200 successful steps (canary `max_steps=200`); checkpoint
- Patches 12 + 13 either landed and verified, or proven
- All commands in `VERIFY.md` (monorepo + simultaneous-translation
- 5000-step run completes (canary -> full config); final loss
- `eval_stage2.py` ASR-BLEU + DNSMOS recorded against
- Verify activation memory is under 4 GB per chip via `diagnose()`.
- If still tight, try moving frozen `MoshiDecoderLayer` to bf16
- Re-run `probe_strategies.py` against the real model with


## 2026-05-08T14:20:04Z | feat/tpu-support@659ecb9 | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-08T14:15:41Z | feat/tpu-support@7948cbb | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/configs/stage2_tpu_v6e_spot.yaml`


## 2026-05-08T14:15:29Z | feat/tpu-support@7948cbb | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/train_hierarchical.py`


## 2026-05-08T12:39:34Z | feat/tpu-support@7948cbb | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-08T12:34:53Z | feat/tpu-support@61b87fb | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/train_hierarchical.py`


## 2026-05-08T12:34:17Z | feat/tpu-support@61b87fb | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/train_hierarchical.py`


## 2026-05-08T12:34:03Z | feat/tpu-support@61b87fb | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/train_hierarchical.py`


## 2026-05-08T11:39:00Z | feat/tpu-support@61b87fb | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-08T10:56:29Z | feat/tpu-support@9131e0f | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/.factory/memories.md`


## 2026-05-08T10:56:16Z | feat/tpu-support@9131e0f | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/.factory/memories.md`


## 2026-05-08T10:55:54Z | feat/tpu-support@9131e0f | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/_artifacts/orch_state.json`


## 2026-05-08T10:55:48Z | feat/tpu-support@9131e0f | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/.factory/PLAN.md`


## 2026-05-08T10:55:39Z | feat/tpu-support@9131e0f | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/docs/tpu-launch-plan.md`


## 2026-05-08T10:55:30Z | feat/tpu-support@9131e0f | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/docs/tpu-changes.md`


## 2026-05-08T10:55:15Z | feat/tpu-support@9131e0f | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/AGENTS.md`


## 2026-05-08T10:55:09Z | feat/tpu-support@9131e0f | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/tpu/README.md`


## 2026-05-08T10:55:00Z | feat/tpu-support@9131e0f | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/tpu/README.md`


## 2026-05-08T10:54:54Z | feat/tpu-support@9131e0f | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/README.md`


## 2026-05-08T10:54:41Z | feat/tpu-support@9131e0f | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/README.md`


## 2026-05-08T10:54:31Z | feat/tpu-support@9131e0f | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/_artifacts/launch_train_v6e_v2.sh`


## 2026-05-08T10:54:04Z | feat/tpu-support@9131e0f | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/configs/stage2_tpu_canary_v6e_spot.yaml`


## 2026-05-08T10:53:50Z | feat/tpu-support@9131e0f | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/configs/stage2_tpu_canary_v6e_spot.yaml`


## 2026-05-08T10:53:36Z | feat/tpu-support@9131e0f | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/configs/stage2_tpu_canary_v6e_spot.yaml`


## 2026-05-08T10:53:30Z | feat/tpu-support@9131e0f | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/configs/stage2_tpu_canary_v6e_spot.yaml`


## 2026-05-08T10:53:25Z | feat/tpu-support@9131e0f | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/configs/stage2_tpu_canary_v6e_spot.yaml`


## 2026-05-08T10:53:19Z | feat/tpu-support@9131e0f | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/configs/stage2_tpu_canary_v6e_spot.yaml`


## 2026-05-08T10:52:51Z | feat/tpu-support@9131e0f | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/configs/stage2_tpu_canary_v6e_spot.yaml`


## 2026-05-08T10:50:59Z | feat/tpu-support@9131e0f | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-08T09:58:10Z | feat/tpu-support@7073d77 | done | edit
created `/home/cataluna84/Workspace/tinyaya-stage2-scale/_artifacts/smoke_profiler.py`


## 2026-05-08T09:50:29Z | feat/tpu-support@7073d77 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/train_hierarchical.py`


## 2026-05-08T09:44:34Z | feat/tpu-support@7073d77 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/_artifacts/launch_train_v6e_v2.sh`


## 2026-05-08T09:44:08Z | feat/tpu-support@7073d77 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/_artifacts/launch_train_v6e_v2.sh`


## 2026-05-08T09:43:37Z | feat/tpu-support@7073d77 | done | edit
created `/home/cataluna84/Workspace/tinyaya-stage2-scale/_artifacts/capture_xla_profile.py`


## 2026-05-08T09:41:11Z | feat/tpu-support@7073d77 | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-08T09:35:09Z | feat/tpu-support@7073d77 | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-08T09:11:53Z | feat/tpu-support@2400ada | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/src/backend/tpu_backend.py`


## 2026-05-08T08:44:48Z | feat/tpu-support@2400ada | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/_artifacts/launch_train_v6e_v2.sh`


## 2026-05-08T08:43:50Z | feat/tpu-support@2400ada | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/train_hierarchical.py`


## 2026-05-08T08:43:12Z | feat/tpu-support@2400ada | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/train_hierarchical.py`


## 2026-05-08T08:42:54Z | feat/tpu-support@2400ada | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/train_hierarchical.py`


## 2026-05-08T08:42:38Z | feat/tpu-support@2400ada | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/src/backend/tpu_backend.py`


## 2026-05-08T08:42:08Z | feat/tpu-support@2400ada | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/src/backend/tpu_backend.py`


## 2026-05-08T08:38:38Z | feat/tpu-support@2400ada | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/src/backend/tpu_backend.py`


## 2026-05-08T08:33:11Z | feat/tpu-support@2400ada | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-08T07:56:19Z | feat/tpu-support@3c629da | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/train_hierarchical.py`


## 2026-05-08T07:55:50Z | feat/tpu-support@3c629da | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/train_hierarchical.py`


## 2026-05-08T07:37:55Z | feat/tpu-support@3c629da | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/train_hierarchical.py`


## 2026-05-08T07:23:50Z | feat/tpu-support@3c629da | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/.factory/memories.md`


## 2026-05-08T07:23:10Z | feat/tpu-support@3c629da | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/_artifacts/orch_state.json`


## 2026-05-08T07:22:37Z | feat/tpu-support@3c629da | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/.factory/PLAN.md`


## 2026-05-08T07:21:33Z | feat/tpu-support@3c629da | done | edit
created `/home/cataluna84/Workspace/tinyaya-stage2-scale/_artifacts/launch_train_v6e_v2.sh`


## 2026-05-08T07:21:00Z | feat/tpu-support@3c629da | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/train_hierarchical.py`


## 2026-05-08T07:20:39Z | feat/tpu-support@3c629da | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/train_hierarchical.py`


## 2026-05-08T07:19:52Z | feat/tpu-support@3c629da | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/train_hierarchical.py`


## 2026-05-08T07:19:34Z | feat/tpu-support@3c629da | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/train_hierarchical.py`


## 2026-05-08T07:19:12Z | feat/tpu-support@3c629da | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/train_hierarchical.py`


## 2026-05-08T07:18:52Z | feat/tpu-support@3c629da | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/src/backend/tpu_backend.py`


## 2026-05-08T07:18:37Z | feat/tpu-support@3c629da | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/configs/stage2_tpu_canary_v6e_spot.yaml`


## 2026-05-08T07:18:24Z | feat/tpu-support@3c629da | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/configs/stage2_tpu_canary_v6e_spot.yaml`


## 2026-05-08T06:46:58Z | feat/tpu-support@3c629da | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-08T06:39:12Z | feat/tpu-support@3c629da | done | edit
created `/home/cataluna84/Workspace/tinyaya-stage2-scale/_artifacts/memory_probe.py`


## 2026-05-08T06:06:32Z | feat/tpu-support@3c629da | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/configs/stage2_tpu_canary_v6e_spot.yaml`


## 2026-05-08T06:03:47Z | feat/tpu-support@3c629da | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/configs/stage2_tpu_canary_v6e_spot.yaml`


## 2026-05-08T05:55:05Z | feat/tpu-support@3c629da | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-08T05:44:53Z | feat/tpu-support@3c629da | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/configs/stage2_tpu_canary_v6e_spot.yaml`


## 2026-05-08T05:44:47Z | feat/tpu-support@3c629da | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/.factory/PLAN.md`


## 2026-05-08T05:44:33Z | feat/tpu-support@3c629da | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/.factory/PLAN.md`


## 2026-05-08T05:42:45Z | feat/tpu-support@3c629da | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-08T05:30:09Z | feat/tpu-support@a7649a0 | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-08T05:29:07Z | feat/tpu-support@a7649a0 | done | edit
created `/home/cataluna84/Workspace/tinyaya-stage2-scale/_artifacts/orch_state.json`


## 2026-05-08T05:28:35Z | feat/tpu-support@a7649a0 | info | session
SessionEnd (other): 26 item(s) carried forward

Next steps:
- >= 200 successful steps (canary `max_steps=200`); first
- Patches 12 + 13 either landed and verified, or proven
- All commands in `VERIFY.md` (monorepo + simultaneous-translation
- 5000-step run completes (canary -> full config); final loss
- `eval_stage2.py` ASR-BLEU + DNSMOS recorded against
- Verify activation memory is under 4 GB per chip via `diagnose()`.
- If still tight, try moving frozen `MoshiDecoderLayer` to bf16
- Re-run `probe_strategies.py` against the real model with


## 2026-05-08T05:28:34Z | feat/tpu-support@a7649a0 | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-08T05:26:22Z | feat/tpu-support@a7649a0 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/.factory/memories.md`


## 2026-05-08T05:25:38Z | feat/tpu-support@a7649a0 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/.factory/droids/tpu-watchdog.md`


## 2026-05-08T05:25:33Z | feat/tpu-support@a7649a0 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/.factory/droids/tpu-watchdog.md`


## 2026-05-08T05:25:25Z | feat/tpu-support@a7649a0 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/.factory/droids/tpu-watchdog.md`


## 2026-05-08T05:25:08Z | feat/tpu-support@a7649a0 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/.factory/droids/tpu-watchdog.md`


## 2026-05-08T05:24:42Z | feat/tpu-support@a7649a0 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/tpu/README.md`


## 2026-05-08T05:24:12Z | feat/tpu-support@a7649a0 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/tpu/README.md`


## 2026-05-08T05:23:52Z | feat/tpu-support@a7649a0 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/.factory/skills/tpu-redeploy/SKILL.md`


## 2026-05-08T05:23:46Z | feat/tpu-support@a7649a0 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/.factory/skills/tpu-redeploy/SKILL.md`


## 2026-05-08T05:23:40Z | feat/tpu-support@a7649a0 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/.factory/skills/tpu-redeploy/SKILL.md`


## 2026-05-08T05:23:29Z | feat/tpu-support@a7649a0 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/.factory/skills/tpu-redeploy/SKILL.md`


## 2026-05-08T05:23:19Z | feat/tpu-support@a7649a0 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/.factory/skills/tpu-redeploy/SKILL.md`


## 2026-05-08T05:23:07Z | feat/tpu-support@a7649a0 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/.factory/skills/tpu-redeploy/SKILL.md`


## 2026-05-08T05:22:43Z | feat/tpu-support@a7649a0 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/.factory/orchestration/SPEC.md`


## 2026-05-08T05:22:38Z | feat/tpu-support@a7649a0 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/.factory/orchestration/SPEC.md`


## 2026-05-08T05:22:31Z | feat/tpu-support@a7649a0 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/.factory/orchestration/SPEC.md`


## 2026-05-08T05:22:17Z | feat/tpu-support@a7649a0 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/.factory/orchestration/SPEC.md`


## 2026-05-08T05:21:56Z | feat/tpu-support@a7649a0 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/docs/tpu-launch-plan.md`


## 2026-05-08T05:21:38Z | feat/tpu-support@a7649a0 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/docs/tpu-changes.md`


## 2026-05-08T05:21:26Z | feat/tpu-support@a7649a0 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/docs/tpu-capacity-log.md`


## 2026-05-08T05:21:13Z | feat/tpu-support@a7649a0 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/docs/tpu-capacity-log.md`


## 2026-05-08T05:21:02Z | feat/tpu-support@a7649a0 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/docs/tpu-trc-allocation.md`


## 2026-05-08T05:20:46Z | feat/tpu-support@a7649a0 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/AGENTS.md`


## 2026-05-08T05:20:29Z | feat/tpu-support@a7649a0 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/AGENTS.md`


## 2026-05-08T05:20:08Z | feat/tpu-support@a7649a0 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/.factory/PLAN.md`


## 2026-05-08T05:19:55Z | feat/tpu-support@a7649a0 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/.factory/PLAN.md`


## 2026-05-08T05:19:30Z | feat/tpu-support@a7649a0 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/.factory/PLAN.md`


## 2026-05-08T05:18:51Z | feat/tpu-support@a7649a0 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/CONTRIBUTING.md`


## 2026-05-08T05:18:33Z | feat/tpu-support@a7649a0 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/README.md`


## 2026-05-08T05:18:04Z | feat/tpu-support@a7649a0 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/README.md`


## 2026-05-08T05:17:51Z | feat/tpu-support@a7649a0 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/README.md`


## 2026-05-08T05:17:33Z | feat/tpu-support@a7649a0 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/README.md`


## 2026-05-08T05:17:21Z | feat/tpu-support@a7649a0 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/README.md`


## 2026-05-08T05:16:59Z | feat/tpu-support@a7649a0 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/README.md`


## 2026-05-08T05:16:50Z | feat/tpu-support@a7649a0 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/README.md`


## 2026-05-08T05:16:36Z | feat/tpu-support@a7649a0 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/README.md`


## 2026-05-08T05:16:23Z | feat/tpu-support@a7649a0 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/README.md`


## 2026-05-08T05:16:16Z | feat/tpu-support@a7649a0 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/README.md`


## 2026-05-08T05:16:08Z | feat/tpu-support@a7649a0 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/README.md`


## 2026-05-08T05:11:31Z | feat/tpu-support@a7649a0 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/.factory/skills/tpu-orchestrate/SKILL.md`


## 2026-05-08T05:11:21Z | feat/tpu-support@a7649a0 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/.factory/skills/tpu-orchestrate/SKILL.md`


## 2026-05-08T05:08:27Z | feat/tpu-support@a7649a0 | info | session
SessionEnd (other): 22 item(s) carried forward

Next steps:
- >= 200 successful steps (canary `max_steps=200`); first
- Patches 12 + 13 either landed and verified, or proven
- All commands in `VERIFY.md` (monorepo + simultaneous-translation
- 5000-step run completes (canary -> full config); final loss
- `eval_stage2.py` ASR-BLEU + DNSMOS recorded against
- Verify activation memory is under 4 GB per chip via `diagnose()`.
- If still tight, try moving frozen `MoshiDecoderLayer` to bf16
- Re-run `probe_strategies.py` against the real model with


## 2026-05-08T05:08:21Z | feat/tpu-support@a7649a0 | info | session
PreCompact (auto): 22 unchecked PLAN items

Top open items:
- >= 200 successful steps (canary `max_steps=200`); first
- Patches 12 + 13 either landed and verified, or proven
- All commands in `VERIFY.md` (monorepo + simultaneous-translation
- 5000-step run completes (canary -> full config); final loss
- `eval_stage2.py` ASR-BLEU + DNSMOS recorded against
- Verify activation memory is under 4 GB per chip via `diagnose()`.
- If still tight, try moving frozen `MoshiDecoderLayer` to bf16
- Re-run `probe_strategies.py` against the real model with
- Decide between `fsdpv2_lora` and `fsdpv2` based on:
- Document the decision in `memories.md` under "TPU strategy


## 2026-05-08T05:08:14Z | feat/tpu-support@a7649a0 | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-08T05:06:00Z | feat/tpu-support@a7649a0 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/src/training/checkpointing.py`


## 2026-05-08T04:43:36Z | feat/tpu-support@a7649a0 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/configs/stage2_tpu_canary_v6e_spot.yaml`


## 2026-05-08T04:39:14Z | feat/tpu-support@a7649a0 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/configs/stage2_tpu_canary_v6e_spot.yaml`


## 2026-05-08T04:39:06Z | feat/tpu-support@a7649a0 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/configs/stage2_tpu_canary_v6e_spot.yaml`


## 2026-05-08T04:38:54Z | feat/tpu-support@a7649a0 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/train_hierarchical.py`


## 2026-05-08T04:38:20Z | feat/tpu-support@a7649a0 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/src/training/checkpointing.py`


## 2026-05-08T04:33:34Z | feat/tpu-support@a7649a0 | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-08T04:32:37Z | feat/tpu-support@dccb7a1 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/configs/stage2_tpu_canary_v6e_spot.yaml`


## 2026-05-08T03:50:44Z | feat/tpu-support@dccb7a1 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/configs/stage2_tpu_canary_v6e_spot.yaml`


## 2026-05-08T03:33:09Z | feat/tpu-support@dccb7a1 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/configs/stage2_tpu_canary_v6e_spot.yaml`


## 2026-05-08T03:32:23Z | feat/tpu-support@dccb7a1 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/src/training/checkpointing.py`


## 2026-05-08T03:13:17Z | feat/tpu-support@dccb7a1 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/.factory/skills/tpu-orchestrate/SKILL.md`


## 2026-05-08T03:07:44Z | feat/tpu-support@dccb7a1 | done | edit
created `/tmp/launch_train_v6e_v2.sh`


## 2026-05-08T03:05:53Z | feat/tpu-support@dccb7a1 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/configs/stage2_tpu_canary_v6e_spot.yaml`


## 2026-05-08T03:05:52Z | feat/tpu-support@dccb7a1 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/configs/stage2_tpu_canary_v6e_spot.yaml`


## 2026-05-08T02:49:31Z | feat/tpu-support@dccb7a1 | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-08T02:29:20Z | feat/tpu-support@dccb7a1 | done | edit
created `/home/cataluna84/Workspace/tinyaya-stage2-scale/_artifacts/v6e8_eu_qr_watch.sh`


## 2026-05-08T01:42:18Z | feat/tpu-support@dccb7a1 | done | edit
created `/home/cataluna84/Workspace/tinyaya-stage2-scale/_artifacts/v6e8_qr_watch.sh`


## 2026-05-08T01:20:39Z | feat/tpu-support@dccb7a1 | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-08T01:19:01Z | feat/tpu-support@dccb7a1 | done | edit
created `/home/cataluna84/Workspace/tinyaya-stage2-scale/_artifacts/v6e_qr_watch.sh`


## 2026-05-08T01:17:48Z | feat/tpu-support@dccb7a1 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/configs/stage2_tpu_canary_v6e_spot.yaml`


## 2026-05-08T01:17:37Z | feat/tpu-support@dccb7a1 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/configs/stage2_tpu_canary_v6e_spot.yaml`


## 2026-05-08T01:12:22Z | feat/tpu-support@dccb7a1 | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-08T01:06:07Z | feat/tpu-support@dccb7a1 | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-08T01:00:56Z | feat/tpu-support@dccb7a1 | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-08T00:49:27Z | feat/tpu-support@dccb7a1 | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-08T00:48:49Z | feat/tpu-support@dccb7a1 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/.factory/memories.md`


## 2026-05-08T00:48:09Z | feat/tpu-support@dccb7a1 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/.factory/PROGRESS.md`


## 2026-05-08T00:48:00Z | feat/tpu-support@dccb7a1 | done | decide
patch 19 staged: end-of-training canonical save (HF #36004 fix)

Root-cause confirmed via Exa research (HF transformers issues #36004,
#29388, #29659): `PEFT.save_pretrained` does NOT support saving
models that are still resident on a TPU device, even when a CPU
state_dict is passed via `state_dict=peft_state` kwarg. The function
internally re-walks the model's submodules (named_parameters /
state_dict introspection inside get_peft_model_state_dict), which on
an FSDPv2-wrapped XLA model triggers SPMD collectives that can never
complete because only host 0 entered the function. Patches 14/16/17
materializing CPU state_dicts on every host were necessary but not
sufficient -- the deadlock is INSIDE save_pretrained itself.

Canonical fix (HF #36004 closed Dec 2025):
    model.to("cpu")           # SPMD gather; ALL hosts call this
    unwrap_model(model).save_pretrained(...)  # now CPU; no XLA hooks

`model.to("cpu")` moves the entire wrapped model onto host RAM in
one collective. After that, save_pretrained walks an ordinary CPU
module with no XLA tensors involved. Trade-off: this is destructive
to FSDPv2 sharding metadata, so we can only invoke it ONCE at the
final step (training cannot continue afterward).

Implementation:
- new `save_checkpoint_canonical_final(model, save_dir, *, is_main)`
  in `src/training/checkpointing.py` -- moves all 5 sub-modules
  (backbone.model, projection, depth_decoder, text_embed,
  audio_heads) to CPU on every host, rendezvous, then host 0 writes
  files via `save_pretrained(safe_serialization=True)` + `torch.save`.
- wired into `scripts/train_hierarchical.py` end-of-training block,
  gated on `cfg.train.final_canonical_save` (default false).
- DEFAULT DISABLED for iter 12 per user direction "validate step 200
  first; fix saves later". Will be enabled for iter 13.

py_compile + ruff clean on both files.


## 2026-05-08T00:46:38Z | feat/tpu-support@dccb7a1 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/train_hierarchical.py`


## 2026-05-08T00:46:22Z | feat/tpu-support@dccb7a1 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/src/training/checkpointing.py`


## 2026-05-08T00:42:00Z | feat/tpu-support@dccb7a1 | block | exec
spot v4-32 QR preempted by SERVICE during iter 11

`tinyaya-stage2-spot-v4-canary-qr` flipped to
`state=SUSPENDED;stateInitiator=SERVICE` while iter 11 was at T+9
(still in compile). All 4 worker VMs returned NOT_FOUND.

Tier 3 escalation per `tpu-orchestrate` skill. Committed dccb7a1
(13 files, +2300 lines) to preserve work. Started backgrounded
QR poller (`_artifacts/qr_resume_poll.sh`, PID 17201) -- polls
every 60s, will append `>>> RESUMED <<<` once state flips to
ACTIVE.

Resuming iter 12+ on QR resurrection.


## 2026-05-08T00:40:19Z | feat/tpu-support@dccb7a1 | done | edit
created `/home/cataluna84/Workspace/tinyaya-stage2-scale/_artifacts/qr_resume_poll.sh`


## 2026-05-08T00:34:45Z | feat/tpu-support@ee01024 | info | session
SessionEnd (other): 22 item(s) carried forward

Next steps:
- >= 200 successful steps (canary `max_steps=200`); first
- Patches 12 + 13 either landed and verified, or proven
- All commands in `VERIFY.md` (monorepo + simultaneous-translation
- 5000-step run completes (canary -> full config); final loss
- `eval_stage2.py` ASR-BLEU + DNSMOS recorded against
- Verify activation memory is under 4 GB per chip via `diagnose()`.
- If still tight, try moving frozen `MoshiDecoderLayer` to bf16
- Re-run `probe_strategies.py` against the real model with


## 2026-05-07T02:47:51Z | feat/tpu-support@ee01024 | info | session
SessionEnd (other): 22 item(s) carried forward

Next steps:
- >= 200 successful steps (canary `max_steps=200`); first
- Patches 12 + 13 either landed and verified, or proven
- All commands in `VERIFY.md` (monorepo + simultaneous-translation
- 5000-step run completes (canary -> full config); final loss
- `eval_stage2.py` ASR-BLEU + DNSMOS recorded against
- Verify activation memory is under 4 GB per chip via `diagnose()`.
- If still tight, try moving frozen `MoshiDecoderLayer` to bf16
- Re-run `probe_strategies.py` against the real model with


## 2026-05-07T02:47:42Z | feat/tpu-support@ee01024 | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-07T02:39:30Z | feat/tpu-support@ee01024 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/train_hierarchical.py`


## 2026-05-07T02:39:06Z | feat/tpu-support@ee01024 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/configs/stage2_tpu_canary_v4_spot.yaml`


## 2026-05-07T01:37:47Z | feat/tpu-support@ee01024 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/train_hierarchical.py`


## 2026-05-07T01:37:20Z | feat/tpu-support@ee01024 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/src/training/checkpointing.py`


## 2026-05-06T21:00:30Z | feat/tpu-support@ee01024 | info | session
SessionEnd (other): 22 item(s) carried forward

Next steps:
- >= 200 successful steps (canary `max_steps=200`); first
- Patches 12 + 13 either landed and verified, or proven
- All commands in `VERIFY.md` (monorepo + simultaneous-translation
- 5000-step run completes (canary -> full config); final loss
- `eval_stage2.py` ASR-BLEU + DNSMOS recorded against
- Verify activation memory is under 4 GB per chip via `diagnose()`.
- If still tight, try moving frozen `MoshiDecoderLayer` to bf16
- Re-run `probe_strategies.py` against the real model with


## 2026-05-06T20:59:56Z | feat/tpu-support@ee01024 | info | session
SessionEnd (other): 22 item(s) carried forward

Next steps:
- >= 200 successful steps (canary `max_steps=200`); first
- Patches 12 + 13 either landed and verified, or proven
- All commands in `VERIFY.md` (monorepo + simultaneous-translation
- 5000-step run completes (canary -> full config); final loss
- `eval_stage2.py` ASR-BLEU + DNSMOS recorded against
- Verify activation memory is under 4 GB per chip via `diagnose()`.
- If still tight, try moving frozen `MoshiDecoderLayer` to bf16
- Re-run `probe_strategies.py` against the real model with


## 2026-05-06T20:59:56Z | feat/tpu-support@ee01024 | info | session
SessionEnd (other): 22 item(s) carried forward

Next steps:
- >= 200 successful steps (canary `max_steps=200`); first
- Patches 12 + 13 either landed and verified, or proven
- All commands in `VERIFY.md` (monorepo + simultaneous-translation
- 5000-step run completes (canary -> full config); final loss
- `eval_stage2.py` ASR-BLEU + DNSMOS recorded against
- Verify activation memory is under 4 GB per chip via `diagnose()`.
- If still tight, try moving frozen `MoshiDecoderLayer` to bf16
- Re-run `probe_strategies.py` against the real model with


## 2026-05-06T20:00:00Z | feat/tpu-support@ee01024 | done | decide
TPU canary v4-32 spot reached step 100 with decreasing loss (iter 7).
First end-to-end Stage 2 success; all SPMD + observability + recompile
fixes validated.

Run: `https://wandb.ai/cataluna84/tinyaya-stage2-tpu/runs/8pse8tzk`
Loss: step 10 = 9.0273 -> step 100 = 7.5983 (decreasing).
Steady-state: 3.41 sec/step from step 30 onwards.
All 4 hosts attached to one wandb umbrella (shared-mode).

Patches that landed (4-11):
- p4: `optimizer_step` strategy-aware (FSDPv2 path: `optimizer.step()`
  + `mark_step()`, replicated path: `xm.optimizer_step()`).
- p5: `xm.mark_step()` before grad clip on TPU.
- p6: skip `clip_grad_norm_` on TPU (FSDPv2 sharded grads + clip norm
  forces a graph break per micro step).
- p7: replace `.item()` with `.detach()` in TPU inner loop; XLA-tensor
  accumulators (`micro_loss_sum_xla`); single materialize at
  log_every. Eliminates the cpu_fallback storm that misdiagnosed iter
  1/2 as "deadlock" (was actually 8 sequential 12-16 min compiles).
- p8: cross-host `is_main_process` =
  `xr.host_index()==0 AND xm.is_master_ordinal()`. Prevents 4 separate
  wandb runs.
- p9: wandb shared-mode rendezvous via GCS
  (`gs://tinyaya-stage2-tpu/wandb-rendezvous/v4-32-spot-canary.id`).
  Worker 0 publishes run_id, workers 1-3 attach via `gsutil cat` retry
  loop (60 x 5s) using `mode=shared, x_primary, x_label=rank_N`.
  Requires wandb >= 0.19.9 (TPU image ships 0.19.11).
- p10: `grad_accum: 2 -> 8` -> hit HBM OOM at iter 4 (34.16G / 31.75G
  by 2.41G).
- p10b: `grad_accum: 8 -> 4` -> hit HBM OOM at iter 5 (over by 41 MB,
  tantalizingly close; static memory dominated, not activations).
- p10c: `grad_accum: 4 -> 2` (revert to iter 3 wiring with patch 7
  fix) -> iter 6 reached step 2 but hit per-batch recompile.
- p11: `collator pad_to=cfg.data.max_frames` (300) on TPU eliminates
  per-batch shape variation. Canonical fix per pytorch/xla
  recompilation guide. Iter 7 reached step 100, sec/step settled to
  3.41 after the warm-up window.
- p12 + p13 (drafted but not yet validated): skip
  `generate_audio_sample` and `run_validation` on TPU during canary;
  they re-trigger XLA recompiles by feeding non-canonical shapes
  through the model.

Iteration timeline (wall-clock minutes-from-deploy):
| Iter | Patches | Outcome | Notes |
|------|---------|---------|-------|
| 1 | (initial) | Misdiagnosed "deadlock" at T+71 | actually compile of `.item()` cpu_fallback storm |
| 2 | FSDPv2 (4,5,6) | Same symptom | confirmed `.item()` was forcing 12-16 min compile each |
| 3 | + 7 (.item() removal) | Compile completed | 4 separate wandb runs (1 per host) |
| 4 | + 8/9/10 (cross-host + shared wandb + grad_accum=8) | OOM at T+76 | 34.16G / 31.75G by 2.41G; fused HLO too large |
| 5 | grad_accum=4 | OOM by 41 MB | static memory dominated; activations not the bottleneck |
| 6 | grad_accum=2 | Step 2 reached | per-batch shape recompiles burned cycles |
| 7 | + 11 (fixed-shape padding to max_frames=300) | **STEP 100, loss decreasing** | sec/step 3.41 steady-state |
| 8 | + 12/13 (skip audio val + run_validation on TPU) | drafted | reduces per-step recompile risk |

Stack diagnostics validated (py-spy 0.4.2 + /proc/PID/stack):
- Real Python PID is the python3 process (not the `uv run` parent;
  `uv run` sleeps).
- Native stack `xla::PjRtCApiClient::CompileAndLoad ->
  InitializeArgsAndCompile -> libtpu.so` = healthy compile, not stall.
- Native stack containing `cpu_fallback / _local_scalar_dense /
  at::native::item` = anti-pattern; redirect to patch 7.

Cross-host SPMD lessons:
- `xr.host_index()` returns 0..N-1 across hosts; `xm.is_master_ordinal
  ()` is local-to-host. Only `host_index==0 AND
  is_master_ordinal()` is the global rank-0.
- wandb shared-mode requires >=0.19.9 (`mode=shared`, `x_primary=True`
  on rank-0, `x_label=rank_N` on others). GCS rendezvous is a
  dependency-free way to share the run_id.

Self-healing orchestrator (Phase 1 commit ee01024) exit metrics:
- Iterations consumed: 8 (5 hot-redeploys without QR re-create).
- Wall-clock total: ~6 hours.
- QRs created: 1 (preserved across iter 1-8).
- Tier-3 escalations: 0.
- User check-ins: 5 (T+15/30/45/60/T+71-deadlock-misdiag, T+63-iter4).


## 2026-05-06T19:39:49Z | feat/tpu-support@ee01024 | info | session
SessionEnd (other): 19 item(s) carried forward

Next steps:
- `scan_layers` enabled around backbone + depth-decoder transformer
- Explicit gradient checkpointing enabled; per-chip HBM usage
- `canary` config restored to `max_frames=300`,
- `fsdpv2_lora` strategy runs **at least 50 successful training
- All commands in `VERIFY.md` pass.
- First successful checkpoint written to GCS and W&B run logged.
- 5000-step run completes; final loss + ASR-BLEU recorded in
- Re-run probe with the real model on `tiny_canary` config; confirm


## 2026-05-06T19:33:04Z | feat/tpu-support@ee01024 | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-06T18:49:29Z | feat/tpu-support@ee01024 | info | session
SessionEnd (other): 19 item(s) carried forward

Next steps:
- `scan_layers` enabled around backbone + depth-decoder transformer
- Explicit gradient checkpointing enabled; per-chip HBM usage
- `canary` config restored to `max_frames=300`,
- `fsdpv2_lora` strategy runs **at least 50 successful training
- All commands in `VERIFY.md` pass.
- First successful checkpoint written to GCS and W&B run logged.
- 5000-step run completes; final loss + ASR-BLEU recorded in
- Re-run probe with the real model on `tiny_canary` config; confirm


## 2026-05-06T16:22:26Z | feat/tpu-support@ee01024 | info | session
SessionEnd (other): 19 item(s) carried forward

Next steps:
- `scan_layers` enabled around backbone + depth-decoder transformer
- Explicit gradient checkpointing enabled; per-chip HBM usage
- `canary` config restored to `max_frames=300`,
- `fsdpv2_lora` strategy runs **at least 50 successful training
- All commands in `VERIFY.md` pass.
- First successful checkpoint written to GCS and W&B run logged.
- 5000-step run completes; final loss + ASR-BLEU recorded in
- Re-run probe with the real model on `tiny_canary` config; confirm


## 2026-05-06T16:04:15Z | feat/tpu-support@ee01024 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/train_hierarchical.py`


## 2026-05-06T16:03:33Z | feat/tpu-support@ee01024 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/train_hierarchical.py`


## 2026-05-06T15:59:59Z | feat/tpu-support@ee01024 | info | session
SessionEnd (other): 19 item(s) carried forward

Next steps:
- `scan_layers` enabled around backbone + depth-decoder transformer
- Explicit gradient checkpointing enabled; per-chip HBM usage
- `canary` config restored to `max_frames=300`,
- `fsdpv2_lora` strategy runs **at least 50 successful training
- All commands in `VERIFY.md` pass.
- First successful checkpoint written to GCS and W&B run logged.
- 5000-step run completes; final loss + ASR-BLEU recorded in
- Re-run probe with the real model on `tiny_canary` config; confirm


## 2026-05-06T15:59:13Z | feat/tpu-support@ee01024 | info | session
SessionEnd (other): 19 item(s) carried forward

Next steps:
- `scan_layers` enabled around backbone + depth-decoder transformer
- Explicit gradient checkpointing enabled; per-chip HBM usage
- `canary` config restored to `max_frames=300`,
- `fsdpv2_lora` strategy runs **at least 50 successful training
- All commands in `VERIFY.md` pass.
- First successful checkpoint written to GCS and W&B run logged.
- 5000-step run completes; final loss + ASR-BLEU recorded in
- Re-run probe with the real model on `tiny_canary` config; confirm


## 2026-05-06T15:59:13Z | feat/tpu-support@ee01024 | info | session
SessionEnd (other): 19 item(s) carried forward

Next steps:
- `scan_layers` enabled around backbone + depth-decoder transformer
- Explicit gradient checkpointing enabled; per-chip HBM usage
- `canary` config restored to `max_frames=300`,
- `fsdpv2_lora` strategy runs **at least 50 successful training
- All commands in `VERIFY.md` pass.
- First successful checkpoint written to GCS and W&B run logged.
- 5000-step run completes; final loss + ASR-BLEU recorded in
- Re-run probe with the real model on `tiny_canary` config; confirm


## 2026-05-06T15:41:40Z | feat/tpu-support@ee01024 | info | session
SessionEnd (other): 19 item(s) carried forward

Next steps:
- `scan_layers` enabled around backbone + depth-decoder transformer
- Explicit gradient checkpointing enabled; per-chip HBM usage
- `canary` config restored to `max_frames=300`,
- `fsdpv2_lora` strategy runs **at least 50 successful training
- All commands in `VERIFY.md` pass.
- First successful checkpoint written to GCS and W&B run logged.
- 5000-step run completes; final loss + ASR-BLEU recorded in
- Re-run probe with the real model on `tiny_canary` config; confirm


## 2026-05-06T12:38:48Z | feat/tpu-support@ee01024 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/train_hierarchical.py`


## 2026-05-06T12:17:35Z | feat/tpu-support@ee01024 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/train_hierarchical.py`


## 2026-05-06T12:17:16Z | feat/tpu-support@ee01024 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/src/data/collator.py`


## 2026-05-06T12:17:10Z | feat/tpu-support@ee01024 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/src/data/collator.py`


## 2026-05-06T12:17:00Z | feat/tpu-support@ee01024 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/src/data/collator.py`


## 2026-05-06T10:47:56Z | feat/tpu-support@ee01024 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/configs/stage2_tpu_canary_v4_spot.yaml`


## 2026-05-06T10:07:00Z | feat/tpu-support@ee01024 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/configs/stage2_tpu_canary_v4_spot.yaml`


## 2026-05-06T09:13:07Z | feat/tpu-support@ee01024 | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-06T08:48:14Z | feat/tpu-support@ee01024 | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-06T08:41:19Z | feat/tpu-support@ee01024 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/configs/stage2_tpu_canary_v4_spot.yaml`


## 2026-05-06T08:41:03Z | feat/tpu-support@ee01024 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/train_hierarchical.py`


## 2026-05-06T08:40:30Z | feat/tpu-support@ee01024 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/src/backend/tpu_backend.py`


## 2026-05-06T08:30:03Z | feat/tpu-support@ee01024 | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-06T08:24:02Z | feat/tpu-support@ee01024 | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-06T08:18:40Z | feat/tpu-support@ee01024 | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-06T08:15:12Z | feat/tpu-support@ee01024 | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-06T08:14:01Z | feat/tpu-support@ee01024 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/train_hierarchical.py`


## 2026-05-06T08:13:53Z | feat/tpu-support@ee01024 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/train_hierarchical.py`


## 2026-05-06T08:13:39Z | feat/tpu-support@ee01024 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/train_hierarchical.py`


## 2026-05-06T08:13:30Z | feat/tpu-support@ee01024 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/train_hierarchical.py`


## 2026-05-06T08:13:22Z | feat/tpu-support@ee01024 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/train_hierarchical.py`


## 2026-05-06T07:50:40Z | feat/tpu-support@ee01024 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/train_hierarchical.py`


## 2026-05-06T07:41:01Z | feat/tpu-support@ee01024 | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-06T07:25:58Z | feat/tpu-support@ee01024 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/train_hierarchical.py`


## 2026-05-06T07:24:46Z | feat/tpu-support@ee01024 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/src/backend/tpu_backend.py`


## 2026-05-06T05:12:20Z | feat/tpu-support@ee01024 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/_artifacts/orch_poll.py`


## 2026-05-06T05:03:19Z | feat/tpu-support@59a8a75 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/tpu/_remote_redeploy.sh`


## 2026-05-06T05:03:03Z | feat/tpu-support@59a8a75 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/tpu/_remote_redeploy.sh`


## 2026-05-06T05:02:55Z | feat/tpu-support@59a8a75 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/tpu/startup_script.sh`


## 2026-05-06T05:02:38Z | feat/tpu-support@59a8a75 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/tpu/startup_script.sh`


## 2026-05-06T05:02:23Z | feat/tpu-support@59a8a75 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/src/model/scan_utils.py`


## 2026-05-06T05:01:18Z | feat/tpu-support@59a8a75 | done | edit
created `/home/cataluna84/Workspace/tinyaya-stage2-scale/_artifacts/orch_state.json`


## 2026-05-06T05:01:17Z | feat/tpu-support@59a8a75 | done | edit
created `/home/cataluna84/Workspace/tinyaya-stage2-scale/_artifacts/scheduled_checkin.py`


## 2026-05-06T05:01:15Z | feat/tpu-support@59a8a75 | done | edit
created `/home/cataluna84/Workspace/tinyaya-stage2-scale/_artifacts/orch_poll.py`


## 2026-05-06T05:00:00Z | feat/tpu-support@59a8a75 | done | edit
created `/home/cataluna84/Workspace/tinyaya-stage2-scale/.factory/droids/tpu-diagnoser.md`


## 2026-05-06T04:59:59Z | feat/tpu-support@59a8a75 | done | edit
created `/home/cataluna84/Workspace/tinyaya-stage2-scale/.factory/droids/tpu-watchdog.md`


## 2026-05-06T04:56:59Z | feat/tpu-support@59a8a75 | done | edit
created `/home/cataluna84/Workspace/tinyaya-stage2-scale/.factory/skills/tpu-redeploy/SKILL.md`


## 2026-05-06T04:56:58Z | feat/tpu-support@59a8a75 | done | edit
created `/home/cataluna84/Workspace/tinyaya-stage2-scale/.factory/skills/tpu-orchestrate/SKILL.md`


## 2026-05-06T04:55:21Z | feat/tpu-support@59a8a75 | done | edit
created `/home/cataluna84/Workspace/tinyaya-stage2-scale/.factory/orchestration/SPEC.md`


## 2026-05-06T04:55:20Z | feat/tpu-support@59a8a75 | done | edit
created `/home/cataluna84/Workspace/tinyaya-stage2-scale/.factory/orchestration/README.md`


## 2026-05-06T04:55:19Z | feat/tpu-support@59a8a75 | done | edit
created `/home/cataluna84/Workspace/tinyaya-stage2-scale/.factory/orchestration/playbook/checkin-protocol.md`


## 2026-05-06T04:55:18Z | feat/tpu-support@59a8a75 | done | edit
created `/home/cataluna84/Workspace/tinyaya-stage2-scale/.factory/orchestration/playbook/tier-definitions.md`


## 2026-05-06T04:55:17Z | feat/tpu-support@59a8a75 | done | edit
created `/home/cataluna84/Workspace/tinyaya-stage2-scale/.factory/orchestration/playbook/diagnosis-table.md`


## 2026-05-06T04:52:57Z | feat/tpu-support@59a8a75 | done | edit
created `/home/cataluna84/Workspace/tinyaya-stage2-scale/.factory/orchestration/diagrams/render.sh`


## 2026-05-06T04:52:56Z | feat/tpu-support@59a8a75 | done | edit
created `/home/cataluna84/Workspace/tinyaya-stage2-scale/.factory/orchestration/diagrams/05-tier3-escalation.mmd`


## 2026-05-06T04:52:54Z | feat/tpu-support@59a8a75 | done | edit
created `/home/cataluna84/Workspace/tinyaya-stage2-scale/.factory/orchestration/diagrams/04-checkin-cadence.mmd`


## 2026-05-06T04:52:53Z | feat/tpu-support@59a8a75 | done | edit
created `/home/cataluna84/Workspace/tinyaya-stage2-scale/.factory/orchestration/diagrams/02-state-machine.mmd`


## 2026-05-06T04:52:52Z | feat/tpu-support@59a8a75 | done | edit
created `/home/cataluna84/Workspace/tinyaya-stage2-scale/.factory/orchestration/diagrams/01-architecture.mmd`


## 2026-05-06T04:52:51Z | feat/tpu-support@59a8a75 | done | edit
created `/home/cataluna84/Workspace/tinyaya-stage2-scale/.factory/orchestration/diagrams/03-sequence.mmd`


## 2026-05-06T04:49:30Z | feat/tpu-support@59a8a75 | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-06T04:41:38Z | feat/tpu-support@59a8a75 | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-06T04:35:04Z | feat/tpu-support@59a8a75 | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-06T04:19:49Z | feat/tpu-support@59a8a75 | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-06T04:15:38Z | feat/tpu-support@1eaa339 | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-06T04:15:19Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/configs/stage2_tpu_v4_spot.yaml`


## 2026-05-06T04:15:12Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/configs/stage2_tpu.yaml`


## 2026-05-06T04:15:01Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/configs/stage2_tpu_canary_v6e_spot.yaml`


## 2026-05-06T04:14:54Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/configs/stage2_tpu_canary_v5e_spot.yaml`


## 2026-05-06T04:14:44Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/configs/stage2_tpu_canary_v4_spot.yaml`


## 2026-05-06T04:14:35Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/configs/stage2_tpu_canary.yaml`


## 2026-05-06T04:09:13Z | feat/tpu-support@1eaa339 | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-06T04:05:46Z | feat/tpu-support@1eaa339 | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-06T03:58:15Z | feat/tpu-support@1eaa339 | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-06T03:50:27Z | feat/tpu-support@1eaa339 | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-06T03:38:17Z | feat/tpu-support@1eaa339 | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-06T00:55:58Z | feat/tpu-support@1eaa339 | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-06T00:24:11Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/src/model/scan_utils.py`


## 2026-05-06T00:05:01Z | feat/tpu-support@1eaa339 | done | edit
created `/tmp/train_poller_postfix.sh`


## 2026-05-06T00:03:27Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/tpu/_remote_redeploy.sh`


## 2026-05-06T00:03:17Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/tpu/_remote_redeploy.sh`


## 2026-05-06T00:03:08Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/tpu/startup_script.sh`


## 2026-05-06T00:02:53Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/tpu/startup_script.sh`


## 2026-05-06T00:02:35Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/src/model/scan_utils.py`


## 2026-05-06T00:02:12Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/src/model/scan_utils.py`


## 2026-05-05T23:56:50Z | feat/tpu-support@1eaa339 | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-05T23:46:42Z | feat/tpu-support@1eaa339 | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-05T23:38:13Z | feat/tpu-support@1eaa339 | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-05T23:37:28Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/docs/tpu-launch-plan.md`


## 2026-05-05T23:36:57Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/docs/tpu-launch-plan.md`


## 2026-05-05T23:27:24Z | feat/tpu-support@1eaa339 | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-05T23:20:40Z | feat/tpu-support@1eaa339 | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-05T23:18:16Z | feat/tpu-support@1eaa339 | done | edit
created `/tmp/train_poller3.sh`


## 2026-05-05T23:16:08Z | feat/tpu-support@1eaa339 | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-05T17:27:36Z | feat/tpu-support@1eaa339 | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-05T16:41:55Z | feat/tpu-support@1eaa339 | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-05T16:11:07Z | feat/tpu-support@1eaa339 | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-05T15:50:49Z | feat/tpu-support@1eaa339 | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-05T15:00:42Z | feat/tpu-support@1eaa339 | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-05T14:57:55Z | feat/tpu-support@1eaa339 | done | edit
created `/tmp/train_poller2.sh`


## 2026-05-05T14:57:13Z | feat/tpu-support@1eaa339 | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-05T14:51:10Z | feat/tpu-support@1eaa339 | done | edit
created `/tmp/train_poller.sh`


## 2026-05-05T14:17:23Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/tpu/startup_script.sh`


## 2026-05-05T14:15:31Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/tpu/startup_script.sh`


## 2026-05-05T14:04:21Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/src/training/checkpointing.py`


## 2026-05-05T13:47:30Z | feat/tpu-support@1eaa339 | done | edit
created `/tmp/qr_poller_v4.sh`


## 2026-05-05T13:17:05Z | feat/tpu-support@1eaa339 | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-05T13:16:47Z | feat/tpu-support@1eaa339 | done | edit
created `/home/cataluna84/Workspace/tinyaya-stage2-scale/_artifacts/gcp-quota-increase-request.md`


## 2026-05-05T13:08:13Z | feat/tpu-support@1eaa339 | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-05T13:01:36Z | feat/tpu-support@1eaa339 | done | edit
created `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/configs/stage2_tpu_canary_v6e_spot.yaml`


## 2026-05-05T12:53:16Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/tpu/launch_qr.sh`


## 2026-05-05T12:53:10Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/tpu/launch_qr.sh`


## 2026-05-05T12:53:03Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/tpu/launch_qr.sh`


## 2026-05-05T12:44:02Z | feat/tpu-support@1eaa339 | info | session
SessionEnd (other): 19 item(s) carried forward

Next steps:
- `scan_layers` enabled around backbone + depth-decoder transformer
- Explicit gradient checkpointing enabled; per-chip HBM usage
- `canary` config restored to `max_frames=300`,
- `fsdpv2_lora` strategy runs **at least 50 successful training
- All commands in `VERIFY.md` pass.
- First successful checkpoint written to GCS and W&B run logged.
- 5000-step run completes; final loss + ASR-BLEU recorded in
- Re-run probe with the real model on `tiny_canary` config; confirm


## 2026-05-05T12:11:52Z | feat/tpu-support@1eaa339 | info | session
SessionEnd (other): 19 item(s) carried forward

Next steps:
- `scan_layers` enabled around backbone + depth-decoder transformer
- Explicit gradient checkpointing enabled; per-chip HBM usage
- `canary` config restored to `max_frames=300`,
- `fsdpv2_lora` strategy runs **at least 50 successful training
- All commands in `VERIFY.md` pass.
- First successful checkpoint written to GCS and W&B run logged.
- 5000-step run completes; final loss + ASR-BLEU recorded in
- Re-run probe with the real model on `tiny_canary` config; confirm


## 2026-05-05T11:56:22Z | feat/tpu-support@1eaa339 | info | session
SessionEnd (other): 19 item(s) carried forward

Next steps:
- `scan_layers` enabled around backbone + depth-decoder transformer
- Explicit gradient checkpointing enabled; per-chip HBM usage
- `canary` config restored to `max_frames=300`,
- `fsdpv2_lora` strategy runs **at least 50 successful training
- All commands in `VERIFY.md` pass.
- First successful checkpoint written to GCS and W&B run logged.
- 5000-step run completes; final loss + ASR-BLEU recorded in
- Re-run probe with the real model on `tiny_canary` config; confirm


## 2026-05-05T10:02:52Z | feat/tpu-support@1eaa339 | info | session
SessionEnd (other): 19 item(s) carried forward

Next steps:
- `scan_layers` enabled around backbone + depth-decoder transformer
- Explicit gradient checkpointing enabled; per-chip HBM usage
- `canary` config restored to `max_frames=300`,
- `fsdpv2_lora` strategy runs **at least 50 successful training
- All commands in `VERIFY.md` pass.
- First successful checkpoint written to GCS and W&B run logged.
- 5000-step run completes; final loss + ASR-BLEU recorded in
- Re-run probe with the real model on `tiny_canary` config; confirm


## 2026-05-05T09:46:41Z | feat/tpu-support@1eaa339 | done | edit
created `/tmp/qr_poller2.sh`


## 2026-05-05T09:45:43Z | feat/tpu-support@1eaa339 | done | edit
created `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/configs/stage2_tpu_canary_v5e_spot.yaml`


## 2026-05-05T09:12:51Z | feat/tpu-support@1eaa339 | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-05T09:11:43Z | feat/tpu-support@1eaa339 | done | edit
created `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/configs/stage2_tpu_v4_spot.yaml`


## 2026-05-05T09:11:26Z | feat/tpu-support@1eaa339 | done | edit
created `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/configs/stage2_tpu_canary_v4_spot.yaml`


## 2026-05-05T09:10:50Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/tpu/launch_qr.sh`


## 2026-05-05T09:10:40Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/tpu/startup_script.sh`


## 2026-05-05T09:10:25Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/tpu/launch_qr.sh`


## 2026-05-05T09:10:03Z | feat/tpu-support@1eaa339 | done | edit
created `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/tpu/launch_spot.sh`


## 2026-05-05T09:09:38Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/.factory/PLAN.md`


## 2026-05-05T09:09:15Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/.factory/memories.md`


## 2026-05-05T09:08:52Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/.factory/PROGRESS.md`


## 2026-05-05T09:10:00Z | feat/tpu-support@1eaa339 | done | decide
TRC quota table refreshed from the original welcome email; falling
back to spot v4-32 in `us-central2-b` because on-demand v4 in that
same zone is currently busy.

- Authoritative quota now lives in
  `simultaneous-translation/docs/tpu-trc-allocation.md` (verbatim
  from `trc-support@google.com` email to `mayankbhaskar007@gmail.com`,
  project `ml-pipelines-315702`, 90-day window).
- Old 5-row table in `docs/tpu-launch-plan.md` §2 marked SUPERSEDED.
- Default spot launch profile: `TRC_PROFILE=v4-32-uc2b`. Same zone /
  IAM / VPC / runtime as the on-demand path; only `--spot` differs.
- Phase 3 / 4 / 5 will run against the spot v4-32 via
  `scripts/tpu/launch_spot.sh` + the new
  `configs/stage2_tpu_canary_v4_spot.yaml` and
  `configs/stage2_tpu_v4_spot.yaml`.


## 2026-05-05T09:08:35Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/docs/tpu-launch-plan.md`


## 2026-05-05T09:08:17Z | feat/tpu-support@1eaa339 | done | edit
created `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/docs/tpu-trc-allocation.md`


## 2026-05-05T09:01:40Z | feat/tpu-support@1eaa339 | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-05T08:54:43Z | feat/tpu-support@1eaa339 | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-05T08:53:57Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/.factory/PROGRESS.md`


## 2026-05-05T08:55:00Z | feat/tpu-support@1eaa339 | done | verify
TPU canary + production code path landed; doc convention applied
across `src/` + `scripts/`.

- Phase 1+2 (PLAN.md): `scan_layers` + `xla_grad_checkpoint` wrappers
  shipped in `src/model/scan_utils.py`; both flags exposed on
  `composite.TinyAyaMoshiComposite` and threaded through
  `scripts/train_hierarchical.py` DEFAULTS.
- Phase 4 (canary fidelity): `configs/stage2_tpu_canary.yaml`
  restored to `max_frames=300`, `depth_chunk_size=16`; both
  `train.use_scan_layers` and `train.xla_grad_checkpoint` are `true`.
- Phase 5 prep: `configs/stage2_tpu.yaml` matches the canary on the
  two new flags; `launch_qr.sh` already plumbs `TPU_STRATEGY` via the
  queued-resource metadata.
- Documentation pass: every `*.py` under `src/` + `scripts/` now uses
  the `WHY THIS EXISTS` + NumPy-docstring convention codified in
  `simultaneous-translation/AGENTS.md` ("TPU code documentation style
  (mandatory)") and the `.factory/skills/tpu-doc-style/SKILL.md` skill.
- Lint + verify: `ruff format --check`, `ruff check` clean across
  `src/` + `scripts/`. `py_compile` clean on every `.py`,
  `yaml.safe_load` clean on every config, `bash -n` clean on every
  shell script. Pre-existing `phase-3-data-generation-pipeline/cli.py`
  `src.config` import error remains out of scope.
- Pending (need live TPU): probe-strategy decision, 5-step + 50-step
  canary, and the 5000-step Phase 5 launch -- runbook delivered to
  the user.




## 2026-05-05T08:53:16Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/.factory/PLAN.md`


## 2026-05-05T08:51:28Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/upload_encoded_dataset.py`


## 2026-05-05T08:51:19Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/prepare_translation_data_fixed.py`


## 2026-05-05T08:51:10Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/prepare_translation_data.py`


## 2026-05-05T08:51:02Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/prepare_data.py`


## 2026-05-05T08:50:52Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/make_splits.py`


## 2026-05-05T08:50:43Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/infer_only.py`


## 2026-05-05T08:50:33Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/generate_demos.py`


## 2026-05-05T08:50:25Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/eval_with_english.py`


## 2026-05-05T08:50:16Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/eval_translation.py`


## 2026-05-05T08:50:07Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/eval_full_codebooks.py`


## 2026-05-05T08:49:57Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/translate_wav.py`


## 2026-05-05T08:49:45Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/train_stage2.py`


## 2026-05-05T08:49:36Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/train_stage1.py`


## 2026-05-05T08:49:27Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/train_and_infer.py`


## 2026-05-05T08:49:18Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/eval_stage2.py`


## 2026-05-05T08:48:38Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/src/data/mimi_encoder.py`


## 2026-05-05T08:48:26Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/src/data/interleaver.py`


## 2026-05-05T08:46:35Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/src/data/collator.py`


## 2026-05-05T08:46:27Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/src/data/dataset.py`


## 2026-05-05T08:45:59Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/src/training/scheduler.py`


## 2026-05-05T08:45:49Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/src/training/translation_loss.py`


## 2026-05-05T08:45:35Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/src/training/checkpointing.py`


## 2026-05-05T08:44:50Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/tpu/probe_strategies.py`


## 2026-05-05T08:44:16Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/src/model/surgery.py`


## 2026-05-05T08:44:05Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/src/model/surgery.py`


## 2026-05-05T08:43:56Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/src/model/surgery.py`


## 2026-05-05T08:43:46Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/src/model/lora_setup.py`


## 2026-05-05T08:43:34Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/src/model/depth_decoder.py`


## 2026-05-05T08:43:22Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/src/model/depth_decoder.py`


## 2026-05-05T08:43:11Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/src/model/depth_decoder.py`


## 2026-05-05T08:42:58Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/src/model/backbone.py`


## 2026-05-05T08:42:30Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/src/model/backbone.py`


## 2026-05-05T08:42:15Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/src/model/backbone.py`


## 2026-05-05T08:41:41Z | feat/tpu-support@1eaa339 | done | edit
created `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/src/backend/tpu_backend.py`


## 2026-05-05T08:39:47Z | feat/tpu-support@1eaa339 | done | edit
created `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/src/backend/gpu_backend.py`


## 2026-05-05T08:39:23Z | feat/tpu-support@1eaa339 | done | edit
created `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/src/backend/base.py`


## 2026-05-05T08:38:06Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/src/training/scheduler.py`


## 2026-05-05T08:37:59Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/src/data/dataset.py`


## 2026-05-05T08:37:51Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/src/backend/tpu_backend.py`


## 2026-05-05T08:37:38Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/src/backend/__init__.py`


## 2026-05-05T08:37:19Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/train_and_infer.py`


## 2026-05-05T08:37:13Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/train_and_infer.py`


## 2026-05-05T08:37:01Z | feat/tpu-support@1eaa339 | done | edit
created `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/test_tpu_training_step.py`


## 2026-05-05T08:36:33Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/eval_with_english.py`


## 2026-05-05T08:36:27Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/eval_stage2.py`


## 2026-05-05T08:35:10Z | feat/tpu-support@1eaa339 | done | edit
created `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/configs/stage2_tpu.yaml`


## 2026-05-05T08:34:53Z | feat/tpu-support@1eaa339 | done | edit
created `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/configs/stage2_tpu_canary.yaml`


## 2026-05-05T08:34:02Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/train_hierarchical.py`


## 2026-05-05T08:33:48Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/train_hierarchical.py`


## 2026-05-05T08:33:38Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/scripts/train_hierarchical.py`


## 2026-05-05T08:33:08Z | feat/tpu-support@1eaa339 | done | edit
created `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/src/model/composite.py`


## 2026-05-05T08:31:43Z | feat/tpu-support@1eaa339 | done | edit
created `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/src/model/scan_utils.py`


## 2026-05-05T08:29:36Z | feat/tpu-support@1eaa339 | done | edit
created `/home/cataluna84/Workspace/tinyaya-stage2-scale/.factory/skills/tpu-doc-style/SKILL.md`


## 2026-05-05T08:28:46Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/.factory/memories.md`


## 2026-05-05T08:28:20Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/AGENTS.md`


## 2026-05-05T08:20:33Z | feat/tpu-support@1eaa339 | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-05T08:10:11Z | feat/tpu-support@1eaa339 | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-05T08:07:39Z | feat/tpu-support@1eaa339 | info | session
SessionEnd (other): 24 item(s) carried forward

Next steps:
- `scan_layers` enabled around backbone + depth-decoder transformer
- Explicit gradient checkpointing enabled; per-chip HBM usage
- `canary` config restored to `max_frames=300`,
- `fsdpv2_lora` strategy runs **at least 50 successful training
- All commands in `VERIFY.md` pass.
- First successful checkpoint written to GCS and W&B run logged.
- 5000-step run completes; final loss + ASR-BLEU recorded in
- Add `scan_layers` wrapper around `CohereDecoderLayer` (backbone,


## 2026-05-05T08:05:45Z | feat/tpu-support@1eaa339 | fail | verify
verify: 11 passed, 1 failed out of 12 on Stop

FAIL [1] # CLI entry point loads and prints help
    ModuleNotFoundError: No module named 'src.config'


## 2026-05-05T08:04:16Z | feat/tpu-support@1eaa339 | done | edit
edited `/home/cataluna84/Workspace/tinyaya-stage2-scale/.gitignore`


## 2026-05-05T08:04:01Z | feat/tpu-support@1eaa339 | done | edit
created `/home/cataluna84/Workspace/tinyaya-stage2-scale/phase-3-data-generation-pipeline/AGENTS.md`


## 2026-05-05T08:03:27Z | feat/tpu-support@1eaa339 | done | edit
created `/home/cataluna84/Workspace/tinyaya-stage2-scale/simultaneous-translation/AGENTS.md`


## 2026-05-05T08:02:44Z | feat/tpu-support@1eaa339 | done | edit
created `/home/cataluna84/Workspace/tinyaya-stage2-scale/AGENTS.md`


## 2026-05-05T08:01:57Z | feat/tpu-support@1eaa339 | done | edit
created `/home/cataluna84/Workspace/tinyaya-stage2-scale/.factory/settings.json`


## 2026-05-05T00:00:00Z | feat/tpu-support@1eaa339 | info | session
Memory system installed. Initial seeding from prior TPU work.

Below entries reconstruct the session that pushed commit `1eaa339`
("feat(tpu): multi-strategy SPMD backend + hot redeploy + bf16 cast").

---

## 2026-05-03T16:00:00Z | feat/tpu-support@1eaa339 | done | session
Pushed commit `1eaa339` to `feat/tpu-support`. Branch in sync with
`origin/feat/tpu-support`. 13 files changed, 753+/73-.

## 2026-05-03T15:55:00Z | feat/tpu-support@1eaa339 | done | edit
Deleted local tarball `_artifacts/tinyaya-with-git.tar.gz`. GCS object
`gs://tinyaya-stage2-tpu/code/tinyaya-with-git.tar.gz` left for user to
delete.

## 2026-05-03T15:50:00Z | feat/tpu-support@a00c11b | done | verify
Tarball compare-and-contrast: all 13 changed files byte-identical
between local working tree and extracted tarball. Sizes match
(415920 b local / 483071 b GCS post-rebuild).

## 2026-05-03T14:30:00Z | feat/tpu-support@a00c11b | fail | exec
fsdpv2_lora compile on real composite (5.17B params) hit 15+ minutes
without progress. Process at 35GB CPU RSS, 440% CPU, futex_wait stack.
Likely cause: 36 CohereDecoderLayer + 6 MoshiDecoderLayer unrolled
into single HLO graph. Mitigation: scan_layers (pending).

## 2026-05-03T14:00:00Z | feat/tpu-support@a00c11b | fail | exec
replicated strategy OOM on real composite: HBM used 25.90GB / limit
15.75GB on v5litepod-16. Mitigation: switch to fsdpv2_lora.

## 2026-05-03T13:30:00Z | feat/tpu-support@a00c11b | done | decide
Cast model to `torch.bfloat16` inside `wrap_model` instead of relying
on `XLA_USE_BF16` (deprecated in torch_xla 2.6+). See
`memories.md` for rationale.

## 2026-05-03T13:00:00Z | feat/tpu-support@a00c11b | done | exec
Probe results on live v5litepod-16 (tiny stand-in model):

| strategy     | compile (s) | step (s) |
|--------------|-------------|----------|
| replicated   | 0.91        | 0.004    |
| fsdpv2_lora  | 0.95        | 0.027    |
| fsdpv2       | 0.97        | 0.052    |

All three strategies validated; partitioner crash from pytorch/xla
#8607 confirmed fixed by `XLA_DISABLE_FUNCTIONALIZATION=0`.

## 2026-05-03T12:00:00Z | feat/tpu-support@a00c11b | done | edit
Added `src/backend/tpu_backend.py` multi-strategy backend with
`TPU_STRATEGY` env var (replicated / fsdpv2 / fsdpv2_lora / auto).
Added `diagnose()`, `mark_sharding()` to base.py.

## 2026-05-03T11:00:00Z | feat/tpu-support@a00c11b | done | edit
Added `scripts/tpu/probe_strategies.py`,
`scripts/tpu/hot_redeploy.sh`, `scripts/tpu/_remote_redeploy.sh`.
Sub-3-minute redeploy without QR re-create.

---

## Next steps (rolled forward by SessionEnd)

- Implement `scan_layers` around `CohereDecoderLayer` (36 backbone
  layers) and `MoshiDecoderLayer` (6 depth-decoder layers) to compile
  one layer's HLO and reuse it. Should cut compile from 25+ min to
  a few min.
- Add explicit gradient checkpointing for forward activation memory.
- Re-test `fsdpv2_lora` compile time with `scan_layers` enabled.
- Restore canary `max_frames` from 64 back to 300 once compile is fast.
- Run full 5000-step training and confirm checkpointing + W&B logging.

### 2026-05-08 04:15 UTC | feat/tpu-support | patch-19-validation

**Status:** SUCCESS
**Kind:** milestone
**Detail:**
Patch 19 (canonical end-of-training save) validated on v6e-8 single-host
europe-west4-a. Training completed 20 steps (fp32, batch=1, accum=2) with
loss 9.94 -> 8.78 (decreasing). The canonical save executed successfully:
`model.to("cpu")` gathered all FSDPv2 shards onto host RAM without deadlock,
then `save_pretrained(safe_serialization=False)` wrote all components.
Checkpoint artifacts confirmed:
  - peft_adapter/adapter_model.bin (7 MB, LoRA adapters)
  - peft_adapter/adapter_config.json (LoRA config)
  - projection.pt (16 MB), depth_decoder.pt (1.4 GB),
    text_embed.pt (1.0 GB), audio_heads.pt (8 MB)
  - metadata.json {"step": "final", "save_kind": "canonical_final"}
  - Total: 2.4 GB

Known issue: save_dir was set to a gs:// URI, but torch.save wrote to a
local directory named `gs:/...` instead of actual GCS. Need to add a
post-save gsutil cp step or use gcsfs.

Earlier in this session:
  - v6e-8 in us-east1-d: preempted during maintenance event
  - v6e-8 in us-east1-d (2nd attempt): SUSPENDED for 28 min, no recovery
  - v6e-8 in europe-west4-a: ACTIVE at T+4:44 (fastest yet)
  - v6e bf16 NaN at step 1: diagnosed as pytorch/xla #4152 (HF attention
    mask torch.finfo(fp32).min -> -inf in bf16) + #8591 (v6e-specific
    batch-size-dependent NaN). Fixed by switching to float32 precision.
  - "Too many open files" crash: systemd startup_script inherited
    LimitNOFILE=100000. Fixed by launching via explicit `ulimit -n 1048576`
    in a dedicated launcher script (/tmp/launch_train_v6e_v2.sh).
  - fd limit root cause: v6e 8-chip topology + libtpu eventfd interrupts
    create ~100k FDs during init, exceeding default ulimit. The systemd-
    managed startup_script had LimitNOFILE=100000 but tmux child inherited
    a lower limit.
