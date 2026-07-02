# W&B Hyperparameter Sweeps — Stage 2

Proxy-first HP search before the expensive 22 h v6e-8 run. See
`docs/next-15k-run-plan.md` §4 for the rationale and `sweep_stage2.yaml`
for the grid.

## Prereqs (Phase 3 — implemented in this PR)
- `wandb agent` passes each swept parameter as a **CLI arg** (the `${args}` in
  `sweep_stage2.yaml` `command`). `train_hierarchical.py` parses them through
  its existing CLI-override path: `lr_lora`/`lr_depth` → `optim`, `text_weight`
  → `loss`, `warmup_steps`/`weight_decay`/`max_steps` → `train`, `val_every`/
  `val_on_tpu` → `logging`, and `lora_r`/`lora_alpha_mult` → `lora.r`/`lora.alpha`
  (`alpha = lora_alpha_mult * lora_r`). `--sweep` also logs a `sweep/text_ok`
  flag (1.0 when `train/text_loss < 11.5`, i.e. the text stream is learning).
- `configs/tpu/stage2_tpu_v6e_proxy.yaml` — a small/short proxy config
  (max_frames 200, `max_steps` 600, cheap val) so each trial is cheap.

## Runbook (steps YOU take)
1. **Create the sweep** (from your workstation; needs `wandb login`):
   ```bash
   wandb sweep sweeps/sweep_stage2.yaml
   # -> prints: wandb: Created sweep with ID: <entity>/<project>/<sweep_id>
   ```
2. **Run agents on the TPU VM** (each agent runs trials sequentially). The
   agent inherits the current env (`${env}`), so **export the full TPU env
   first** — same vars as the launcher:
   ```bash
   # on the v6e-8 (tmux), as the run user:
   cd /opt/tinyaya
   export DEVICE_BACKEND=tpu PJRT_DEVICE=TPU XLA_DISABLE_FUNCTIONALIZATION=0 \
          XLA_NO_SPECIAL_SCALARS=1 TPU_STRATEGY=fsdpv2_lora PYTHONUNBUFFERED=1 \
          LD_LIBRARY_PATH="$(dirname "$(find ~/.local/share/uv/python -name libpython3.12.so.1.0 | head -1)"):$LD_LIBRARY_PATH"
   set -a; . /opt/tinyaya/.env; set +a   # HF_TOKEN / WANDB_API_KEY
   wandb agent <entity>/<project>/<sweep_id> --count 8
   ```
   On a single v6e-8 trials are sequential (one SPMD program/host) and each
   pays the ~18 min cold compile — keep `max_steps` small and let hyperband
   prune. For the LR-range test, prefer a tiny CPU/GPU proxy to avoid the
   TPU compile tax.
3. **Pick the winner** in the W&B sweep dashboard (lowest `val/audio_loss`;
   reject any trial whose `val/text_loss` ≈ ln(V) = the data bug, and any
   whose `sweep/text_ok` stayed 0).
4. **Promote to the full run** — write the winning HPs into the prod config
   with the helper (no hand-editing; it patches only the swept keys):
   ```bash
   python scripts/promote_sweep_winner.py \
       --sweep <entity>/<project>/<sweep_id> \
       --config configs/tpu/stage2_tpu_v6e_v2.yaml
   # or pass HPs explicitly if you picked by eye:
   python scripts/promote_sweep_winner.py \
       --config configs/tpu/stage2_tpu_v6e_v2.yaml \
       --set lr_lora=2e-4 lr_depth=3e-4 lora_r=32 lora_alpha_mult=2 \
             text_weight=0.5 warmup_steps=300 weight_decay=1e-3
   ```
   Then launch the release run:
   ```bash
   bash scripts/tpu/launch_release.sh configs/tpu/stage2_tpu_v6e_v2.yaml
   ```

## Notes
- `method: bayes` + `early_terminate: hyperband` ≈ ASHA. Switch to
  `method: grid` for a deterministic coarse pass first if preferred.
- Flip `metric.name` to `val/loss` once the text stream learns.

---

## Sweep results (`9ba8h0ho`)

First production sweep — **8 bayes + hyperband trials** on `stage2_tpu_v6e_proxy.yaml`
(600 steps each) on a single v6e-8, ranking hyperparameters before the 15k run.

**📊 W&B dashboard:** https://wandb.ai/cataluna84/tinyaya-stage2-tpu/sweeps/9ba8h0ho

The dashboard is the live source of truth. It has:
- **Per-trial loss curves** — `val/audio_loss` (the optimization target), `val/text_loss`,
  and the on-device per-group stability `diag/*` metrics streamed during each trial.
- **Parallel-coordinates view** — all 8 trials' hyperparameters (`lr_lora`, `lr_depth`,
  `lora_r`, `lora_alpha_mult`, `text_weight`, `warmup_steps`, `weight_decay`) plotted
  against the metric, so the winning region is visible at a glance.
- **Importance / correlation panel** — which knobs actually move `val/audio_loss`
  (`lora_r` dominates).
- **`sweep/text_ok` flag** per trial — 1.0 when the text stream is learning
  (`train/text_loss < 11.5`), i.e. the Phase 0 fix is holding.

**🏆 Winning run `ryvims4h`:** https://wandb.ai/cataluna84/tinyaya-stage2-tpu/runs/ryvims4h

### Ranked results (by `val/audio_loss`, the sweep metric)

| rk | run | val/audio | val/text | text_ok | lr_lora | lr_depth | r | α | text_w | warmup | wd |
|---:|-----|----------:|---------:|:-------:|--------:|---------:|--:|--:|--:|--:|--:|
| **1** | **`ryvims4h`** | **4.8899** | 4.140 | ✅ | 4.60e-4 | 1.10e-4 | 64 | 128 | 0.1 | 300 | 0.01 |
| 2 | `6g6o1920` | 4.9237 | 4.304 | ✅ | 3.79e-4 | 1.04e-4 | 64 | 64 | 0.1 | 300 | 0.01 |
| 3 | `5woejq1o` | 4.9780 | 4.375 | ✅ | 2.10e-4 | 1.06e-4 | 64 | 128 | 0.1 | 100 | 0.01 |
| 4 | `icxhcvgx` | 5.2666 | 3.734 | ✅ | 3.27e-4 | 1.12e-4 | 64 | 64 | 0.5 | 600 | 0.01 |
| 5 | `ww1dsyui` | 5.3570 | 3.705 | ✅ | 3.47e-4 | 1.32e-4 | 64 | 64 | 2 | 600 | 0.01 |
| 6 | `5x7a8nwy` | 5.3714 | 3.673 | ✅ | 3.58e-4 | 1.17e-4 | 64 | 64 | 2 | 600 | 0.001 |
| 7 | `l4eu8lx4` | 5.5628 | 3.798 | ✅ | 7.43e-5 | 2.01e-4 | 32 | 64 | 1 | 600 | 0.001 |
| 8 | `h0v2b7wh` | 5.6747 | 3.832 | ✅ | 6.21e-5 | 5.47e-4 | 16 | 16 | 2 | 300 | 0.001 |

(α = `lora_alpha_mult` × `lora_r`. All values at the 600-step proxy horizon.)

### Winner: `ryvims4h` (val/audio_loss = 4.8899)

```yaml
lora:   { r: 64, alpha: 128 }          # alpha = lora_alpha_mult(2) * r(64)
optim:  { lr_lora: 4.6e-4, lr_depth: 1.1e-4 }
loss:   { text_weight: 0.1 }
train:  { warmup_steps: 300, weight_decay: 0.01 }
```

**Why this recipe:**
- **Every trial learned the text stream** (`text_ok=1`, val/text 3.67–4.38, all far below
  ln(V) ≈ 12.5) — independent confirmation the Phase 0 padding-weighted text loss is
  robust across the whole hyperparameter space, not a single-config fluke.
- **`lora_r=64` dominates** — the six r=64 trials take the top six slots; the r=32 and
  r=16 trials are last. LoRA capacity is the single biggest lever for the audio stream.
- **The sweep converged** on a tight region for the leaders (`r=64`, `lr_lora≈3.8–4.6e-4`,
  `lr_depth≈1.0e-4`, `warmup=300`, `wd=0.01`, `text_weight=0.1`). The winner is its
  aggressive end (highest `lr_lora`, `alpha=128`).
- **Metric choice:** the sweep minimizes `val/audio_loss` because the model's *output* is
  target-language audio; text is an auxiliary stream. Now that text reliably learns,
  `val/loss` is a reasonable secondary metric.

**Audio↔text trade-off:** low `text_weight=0.1` (ranks 1–3) maximizes audio quality but
gives weaker text (~4.1–4.4); high `text_weight=1–2` (ranks 4–8) flips it (text ~3.67,
audio worse). For the 15k production run, consider nudging `text_weight` to ~0.2 to keep
the text stream improving over the longer horizon; rank-2 `6g6o1920` is a near-tied
(Δaudio 0.034), gentler alternative (`lr_lora=3.79e-4, alpha=64`) if long-run stability is
prioritized over the marginal audio gain.

### Promote the winner → launch the 15k run

```bash
# write the winning HPs into the prod config (preserves comments; same flat->nested
# mapping as train_hierarchical.py)
python scripts/promote_sweep_winner.py \
    --sweep cataluna84/tinyaya-stage2-tpu/9ba8h0ho \
    --config configs/tpu/stage2_tpu_v6e_v2.yaml
# then launch the 15k release run on the v6e-8
bash scripts/tpu/launch_release.sh configs/tpu/stage2_tpu_v6e_v2.yaml
```

## v0.3 sweep (parallel, on the composite metric)

`sweeps/sweep_stage2_v3.yaml` searches the **regularization** knobs (`lr_lora`,
`lora_r∈{8,16,32}`, `lora_dropout`, `weight_decay`) with all v0.3 levers (A/C/D)
ON, optimising **`val/composite`** (logged `summary="min"`). It points at the
single-host proxy `configs/tpu/stage2_tpu_v6e_v3_proxy.yaml`.

Each trial is independent, so run it **N-wide** rather than on a bigger mesh
(LoRA generalises worse past batch ~128, so a 64-chip SPMD trial is the wrong
tool — see the brainstorm in the PR):

```bash
# 1. create the sweep (workstation)
wandb sweep sweeps/sweep_stage2_v3.yaml          # -> ENTITY/PROJECT/SWEEP_ID

# 2a. v6e-64: one agent per host (8 parallel trials). VALIDATE isolation --
#     the first trial must log `[tpu_backend][post-wrap] global=8` (NOT 64).
SWEEP_ID=ENTITY/PROJECT/SWEEP_ID \
NODE_ID=tinyaya-stage2-spot-v6e64-eu ZONE=europe-west4-a \
    bash scripts/tpu/launch_sweep.sh

# 2b. ROBUST fallback: N separate v6e-8 spot slices, one agent each (WORKER=0).
```

Promote the winner into the **production v3** config (default `--metric
val/composite`), then run the single 15k v0.3 run on a **v6e-8**:

```bash
python scripts/promote_sweep_winner.py \
    --sweep ENTITY/PROJECT/SWEEP_ID \
    --config configs/tpu/stage2_tpu_v6e_v3.yaml
```
