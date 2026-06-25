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
- `configs/stage2_tpu_v6e_proxy.yaml` — a small/short proxy config (max_frames
  200, `max_steps` 600, cheap val) so each trial is cheap.

## Runbook (steps YOU take)
1. **Create the sweep** (from your workstation; needs `wandb login`):
   ```bash
   wandb sweep simultaneous-translation/sweeps/sweep_stage2.yaml
   # -> prints: wandb: Created sweep with ID: <entity>/<project>/<sweep_id>
   ```
2. **Run agents on the TPU VM** (each agent runs trials sequentially). The
   agent inherits the current env (`${env}`), so **export the full TPU env
   first** — same vars as the launcher:
   ```bash
   # on the v6e-8 (tmux), as the run user:
   cd /opt/tinyaya/simultaneous-translation
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
   reject any trial whose `val/text_loss` ≈ ln(V) = the data bug).
4. **Promote to the full run**: copy the winning HPs into
   `configs/stage2_tpu_v6e_v2.yaml` and launch the 22 h release run via
   `scripts/tpu/launch_release.sh`.

## Notes
- `method: bayes` + `early_terminate: hyperband` ≈ ASHA. Switch to
  `method: grid` for a deterministic coarse pass first if preferred.
- Flip `metric.name` to `val/loss` once the text stream learns.
