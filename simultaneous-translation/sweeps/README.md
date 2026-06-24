# W&B Hyperparameter Sweeps — Stage 2

Proxy-first HP search before the expensive 22 h v6e-8 run. See
`docs/next-15k-run-plan.md` §4 for the rationale and `sweep_stage2.yaml`
for the grid.

## Prereqs (Phase 3 — implemented in this PR)
- `--sweep` flag in `scripts/train_hierarchical.py` that maps flat
  `wandb.config` keys (`lr_lora`, `lora_r`, `lora_alpha_mult`, `text_weight`,
  …) onto nested `cfg` overrides before training starts.
- `configs/stage2_tpu_v6e_proxy.yaml` — a small/short proxy config (fewer
  steps, smaller `val_max_batches`) so each trial is cheap.

## Runbook (steps YOU take)
1. **Create the sweep** (from your workstation; needs `wandb login`):
   ```bash
   wandb sweep simultaneous-translation/sweeps/sweep_stage2.yaml
   # -> prints: wandb: Created sweep with ID: <entity>/<project>/<sweep_id>
   ```
2. **Run agents on the TPU VM** (each agent runs trials sequentially):
   ```bash
   # on the v6e-8 (tmux), with the TPU env exported:
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
