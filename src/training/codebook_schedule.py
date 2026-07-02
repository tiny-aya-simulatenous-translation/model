"""Per-codebook loss-weight schedule (Phase C: deep-codebook learning).

WHY THIS EXISTS
---------------
v0.2's deeper RVQ codebooks collapsed (cb0 ~14% top-1, cb1-7 <4%). Two levers from
the audio-LM literature (Moshi / SoundStorm / MusicGen):

* **Per-codebook loss multipliers** -- weight each codebook's CE so the optimiser
  doesn't treat all 8 equally (the first codebook carries semantics; deeper ones
  carry fine acoustic residual).
* **Progressive coarse->fine unmasking** -- supervise only the first ``k0`` codebooks
  early, then ramp the number of active codebooks up to ``num_codebooks`` over the
  first ``unmask_fraction`` of training. Masked (weight 0) codebooks contribute no
  gradient, so the model stabilises the coarse structure before fine detail.

This is a PURE function (no torch) so it is unit-testable and is the single source
of truth for the weight vector. The training loop turns the returned list into a
device tensor (cached by value so the unmask ramp only recompiles a handful of
times -- see train_hierarchical.py).
"""

from __future__ import annotations

import math


def active_codebooks(step: int, total_steps: int, num_codebooks: int,
                     unmask_fraction: float = 0.0, k0: int = 1) -> int:
    """Number of supervised codebooks at ``step`` under progressive unmasking.

    Ramps ``k0 -> num_codebooks`` linearly over the first ``unmask_fraction`` of
    ``total_steps``. ``unmask_fraction <= 0`` disables unmasking (all active).
    """
    if num_codebooks <= 0:
        return 0
    k0 = max(1, min(num_codebooks, k0))
    if unmask_fraction and unmask_fraction > 0 and total_steps > 0:
        unmask_steps = max(1, int(unmask_fraction * total_steps))
        progress = min(1.0, max(0.0, step) / unmask_steps)
        k = k0 + math.ceil((num_codebooks - k0) * progress)
    else:
        k = num_codebooks
    return max(1, min(num_codebooks, k))


def codebook_weights(step: int, total_steps: int, num_codebooks: int,
                     multipliers: list[float] | None = None,
                     unmask_fraction: float = 0.0, k0: int = 1) -> list[float]:
    """Per-codebook loss weights at ``step`` (length ``num_codebooks``).

    weight[c] = multipliers[c]  if c is active at this step, else 0.0.
    Active set = the first ``active_codebooks(step, ...)`` codebooks.
    ``multipliers=None`` => all-ones (pure unmasking, no re-weighting).
    """
    if multipliers is None:
        multipliers = [1.0] * num_codebooks
    if len(multipliers) != num_codebooks:
        raise ValueError(
            f"multipliers has {len(multipliers)} entries, expected {num_codebooks}"
        )
    k = active_codebooks(step, total_steps, num_codebooks, unmask_fraction, k0)
    return [float(multipliers[c]) if c < k else 0.0 for c in range(num_codebooks)]
