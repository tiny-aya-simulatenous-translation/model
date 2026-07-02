"""Phase D — multitask balance: composite val metric + text-weight curriculum.

WHY THIS EXISTS
---------------
v0.2 overfit and its audio codebooks collapsed while the text stream memorised.
Selecting ``best_by_val`` / early-stopping on the raw weighted ``val/loss`` lets
whichever stream dominates the weighting decide the checkpoint. Phase D instead:

* scores validation with an explicit **composite** of the two *raw* stream losses
  (audio weighted higher — it is the bottleneck), so a run that sacrifices the
  inner-monologue to chase audio (or vice-versa) is penalised; and
* anneals the **train** ``text_weight`` from a higher start to a lower end over
  the first slice of training, so the text stream is protected early (while the
  audio heads are still random) and capacity shifts to audio later.

TPU NOTE
--------
``text_weight`` is multiplied into the loss as a Python scalar, so each *distinct*
value bakes a new constant into the HLO and triggers one XLA recompile. The
curriculum is therefore **quantised** (piecewise-constant) — at most
``ceil(|start-end|/quantum)+1`` distinct values across the whole run — rather than
a smooth per-step ramp that would recompile every step.

Torch-free on purpose: CI installs only pytest + pyyaml (no torch), and this is
pure arithmetic, so the math is unit-tested without a TPU/GPU.
"""

from __future__ import annotations

import math


def composite_val_loss(
    text_loss: float,
    audio_loss: float,
    text_w: float = 0.4,
    audio_w: float = 0.6,
) -> float:
    """Weighted validation objective for ``best_by_val`` + early stopping.

    A convex combination of the *raw* per-stream val losses, normalised by the
    weight sum so the scale stays comparable to a plain mean when the weights do
    not add to 1. Audio carries the larger default weight because CB1-7 are the
    hard, collapse-prone part of the model.
    """
    denom = text_w + audio_w
    if denom <= 0:
        return 0.5 * text_loss + 0.5 * audio_loss
    return (text_w * text_loss + audio_w * audio_loss) / denom


def text_weight_at(
    step: int,
    total_steps: int,
    start: float = 0.5,
    end: float = 0.3,
    frac: float = 0.25,
    quantum: float = 0.05,
) -> float:
    """Quantised linear anneal of the train ``text_weight``.

    Ramps ``start -> end`` over the first ``frac`` of ``total_steps``, then holds
    at ``end``. The result is snapped to a ``quantum`` grid so only a handful of
    distinct values occur over the run (bounded XLA recompiles — see module docstring).

    ``frac <= 0`` (or a zero-length run) disables the ramp and returns ``end``.
    """
    if frac <= 0 or total_steps <= 0:
        return _snap(end, quantum)
    ramp_steps = max(1.0, frac * total_steps)
    t = min(1.0, max(0.0, step / ramp_steps))
    raw = start + (end - start) * t
    return _snap(raw, quantum)


def _snap(value: float, quantum: float) -> float:
    """Round ``value`` to the nearest ``quantum`` (no-op if ``quantum <= 0``)."""
    if quantum <= 0:
        return float(value)
    return round(round(value / quantum) * quantum, 10)
