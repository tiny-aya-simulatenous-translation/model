"""Pure early-stopping decision helper (Phase A).

WHY THIS EXISTS
---------------
v0.2 overfit: validation loss bottomed early and rose while training loss fell.
Early stopping halts a run once validation stops improving. This module holds the
*decision* as a single pure function so it is unit-testable without torch / a TPU
and is the single source of truth used by the training loop.

Important: stopping never degrades the released model -- ``best_by_val`` is saved
on every improvement regardless of patience. Patience only controls *when the run
halts* (and a longer patience can catch a late second-descent from LR decay).
"""

from __future__ import annotations

from typing import NamedTuple


class EarlyStopDecision(NamedTuple):
    improved: bool          # val improved by > min_delta -> save best_by_val + reset patience
    best_val: float         # updated best metric
    patience_left: int      # cycles remaining before stop
    should_stop: bool       # patience exhausted -> halt the run


def early_stop_step(
    val_loss: float,
    best_val: float,
    patience_left: int,
    patience: int,
    min_delta: float = 0.0,
) -> EarlyStopDecision:
    """Apply one early-stopping update from a validation result (lower = better).

    Args:
        val_loss: the just-measured validation metric.
        best_val: best metric seen so far (use ``float('inf')`` initially).
        patience_left: cycles remaining before stopping.
        patience: configured patience in val cycles (``<= 0`` disables stopping).
        min_delta: minimum improvement over ``best_val`` to count as a new best.

    Returns:
        EarlyStopDecision(improved, best_val, patience_left, should_stop).
    """
    if val_loss < best_val - min_delta:
        # significant new best -> save it, reset the patience counter
        return EarlyStopDecision(True, val_loss, patience, False)
    if patience <= 0:
        # early stopping disabled -> never stops, best_val unchanged
        return EarlyStopDecision(False, best_val, patience_left, False)
    new_left = patience_left - 1
    return EarlyStopDecision(False, best_val, new_left, new_left <= 0)
