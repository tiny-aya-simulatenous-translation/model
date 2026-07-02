"""SpecAugment adapted to discrete Mimi codes (Phase E, flag-gated).

Classic SpecAugment masks time/frequency bands of a spectrogram. RVQ codes have
no spectrogram, so we adapt it to two discrete operations on the INPUT (user)
audio stream:

* **time masking** — blank random spans of frames (set to ``SILENCE_TOKEN``), and
* **codebook dropout** — blank random whole codebooks (a discrete analogue of
  frequency masking: drops a residual-quantizer level for the whole utterance).

Applied to the user stream ONLY (never the targets / never val), train-loader
only, as a robustness regularizer.

The *planning* functions (which positions to mask) are pure-Python / torch-free
so CI can unit-test the logic without a tensor backend; ``apply_spec_augment``
does the in-place tensor write and short-circuits before touching the tensor
when disabled, so it too is importable/testable without torch.
"""

from __future__ import annotations

import random


def plan_time_masks(
    T: int, num_masks: int, max_span: int, rng: random.Random
) -> list[tuple[int, int]]:
    """Up to ``num_masks`` ``(start, length)`` spans within ``[0, T)``.

    Each length is in ``[1, min(max_span, T)]`` at a random start; spans may
    overlap. Returns ``[]`` when ``T<=0``, ``num_masks<=0`` or ``max_span<=0``.
    """
    if T <= 0 or num_masks <= 0 or max_span <= 0:
        return []
    spans: list[tuple[int, int]] = []
    for _ in range(num_masks):
        span = rng.randint(1, min(max_span, T))
        start = rng.randint(0, T - span)
        spans.append((start, span))
    return spans


def plan_codebook_drops(
    num_codebooks: int, max_drops: int, rng: random.Random
) -> list[int]:
    """Sorted list of ``<= max_drops`` distinct codebooks to drop.

    Never drops ALL codebooks (keeps at least one) so the stream is not fully
    blanked. Returns ``[]`` when ``num_codebooks<=1`` or ``max_drops<=0``.
    """
    if num_codebooks <= 1 or max_drops <= 0:
        return []
    k = min(max_drops, num_codebooks - 1)
    return sorted(rng.sample(range(num_codebooks), k))


def apply_spec_augment(
    stream, cfg: dict | None, rng: random.Random, *, silence_token: int, lengths=None
):
    """In-place SpecAugment on a ``[B, CB, T]`` long tensor (user audio stream).

    ``cfg`` keys: ``enabled``, ``prob``, ``time_mask_count``, ``time_mask_max``,
    ``codebook_drop_count``. ``lengths[i]`` (optional) caps masking to sample
    ``i``'s real (un-padded) frames. No-op (returns ``stream`` untouched) when
    ``cfg`` is falsy or ``enabled`` is not set. Returns ``stream``.
    """
    if not cfg or not cfg.get("enabled"):
        return stream
    B, CB, T = stream.shape
    prob = float(cfg.get("prob", 1.0))
    tmc = int(cfg.get("time_mask_count", 0))
    tmm = int(cfg.get("time_mask_max", 0))
    cdc = int(cfg.get("codebook_drop_count", 0))
    for i in range(B):
        if rng.random() > prob:
            continue
        Ti = int(lengths[i]) if lengths is not None else T
        Ti = max(0, min(Ti, T))
        for start, span in plan_time_masks(Ti, tmc, tmm, rng):
            stream[i, :, start:start + span] = silence_token
        for cb in plan_codebook_drops(CB, cdc, rng):
            stream[i, cb, :Ti] = silence_token
    return stream
