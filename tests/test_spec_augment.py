"""Tests for Phase E SpecAugment-on-codes planning + gating (torch-free)."""

from __future__ import annotations

import importlib.util
import pathlib
import random

REPO = pathlib.Path(__file__).resolve().parents[1]


def _load():
    path = REPO / "src" / "data" / "spec_augment.py"
    spec = importlib.util.spec_from_file_location("spec_augment", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sa = _load()


# ---- plan_time_masks -------------------------------------------------------

def test_time_masks_count_and_bounds():
    rng = random.Random(0)
    spans = sa.plan_time_masks(T=100, num_masks=3, max_span=20, rng=rng)
    assert len(spans) == 3
    for start, span in spans:
        assert 1 <= span <= 20
        assert 0 <= start and start + span <= 100  # never runs past T


def test_time_masks_disabled_returns_empty():
    rng = random.Random(0)
    assert sa.plan_time_masks(100, 0, 20, rng) == []
    assert sa.plan_time_masks(100, 3, 0, rng) == []
    assert sa.plan_time_masks(0, 3, 20, rng) == []


def test_time_masks_span_clipped_to_T():
    rng = random.Random(1)
    spans = sa.plan_time_masks(T=5, num_masks=4, max_span=50, rng=rng)
    for start, span in spans:
        assert span <= 5 and start + span <= 5


def test_time_masks_deterministic_with_seed():
    a = sa.plan_time_masks(100, 3, 20, random.Random(42))
    b = sa.plan_time_masks(100, 3, 20, random.Random(42))
    assert a == b


# ---- plan_codebook_drops ---------------------------------------------------

def test_codebook_drops_distinct_sorted_never_all():
    rng = random.Random(0)
    drops = sa.plan_codebook_drops(num_codebooks=8, max_drops=3, rng=rng)
    assert len(drops) == 3
    assert drops == sorted(drops)
    assert len(set(drops)) == len(drops)
    assert all(0 <= c < 8 for c in drops)


def test_codebook_drops_keeps_at_least_one():
    rng = random.Random(0)
    # asking to drop >= num_codebooks must still keep one
    drops = sa.plan_codebook_drops(num_codebooks=4, max_drops=10, rng=rng)
    assert len(drops) == 3  # 4 - 1


def test_codebook_drops_disabled_returns_empty():
    rng = random.Random(0)
    assert sa.plan_codebook_drops(8, 0, rng) == []
    assert sa.plan_codebook_drops(1, 2, rng) == []


# ---- apply_spec_augment gating (no tensor needed when disabled) -----------

def test_apply_noop_when_disabled():
    sentinel = object()  # never touched -> proves no .shape access
    rng = random.Random(0)
    assert sa.apply_spec_augment(sentinel, None, rng, silence_token=2048) is sentinel
    assert sa.apply_spec_augment(sentinel, {"enabled": False}, rng, silence_token=2048) is sentinel


def test_apply_masks_user_stream_when_enabled():
    # Minimal list-backed stand-in for a [B, CB, T] tensor supporting the slice
    # assignments apply_spec_augment performs.
    class FakeStream:
        def __init__(self, B, CB, T):
            self.shape = (B, CB, T)
            self.data = [[[1] * T for _ in range(CB)] for _ in range(B)]

        def __setitem__(self, idx, val):
            i, cb, sl = idx
            cbs = range(self.shape[1]) if cb == slice(None) else [cb]
            start = sl.start or 0
            stop = sl.stop if sl.stop is not None else self.shape[2]
            for c in cbs:
                for t in range(start, stop):
                    self.data[i][c][t] = val

    s = FakeStream(1, 8, 50)
    cfg = {"enabled": True, "prob": 1.0, "time_mask_count": 2,
           "time_mask_max": 10, "codebook_drop_count": 1}
    sa.apply_spec_augment(s, cfg, random.Random(0), silence_token=2048, lengths=[50])
    flat = [v for cb in s.data[0] for v in cb]
    assert 2048 in flat   # something got masked
    assert 1 in flat      # but not everything


if __name__ == "__main__":
    import sys
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    sys.exit(1 if failed else 0)
