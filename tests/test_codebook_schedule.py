"""Tests for the Phase C per-codebook weight schedule (torch-free)."""

from __future__ import annotations

import importlib.util
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]


def _load():
    path = REPO / "src" / "training" / "codebook_schedule.py"
    spec = importlib.util.spec_from_file_location("codebook_schedule", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cs = _load()


# ---- active_codebooks (the unmask ramp) -----------------------------------

def test_unmask_off_all_active():
    assert cs.active_codebooks(0, 1000, 8, unmask_fraction=0.0) == 8
    assert cs.active_codebooks(500, 1000, 8, unmask_fraction=0.0) == 8


def test_ramp_starts_at_k0_and_reaches_full():
    # unmask over first 10% of 1000 steps = 100 steps; k0=1 -> 8
    assert cs.active_codebooks(0, 1000, 8, unmask_fraction=0.10, k0=1) == 1
    assert cs.active_codebooks(100, 1000, 8, unmask_fraction=0.10, k0=1) == 8
    assert cs.active_codebooks(1000, 1000, 8, unmask_fraction=0.10, k0=1) == 8  # past ramp


def test_ramp_is_monotonic_nondecreasing():
    ks = [cs.active_codebooks(s, 1000, 8, unmask_fraction=0.10, k0=1) for s in range(0, 120, 5)]
    assert ks == sorted(ks)
    assert ks[0] == 1 and ks[-1] == 8


def test_k0_clamped_into_range():
    assert cs.active_codebooks(0, 1000, 8, unmask_fraction=0.10, k0=0) == 1   # >=1
    assert cs.active_codebooks(0, 1000, 8, unmask_fraction=0.10, k0=99) == 8  # <=num_cb


# ---- codebook_weights ------------------------------------------------------

def test_weights_default_multipliers_all_one_when_unmasked():
    w = cs.codebook_weights(0, 1000, 8, multipliers=None, unmask_fraction=0.0)
    assert w == [1.0] * 8


def test_masked_codebooks_are_zero_during_ramp():
    # step 0, k0=1 -> only codebook 0 active
    w = cs.codebook_weights(0, 1000, 8, multipliers=None, unmask_fraction=0.10, k0=1)
    assert w[0] == 1.0
    assert w[1:] == [0.0] * 7


def test_multipliers_applied_to_active_codebooks():
    mult = [1.0, 1.0, 1.5, 1.5, 2.0, 2.0, 2.0, 2.0]
    # past the ramp: all active -> full multipliers
    w = cs.codebook_weights(1000, 1000, 8, multipliers=mult, unmask_fraction=0.10, k0=1)
    assert w == mult
    # at step 0: only cb0 active -> its multiplier, rest 0
    w0 = cs.codebook_weights(0, 1000, 8, multipliers=mult, unmask_fraction=0.10, k0=1)
    assert w0 == [1.0] + [0.0] * 7


def test_wrong_multiplier_length_raises():
    try:
        cs.codebook_weights(0, 1000, 8, multipliers=[1.0, 1.0])
    except ValueError:
        return
    raise AssertionError("expected ValueError for wrong multiplier length")


def test_weight_vectors_are_hashable_for_caching():
    # the train loop caches device tensors keyed by tuple(weights) -> must be hashable
    w = cs.codebook_weights(50, 1000, 8, multipliers=[1.0] * 8, unmask_fraction=0.10, k0=1)
    assert hash(tuple(w)) == hash(tuple(w))


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
