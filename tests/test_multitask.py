"""Tests for Phase D multitask balance helpers (torch-free)."""

from __future__ import annotations

import importlib.util
import math
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]


def _load():
    path = REPO / "src" / "training" / "multitask.py"
    spec = importlib.util.spec_from_file_location("multitask", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mt = _load()


# ---- composite_val_loss ----------------------------------------------------

def test_composite_default_weights_audio_heavier():
    # audio weighted 0.6, text 0.4 -> closer to audio
    c = mt.composite_val_loss(text_loss=10.0, audio_loss=5.0)
    assert math.isclose(c, 0.4 * 10.0 + 0.6 * 5.0)  # weights sum to 1
    assert c < 7.5  # pulled below the plain mean toward audio


def test_composite_normalises_when_weights_not_unit_sum():
    # weights 1.0/1.0 -> plain mean regardless of magnitude
    c = mt.composite_val_loss(4.0, 8.0, text_w=1.0, audio_w=1.0)
    assert math.isclose(c, 6.0)


def test_composite_zero_weights_falls_back_to_mean():
    c = mt.composite_val_loss(4.0, 8.0, text_w=0.0, audio_w=0.0)
    assert math.isclose(c, 6.0)


# ---- text_weight_at --------------------------------------------------------

def test_text_weight_starts_high_ends_low():
    assert mt.text_weight_at(0, 1000, start=0.5, end=0.3, frac=0.25) == 0.5
    # at/after the ramp end -> hold at end
    assert mt.text_weight_at(250, 1000, start=0.5, end=0.3, frac=0.25) == 0.3
    assert mt.text_weight_at(900, 1000, start=0.5, end=0.3, frac=0.25) == 0.3


def test_text_weight_monotone_nonincreasing_over_ramp():
    vals = [mt.text_weight_at(s, 1000, 0.5, 0.3, 0.25) for s in range(0, 260, 10)]
    assert all(b <= a + 1e-9 for a, b in zip(vals, vals[1:]))
    assert vals[0] == 0.5 and vals[-1] == 0.3


def test_text_weight_quantised_to_few_values():
    # 0.5 -> 0.3 on a 0.05 grid => at most {0.5,0.45,0.4,0.35,0.3}
    distinct = {mt.text_weight_at(s, 1000, 0.5, 0.3, 0.25, quantum=0.05)
                for s in range(0, 1000)}
    assert distinct.issubset({0.5, 0.45, 0.4, 0.35, 0.3})
    assert len(distinct) <= 5


def test_text_weight_frac_zero_disables():
    assert mt.text_weight_at(0, 1000, 0.5, 0.3, frac=0.0) == 0.3
    assert mt.text_weight_at(123, 1000, 0.5, 0.3, frac=0.0) == 0.3


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
