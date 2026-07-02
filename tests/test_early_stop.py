"""Tests for the Phase A early-stopping decision helper (torch-free)."""

from __future__ import annotations

import importlib.util
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]


def _load():
    path = REPO / "src" / "training" / "early_stop.py"
    spec = importlib.util.spec_from_file_location("early_stop", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


es = _load()
INF = float("inf")


def test_improvement_resets_patience_and_updates_best():
    d = es.early_stop_step(val_loss=3.0, best_val=INF, patience_left=2, patience=8, min_delta=0.001)
    assert d.improved is True
    assert d.best_val == 3.0
    assert d.patience_left == 8        # reset to full patience
    assert d.should_stop is False


def test_non_improvement_decrements_patience():
    d = es.early_stop_step(val_loss=4.0, best_val=3.0, patience_left=8, patience=8, min_delta=0.001)
    assert d.improved is False
    assert d.best_val == 3.0           # unchanged
    assert d.patience_left == 7
    assert d.should_stop is False


def test_min_delta_filters_trivial_improvement():
    # 2.9995 is better than 3.0 but only by 0.0005 < min_delta 0.001 -> NOT a new best
    d = es.early_stop_step(val_loss=2.9995, best_val=3.0, patience_left=5, patience=8, min_delta=0.001)
    assert d.improved is False
    assert d.best_val == 3.0
    assert d.patience_left == 4


def test_significant_improvement_past_min_delta():
    d = es.early_stop_step(val_loss=2.99, best_val=3.0, patience_left=5, patience=8, min_delta=0.001)
    assert d.improved is True
    assert d.best_val == 2.99


def test_stops_when_patience_exhausted():
    d = es.early_stop_step(val_loss=4.0, best_val=3.0, patience_left=1, patience=8, min_delta=0.001)
    assert d.improved is False
    assert d.patience_left == 0
    assert d.should_stop is True       # 1 -> 0 triggers stop


def test_patience_zero_disables_stopping():
    d = es.early_stop_step(val_loss=9.0, best_val=3.0, patience_left=0, patience=0, min_delta=0.001)
    assert d.should_stop is False      # disabled -> never stops
    assert d.best_val == 3.0


def test_full_v3_sequence_8_patience_stops_after_8_bad_cycles():
    """Simulate v0.2-style monotonic val rise: best at cycle 0, then 8 worse cycles."""
    best, left, stops = INF, 8, []
    # cycle 0: new best
    d = es.early_stop_step(3.0, best, left, patience=8, min_delta=0.001)
    best, left = d.best_val, d.patience_left
    assert d.improved and left == 8
    # 8 consecutive worse cycles -> should stop exactly on the 8th
    for i in range(1, 9):
        d = es.early_stop_step(3.0 + 0.1 * i, best, left, patience=8, min_delta=0.001)
        best, left = d.best_val, d.patience_left
        stops.append(d.should_stop)
    assert best == 3.0                 # best preserved throughout
    assert stops == [False] * 7 + [True]   # stops on the 8th bad cycle


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
