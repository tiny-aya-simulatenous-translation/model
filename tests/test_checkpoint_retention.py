"""Tests for unlimited / bounded checkpoint retention (`prune_checkpoints`).

WHY THIS EXISTS
---------------
The 15k run discovered two retention bugs: (1) the prune call site hard-coded
`keep_last=3`, ignoring the `keep_last_n` config; (2) there was no way to keep
*all* checkpoints. The fix adds `keep_last<=0 => unlimited` semantics. These
torch-free tests pin the local-path pruning behavior so it can't regress.

Run: `python -m pytest tests/test_checkpoint_retention.py -v`  (or run directly)
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
from unittest.mock import MagicMock

REPO = pathlib.Path(__file__).resolve().parents[1]


def _load_ckpt():
    """Import src/training/checkpointing.py with a stub `torch`.

    The module does a top-level `import torch` (and uses `torch.nn.Module` in an
    annotation), but `prune_checkpoints` (the unit under test) is pure-Python
    file ops and never touches torch. A MagicMock satisfies any `torch.*` access
    at import time, so the module loads on a CPU box / lightweight CI with no
    torch installed (the module defines no classes that subclass torch).
    """
    if "torch" not in sys.modules:
        sys.modules["torch"] = MagicMock()
    path = REPO / "src" / "training" / "checkpointing.py"
    spec = importlib.util.spec_from_file_location("checkpointing", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ckpt = _load_ckpt()


def _make_ckpt_tree(root: pathlib.Path, steps, with_best=True):
    for s in steps:
        d = root / f"step_{s:06d}"
        d.mkdir(parents=True)
        (d / "metadata.json").write_text("{}")
    if with_best:
        b = root / "best_by_val"
        b.mkdir()
        (b / "metadata.json").write_text("{}")


def _surviving_steps(root: pathlib.Path):
    return sorted(int(p.name.split("_")[1]) for p in root.glob("step_*") if p.is_dir())


def test_keep_last_zero_is_unlimited(tmp_path):
    _make_ckpt_tree(tmp_path, [1000, 2000, 3000, 4000, 5000])
    ckpt.prune_checkpoints(str(tmp_path), keep_last=0, keep_best="best_by_val")
    assert _surviving_steps(tmp_path) == [1000, 2000, 3000, 4000, 5000]
    assert (tmp_path / "best_by_val").exists()


def test_keep_last_negative_is_unlimited(tmp_path):
    _make_ckpt_tree(tmp_path, [1000, 2000, 3000])
    ckpt.prune_checkpoints(str(tmp_path), keep_last=-1)
    assert _surviving_steps(tmp_path) == [1000, 2000, 3000]


def test_keep_last_none_is_unlimited(tmp_path):
    _make_ckpt_tree(tmp_path, [1000, 2000])
    ckpt.prune_checkpoints(str(tmp_path), keep_last=None)
    assert _surviving_steps(tmp_path) == [1000, 2000]


def test_bounded_keeps_last_n_plus_best(tmp_path):
    _make_ckpt_tree(tmp_path, [1000, 2000, 3000, 4000, 5000])
    ckpt.prune_checkpoints(str(tmp_path), keep_last=2, keep_best="best_by_val")
    assert _surviving_steps(tmp_path) == [4000, 5000]
    assert (tmp_path / "best_by_val").exists()  # best survives pruning


if __name__ == "__main__":
    import sys
    import tempfile
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        with tempfile.TemporaryDirectory() as td:
            try:
                fn(pathlib.Path(td))
                print(f"PASS {fn.__name__}")
            except Exception:
                failed += 1
                print(f"FAIL {fn.__name__}")
                traceback.print_exc()
    sys.exit(1 if failed else 0)
