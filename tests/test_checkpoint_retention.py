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


# ---- B1 resume: latest-checkpoint discovery + metadata reader ----

def test_find_latest_checkpoint_picks_highest_step_not_best(tmp_path):
    _make_ckpt_tree(tmp_path, [500, 1000, 1500], with_best=True)
    latest = ckpt.find_latest_checkpoint(str(tmp_path))
    assert latest is not None
    assert latest.rstrip("/").endswith("step_001500")  # not best_by_val


def test_get_checkpoint_dirs_excludes_best_and_sorts(tmp_path):
    _make_ckpt_tree(tmp_path, [2000, 500, 1000], with_best=True)
    dirs = ckpt.get_checkpoint_dirs(str(tmp_path))
    names = [d.rstrip("/").rsplit("/", 1)[-1] for d in dirs]
    assert names == ["step_000500", "step_001000", "step_002000"]  # sorted, no best


def test_get_checkpoint_dirs_empty_when_absent(tmp_path):
    assert ckpt.get_checkpoint_dirs(str(tmp_path / "nope")) == []


def test_step_of_parses_and_defaults(tmp_path):
    assert ckpt._step_of("gs://b/run/step_000750") == 750
    assert ckpt._step_of("/x/step_000750/") == 750
    assert ckpt._step_of("/x/best_by_val") == -1


def test_read_checkpoint_metadata_local(tmp_path):
    d = tmp_path / "step_000500"
    d.mkdir()
    (d / "metadata.json").write_text('{"step": 500, "wandb_run_id": "abc123"}')
    meta = ckpt.read_checkpoint_metadata(str(d))
    assert meta.get("wandb_run_id") == "abc123"
    assert meta.get("step") == 500


def test_read_checkpoint_metadata_missing_returns_empty(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    assert ckpt.read_checkpoint_metadata(str(d)) == {}


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
