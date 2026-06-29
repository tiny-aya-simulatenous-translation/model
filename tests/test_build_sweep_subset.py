"""Tests for the v0.3 sweep subset builder (torch-free)."""

from __future__ import annotations

import importlib.util
import pathlib
import random
from collections import Counter

REPO = pathlib.Path(__file__).resolve().parents[1]


def _load():
    path = REPO / "scripts" / "build_sweep_subset.py"
    spec = importlib.util.spec_from_file_location("build_sweep_subset", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bs = _load()


def test_source_of():
    assert bs.source_of("conv_13222") == "conv"
    assert bs.source_of("flores_dev_104") == "flores"
    assert bs.source_of("flores_devtest_61") == "flores"
    assert bs.source_of("opus_900") == "opus"
    assert bs.source_of("fleurs_221") == "fleurs"
    assert bs.source_of("") == "other"


def _make_rows():
    rows = []
    # 7000 conv, 2000 flores, 1000 opus (natural-ish proportions), ~50/50 dirs
    for src, count in (("conv", 7000), ("flores", 2000), ("opus", 1000)):
        for i in range(count):
            d = "hi->tr" if i % 2 == 0 else "tr->hi"
            rows.append({"pair_id": f"{src}_{i}", "direction": d,
                         "pt_path": f"encoded/{src}_{i}.pt",
                         "src_align_path": f"encoded/{src}_{i}_src.json",
                         "tgt_align_path": f"encoded/{src}_{i}_tgt.json"})
    return rows


def test_subset_size_exact():
    rows = _make_rows()  # 10000
    out = bs.stratified_subset(rows, 1000, random.Random(0))
    assert len(out) == 1000


def test_subset_preserves_source_proportions():
    rows = _make_rows()  # 70/20/10
    out = bs.stratified_subset(rows, 1000, random.Random(0))
    c = Counter(bs.source_of(r["pair_id"]) for r in out)
    # within ~3% of the parent proportions
    assert abs(c["conv"] / 1000 - 0.70) < 0.03
    assert abs(c["flores"] / 1000 - 0.20) < 0.03
    assert abs(c["opus"] / 1000 - 0.10) < 0.03


def test_subset_direction_roughly_balanced():
    rows = _make_rows()
    out = bs.stratified_subset(rows, 1000, random.Random(0))
    c = Counter(r["direction"] for r in out)
    assert abs(c["hi->tr"] - c["tr->hi"]) < 0.1 * 1000  # within 10%


def test_subset_no_duplicates():
    rows = _make_rows()
    out = bs.stratified_subset(rows, 1000, random.Random(0))
    ids = [r["pair_id"] for r in out]
    assert len(ids) == len(set(ids))


def test_subset_n_ge_len_returns_all():
    rows = _make_rows()
    out = bs.stratified_subset(rows, 99999, random.Random(0))
    assert len(out) == len(rows)


def test_subset_deterministic_with_seed():
    rows = _make_rows()
    a = bs.stratified_subset(rows, 1000, random.Random(7))
    b = bs.stratified_subset(rows, 1000, random.Random(7))
    assert [r["pair_id"] for r in a] == [r["pair_id"] for r in b]


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
