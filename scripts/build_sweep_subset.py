"""Build a small, balanced subset of a training split for the v0.3 sweep.

WHY THIS EXISTS
---------------
The full synthetic corpus is ~1.24M samples (~4M files). The Phase-E sweep only
needs enough data to *rank* hyperparameters, and an 8-wide v6e-8 fleet would
otherwise stage the full corpus 8x (HF rate-limits + disk). This script carves a
~200K subset that is:

  * **source-stratified** — keeps the natural conv/flores/opus proportions, so the
    sweep sees the same data mix as the full run; and
  * **direction-balanced** — ~50/50 hi->tr / tr->hi within each source.

The subset JSONL references the same ``encoded/<name>.pt`` files as the parent
split, so staging only needs those files (see scripts/tpu/stage_sweep_subset.sh).

Pure-Python / torch-free so the sampling logic is unit-tested in CI.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict


def source_of(pair_id: str) -> str:
    """Coarse data source from a pair_id (conv_*, flores_*, opus_*)."""
    pid = pair_id or ""
    if pid.startswith("conv_"):
        return "conv"
    if pid.startswith("flores"):
        return "flores"
    if pid.startswith("opus"):
        return "opus"
    if pid.startswith("fleurs"):
        return "fleurs"
    return "other"


def stratified_subset(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Sample ``n`` rows, stratified by source and balanced by direction.

    Per-source quota is proportional to that source's share of ``rows``; within a
    source the quota is split evenly across its directions. Sampling is without
    replacement; if a (source, direction) cell is short, the deficit is back-filled
    from the global remainder so the result still totals ``min(n, len(rows))``.
    """
    if n >= len(rows):
        out = list(rows)
        rng.shuffle(out)
        return out

    by_source: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_source[source_of(r.get("pair_id", ""))].append(r)

    total = len(rows)
    picked: list[dict] = []
    chosen_ids: set[int] = set()

    for source, srows in by_source.items():
        quota = round(n * len(srows) / total)
        by_dir: dict[str, list[dict]] = defaultdict(list)
        for r in srows:
            by_dir[r.get("direction", "?")].append(r)
        dirs = list(by_dir)
        per_dir = max(1, quota // max(1, len(dirs)))
        for d in dirs:
            pool = by_dir[d]
            k = min(per_dir, len(pool))
            for r in rng.sample(pool, k):
                picked.append(r)
                chosen_ids.add(id(r))

    # Back-fill / trim to exactly min(n, len(rows)).
    target = min(n, len(rows))
    if len(picked) < target:
        remainder = [r for r in rows if id(r) not in chosen_ids]
        rng.shuffle(remainder)
        picked.extend(remainder[: target - len(picked)])
    elif len(picked) > target:
        rng.shuffle(picked)
        picked = picked[:target]
    rng.shuffle(picked)
    return picked


def _read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _composition(rows: list[dict]) -> dict:
    src = defaultdict(int)
    direction = defaultdict(int)
    for r in rows:
        src[source_of(r.get("pair_id", ""))] += 1
        direction[r.get("direction", "?")] += 1
    return {"sources": dict(src), "directions": dict(direction)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="parent split jsonl (e.g. train.jsonl)")
    p.add_argument("--output", required=True, help="subset jsonl to write")
    p.add_argument("--n", type=int, default=200_000, help="target subset size")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--filelist", default=None,
                   help="optional: write the subset's encoded file paths here (for staging)")
    args = p.parse_args()

    rows = _read_jsonl(args.input)
    subset = stratified_subset(rows, args.n, random.Random(args.seed))
    with open(args.output, "w") as f:
        for r in subset:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(subset)} rows -> {args.output}")
    print(f"  parent: {len(rows)}  composition: {_composition(rows)}")
    print(f"  subset: {len(subset)}  composition: {_composition(subset)}")

    if args.filelist:
        keys = ("pt_path", "src_align_path", "tgt_align_path")
        with open(args.filelist, "w") as f:
            for r in subset:
                for k in keys:
                    if r.get(k):
                        f.write(r[k] + "\n")
        print(f"  wrote file list -> {args.filelist}")


if __name__ == "__main__":
    main()
