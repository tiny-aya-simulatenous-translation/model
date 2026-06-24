"""Static-shape bucket sampling for TPU training.

WHY THIS EXISTS
---------------
On GPU, each mini-batch can pad to its own longest example and cuDNN
will execute that dynamic shape directly. TPU/XLA instead compiles a
separate graph for every new shape, so Stage 2 normally pads all examples
to `max_frames=400`. This sampler is an opt-in middle ground: it groups
examples into a small set of static sequence-length buckets and yields
whole gradient-accumulation macro-steps from one bucket at a time. XLA
then sees only a few prewarmed shapes instead of one shape per batch.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence

from torch.utils.data import Sampler


def normalize_buckets(bucket_frames: Sequence[int], *, max_frames: int) -> tuple[int, ...]:
    """Return sorted unique buckets that include `max_frames`."""
    buckets = tuple(sorted({int(b) for b in bucket_frames}))
    if not buckets:
        raise ValueError("bucket_frames must contain at least one bucket")
    if buckets[-1] != int(max_frames):
        raise ValueError(
            f"largest bucket_frames entry must equal max_frames={max_frames}; got {buckets}"
        )
    if buckets[0] <= 0:
        raise ValueError(f"bucket_frames must be positive; got {buckets}")
    return buckets


def bucket_for_length(length: int, buckets: Sequence[int]) -> int:
    """Return the smallest configured bucket that can hold `length` frames."""
    for bucket in buckets:
        if length <= bucket:
            return int(bucket)
    return int(buckets[-1])


class BucketedMacroBatchSampler(Sampler[list[int]]):
    """Yield static bucketed micro-batches aligned to macro-step boundaries."""

    def __init__(
        self,
        lengths: Sequence[int],
        bucket_frames: Sequence[int],
        *,
        batch_size: int,
        grad_accum: int,
        seed: int = 42,
        shuffle: bool = True,
        warmup_first: bool = True,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive; got {batch_size}")
        if grad_accum <= 0:
            raise ValueError(f"grad_accum must be positive; got {grad_accum}")
        if not lengths:
            raise ValueError("lengths must be non-empty")

        self.lengths = [int(length) for length in lengths]
        self.bucket_frames = tuple(int(bucket) for bucket in bucket_frames)
        self.batch_size = int(batch_size)
        self.grad_accum = int(grad_accum)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.warmup_first = bool(warmup_first)
        self.epoch = 0

        self._macro_size = self.batch_size * self.grad_accum
        self.bucket_counts = Counter(
            bucket_for_length(length, self.bucket_frames) for length in self.lengths
        )
        self._num_macro_groups = sum(
            math.ceil(count / self._macro_size) for count in self.bucket_counts.values()
        )
        self.padded_examples_per_epoch = self._num_macro_groups * self._macro_size

    def __len__(self) -> int:
        return self._num_macro_groups * self.grad_accum

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        by_bucket: dict[int, list[int]] = defaultdict(list)
        for idx, length in enumerate(self.lengths):
            by_bucket[bucket_for_length(length, self.bucket_frames)].append(idx)

        warmup_groups: list[tuple[int, list[int]]] = []
        macro_groups: list[tuple[int, list[int]]] = []
        for bucket in self.bucket_frames:
            indices = by_bucket.get(bucket, [])
            if not indices:
                continue
            indices = list(indices)
            if self.shuffle:
                rng.shuffle(indices)

            remainder = len(indices) % self._macro_size
            if remainder:
                needed = self._macro_size - remainder
                repeats = (needed + len(indices) - 1) // len(indices)
                indices.extend((indices * repeats)[:needed])

            groups = [
                (bucket, indices[start : start + self._macro_size])
                for start in range(0, len(indices), self._macro_size)
            ]
            if self.warmup_first and groups:
                warmup_groups.append(groups[0])
                macro_groups.extend(groups[1:])
            else:
                macro_groups.extend(groups)

        if self.shuffle:
            rng.shuffle(macro_groups)
        ordered = warmup_groups + macro_groups

        for _bucket, group in ordered:
            for start in range(0, len(group), self.batch_size):
                yield group[start : start + self.batch_size]
