"""Batch collation for interleaved audio-text sequences.

WHY THIS EXISTS
---------------
PyTorch's default collator stacks tensors only when shapes match. Our
samples have variable frame lengths, so :class:`InterleavedCollator`
right-pads each batch element to the longest sequence and emits the
attention mask + per-sample loss mask the loss function expects.

For TPU/XLA callers, the collator can also pad the batch axis itself.
That keeps the SPMD graph static even for the final short DataLoader
batch at an epoch boundary.

The collator is shared by Stage 1 and Stage 2 datasets. Padding is done
on host CPU and the DataLoader hands the result to the device just like
on GPU.
"""

import random
from collections.abc import Sequence

import torch

from src.data.spec_augment import apply_spec_augment

# Must match TinyAyaBackbone special tokens
ZERO_PADDING = 262146
SILENCE_TOKEN = 2048  # audio_vocab_size; pads the parallel-stream audio codes


class InterleavedCollator:
    """Collates variable-length interleaved audio+text samples into padded batches.

    Works for both Stage 1 (audio understanding) and Stage 2 (translation).
    """

    def __init__(
        self,
        audio_pad_id: int = 0,
        text_pad_id: int = ZERO_PADDING,
        pad_to: int | Sequence[int] | None = None,
        batch_pad_to: int | None = None,
        expected_num_codebooks: int | None = None,
        spec_augment: dict | None = None,
    ):
        self.audio_pad_id = audio_pad_id
        self.text_pad_id = text_pad_id
        self.pad_to = tuple(pad_to) if isinstance(pad_to, Sequence) else pad_to
        self.batch_pad_to = batch_pad_to
        self.expected_num_codebooks = expected_num_codebooks
        # Phase E: SpecAugment-on-codes (train collator only; default off). When
        # enabled, masks the INPUT user-audio stream (never targets/val).
        self.spec_augment = spec_augment if (spec_augment and spec_augment.get("enabled")) else None
        self._sa_rng = random.Random() if self.spec_augment else None

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        if not batch:
            raise ValueError("InterleavedCollator received an empty batch")

        lengths = [item["num_frames"] for item in batch]
        # When pad_to is set, every batch is padded to a static length so the
        # XLA tracer compiles a bounded set of graphs. An integer means one
        # fixed shape; a sequence means "round this batch up to the smallest
        # allowed bucket" (GPU analogue: dynamic per-batch padding).
        if isinstance(self.pad_to, tuple):
            batch_max = max(lengths)
            max_len = next(
                (bucket for bucket in self.pad_to if batch_max <= bucket), self.pad_to[-1]
            )
        else:
            max_len = self.pad_to if self.pad_to is not None else max(lengths)
        # Truncate any over-long sample (defensive; dataset already truncates)
        if self.pad_to is not None:
            lengths = [min(length, max_len) for length in lengths]
        real_b = len(batch)
        B = self.batch_pad_to if self.batch_pad_to is not None else real_b
        if real_b > B:
            raise ValueError(f"batch has {real_b} rows but batch_pad_to={B}")

        # Get number of codebooks from first sample
        num_codebooks = self.expected_num_codebooks or batch[0]["audio_codes"].shape[0]

        # Pad audio codes: [B, CB, T_max]
        audio_codes = torch.full((B, num_codebooks, max_len), self.audio_pad_id, dtype=torch.long)
        for i, item in enumerate(batch):
            T = min(item["audio_codes"].shape[1], max_len)
            if item["audio_codes"].shape[0] < num_codebooks:
                raise ValueError(
                    f"sample has {item['audio_codes'].shape[0]} codebooks; "
                    f"expected at least {num_codebooks}"
                )
            audio_codes[i, :, :T] = item["audio_codes"][:num_codebooks, :T]

        # Pad text IDs: [B, T_max]
        text_ids = torch.full((B, max_len), self.text_pad_id, dtype=torch.long)
        for i, item in enumerate(batch):
            T = min(item["text_ids"].shape[0], max_len)
            text_ids[i, :T] = item["text_ids"][:T]

        # Attention mask: [B, T_max]
        attention_mask = torch.zeros(B, max_len, dtype=torch.long)
        for i, T in enumerate(lengths):
            attention_mask[i, :T] = 1
        if B > real_b:
            lengths = lengths + [0] * (B - real_b)

        result = {
            "audio_codes": audio_codes,
            "text_ids": text_ids,
            "attention_mask": attention_mask,
            "lengths": torch.tensor(lengths, dtype=torch.long),
        }

        # Loss mask (for Stage 2 translation)
        if "loss_mask" in batch[0]:
            loss_mask = torch.zeros(B, max_len, dtype=torch.long)
            for i, item in enumerate(batch):
                T = min(item["loss_mask"].shape[0], max_len)
                loss_mask[i, :T] = item["loss_mask"][:T]
            result["loss_mask"] = loss_mask

        # Parallel two-stream codes (Moshi-style). The dataset builds these
        # (user = source+silence, model = silence+target, with codebook delay)
        # but WITHOUT passing them here the training/val forward fell back to
        # single-stream mode -- model_audio_embed was never called (grad 0.00).
        # Pad like audio_codes, with SILENCE in the padding so the forward's
        # is-silence handling stays correct.
        for key in ("user_audio_codes", "model_audio_codes"):
            if key in batch[0]:
                stream = torch.full(
                    (B, num_codebooks, max_len), SILENCE_TOKEN, dtype=torch.long
                )
                for i, item in enumerate(batch):
                    T = min(item[key].shape[1], max_len)
                    stream[i, :, :T] = item[key][:num_codebooks, :T]
                result[key] = stream

        # Phase E: SpecAugment masks the INPUT (user) stream only, capped to each
        # sample's real frame count so padding stays SILENCE. Off unless enabled
        # on this collator (train loader).
        if self.spec_augment is not None and "user_audio_codes" in result:
            apply_spec_augment(
                result["user_audio_codes"], self.spec_augment, self._sa_rng,
                silence_token=SILENCE_TOKEN, lengths=lengths,
            )

        return result
