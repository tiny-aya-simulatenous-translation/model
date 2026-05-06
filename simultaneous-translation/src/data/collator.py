"""Batch collation for interleaved audio-text sequences.

WHY THIS EXISTS
---------------
PyTorch's default collator stacks tensors only when shapes match. Our
samples have variable frame lengths, so :class:`InterleavedCollator`
right-pads each batch element to the longest sequence and emits the
attention mask + per-sample loss mask the loss function expects.

The collator is shared by Stage 1 and Stage 2 datasets. There is no
TPU-specific behaviour here -- padding is done on host CPU and the
DataLoader hands the result to the device just like on GPU.
"""

import torch

# Must match TinyAyaBackbone special tokens
ZERO_PADDING = 262146


class InterleavedCollator:
    """Collates variable-length interleaved audio+text samples into padded batches.

    Works for both Stage 1 (audio understanding) and Stage 2 (translation).
    """

    def __init__(self, audio_pad_id: int = 0, text_pad_id: int = ZERO_PADDING):
        self.audio_pad_id = audio_pad_id
        self.text_pad_id = text_pad_id

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        lengths = [item["num_frames"] for item in batch]
        max_len = max(lengths)
        B = len(batch)

        # Get number of codebooks from first sample
        num_codebooks = batch[0]["audio_codes"].shape[0]

        # Pad audio codes: [B, CB, T_max]
        audio_codes = torch.full((B, num_codebooks, max_len), self.audio_pad_id, dtype=torch.long)
        for i, item in enumerate(batch):
            T = item["audio_codes"].shape[1]
            audio_codes[i, :, :T] = item["audio_codes"]

        # Pad text IDs: [B, T_max]
        text_ids = torch.full((B, max_len), self.text_pad_id, dtype=torch.long)
        for i, item in enumerate(batch):
            T = item["text_ids"].shape[0]
            text_ids[i, :T] = item["text_ids"]

        # Attention mask: [B, T_max]
        attention_mask = torch.zeros(B, max_len, dtype=torch.long)
        for i, T in enumerate(lengths):
            attention_mask[i, :T] = 1

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
                T = item["loss_mask"].shape[0]
                loss_mask[i, :T] = item["loss_mask"]
            result["loss_mask"] = loss_mask

        return result
