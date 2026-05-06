"""Dataset classes for Stage 1 (audio understanding) and Stage 2 (translation).

WHY THIS EXISTS
---------------
Stage 2 ingests pre-encoded ``.pt`` files holding Mimi audio codes
plus optional word-level alignment JSONs. This module exposes two
``torch.utils.data.Dataset`` subclasses:

* :class:`InterleavedAudioDataset` -- Stage 1 single-language
  audio-understanding dataset.
* :class:`StreamingTranslationDataset` (defined later in the file)
  -- Stage 2 streaming translation dataset that lazily reads
  ``.pt`` shards from local disk or GCS.

The dataset code is device-agnostic: tensors come out of ``__getitem__``
on the host CPU and the collator + DataLoader move them onto the
device. On TPU SPMD the device move happens implicitly via
``backend.mark_sharding`` in the training loop.
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .interleaver import Interleaver, load_alignments


class InterleavedAudioDataset(Dataset):
    """Stage 1: Loads pre-encoded audio + alignment JSONs.

    Each .pt file contains:
        audio_codes: [num_codebooks, T]
        text: str (transcript)
        language: str
        duration_s: float

    Each .json file (same stem) contains word-level alignments from Whisper.
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer,
        max_frames: int = 250,
        audio_frame_rate: float = 12.5,
    ):
        self.data_dir = Path(data_dir)
        self.max_frames = max_frames
        self.interleaver = Interleaver(tokenizer, audio_frame_rate=audio_frame_rate)

        # Find all .pt files
        self.samples = sorted(self.data_dir.glob("**/*.pt"))
        if not self.samples:
            raise FileNotFoundError(f"No .pt files found in {data_dir}")
        print(f"InterleavedAudioDataset: {len(self.samples)} samples from {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pt_path = self.samples[idx]
        data = torch.load(pt_path, weights_only=False)

        audio_codes = data["audio_codes"]  # [num_codebooks, T]
        T = audio_codes.shape[1]

        # Truncate if needed
        if T > self.max_frames:
            audio_codes = audio_codes[:, : self.max_frames]
            T = self.max_frames

        # Load alignment JSON (same stem as .pt)
        json_path = pt_path.with_suffix(".json")
        if json_path.exists():
            alignments = load_alignments(json_path)
            text_ids = self.interleaver.prepare_item(alignments, T)
        else:
            # No alignment file — use zero_padding (no text)
            text_ids = torch.full((T,), self.interleaver.zero_padding, dtype=torch.long)

        return {
            "audio_codes": audio_codes,  # [num_codebooks, T]
            "text_ids": text_ids,  # [T]
            "num_frames": T,
            "language": data.get("language", "unknown"),
        }


class TranslationDataset(Dataset):
    """Stage 2: Loads paired source + target pre-encoded audio.

    Each .pt file contains:
        source_audio_codes: [num_codebooks, T_src]
        target_audio_codes: [num_codebooks, T_tgt]
        source_text: str
        target_text: str
        source_language: str
        target_language: str
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer,
        max_frames: int = 400,
        audio_frame_rate: float = 12.5,
    ):
        self.data_dir = Path(data_dir)
        self.max_frames = max_frames
        self.interleaver = Interleaver(tokenizer, audio_frame_rate=audio_frame_rate)

        self.samples = sorted(self.data_dir.glob("**/*.pt"))
        if not self.samples:
            raise FileNotFoundError(f"No .pt files found in {data_dir}")
        print(f"TranslationDataset: {len(self.samples)} samples from {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pt_path = self.samples[idx]
        data = torch.load(pt_path, weights_only=False)

        src_codes = data["source_audio_codes"]  # [CB, T_src]
        tgt_codes = data["target_audio_codes"]  # [CB, T_tgt]

        T_src = src_codes.shape[1]
        T_tgt = tgt_codes.shape[1]

        # Truncate to fit max_frames
        total = T_src + T_tgt
        if total > self.max_frames:
            # Give at least 50% to source
            max_src = max(self.max_frames // 2, self.max_frames - T_tgt)
            T_src = min(T_src, max_src)
            T_tgt = min(T_tgt, self.max_frames - T_src)
            src_codes = src_codes[:, :T_src]
            tgt_codes = tgt_codes[:, :T_tgt]

        # Concatenate source + target
        combined_codes = torch.cat([src_codes, tgt_codes], dim=1)  # [CB, T_total]
        T_total = combined_codes.shape[1]

        # Load alignments if available. ``src_json`` is intentionally
        # not consumed (source text is teacher-forced as zero-padding
        # below); we resolve it anyway so the path layout is documented
        # right next to ``tgt_json`` and a future regression that
        # introduces source-side teacher forcing has a one-line edit.
        src_json = pt_path.with_name(pt_path.stem + "_source.json")  # noqa: F841
        tgt_json = pt_path.with_name(pt_path.stem + "_target.json")

        # Source text (zero padding for now — source is teacher-forced)
        src_text = torch.full((T_src,), self.interleaver.zero_padding, dtype=torch.long)

        # Target text with alignment
        if tgt_json.exists():
            tgt_alignments = load_alignments(tgt_json)
            tgt_text = self.interleaver.prepare_item(tgt_alignments, T_tgt)
        else:
            tgt_text = torch.full((T_tgt,), self.interleaver.zero_padding, dtype=torch.long)

        text_ids = torch.cat([src_text, tgt_text])  # [T_total]

        # Loss mask: 1 for target positions only
        loss_mask = torch.zeros(T_total, dtype=torch.long)
        loss_mask[T_src:] = 1

        return {
            "audio_codes": combined_codes,  # [CB, T_total]
            "text_ids": text_ids,  # [T_total]
            "loss_mask": loss_mask,  # [T_total]
            "num_frames": T_total,
            "source_length": T_src,
            "target_length": T_tgt,
        }


class StreamingTranslationDataset(Dataset):
    """Stage 2 streaming dataset: one .pt + two alignment JSONs per row.

    Source of truth is a split JSONL written by scripts/make_splits.py. Each row:
        {"pt_path": "...", "src_align_path": "...", "tgt_align_path": "...",
         "direction": "tr->hi"|"hi->tr", "sentence_id": int,
         "source_type": "fleurs_real"|"fleurs_tts", "tts_voice": "..."}

    __getitem__ loads one .pt (torch.load, mmap when supported) + two JSONs, runs
    Interleaver on src/tgt, concatenates prefix-LM style, and builds a
    target-only loss_mask. No preloading.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer,
        max_frames: int = 300,
        audio_frame_rate: float = 12.5,
        encoded_dir: str | Path | None = None,
    ):
        self.jsonl_path = Path(jsonl_path)
        self.max_frames = max_frames
        self.interleaver = Interleaver(tokenizer, audio_frame_rate=audio_frame_rate)
        self.encoded_dir = Path(encoded_dir) if encoded_dir else None

        self.rows = []
        with open(self.jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))
        if not self.rows:
            raise FileNotFoundError(f"No rows in {self.jsonl_path}")
        print(f"StreamingTranslationDataset: {len(self.rows)} rows from {self.jsonl_path}")

    def __len__(self):
        return len(self.rows)

    def _resolve(self, p: str) -> Path:
        pp = Path(p)
        if pp.exists():
            return pp
        if self.encoded_dir:
            cand = self.encoded_dir / pp.name
            if cand.exists():
                return cand
        return pp

    def __getitem__(self, idx):
        row = self.rows[idx]
        pt_path = self._resolve(row["pt_path"])
        try:
            data = torch.load(pt_path, weights_only=False, mmap=True)
        except (TypeError, RuntimeError):
            data = torch.load(pt_path, weights_only=False)

        src_codes = data["src_codes"].long()  # [CB, T_src]
        tgt_codes = data["tgt_codes"].long()  # [CB, T_tgt]
        T_src = src_codes.shape[1]
        T_tgt = tgt_codes.shape[1]

        # Load alignments
        src_align_path = self._resolve(row["src_align_path"])
        tgt_align_path = self._resolve(row["tgt_align_path"])
        src_align = load_alignments(src_align_path) if src_align_path.exists() else None
        tgt_align = load_alignments(tgt_align_path) if tgt_align_path.exists() else None

        src_text = (
            self.interleaver.prepare_item(src_align, T_src)
            if src_align is not None
            else torch.full((T_src,), self.interleaver.zero_padding, dtype=torch.long)
        )
        tgt_text = (
            self.interleaver.prepare_item(tgt_align, T_tgt)
            if tgt_align is not None
            else torch.full((T_tgt,), self.interleaver.zero_padding, dtype=torch.long)
        )

        # Truncate to max_frames — target first; preserve source
        total = T_src + T_tgt
        if total > self.max_frames:
            new_T_tgt = max(1, self.max_frames - T_src)
            if T_src > self.max_frames - 1:
                # very long source: preserve at least 1 frame of target
                T_src = self.max_frames - 1
                src_codes = src_codes[:, :T_src]
                src_text = src_text[:T_src]
                new_T_tgt = 1
            tgt_codes = tgt_codes[:, :new_T_tgt]
            tgt_text = tgt_text[:new_T_tgt]
            T_tgt = new_T_tgt

        audio_codes = torch.cat([src_codes, tgt_codes], dim=1).contiguous()
        text_ids = torch.cat([src_text, tgt_text]).contiguous()
        T_total = audio_codes.shape[1]
        loss_mask = torch.zeros(T_total, dtype=torch.long)
        loss_mask[T_src:] = 1

        return {
            "audio_codes": audio_codes,
            "text_ids": text_ids,
            "loss_mask": loss_mask,
            "num_frames": T_total,
            "source_length": T_src,
            "target_length": T_tgt,
            "sentence_id": row.get("sentence_id", -1),
            "direction": row.get("direction", ""),
        }
