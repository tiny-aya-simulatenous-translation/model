"""Text-audio interleaver -- maps word timestamps to audio frame positions.

WHY THIS EXISTS
---------------
The Moshi-style summed-embedding scheme requires the text and audio
streams to be aligned at the *frame* level. Mimi runs at 12.5 Hz so
each frame is 80 ms wide. Given a word-level alignment (``word``,
``start``, ``end`` triples produced by the data pipeline), this
module places the first text token of each word at the frame where
the word begins, and fills the remaining frames with padding tokens.

Ported from ``moshi-finetune/finetune/data/interleaver.py`` and
adapted to use TinyAya's HF tokenizer instead of SentencePiece.

This file does not touch GPU or TPU; it runs entirely on host CPU
during DataLoader iteration.
"""

import json
from collections import deque
from pathlib import Path

import torch

# Special token IDs (must match TinyAyaBackbone)
TEXT_PADDING = 262144
END_OF_TEXT_PADDING = 262145
ZERO_PADDING = 262146
IN_WORD_PADDING = 262147

Alignment = tuple[str, tuple[float, float], str]
TokenizedAlignment = tuple[list[int], tuple[float, float], str]


class Interleaver:
    """Builds a text token stream aligned to audio frames.

    Given word-level timestamps from Whisper, places text tokens at the
    audio frame positions where words are spoken. Between words, inserts
    padding tokens.
    """

    def __init__(
        self,
        tokenizer,
        audio_frame_rate: float = 12.5,
        text_padding: int = TEXT_PADDING,
        end_of_text_padding: int = END_OF_TEXT_PADDING,
        zero_padding: int = ZERO_PADDING,
        in_word_padding: int = IN_WORD_PADDING,
        device: str = "cpu",
    ):
        self.tokenizer = tokenizer
        self.audio_frame_rate = audio_frame_rate
        self.text_padding = text_padding
        self.end_of_text_padding = end_of_text_padding
        self.zero_padding = zero_padding
        self.in_word_padding = in_word_padding
        self.device = device

    def _tokenize(self, alignments: list[Alignment]) -> list[TokenizedAlignment]:
        """Tokenize each word individually."""
        out = []
        for word, ts, speaker in alignments:
            toks = self.tokenizer.encode(word.strip(), add_special_tokens=False)
            if toks:
                out.append((toks, ts, speaker))
        return out

    def _keep_those_with_duration(self, alignments):
        return [a for a in alignments if a[1][0] < a[1][1]]

    def build_token_stream(
        self,
        alignments: list[TokenizedAlignment] | None,
        num_frames: int,
    ) -> torch.Tensor:
        """Build text token stream aligned to audio frames.

        For each audio frame, determines which text token (if any) should
        appear at that position based on word timestamps.
        """
        if alignments is None:
            text_tokens = [self.zero_padding] * num_frames
        else:
            text_tokens = [self.text_padding] * num_frames
            i = 0
            to_append_stack: deque = deque()
            last_word_end = -1

            for t in range(num_frames):
                # Check if any word starts at this frame
                while i < len(alignments) and alignments[i][1][0] * self.audio_frame_rate < t + 1:
                    tokenized = alignments[i][0]
                    last_word_end = int(alignments[i][1][1] * self.audio_frame_rate)
                    to_append_stack = deque(tokenized)
                    i += 1

                if to_append_stack:
                    # Mark end of previous padding
                    if t > 0 and text_tokens[t - 1] in [
                        self.text_padding,
                        self.in_word_padding,
                    ]:
                        text_tokens[t - 1] = self.end_of_text_padding
                    text_tokens[t] = to_append_stack.popleft()
                elif t <= last_word_end:
                    text_tokens[t] = self.in_word_padding

        return torch.tensor(text_tokens, dtype=torch.long, device=self.device)

    def prepare_item(
        self,
        alignments: list[Alignment] | None,
        num_frames: int,
    ) -> torch.Tensor:
        """Process alignments and build token stream.

        Args:
            alignments: list of (word, (start_sec, end_sec), speaker) tuples
            num_frames: number of audio frames

        Returns:
            text_ids: [num_frames] tensor of text token IDs
        """
        if alignments is None:
            tokenized = None
        else:
            tokenized = self._tokenize(sorted(alignments, key=lambda x: x[1][0]))
            tokenized = self._keep_those_with_duration(tokenized)

        return self.build_token_stream(tokenized, num_frames)


def load_alignments(json_path: str | Path) -> list[Alignment]:
    """Load word-level alignments from a JSON file.

    Expected format (from whisper_timestamped / annotate.py):
    {"alignments": [["word", [start, end], "SPEAKER_MAIN"], ...]}
    """
    with open(json_path) as f:
        data = json.load(f)

    alignments = []
    for item in data["alignments"]:
        word = item[0]
        ts = (item[1][0], item[1][1])
        speaker = item[2] if len(item) > 2 else "SPEAKER_MAIN"
        alignments.append((word, ts, speaker))

    return alignments


def dicho(alignment, val, i=0, j=None):
    """Binary search for alignment position."""
    if j is None:
        j = len(alignment)
    if i == j:
        return i
    k = (i + j) // 2
    if alignment[k][1][0] < val:
        return dicho(alignment, val, k + 1, j)
    else:
        return dicho(alignment, val, i, k)
