"""Mimi codec wrapper for encoding / decoding audio.

WHY THIS EXISTS
---------------
The data pipeline encodes raw waveform into Mimi audio codes once,
offline, and persists the result in ``.pt`` shards consumed by the
training datasets. This module is the helper used by the offline
encoder (``scripts/prepare_data.py``) and by the inference scripts
that need to round-trip audio for ad-hoc demos.

The Mimi model itself is a separate HF checkpoint (``kyutai/mimi``)
and is NOT part of the trained composite. We only ever call its
``encode``/``decode`` methods; its weights are frozen.

GPU vs TPU
----------
The default ``device="cuda"`` is fine for offline encoding on a GPU
host. TPU users should encode their data on a GPU box (faster + the
TPU runtime does not currently support 1-D conv1d kernels well) and
upload the resulting shards to GCS. The training loop never touches
``MimiEncoder`` directly.
"""

import numpy as np
import torch
from scipy.signal import resample_poly
from transformers import AutoFeatureExtractor, MimiModel


class MimiEncoder:
    """Encode/decode audio using the Mimi neural codec."""

    def __init__(self, model_name: str = "kyutai/mimi", device: str = "cuda"):
        self.device = device
        self.model = MimiModel.from_pretrained(model_name).to(device).eval()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.sample_rate = self.feature_extractor.sampling_rate  # 24000
        self.frame_rate = 12.5  # Hz

    @torch.no_grad()
    def encode(self, audio: torch.Tensor, sr: int | None = None) -> torch.Tensor:
        """Encode audio waveform to Mimi codes.

        Args:
            audio: [num_samples] or [1, num_samples] waveform
            sr: sample rate of input (resampled to 24kHz if different)

        Returns:
            codes: [num_codebooks, T] integer codes (0..2047)
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        if sr is not None and sr != self.sample_rate:
            # Resample using scipy (avoids torchaudio CUDA dep issues)
            from math import gcd

            g = gcd(sr, self.sample_rate)
            audio_np = audio.squeeze().numpy()
            audio_np = resample_poly(audio_np, self.sample_rate // g, sr // g)
            audio = torch.from_numpy(audio_np.astype(np.float32)).unsqueeze(0)

        inputs = self.feature_extractor(
            raw_audio=audio.squeeze(0).numpy(),
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        )
        input_values = inputs["input_values"].to(self.device)

        outputs = self.model.encode(input_values)
        codes = outputs.audio_codes.squeeze(0)  # [num_codebooks, T]
        return codes.cpu()

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode Mimi codes back to audio waveform.

        Args:
            codes: [num_codebooks, T] integer codes

        Returns:
            audio: [num_samples] waveform at 24kHz
        """
        if codes.dim() == 2:
            codes = codes.unsqueeze(0)  # [1, num_codebooks, T]
        codes = codes.to(self.device)

        output = self.model.decode(codes)
        # MimiDecoderOutput has .audio_values attribute
        audio = output.audio_values if hasattr(output, "audio_values") else output
        return audio.squeeze().cpu()
