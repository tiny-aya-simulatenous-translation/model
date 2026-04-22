import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm

from src.config import PipelineConfig
from src.manifest import ManifestEntry, read_manifest


def _load_mimi(model_id: str, device: str = "cuda"):
    """Load Mimi codec model."""
    from transformers import AutoModel
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    model = model.to(device).eval()
    return model


def _encode_audio(model, audio_path: str, num_codebooks: int, device: str = "cuda") -> torch.Tensor:
    """Encode audio file to Mimi codebook tokens. Returns [num_codebooks, T]."""
    wav, sr = torchaudio.load(audio_path)
    # Mimi expects 24kHz mono
    if sr != 24000:
        wav = torchaudio.functional.resample(wav, sr, 24000)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)

    wav = wav.unsqueeze(0).to(device)  # [1, 1, samples]
    with torch.no_grad():
        out = model.encode(wav, num_quantizers=num_codebooks)
    codes = out.audio_codes if hasattr(out, "audio_codes") else out[0]
    return codes[0, :num_codebooks, :].cpu()  # [num_codebooks, T]


def encode_mimi(config: PipelineConfig) -> None:
    """Encode accepted pairs through Mimi codec. Saves .pt files."""
    input_path = config.manifest_dir / "accepted_manifest.jsonl"
    entries = read_manifest(input_path)
    print(f"Loaded {len(entries)} entries from {input_path}")

    config.ensure_dirs()
    model = _load_mimi(config.mimi_model_id)
    num_cb = config.mimi_num_codebooks

    encoded_count = 0
    for entry in tqdm(entries, desc="Encoding with Mimi"):
        out_path = config.encoded_dir / f"{entry.pair_id}_{entry.src_lang}{entry.tgt_lang}.pt"
        if out_path.exists():
            continue

        src_codes = _encode_audio(model, entry.src_audio_path, num_cb)
        tgt_codes = _encode_audio(model, entry.tgt_audio_path, num_cb)

        torch.save({
            "pair_id": entry.pair_id,
            "src_lang": entry.src_lang,
            "tgt_lang": entry.tgt_lang,
            "src_text": entry.src_text,
            "tgt_text": entry.tgt_text,
            "src_codes": src_codes.cpu(),
            "tgt_codes": tgt_codes.cpu(),
        }, out_path)
        encoded_count += 1

    print(f"Encoded {encoded_count} new pairs to {config.encoded_dir}")
