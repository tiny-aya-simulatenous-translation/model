"""Whisper word-level alignment for the Moshi-style interleaver.

Writes two sidecar JSONs per accepted pair:
    {pair_id}_{src}{tgt}.src.alignments.json
    {pair_id}_{src}{tgt}.tgt.alignments.json

Schema: {"alignments": [[word, [start_sec, end_sec], "SPEAKER_MAIN"], ...]}

Implementation notes:
- Uses openai-whisper via whisper_timestamped (transformers 5.0 backend is
  broken on WhisperConfig.max_length — filed upstream, skipping).
- Deduplicates by wav path across pairs to avoid aligning the same FLEURS clip
  twice when it is the source of multiple TTS variants.
- Model: whisper-base for both TR and HI. base is chosen for throughput
  (~0.4 s/file on G4 vs ~1 s/file for large-v3); the interleaver only needs
  coarse word boundaries, not near-perfect transcripts.
"""

from __future__ import annotations

import gc
import json
import shutil
from pathlib import Path

import librosa
import torch
import whisper_timestamped as wt
from tqdm import tqdm

from src.config import PipelineConfig
from src.manifest import read_manifest


MODEL_NAME = "base"


def _load_wav_16k(path: str):
    y, _ = librosa.load(path, sr=16000, mono=True)
    return y


def _align_one(model, wav, language: str):
    try:
        r = wt.transcribe_timestamped(
            model, wav, language=language, vad=False, verbose=None,
            beam_size=1, best_of=1, detect_disfluencies=False,
        )
    except Exception:
        r = wt.transcribe(model, wav, language=language)
    out = []
    for seg in r.get("segments", []):
        for w in seg.get("words", []):
            text = (w.get("text") or "").strip()
            start = float(w.get("start", 0.0))
            end = float(w.get("end", start))
            if not text or end <= start:
                continue
            out.append([text, [start, end], "SPEAKER_MAIN"])
    return out


def align_manifest(config: PipelineConfig, batch_size: int = 8) -> None:
    entries = read_manifest(config.manifest_dir / "accepted_manifest.jsonl")
    print(f"Accepted entries: {len(entries)}")
    encoded_dir = config.encoded_dir

    # (wav_path, lang) → list of output json paths to write
    buckets: dict[tuple[str, str], list[Path]] = {}
    for e in entries:
        stem = f"{e.pair_id}_{e.src_lang}{e.tgt_lang}"
        src_out = encoded_dir / f"{stem}.src.alignments.json"
        tgt_out = encoded_dir / f"{stem}.tgt.alignments.json"
        buckets.setdefault((e.src_audio_path, e.src_lang), []).append(src_out)
        buckets.setdefault((e.tgt_audio_path, e.tgt_lang), []).append(tgt_out)

    # Stage by language
    by_lang: dict[str, list[tuple[str, list[Path]]]] = {"tr": [], "hi": []}
    for (wav, lang), outs in buckets.items():
        if lang not in by_lang:
            continue
        # filter to work that is actually missing
        outs = [o for o in outs if not o.exists()]
        if outs:
            by_lang[lang].append((wav, outs))

    for lang, work in by_lang.items():
        if not work:
            continue
        print(f"[{lang}] unique wavs to align: {len(work)}")
        model = wt.load_model(MODEL_NAME, device="cuda", backend="openai-whisper")
        first_out_for_wav: dict[str, Path] = {}
        for wav_path, outs in tqdm(work, desc=f"whisper-{lang}"):
            try:
                wav = _load_wav_16k(wav_path)
                aln = _align_one(model, wav, lang)
            except Exception as ex:
                print(f"  skip {wav_path}: {ex}")
                aln = []
            # Write first, then hardlink/copy rest
            first = outs[0]
            first.parent.mkdir(parents=True, exist_ok=True)
            with open(first, "w") as f:
                json.dump({"alignments": aln}, f, ensure_ascii=False)
            first_out_for_wav[wav_path] = first
            for extra in outs[1:]:
                extra.parent.mkdir(parents=True, exist_ok=True)
                try:
                    if extra.exists():
                        extra.unlink()
                    extra.hardlink_to(first)
                except OSError:
                    shutil.copy2(first, extra)
        del model
        gc.collect()
        torch.cuda.empty_cache()
