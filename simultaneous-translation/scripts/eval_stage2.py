"""Final eval for Stage 2: ASR-BLEU + DNSMOS + demo wavs.

WHY THIS EXISTS
---------------
This script is the canonical evaluation harness for a trained
Stage-2 checkpoint. It loads the composite, runs autoregressive
generation across the validation split, and reports two metrics:

* ASR-BLEU -- transcribe the generated target audio with Whisper,
  BLEU-score the transcripts against ground-truth target text.
* DNSMOS -- mean-opinion-score predictor for naturalness.

It also dumps a handful of source / generated / reference wav demos
under ``--out_dir`` for human listening.

GPU-only by design
------------------
Eval runs on a GPU box (typically the same machine the data pipeline
ran on). Loading the composite onto a TPU just for inference is
overkill: the autoregressive generation loop is sequential and cannot
benefit from SPMD sharding. Train on TPU, evaluate on GPU.

Usage::

    python scripts/eval_stage2.py --checkpoint checkpoints/stage2_scale/best_by_val \\
        --val_split data/splits/val.jsonl --out_dir eval_outputs/stage2_scale
"""

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import StreamingTranslationDataset
from src.data.mimi_encoder import MimiEncoder
from src.model.backbone import TinyAyaBackbone
from src.model.composite import TinyAyaMoshiComposite
from src.model.lora_setup import apply_lora, register_embedding_grad_mask
from src.training.checkpointing import load_checkpoint

# ---- text normalization (from hindi-tts-probe) ------------------------------

PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
SPACE_RE = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFC", s).lower().strip()
    s = PUNCT_RE.sub(" ", s)
    return SPACE_RE.sub(" ", s).strip()


# ---- autoregressive hierarchical generation ---------------------------------


@torch.no_grad()
def ar_generate(model, src_codes, device, num_codebooks, max_new_frames: int):
    """Generate target audio codes autoregressively (hierarchical)."""
    src = src_codes[0:1, :].to(device)
    T_src = src.shape[1]
    generated_cb0 = src.clone()
    text_ids = torch.full((1, T_src), TinyAyaBackbone.ZERO_PADDING, dtype=torch.long, device=device)
    all_gen = []

    for _ in range(max_new_frames):
        mask = torch.ones(1, generated_cb0.shape[1], dtype=torch.long, device=device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            backbone_out = model.backbone(
                text_ids=text_ids, audio_codes=generated_cb0, attention_mask=mask
            )
            projected = model.projection(backbone_out["hidden_states"])
            ctx = projected[:, -1:, :].expand(1, num_codebooks, -1).contiguous()

            depth_input = torch.zeros(1, num_codebooks, dtype=torch.long, device=device)
            frame = []
            for cb_idx in range(num_codebooks):
                depth_out = model.depth_decoder(
                    input_ids=depth_input,
                    last_hidden_state=ctx,
                    use_cache=False,
                    return_dict=True,
                )
                tok = depth_out.logits[0, cb_idx, :].argmax(dim=-1)
                frame.append(tok.cpu())
                if cb_idx + 1 < num_codebooks:
                    depth_input[0, cb_idx + 1] = tok
        next_tokens = torch.stack(frame)
        all_gen.append(next_tokens)
        next_cb0 = next_tokens[0].unsqueeze(0).unsqueeze(0).to(device)
        generated_cb0 = torch.cat([generated_cb0, next_cb0], dim=1)
        text_ids = torch.cat(
            [
                text_ids,
                torch.full((1, 1), TinyAyaBackbone.ZERO_PADDING, dtype=torch.long, device=device),
            ],
            dim=1,
        )
    return torch.stack(all_gen, dim=1)  # [CB, T]


# ---- Whisper ASR ------------------------------------------------------------


def load_asr(lang: str, device: str):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    model_id = {"tr": "openai/whisper-large-v3", "hi": "vasista22/whisper-hindi-large-v2"}[lang]
    proc = WhisperProcessor.from_pretrained(model_id)
    mdl = (
        WhisperForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16)
        .to(device)
        .eval()
    )
    mdl.generation_config.forced_decoder_ids = proc.get_decoder_prompt_ids(
        language=lang, task="transcribe"
    )
    return proc, mdl


@torch.no_grad()
def asr_transcribe(proc, mdl, wav_np: np.ndarray, sr: int, device: str) -> str:
    if sr != 16000:
        wav_t = torchaudio.functional.resample(torch.from_numpy(wav_np).float(), sr, 16000).numpy()
    else:
        wav_t = wav_np
    inputs = proc(wav_t, sampling_rate=16000, return_tensors="pt")
    feats = inputs.input_features.to(device, dtype=torch.float16)
    gen = mdl.generate(feats, max_new_tokens=440)
    return proc.batch_decode(gen, skip_special_tokens=True)[0].strip()


# ---- main -------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--val_split", required=True)
    ap.add_argument("--encoded_dir", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--num_demos", type=int, default=5)
    ap.add_argument("--max_new_frames_cap", type=int, default=300)
    ap.add_argument("--num_codebooks", type=int, default=8)
    args = ap.parse_args()

    device = "cuda"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    demos_dir = out_dir / "demos"
    demos_dir.mkdir(exist_ok=True)

    # ---- load model
    print("Loading model...")
    model = TinyAyaMoshiComposite(num_codebooks=args.num_codebooks)
    model.backbone = apply_lora(model.backbone, r=16)
    register_embedding_grad_mask(model.backbone)
    model = model.to(device)
    load_checkpoint(model, None, None, args.checkpoint)
    model.eval()

    # ---- dataset
    ds = StreamingTranslationDataset(
        args.val_split,
        model.backbone.tokenizer,
        max_frames=10_000,  # no truncation at eval
        encoded_dir=args.encoded_dir,
    )
    rows = ds.rows
    if args.max_samples:
        rows = rows[: args.max_samples]

    # ---- load ASR models (both, since both directions present)
    asr = {
        "tr": load_asr("tr", device),
        "hi": load_asr("hi", device),
    }

    mimi = MimiEncoder(device=device)

    # ---- iterate
    per_sample = []
    dnsmos_gen = []
    dnsmos_gt = []
    demos_per_dir = {"tr->hi": 0, "hi->tr": 0}

    for i, row in enumerate(tqdm(rows, desc="eval")):
        try:
            d = torch.load(row["pt_path"], weights_only=False)
        except Exception:
            continue
        src_codes = d["src_codes"].long()
        tgt_codes = d["tgt_codes"].long()
        direction = row["direction"]  # "tr->hi"
        tgt_lang = direction.split("->")[1]

        max_new = min(args.max_new_frames_cap, max(10, int(tgt_codes.shape[1] * 1.2)))
        gen_codes = ar_generate(model, src_codes, device, args.num_codebooks, max_new)

        src_wav = mimi.decode(src_codes).numpy()
        tgt_wav = mimi.decode(tgt_codes).numpy()
        gen_wav = mimi.decode(gen_codes).numpy()

        # ASR on generated + reference text
        proc, mdl = asr[tgt_lang]
        hyp = asr_transcribe(proc, mdl, gen_wav, 24000, device)
        ref = d.get("tgt_text", "")

        # DNSMOS
        try:
            from speechmos import dnsmos

            s_gen = dnsmos.run(gen_wav.astype(np.float32), sr=24000)
            s_gt = dnsmos.run(tgt_wav.astype(np.float32), sr=24000)
            dnsmos_gen.append([s_gen["sig"], s_gen["bak"], s_gen["ovrl"]])
            dnsmos_gt.append([s_gt["sig"], s_gt["bak"], s_gt["ovrl"]])
        except Exception:
            pass

        per_sample.append(
            {
                "pair_id": d.get("pair_id"),
                "sentence_id": row.get("sentence_id"),
                "direction": direction,
                "ref": ref,
                "hyp": hyp,
                "ref_norm": normalize_text(ref),
                "hyp_norm": normalize_text(hyp),
            }
        )

        # demos — save first N per direction
        if demos_per_dir.get(direction, 0) < args.num_demos:
            dd = demos_dir / direction / str(row.get("sentence_id", i))
            dd.mkdir(parents=True, exist_ok=True)
            sf.write(dd / "1_source.wav", src_wav, 24000)
            sf.write(dd / "2_generated.wav", gen_wav, 24000)
            sf.write(dd / "3_groundtruth.wav", tgt_wav, 24000)
            (dd / "4_text.json").write_text(
                json.dumps(
                    {
                        "src_text": d.get("src_text"),
                        "tgt_text": ref,
                        "hyp_text": hyp,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            demos_per_dir[direction] = demos_per_dir.get(direction, 0) + 1

    # ---- BLEU
    import sacrebleu

    bleu_by_dir = {}
    for direction in ("tr->hi", "hi->tr"):
        ys = [r for r in per_sample if r["direction"] == direction]
        if not ys:
            continue
        hyps = [r["hyp_norm"] for r in ys]
        refs = [r["ref_norm"] for r in ys]
        bleu_by_dir[direction] = sacrebleu.corpus_bleu(hyps, [refs]).score
    all_hyps = [r["hyp_norm"] for r in per_sample]
    all_refs = [r["ref_norm"] for r in per_sample]
    overall_bleu = sacrebleu.corpus_bleu(all_hyps, [all_refs]).score if per_sample else 0.0

    # ---- DNSMOS aggregation (mean + bootstrap 95% CI)
    def _stats(arr):
        if not arr:
            return None
        a = np.array(arr)
        means = a.mean(axis=0)
        rng = np.random.default_rng(0)
        boot = np.stack([a[rng.integers(0, len(a), len(a))].mean(axis=0) for _ in range(200)])
        lo, hi = np.quantile(boot, [0.025, 0.975], axis=0)
        return {
            "sig": [float(means[0]), float(lo[0]), float(hi[0])],
            "bak": [float(means[1]), float(lo[1]), float(hi[1])],
            "ovrl": [float(means[2]), float(lo[2]), float(hi[2])],
        }

    summary = {
        "n": len(per_sample),
        "bleu_by_direction": bleu_by_dir,
        "bleu_overall": overall_bleu,
        "dnsmos_generated": _stats(dnsmos_gen),
        "dnsmos_groundtruth": _stats(dnsmos_gt),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    with open(out_dir / "per_sample.jsonl", "w") as f:
        for r in per_sample:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nArtifacts saved to {out_dir}/")


if __name__ == "__main__":
    main()
