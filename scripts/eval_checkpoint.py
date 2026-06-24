"""Evaluate a TinyAya S2S checkpoint: translation quality + audio quality.

Usage:
    python scripts/eval_checkpoint.py \
        --checkpoint checkpoints/stage2_26k_parallel/best_by_val \
        --val_jsonl /path/to/val_500.jsonl \
        --encoded_dir /path/to/encoded \
        --num_samples 30 \
        --output_dir eval_results/step_3000
"""
import argparse
import json
import os
import sys
import time

import torch
import soundfile as sf
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model.composite import TinyAyaMoshiComposite
from src.model.lora_setup import apply_lora
from src.training.checkpointing import load_checkpoint
from src.data.dataset import StreamingTranslationDataset, undo_codebook_delay, SILENCE_TOKEN
from src.data.mimi_encoder import MimiEncoder
from transformers import AutoTokenizer


def generate_teacher_forced(model, sample, device, num_codebooks=8):
    """Teacher-forced generation — upper bound on quality."""
    T = sample["text_ids"].shape[0]
    src_len = sample["source_length"]

    user_cb0 = sample["user_audio_codes"][0].unsqueeze(0).to(device)
    model_cb0 = sample["model_audio_codes"][0].unsqueeze(0).to(device)
    text = sample["text_ids"].unsqueeze(0).to(device)
    mask = torch.ones(1, T, dtype=torch.long, device=device)
    full_model = sample["model_audio_codes"].unsqueeze(0).to(device)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        output = model(
            text_ids=text, audio_codes=user_cb0, model_audio_codes=model_cb0,
            attention_mask=mask, full_audio_codes=full_model[:, :num_codebooks, :],
            depth_chunk_size=16,
        )
        _, audio_logits, _ = output

    # Next-token predictions for target region
    pred_codes = audio_logits[0, :, src_len - 1:-1, :].argmax(dim=-1)  # [8, tgt_len]
    return pred_codes.cpu()


def generate_autoregressive(model, sample, device, temp=0.8, top_p=0.9):
    """Autoregressive generation — realistic quality."""
    T = sample["text_ids"].shape[0]
    src_len = sample["source_length"]

    user_stream = sample["user_audio_codes"][0].unsqueeze(0).to(device)
    model_stream = torch.full((1, T), SILENCE_TOKEN, dtype=torch.long, device=device)
    ar_text = sample["text_ids"].unsqueeze(0).to(device)
    gen_all = torch.full((8, T), SILENCE_TOKEN, dtype=torch.long, device=device)

    for t in range(src_len, T):
        ar_mask = torch.ones(1, t + 1, dtype=torch.long, device=device)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            bb_out = model.backbone(
                text_ids=ar_text[:, :t + 1],
                audio_codes=user_stream[:, :t + 1],
                model_audio_codes=model_stream[:, :t + 1],
                attention_mask=ar_mask,
            )
            hidden = bb_out["hidden_states"]

            # CB0 from backbone
            cb0_logits = model.backbone.audio_heads[0](hidden[:, -1:, :]).squeeze()
            cb0_tok = _sample_top_p(cb0_logits, temp, top_p)
            model_stream[0, t] = cb0_tok
            gen_all[0, t] = cb0_tok

            # CB1-7 from depth decoder
            projected = model.projection(hidden[:, -1:, :])
            ctx_expanded = projected.expand(1, model.num_codebooks, -1).contiguous()
            depth_input = torch.zeros(1, model.num_codebooks, dtype=torch.long, device=device)
            depth_input[0, 1] = cb0_tok

            depth_out = model.depth_decoder(
                input_ids=depth_input, last_hidden_state=ctx_expanded,
                use_cache=False, return_dict=True,
            )

            for cb_idx in range(7):
                logits = depth_out.logits[0, cb_idx + 1, :].float()
                tok = _sample_top_p(logits, temp, top_p)
                gen_all[cb_idx + 1, t] = tok
                if cb_idx + 2 < model.num_codebooks:
                    depth_input[0, cb_idx + 2] = tok

    return gen_all[:, src_len:T].cpu()


def _sample_top_p(logits, temp, top_p):
    probs = torch.softmax(logits.float() / temp, dim=-1)
    sorted_p, sorted_i = torch.sort(probs, descending=True)
    cum = torch.cumsum(sorted_p, dim=-1)
    mask = cum - sorted_p > top_p
    sorted_p[mask] = 0.0
    sorted_p = sorted_p / sorted_p.sum()
    return sorted_i[torch.multinomial(sorted_p, 1)].squeeze(-1)


def run_asr(audio_path, model_size="large-v3", language=None):
    """Run Whisper ASR on audio file, return transcript."""
    from faster_whisper import WhisperModel
    if not hasattr(run_asr, "_model"):
        print(f"  Loading Whisper {model_size}...", flush=True)
        run_asr._model = WhisperModel(model_size, device="cuda", compute_type="float16")
    segments, info = run_asr._model.transcribe(audio_path, language=language, beam_size=5)
    return " ".join(s.text.strip() for s in segments)


def compute_bleu(hypothesis, reference):
    """Simple corpus BLEU (sacrebleu-style)."""
    try:
        import sacrebleu
        result = sacrebleu.sentence_bleu(hypothesis, [reference])
        return result.score
    except ImportError:
        # Fallback: simple unigram precision
        hyp_tokens = hypothesis.lower().split()
        ref_tokens = reference.lower().split()
        if not hyp_tokens:
            return 0.0
        matches = sum(1 for t in hyp_tokens if t in ref_tokens)
        return (matches / len(hyp_tokens)) * 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument("--encoded_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=30)
    parser.add_argument("--output_dir", default="eval_results")
    parser.add_argument("--whisper_model", default="large-v3")
    parser.add_argument("--skip_ar", action="store_true", help="Skip slow autoregressive generation")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda"

    # Load model
    print("Loading model...", flush=True)
    model = TinyAyaMoshiComposite(num_codebooks=8)
    model.backbone = apply_lora(model.backbone, r=16, num_full_ft_layers=0)
    load_checkpoint(model, None, None, args.checkpoint)
    model = model.to(device).to(torch.bfloat16).eval()

    # Load dataset + Mimi decoder
    tokenizer = AutoTokenizer.from_pretrained("CohereLabs/tiny-aya-base", trust_remote_code=True)
    ds = StreamingTranslationDataset(args.val_jsonl, tokenizer, max_frames=300, encoded_dir=args.encoded_dir)
    mimi = MimiEncoder(device=device)

    # Load metadata for reference texts
    with open(args.val_jsonl) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    num_samples = min(args.num_samples, len(ds))
    results = []

    for i in range(num_samples):
        print(f"\n{'='*60}", flush=True)
        print(f"Sample {i+1}/{num_samples}", flush=True)

        sample = ds[i]
        row = rows[i]
        src_len = sample["source_length"]
        tgt_len = sample["target_length"]
        direction = row.get("direction", "unknown")

        # Determine target language for ASR
        tgt_lang = "hi" if "hi" in direction.split("->")[-1] else "tr"

        sample_dir = os.path.join(args.output_dir, f"sample_{i:03d}")
        os.makedirs(sample_dir, exist_ok=True)

        # Decode source + ground truth
        raw_audio = sample["audio_codes"]
        src_wav = mimi.decode(raw_audio[:, :src_len]).numpy()
        tgt_wav = mimi.decode(raw_audio[:, src_len:]).numpy()
        sf.write(os.path.join(sample_dir, "source.wav"), src_wav, 24000)
        sf.write(os.path.join(sample_dir, "target_gt.wav"), tgt_wav, 24000)

        # Teacher-forced generation
        t0 = time.time()
        tf_codes = generate_teacher_forced(model, sample, device)
        tf_codes_clean = tf_codes.clamp(max=2047)
        tf_wav = mimi.decode(tf_codes_clean.to(device)).numpy()
        sf.write(os.path.join(sample_dir, "teacher_forced.wav"), tf_wav, 24000)
        tf_time = time.time() - t0

        # Per-codebook accuracy (teacher-forced)
        gt_delayed = sample["model_audio_codes"][:, src_len:src_len + tf_codes.shape[1]]
        cb_accs = []
        for cb in range(8):
            acc = (tf_codes[cb] == gt_delayed[cb]).float().mean().item()
            cb_accs.append(acc)

        # Autoregressive generation
        ar_wav = None
        ar_time = 0
        if not args.skip_ar:
            t0 = time.time()
            ar_codes = generate_autoregressive(model, sample, device)
            ar_codes_clean = ar_codes.clamp(max=2047)
            ar_wav = mimi.decode(ar_codes_clean.to(device)).numpy()
            sf.write(os.path.join(sample_dir, "autoregressive.wav"), ar_wav, 24000)
            ar_time = time.time() - t0

        # ASR on generated audio
        print(f"  Running ASR...", flush=True)
        gt_transcript = run_asr(os.path.join(sample_dir, "target_gt.wav"), args.whisper_model, language=tgt_lang)
        tf_transcript = run_asr(os.path.join(sample_dir, "teacher_forced.wav"), args.whisper_model, language=tgt_lang)
        ar_transcript = run_asr(os.path.join(sample_dir, "autoregressive.wav"), args.whisper_model, language=tgt_lang) if ar_wav is not None else ""

        # BLEU scores
        tf_bleu = compute_bleu(tf_transcript, gt_transcript)
        ar_bleu = compute_bleu(ar_transcript, gt_transcript) if ar_transcript else 0.0

        result = {
            "sample_idx": i,
            "direction": direction,
            "src_frames": src_len,
            "tgt_frames": tgt_len,
            "cb0_acc": cb_accs[0],
            "cb_avg_acc": np.mean(cb_accs),
            "per_cb_acc": cb_accs,
            "tf_bleu": tf_bleu,
            "ar_bleu": ar_bleu,
            "gt_transcript": gt_transcript,
            "tf_transcript": tf_transcript,
            "ar_transcript": ar_transcript,
            "tf_time": tf_time,
            "ar_time": ar_time,
        }
        results.append(result)

        print(f"  Direction: {direction}", flush=True)
        print(f"  CB0 acc: {cb_accs[0]*100:.1f}%, Avg: {np.mean(cb_accs)*100:.1f}%", flush=True)
        print(f"  GT:  {gt_transcript[:80]}", flush=True)
        print(f"  TF:  {tf_transcript[:80]}", flush=True)
        print(f"  AR:  {ar_transcript[:80]}", flush=True)
        print(f"  BLEU — TF: {tf_bleu:.1f}, AR: {ar_bleu:.1f}", flush=True)

    # Summary
    print(f"\n{'='*60}", flush=True)
    print(f"SUMMARY ({num_samples} samples)", flush=True)
    print(f"{'='*60}", flush=True)
    avg_cb0 = np.mean([r["cb0_acc"] for r in results])
    avg_cb_all = np.mean([r["cb_avg_acc"] for r in results])
    avg_tf_bleu = np.mean([r["tf_bleu"] for r in results])
    avg_ar_bleu = np.mean([r["ar_bleu"] for r in results])
    print(f"  Avg CB0 accuracy:     {avg_cb0*100:.1f}%", flush=True)
    print(f"  Avg all-CB accuracy:  {avg_cb_all*100:.1f}%", flush=True)
    print(f"  Avg TF BLEU:          {avg_tf_bleu:.1f}", flush=True)
    print(f"  Avg AR BLEU:          {avg_ar_bleu:.1f}", flush=True)

    # Save results
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump({"summary": {"avg_cb0_acc": avg_cb0, "avg_cb_all_acc": avg_cb_all,
                                "avg_tf_bleu": avg_tf_bleu, "avg_ar_bleu": avg_ar_bleu,
                                "num_samples": num_samples},
                   "samples": results}, f, indent=2)
    print(f"\nResults saved to {args.output_dir}/results.json", flush=True)


if __name__ == "__main__":
    main()
