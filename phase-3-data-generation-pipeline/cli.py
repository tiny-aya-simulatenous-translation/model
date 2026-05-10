"""
Data generation pipeline for TR↔HI parallel speech pairs.

Usage:
    python cli.py extract [--splits train,validation,test]
    python cli.py generate
    python cli.py validate
    python cli.py filter [--max-wer 0.15] [--max-cer 0.10]
    python cli.py encode
    python cli.py run-all
"""
from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config import PipelineConfig


def cmd_extract(args, config: PipelineConfig):
    from src.sources.fleurs import extract_fleurs
    splits = args.splits.split(",") if args.splits else None
    extract_fleurs(config, splits=splits)


def cmd_generate(args, config: PipelineConfig):
    from src.tts.generate import generate_tts
    generate_tts(config)


def cmd_validate(args, config: PipelineConfig):
    from src.validation.stt import validate_stt
    validate_stt(config)


def cmd_filter(args, config: PipelineConfig):
    if args.max_wer is not None:
        config.max_wer = args.max_wer
    if args.max_cer is not None:
        config.max_cer = args.max_cer
    from src.validation.filter import run_filter
    run_filter(config)


def cmd_encode(args, config: PipelineConfig):
    from src.encoding.mimi import encode_mimi
    encode_mimi(config)


def cmd_align(args, config: PipelineConfig):
    from src.encoding.whisper_align import align_manifest
    align_manifest(config, batch_size=args.batch_size)


def cmd_run_all(args, config: PipelineConfig):
    print("=== Stage 1: Extract FLEURS ===")
    cmd_extract(args, config)
    print("\n=== Stage 2: Generate TTS ===")
    cmd_generate(args, config)
    print("\n=== Stage 3: Validate STT ===")
    cmd_validate(args, config)
    print("\n=== Stage 4: Filter ===")
    cmd_filter(args, config)
    print("\n=== Pipeline complete ===")


def main():
    parser = argparse.ArgumentParser(description="TR↔HI data generation pipeline")
    parser.add_argument("--data-dir", type=str, default="data", help="Base data directory")
    sub = parser.add_subparsers(dest="command", required=True)

    # extract
    p_extract = sub.add_parser("extract", help="Extract FLEURS parallel pairs")
    p_extract.add_argument("--splits", type=str, default=None,
                           help="Comma-separated FLEURS splits (default: all)")

    # generate
    sub.add_parser("generate", help="Generate TTS audio variants")

    # validate
    sub.add_parser("validate", help="Validate TTS with STT round-trip")

    # filter
    p_filter = sub.add_parser("filter", help="Apply quality thresholds")
    p_filter.add_argument("--max-wer", type=float, default=None)
    p_filter.add_argument("--max-cer", type=float, default=None)

    # encode
    sub.add_parser("encode", help="Encode audio with Mimi codec (optional)")

    # align (Whisper word timestamps)
    p_align = sub.add_parser("align", help="Whisper word-level alignments")
    p_align.add_argument("--batch-size", type=int, default=8)

    # run-all
    p_all = sub.add_parser("run-all", help="Run full pipeline (stages 1-4)")
    p_all.add_argument("--splits", type=str, default=None)
    p_all.add_argument("--max-wer", type=float, default=None)
    p_all.add_argument("--max-cer", type=float, default=None)

    parser.add_argument("--ref-dir", type=str, default="ref-audios",
                        help="Directory with voice reference audio files")
    parser.add_argument("--max-voices", type=int, default=2,
                        help="Max voices per Turkish TTS model (default: 2)")

    args = parser.parse_args()
    from pathlib import Path
    from src.config import PipelineConfig

    config = PipelineConfig(data_dir=Path(args.data_dir))
    config.load_voice_refs(Path(args.ref_dir), max_voices_per_model=args.max_voices)

    if config.xttsv2_voice_refs:
        print(f"Turkish voices: {list(config.xttsv2_voice_refs.keys())} (XTTSv2), {list(config.chatterbox_voice_refs.keys())} (Chatterbox)")

    commands = {
        "extract": cmd_extract,
        "generate": cmd_generate,
        "validate": cmd_validate,
        "filter": cmd_filter,
        "encode": cmd_encode,
        "align": cmd_align,
        "run-all": cmd_run_all,
    }
    commands[args.command](args, config)


if __name__ == "__main__":
    main()
