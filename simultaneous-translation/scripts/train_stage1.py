"""Stage 1: audio-understanding training (single-host GPU).

WHY THIS EXISTS
---------------
Stage 1 is a one-language-at-a-time pretraining pass: TinyAya learns
to predict the next audio token (codebook 0 only) and the next text
token aligned with the audio frame. Stage 2 starts from a Stage-1
checkpoint, so this script is the upstream dependency of the whole
translation pipeline.

This script targets a single GPU host. The TPU version of Stage 2 is
``scripts/train_hierarchical.py``; we never trained Stage 1 on TPU
because the codebase 0-only loss was small enough to fit on a single
H100.

Usage:
    python scripts/train_stage1.py --data_dir data/stage1 --max_steps 2000
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.collator import InterleavedCollator
from src.data.dataset import InterleavedAudioDataset
from src.model.backbone import TinyAyaBackbone
from src.model.lora_setup import apply_lora, get_parameter_groups, register_embedding_grad_mask
from src.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/stage1")
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument("--max_seq_len", type=int, default=200)
    parser.add_argument("--lr_lora", type=float, default=3e-4)
    parser.add_argument("--lr_full_ft", type=float, default=1e-4)
    parser.add_argument("--lr_audio_embed", type=float, default=1e-3)
    parser.add_argument("--lr_text_embed", type=float, default=1e-3)
    parser.add_argument("--lr_audio_head", type=float, default=1e-3)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--text_weight", type=float, default=1.0)
    parser.add_argument("--audio_weight", type=float, default=1.0)
    parser.add_argument("--text_padding_weight", type=float, default=0.01)
    parser.add_argument("--save_dir", type=str, default="checkpoints/stage1")
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="simul-translation")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # === Model ===
    print("\n=== Loading TinyAya backbone ===")
    backbone = TinyAyaBackbone(load_in_bf16=True)

    print("\n=== Applying LoRA ===")
    backbone = apply_lora(backbone, r=args.lora_r)
    register_embedding_grad_mask(backbone)

    backbone = backbone.to(device)

    # Enable gradient checkpointing to save memory
    backbone.gradient_checkpointing_enable()

    total_params = sum(p.numel() for p in backbone.parameters())
    trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"\nTotal params: {total_params / 1e9:.2f}B")
    print(
        f"Trainable params: {trainable_params / 1e6:.1f}M ({100 * trainable_params / total_params:.1f}%)"
    )

    # === Dataset ===
    print("\n=== Loading dataset ===")
    dataset = InterleavedAudioDataset(
        data_dir=args.data_dir,
        tokenizer=backbone.tokenizer,
        max_frames=args.max_seq_len,
    )

    collator = InterleavedCollator()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,  # simple for overfitting
        pin_memory=True,
    )

    # === Optimizer ===
    print("\n=== Creating optimizer ===")
    param_groups = get_parameter_groups(
        backbone,
        lr_lora=args.lr_lora,
        lr_full_ft=args.lr_full_ft,
        lr_audio_embed=args.lr_audio_embed,
        lr_text_embed=args.lr_text_embed,
        lr_audio_head=args.lr_audio_head,
    )
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

    # === W&B ===
    if args.use_wandb:
        import wandb

        wandb.init(project=args.wandb_project, config=vars(args))

    # === Train ===
    trainer = Trainer(
        model=backbone,
        optimizer=optimizer,
        device=device,
        max_grad_norm=1.0,
        gradient_accumulation_steps=args.accum_steps,
        log_every=args.log_every,
        save_every=args.save_every,
        save_dir=args.save_dir,
        use_wandb=args.use_wandb,
        text_weight=args.text_weight,
        audio_weight=args.audio_weight,
        text_padding_weight=args.text_padding_weight,
    )

    trainer.train(dataloader, max_steps=args.max_steps)

    print("\nDone!")


if __name__ == "__main__":
    main()
