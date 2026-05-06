"""Stage 2: translation training (TR <-> HI), single-host GPU.

WHY THIS EXISTS
---------------
Pre-TPU Stage-2 driver. Trains the composite on paired
source/target audio using prefix-LM concatenation, with the loss
masked to target positions only.

Superseded for production by ``scripts/train_hierarchical.py``
which adds the backend abstraction and TPU SPMD support. We keep
this file around as a reference single-GPU implementation -- it is
shorter, easier to step through in a debugger, and doesn't require
the ``torch_xla`` wheel.

Loss is computed only on target positions (the tail half of each
sample, marked by ``loss_mask``).

Usage:
    python scripts/train_stage2.py --data_dir data/stage2 --max_steps 1000 \
        --checkpoint checkpoints/stage1/checkpoint_step_500
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.collator import InterleavedCollator
from src.data.dataset import TranslationDataset
from src.model.backbone import TinyAyaBackbone
from src.model.lora_setup import apply_lora, get_parameter_groups, register_embedding_grad_mask
from src.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/stage2")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Stage 1 checkpoint to resume from"
    )
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument("--max_seq_len", type=int, default=300)
    parser.add_argument("--lr_lora", type=float, default=1.5e-4)
    parser.add_argument("--lr_full_ft", type=float, default=5e-5)
    parser.add_argument("--lr_audio_embed", type=float, default=5e-4)
    parser.add_argument("--lr_text_embed", type=float, default=5e-4)
    parser.add_argument("--lr_audio_head", type=float, default=5e-4)
    parser.add_argument(
        "--text_weight", type=float, default=0.1, help="Text is auxiliary in translation"
    )
    parser.add_argument("--audio_weight", type=float, default=1.0)
    parser.add_argument("--save_dir", type=str, default="checkpoints/stage2")
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=250)
    args = parser.parse_args()

    device = "cuda"
    print(f"Device: {device}")

    # === Model ===
    print("\n=== Loading TinyAya backbone ===")
    backbone = TinyAyaBackbone(load_in_bf16=True)

    print("\n=== Applying LoRA ===")
    backbone = apply_lora(backbone, r=16)
    register_embedding_grad_mask(backbone)

    # Load Stage 1 checkpoint if provided
    if args.checkpoint:
        print(f"\n=== Loading checkpoint: {args.checkpoint} ===")
        ckpt = torch.load(
            Path(args.checkpoint) / "training_state.pt",
            map_location="cpu",
            weights_only=False,
        )
        backbone.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"  Loaded from step {ckpt['step']}")

    backbone = backbone.to(device)
    backbone.gradient_checkpointing_enable()

    # === Dataset ===
    print("\n=== Loading translation dataset ===")
    dataset = TranslationDataset(
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
        num_workers=0,
        pin_memory=True,
    )

    # === Optimizer ===
    param_groups = get_parameter_groups(
        backbone,
        lr_lora=args.lr_lora,
        lr_full_ft=args.lr_full_ft,
        lr_audio_embed=args.lr_audio_embed,
        lr_text_embed=args.lr_text_embed,
        lr_audio_head=args.lr_audio_head,
    )
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

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
        text_weight=args.text_weight,
        audio_weight=args.audio_weight,
    )

    trainer.train(dataloader, max_steps=args.max_steps)
    print("\nDone!")


if __name__ == "__main__":
    main()
