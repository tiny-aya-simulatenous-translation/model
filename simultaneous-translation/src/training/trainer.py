"""Training loop with gradient accumulation, logging, and checkpointing."""

import time
from pathlib import Path

import torch


class Trainer:
    """Simple training loop for the interleaved audio-text model."""

    def __init__(
        self,
        model,
        optimizer,
        scheduler=None,
        device: str = "cuda",
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
        log_every: int = 5,
        save_every: int = 500,
        save_dir: str = "checkpoints",
        use_wandb: bool = False,
        text_weight: float = 1.0,
        audio_weight: float = 1.0,
        text_padding_weight: float = 0.01,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.accum_steps = gradient_accumulation_steps
        self.log_every = log_every
        self.save_every = save_every
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb
        self.text_weight = text_weight
        self.audio_weight = audio_weight
        self.text_padding_weight = text_padding_weight

        if use_wandb:
            import wandb

            self.wandb = wandb

    def train(self, dataloader, max_steps: int):
        """Run training loop."""
        from .loss import compute_interleaved_loss

        self.model.train()
        global_step = 0
        micro_step = 0
        data_iter = iter(dataloader)
        running_loss = 0.0
        running_text_loss = 0.0
        running_audio_loss = 0.0

        print(f"\n{'=' * 60}")
        print(f"Starting training: {max_steps} steps, accum={self.accum_steps}")
        print(f"{'=' * 60}\n")

        t0 = time.time()

        while global_step < max_steps:
            # Get batch (cycle if dataset is small)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            text_ids = batch["text_ids"].to(self.device)
            all_audio_codes = batch["audio_codes"].to(self.device)  # [B, CB, T]
            audio_codes_cb0 = all_audio_codes[:, 0, :]  # [B, T] for backbone input
            attention_mask = batch["attention_mask"].to(self.device)
            loss_mask = batch.get("loss_mask")
            if loss_mask is not None:
                loss_mask = loss_mask.to(self.device)

            # Forward (backbone input uses codebook 0 only)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                output = self.model(
                    text_ids=text_ids,
                    audio_codes=audio_codes_cb0,
                    attention_mask=attention_mask,
                )
                text_logits, audio_logits, _ = output
                losses = compute_interleaved_loss(
                    text_logits=text_logits,
                    audio_logits=audio_logits,
                    text_targets=text_ids,
                    audio_targets=all_audio_codes,
                    attention_mask=attention_mask,
                    loss_mask=loss_mask,
                    text_weight=self.text_weight,
                    audio_weight=self.audio_weight,
                    text_padding_weight=self.text_padding_weight,
                )
                loss = losses["loss"] / self.accum_steps

            # Backward
            loss.backward()
            micro_step += 1

            running_loss += losses["loss"].item()
            running_text_loss += losses["text_loss"].item()
            running_audio_loss += losses["audio_loss"].item()

            # Optimizer step
            if micro_step % self.accum_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.scheduler is not None:
                    self.scheduler.step()

                global_step += 1

                # Logging
                if global_step % self.log_every == 0:
                    avg_loss = running_loss / self.log_every
                    avg_text = running_text_loss / self.log_every
                    avg_audio = running_audio_loss / self.log_every
                    elapsed = time.time() - t0
                    steps_per_sec = global_step / elapsed

                    print(
                        f"step {global_step:5d} | "
                        f"loss {avg_loss:.4f} | "
                        f"text {avg_text:.4f} | "
                        f"audio {avg_audio:.4f} | "
                        f"grad_norm {grad_norm:.3f} | "
                        f"{steps_per_sec:.2f} steps/s"
                    )

                    if self.use_wandb:
                        self.wandb.log(
                            {
                                "train/loss": avg_loss,
                                "train/text_loss": avg_text,
                                "train/audio_loss": avg_audio,
                                "train/grad_norm": grad_norm.item(),
                                "train/step": global_step,
                            }
                        )

                    running_loss = 0.0
                    running_text_loss = 0.0
                    running_audio_loss = 0.0

                # Save checkpoint
                if global_step % self.save_every == 0:
                    self._save_checkpoint(global_step)

        print(f"\nTraining complete: {global_step} steps in {time.time() - t0:.1f}s")
        self._save_checkpoint(global_step)

    def _save_checkpoint(self, step):
        """Save model checkpoint."""
        path = self.save_dir / f"checkpoint_step_{step}"
        path.mkdir(exist_ok=True)

        # Save model state
        state = {
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(state, path / "training_state.pt")
        print(f"  Checkpoint saved: {path}")
