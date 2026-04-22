"""Weighted text + audio loss with padding reduction and codebook upweighting."""

import torch
import torch.nn.functional as F

# Special token IDs
TEXT_PADDING = 262144
END_OF_TEXT_PADDING = 262145
ZERO_PADDING = 262146
IN_WORD_PADDING = 262147
SPECIAL_TOKENS = {TEXT_PADDING, END_OF_TEXT_PADDING, ZERO_PADDING, IN_WORD_PADDING}


def compute_interleaved_loss(
    text_logits: torch.Tensor,
    audio_logits: torch.Tensor,
    text_targets: torch.Tensor,
    audio_targets: torch.Tensor,
    attention_mask: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
    text_weight: float = 1.0,
    audio_weight: float = 1.0,
    text_padding_weight: float = 0.01,
    zero_padding_weight: float = 0.0,
):
    """Compute weighted text + audio cross-entropy loss.

    Args:
        text_logits: [B, T, text_vocab_size] from backbone LM head
        audio_logits: [B, T, audio_vocab_size] from audio head
        text_targets: [B, T] next text token targets
        audio_targets: [B, T] next audio codebook-0 targets
        attention_mask: [B, T] valid positions
        loss_mask: [B, T] optional, for Stage 2 (1=compute loss, 0=skip)
        text_weight: scalar weight for text loss
        audio_weight: scalar weight for audio loss
        text_padding_weight: weight for text padding tokens (< 1 to downweight)
        zero_padding_weight: weight for zero_padding positions (no text = no text loss)
    """
    B, T = attention_mask.shape

    # Shift for next-token prediction
    # Predict position t+1 from position t
    text_logits = text_logits[:, :-1].contiguous()
    text_targets = text_targets[:, 1:].contiguous()
    mask = attention_mask[:, 1:].bool()

    # Audio shift depends on shape
    if audio_logits.dim() == 4:
        # [B, C, T, vocab] -> [B, C, T-1, vocab]
        audio_logits = audio_logits[:, :, :-1].contiguous()
        audio_targets = audio_targets[:, :, 1:].contiguous()
    else:
        audio_logits = audio_logits[:, :-1].contiguous()
        audio_targets = audio_targets[:, 1:].contiguous()

    if loss_mask is not None:
        mask = mask & loss_mask[:, 1:].bool()

    # === Text loss with weighted padding ===
    # Create per-token weights
    text_token_weights = torch.ones_like(text_targets, dtype=torch.float32)
    # Downweight padding tokens
    for tok_id in (TEXT_PADDING, END_OF_TEXT_PADDING, IN_WORD_PADDING):
        text_token_weights[text_targets == tok_id] = text_padding_weight
    # Zero weight for zero_padding (no text to predict)
    text_token_weights[text_targets == ZERO_PADDING] = zero_padding_weight

    # CE loss per token
    text_loss_per_token = F.cross_entropy(
        text_logits.view(-1, text_logits.size(-1)),
        text_targets.view(-1),
        reduction="none",
    ).view(B, -1)

    # Apply mask and weights
    text_loss_masked = text_loss_per_token * mask.float() * text_token_weights
    text_denom = (mask.float() * text_token_weights).sum().clamp(min=1.0)
    text_loss = text_loss_masked.sum() / text_denom

    # === Audio loss across all codebooks ===
    # audio_logits: [B, num_codebooks, T-1, vocab] or [B, T-1, vocab] (legacy)
    # audio_targets: [B, num_codebooks, T-1] or [B, T-1] (legacy)
    if audio_logits.dim() == 4:
        # Multi-codebook: [B, C, T-1, vocab], targets [B, C, T-1]
        num_cb = audio_logits.shape[1]
        cb_losses = []
        for c in range(num_cb):
            cb_logits = audio_logits[:, c]  # [B, T-1, vocab]
            cb_targets = audio_targets[:, c]  # [B, T-1]
            cb_loss = F.cross_entropy(
                cb_logits.reshape(-1, cb_logits.size(-1)),
                cb_targets.reshape(-1),
                reduction="none",
            ).view(B, -1)
            cb_loss_masked = (cb_loss * mask.float()).sum() / mask.float().sum().clamp(min=1.0)
            cb_losses.append(cb_loss_masked)
        audio_loss = torch.stack(cb_losses).mean()
    else:
        # Single codebook fallback
        audio_loss_per_token = F.cross_entropy(
            audio_logits.view(-1, audio_logits.size(-1)),
            audio_targets.view(-1),
            reduction="none",
        ).view(B, -1)
        audio_loss_masked = audio_loss_per_token * mask.float()
        audio_loss = audio_loss_masked.sum() / mask.float().sum().clamp(min=1.0)

    # Combined
    total_loss = text_weight * text_loss + audio_weight * audio_loss

    return {
        "loss": total_loss,
        "text_loss": text_loss.detach(),
        "audio_loss": audio_loss.detach(),
    }
