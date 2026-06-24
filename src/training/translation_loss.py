"""Hierarchical translation loss: target-only CE on text + N audio codebooks.

WHY THIS EXISTS
---------------
The composite predicts both text and ``num_codebooks`` audio token
streams. The translation training only supervises the *target*
positions (the second half of each sample, after ``loss_mask`` flips
to 1). Loss is the weighted sum of:

* a single text cross-entropy over the target positions,
* the per-codebook mean cross-entropy over the target positions.

The next-token shift convention is ``logits[:, :-1] vs targets[:, 1:]``
applied symmetrically to the text and audio streams.

GPU vs TPU
----------
The function is plain PyTorch ``F.cross_entropy`` calls; XLA traces
each call into HLO ``select`` + ``log_softmax`` ops with no special
treatment required. The ``per_codebook`` Python loop unrolls into
``num_codebooks`` HLO subgraphs at compile time -- a one-time cost
because the codebook dimension is small (8) and stable.
"""

import torch
import torch.nn.functional as F


def compute_hierarchical_translation_loss(
    text_logits: torch.Tensor,  # [B, T, text_vocab]
    audio_logits: torch.Tensor,  # [B, CB, T, audio_vocab]
    text_targets: torch.Tensor,  # [B, T]
    audio_targets: torch.Tensor,  # [B, CB, T]
    attention_mask: torch.Tensor,  # [B, T]
    loss_mask: torch.Tensor,  # [B, T] — 1 ONLY on target positions
    text_weight: float = 0.1,
    audio_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    # Next-token shift
    tl = text_logits[:, :-1].contiguous()
    tt = text_targets[:, 1:].contiguous()
    al = audio_logits[:, :, :-1].contiguous()
    at = audio_targets[:, :, 1:].contiguous()

    am = attention_mask[:, 1:].bool()
    lm = loss_mask[:, 1:].bool()
    mask = am & lm  # [B, T-1]
    denom = mask.float().sum().clamp(min=1.0)
    zero = torch.zeros((), device=tl.device)

    def _masked_sum(ce: torch.Tensor) -> torch.Tensor:
        # Zero out non-target positions with ``where`` (NOT ``ce * mask``):
        # a padded position can yield ``inf`` cross-entropy, and ``inf * 0``
        # is NaN, which would poison the whole sum even though the position
        # is masked. ``where`` keeps masked entries at exactly 0.
        masked = torch.where(mask, ce, zero)
        # Defense-in-depth for the inline-TPU-val NaN: drop any non-finite that
        # still leaks into a target position so one bad element can't NaN the
        # whole val metric. NOTE: this (and the ``where`` above) only actually
        # fire when XLA is NOT assuming finiteness -- the launchers set
        # ``XLA_NO_SPECIAL_SCALARS=1`` so the assume-no-NaN/Inf algebraic
        # rewrites that previously elided these guards stay off.
        masked = torch.nan_to_num(masked, nan=0.0, posinf=0.0, neginf=0.0)
        return masked.sum()

    B, Tm1, V_text = tl.shape
    num_cb = al.shape[1]
    V_audio = al.shape[-1]

    # Text loss on target positions
    if text_weight > 0:
        tl_flat = tl.reshape(-1, V_text)
        tt_flat = tt.reshape(-1)
        ce = F.cross_entropy(tl_flat.float(), tt_flat, reduction="none").view(B, Tm1)
        text_loss = _masked_sum(ce) / denom
    else:
        text_loss = torch.zeros((), device=tl.device)

    # Audio loss — per-codebook mean CE on target positions.
    # Targets carry SILENCE_TOKEN (audio_vocab_size, e.g. 2048) at non-speech
    # / pad frames, which is OUT OF BOUNDS for the V_audio-class head. On XLA
    # an out-of-bounds cross_entropy gather is undefined behaviour and yields
    # non-deterministic NaN (finite when materialised alone, NaN when fused) --
    # the validation NaN. Those positions are masked out by loss_mask, so
    # clamping the target into range changes nothing supervised; it just
    # removes the UB.
    at_safe = at.clamp(max=V_audio - 1)
    per_cb = []
    for c in range(num_cb):
        ce_c = F.cross_entropy(
            al[:, c].reshape(-1, V_audio).float(),
            at_safe[:, c].reshape(-1),
            reduction="none",
        ).view(B, Tm1)
        per_cb.append(_masked_sum(ce_c) / denom)
    per_codebook = torch.stack(per_cb)  # [CB]
    audio_loss = per_codebook.mean()

    loss = text_weight * text_loss + audio_weight * audio_loss
    return {
        "loss": loss,
        "text_loss": text_loss.detach(),
        "audio_loss": audio_loss.detach(),
        "per_codebook_loss": per_codebook.detach(),
    }
