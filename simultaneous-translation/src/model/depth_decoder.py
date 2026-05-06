"""Depth decoder: creates ``MoshiDepthDecoder`` with optional codebook extension.

WHY THIS EXISTS
---------------
Moshi's depth decoder is the second stage of audio generation: given
the backbone's hidden state at frame ``t``, it autoregressively
produces ``num_codebooks`` audio tokens that together represent one
frame of Mimi-encoded audio. Stage 2 reuses the pretrained Moshi
depth decoder via ``surgery.extract_depth_decoder_state_dict`` and
optionally extends its codebook count with ``extend_weights`` below.

This file is "TPU-adjacent": it constructs a vanilla HuggingFace
module that the composite later patches via ``scan_utils`` for TPU
compile-time and memory wins. There is no ``torch_xla`` import here.
"""

import torch
from transformers.models.moshi.configuration_moshi import MoshiDepthConfig
from transformers.models.moshi.modeling_moshi import MoshiDepthDecoder


def extend_weights(state_dict: dict, orig_q: int, new_q: int) -> dict:
    """Extend depth-decoder weights from ``orig_q`` to ``new_q`` codebooks.

    Parameters
    ----------
    state_dict : dict
        The original depth-decoder state dict.
    orig_q : int
        Original codebook count baked into the pretrained weights.
    new_q : int
        Target codebook count for the composite. May be equal to
        ``orig_q`` (then the state dict is just cloned).

    Returns
    -------
    dict
        A new state dict with FlexibleLinear weights and
        ``embed_tokens`` entries replicated cyclically to the new
        codebook count.

    Notes
    -----
    ``MoshiFlexibleLinear`` weights have shape
    ``[num_codebooks, ...]``. We use cyclic copying so that new
    codebook ``k`` starts from ``k % orig_q`` -- a heuristic that
    keeps initial activations roughly in distribution.
    """
    if orig_q == new_q:
        return {k: v.clone() for k, v in state_dict.items()}

    print(f"  Extending depth decoder: {orig_q} → {new_q} codebooks")
    extended = {}

    for key, weight in state_dict.items():
        if key.startswith("embed_tokens."):
            extended[key] = weight.clone()
            continue

        if weight.dim() >= 2 and weight.shape[0] == orig_q:
            # FlexibleLinear weight — extend first dim
            new_shape = list(weight.shape)
            new_shape[0] = new_q
            new_weight = torch.zeros(new_shape, dtype=weight.dtype)
            new_weight[:orig_q] = weight
            for k in range(orig_q, new_q):
                new_weight[k] = weight[k % orig_q]
            extended[key] = new_weight
        else:
            extended[key] = weight.clone()

    # Extend embed_tokens: orig_q-1 → new_q-1 audio embeddings
    n_orig = orig_q - 1
    n_new = new_q - 1
    for new_idx in range(n_orig, n_new):
        src_key = f"embed_tokens.{new_idx % n_orig}.weight"
        dst_key = f"embed_tokens.{new_idx}.weight"
        extended[dst_key] = state_dict[src_key].clone()

    print(f"  Extended FlexibleLinear weights and embed_tokens ({n_orig} → {n_new})")
    return extended


def create_depth_decoder(
    state_dict: dict[str, torch.Tensor],
    num_codebooks: int = 8,
    hidden_size: int = 1024,
    input_size: int = 4096,
    num_layers: int = 6,
    num_heads: int = 16,
    ffn_dim: int = 5632,
    audio_vocab_size: int = 2048,
    text_vocab_size: int = 32000,
    orig_num_codebooks: int = 8,
) -> MoshiDepthDecoder:
    """Create a ``MoshiDepthDecoder`` from a state dict, optionally
    extending its codebook count.

    Parameters
    ----------
    state_dict : dict[str, torch.Tensor]
        Output of :func:`surgery.extract_depth_decoder_state_dict`.
    num_codebooks : int, default 8
        Target codebook count. Must be >= ``orig_num_codebooks``.
    hidden_size, input_size, num_layers, num_heads, ffn_dim, audio_vocab_size,
        text_vocab_size, orig_num_codebooks : int
        Architectural constants matching the pretrained Moshi config.
        Override only when porting to a different Moshi variant.

    Returns
    -------
    MoshiDepthDecoder
        The HF module, weights loaded with ``strict=False``. Missing
        and unexpected key counts are printed for sanity.
    """

    # Extend weights if needed
    if num_codebooks != orig_num_codebooks:
        state_dict = extend_weights(state_dict, orig_num_codebooks, num_codebooks)

    config = MoshiDepthConfig(
        vocab_size=text_vocab_size,
        hidden_size=hidden_size,
        input_size=input_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
        max_position_embeddings=num_codebooks + 1,
        hidden_act="silu",
        head_dim=hidden_size // num_heads,
        ffn_dim=ffn_dim,
        rms_norm_eps=1e-8,
        num_codebooks=num_codebooks,
        audio_vocab_size=audio_vocab_size,
        sliding_window=num_codebooks,
    )

    print(f"Creating MoshiDepthDecoder: {num_codebooks} codebooks, {num_layers} layers")
    decoder = MoshiDepthDecoder(config)

    missing, unexpected = decoder.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")

    print(f"  input_projections: {tuple(decoder.input_projections.weight.shape)}")
    print(f"  lm_heads: {tuple(decoder.lm_heads.weight.shape)}")
    print(f"  embed_tokens: {len(decoder.embed_tokens)} audio embeddings")
    print(f"  Total params: {sum(p.numel() for p in decoder.parameters()) / 1e6:.0f}M")

    return decoder
