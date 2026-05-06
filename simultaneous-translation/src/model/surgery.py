"""Model surgery: extract the depth decoder from Moshiko, build the projection.

WHY THIS EXISTS
---------------
The composite stitches two pretrained models together. We don't want
to ship Moshi-the-whole-thing as a dependency; we only need its
*depth decoder*. This module pulls the depth decoder's state dict out
of the Moshiko checkpoint on host CPU and frees the rest. The output
state dict is then handed to ``depth_decoder.create_depth_decoder``.

The ``create_projection`` helper builds the small Linear layer that
bridges TinyAya's hidden size (2048) to the depth decoder's input
size (4096). It is always trained.

This file runs once at composite construction; it is fully
device-agnostic and never imports ``torch_xla``.
"""

import torch
import torch.nn as nn
from transformers import MoshiForConditionalGeneration


def extract_depth_decoder_state_dict(moshiko_path: str = "kmhf/hf-moshiko") -> dict:
    """Pull the depth-decoder state dict out of Moshiko.

    Parameters
    ----------
    moshiko_path : str, default ``"kmhf/hf-moshiko"``
        HF model ID or local path for the source Moshi checkpoint.

    Returns
    -------
    dict
        Mapping of parameter / buffer names to detached CPU tensors.

    Notes
    -----
    The full Moshi model is loaded only to reach its depth decoder;
    we delete it afterwards. ``torch.cuda.empty_cache`` is called
    even on TPU/CPU runs -- it's a no-op there and saves a branch.
    """
    print("Loading Moshiko for depth decoder extraction...")
    moshiko = MoshiForConditionalGeneration.from_pretrained(
        moshiko_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    depth_sd = {}
    for name, param in moshiko.depth_decoder.named_parameters():
        depth_sd[name] = param.data.clone()
    for name, buf in moshiko.depth_decoder.named_buffers():
        depth_sd[name] = buf.clone()

    print(f"Extracted {len(depth_sd)} tensors from depth decoder")
    print(f"  input_projections: {tuple(depth_sd['input_projections.weight'].shape)}")
    print(f"  lm_heads: {tuple(depth_sd['lm_heads.weight'].shape)}")

    del moshiko
    torch.cuda.empty_cache()
    return depth_sd


def create_projection(in_features: int = 2048, out_features: int = 4096) -> nn.Linear:
    """Build the trainable Linear bridge from backbone hidden to depth input.

    Parameters
    ----------
    in_features : int, default 2048
        TinyAya hidden size.
    out_features : int, default 4096
        Moshi depth-decoder input size.

    Returns
    -------
    nn.Linear
        Bias-free Linear with Xavier-uniform initialisation. Always
        trained (the training script flips ``requires_grad=True`` on
        every parameter of this module).
    """
    proj = nn.Linear(in_features, out_features, bias=False)
    nn.init.xavier_uniform_(proj.weight)
    print(f"Projection: Linear({in_features}, {out_features}, bias=False)")
    return proj
