"""Model surgery: extract depth decoder from Moshiko, create projection."""

import torch
import torch.nn as nn
from transformers import MoshiForConditionalGeneration


def extract_depth_decoder_state_dict(moshiko_path="kmhf/hf-moshiko"):
    """Extract depth decoder weights from Moshiko."""
    print("Loading Moshiko for depth decoder extraction...")
    moshiko = MoshiForConditionalGeneration.from_pretrained(
        moshiko_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
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


def create_projection(in_features=2048, out_features=4096):
    """Create projection: TinyAya hidden -> depth decoder input."""
    proj = nn.Linear(in_features, out_features, bias=False)
    nn.init.xavier_uniform_(proj.weight)
    print(f"Projection: Linear({in_features}, {out_features}, bias=False)")
    return proj
