"""Pure parameter-name classification for optimizer groups (torch-free).

WHY THIS EXISTS
---------------
The optimizer splits params into LR groups by name. Phase C3 adds a distinction:
the depth decoder's *transformer blocks* (when unfrozen for low-LR adaptation) must
go in their own group, separate from the depth I/O (input_projections / embed_tokens
/ lm_heads) which keep the normal depth LR. Keeping the predicate here as a pure
function makes the rule the single source of truth and unit-testable without torch.

Depth-decoder param names on the unwrapped model look like:
  depth_decoder.layers.<i>.self_attn...   -> a transformer BLOCK  (low-LR group)
  depth_decoder.input_projections...      -> I/O                  (normal depth LR)
  depth_decoder.embed_tokens / lm_heads   -> I/O                  (normal depth LR)
"""

from __future__ import annotations


def is_depth_block(name: str) -> bool:
    """True if ``name`` is a depth-decoder TRANSFORMER-BLOCK param (not I/O)."""
    return "depth_decoder" in name and ".layers." in name
