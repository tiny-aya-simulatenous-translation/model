"""Pure parameter-name classification for optimizer groups (torch-free).

WHY THIS EXISTS
---------------
The optimizer splits params into LR groups by name. Phase C3 adds a distinction:
the depth decoder's *transformer blocks* (when unfrozen for low-LR adaptation) must
go in their own group, separate from the depth I/O (input_projections / embed_tokens
/ lm_heads) which keep the normal depth LR. Keeping the predicate here as a pure
function makes the rule the single source of truth and unit-testable without torch.

The TPU smoke (2026-06-28) showed why module-walk (``model.depth_decoder.layers``)
was the wrong handle for the *unfreeze*: it reported 0M. The freeze loop, which
enumerates ``model.depth_decoder.named_parameters()``, correctly sees all 617M of
block params. So both the unfreeze and the optimizer-group split are name-based,
keyed off the same enumeration, via ``depth_block_layer_index`` here.

Depth-decoder param names (the TPU run scan-wraps `.layers`, so the index moves
under a ``layers_list`` segment -- the smoke proved this is the real layout):
  layers.<i>.self_attn...                       no-scan (GPU fallback)
  layers.layers_list.<i>.layer.self_attn...     scan / grad-checkpoint proxy (TPU)
  depth_decoder.<above>                          full-model prefix (model.named_parameters())
  depth_decoder.input_projections / embed_tokens / lm_heads  -> I/O (normal depth LR)
"""

from __future__ import annotations

import re

# Matches a ``layers.<i>.`` OR ``layers_list.<i>.`` segment (start or after a dot),
# capturing the layer index <i>. The optional ``_list`` covers the scan/grad-ckpt
# proxy (_ScannedLayerStack) that renames blocks to ``layers.layers_list.<i>.layer``.
_LAYER_RE = re.compile(r"(?:^|\.)layers(?:_list)?\.(\d+)\.")


def depth_block_layer_index(name: str) -> int | None:
    """Layer index ``i`` if ``name`` is a depth-decoder TRANSFORMER-BLOCK param.

    Accepts both the relative (``layers.<i>...``) and full-model
    (``depth_decoder.layers.<i>...``) name forms. Returns None for depth I/O and
    for non-depth params. A backbone block (``model.layers.<i>...``) has no
    ``depth_decoder`` segment, so it is NOT treated as a depth block.
    """
    if "depth_decoder.layers." not in name and not name.startswith("layers."):
        return None
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else None  # captures the index after layers[_list]


def is_depth_block(name: str) -> bool:
    """True if ``name`` is a depth-decoder transformer-block param (not I/O).

    Used on FULL-model names in get_param_groups, so it requires the
    ``depth_decoder`` segment to avoid matching backbone ``model.layers.*``.
    """
    return "depth_decoder" in name and depth_block_layer_index(name) is not None
