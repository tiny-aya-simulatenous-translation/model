"""Tests for Phase C3 param-name classification (torch-free)."""

from __future__ import annotations

import importlib.util
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]


def _load():
    path = REPO / "src" / "training" / "param_classify.py"
    spec = importlib.util.spec_from_file_location("param_classify", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pc = _load()


def test_depth_transformer_blocks_are_depth_blocks():
    assert pc.is_depth_block("depth_decoder.layers.0.self_attn.q_proj.weight") is True
    assert pc.is_depth_block("depth_decoder.layers.5.mlp.fc1.weight") is True


def test_depth_io_is_not_a_block():
    # input_projections / embed_tokens / lm_heads keep the normal depth LR
    assert pc.is_depth_block("depth_decoder.input_projections.0.weight") is False
    assert pc.is_depth_block("depth_decoder.embed_tokens.weight") is False
    assert pc.is_depth_block("depth_decoder.lm_heads.3.weight") is False


def test_backbone_and_other_params_are_not_depth_blocks():
    assert pc.is_depth_block("model.layers.5.self_attn.q_proj.lora_A.weight") is False
    assert pc.is_depth_block("projection.weight") is False
    assert pc.is_depth_block("model_audio_embed.weight") is False


# ---- depth_block_layer_index (the name-based unfreeze handle) --------------

def test_layer_index_full_model_name():
    # full-model name (model.named_parameters())
    assert pc.depth_block_layer_index("depth_decoder.layers.0.self_attn.q_proj.weight") == 0
    assert pc.depth_block_layer_index("depth_decoder.layers.5.mlp.fc1.weight") == 5


def test_layer_index_relative_name():
    # relative name (model.depth_decoder.named_parameters()) -> the enumeration
    # the unfreeze actually iterates; smoke showed module-walk was the wrong handle
    assert pc.depth_block_layer_index("layers.0.self_attn.q_proj.weight") == 0
    assert pc.depth_block_layer_index("layers.5.input_layernorm.weight") == 5


def test_layer_index_scan_wrapped_name():
    # TPU run scan-wraps .layers -> blocks live under layers_list.<i>.layer (the
    # real layout the smoke revealed; the original regex missed this)
    assert pc.depth_block_layer_index(
        "layers.layers_list.0.layer.self_attn.q_proj.linear.weight") == 0
    assert pc.depth_block_layer_index(
        "depth_decoder.layers.layers_list.5.layer.mlp.fc1.weight") == 5
    assert pc.is_depth_block(
        "depth_decoder.layers.layers_list.5.layer.mlp.fc1.weight") is True
    # backbone scan-wrapped block has layers_list but no depth_decoder -> not depth
    assert pc.is_depth_block(
        "backbone.model.model.layers.layers_list.3.layer.self_attn.q_proj.weight") is False


def test_layer_index_none_for_io_and_backbone():
    assert pc.depth_block_layer_index("depth_decoder.input_projections.weight") is None
    assert pc.depth_block_layer_index("depth_decoder.lm_heads.weight") is None
    assert pc.depth_block_layer_index("depth_decoder.embed_tokens.3.weight") is None
    # backbone block: has layers.<i> but no depth_decoder -> not a depth block
    assert pc.depth_block_layer_index("model.layers.5.self_attn.q_proj.weight") is None


def test_unfreeze_last_n_selection_matches_block_params():
    # simulate selecting the last 2 of 6 layers by index, as the train loop does
    names = [f"depth_decoder.layers.{i}.self_attn.q_proj.weight" for i in range(6)]
    names += ["depth_decoder.input_projections.weight", "depth_decoder.lm_heads.weight"]
    idxs = {pc.depth_block_layer_index(n) for n in names}
    idxs.discard(None)
    n_layers = max(idxs) + 1
    assert n_layers == 6
    threshold = n_layers - 2
    selected = [n for n in names if (pc.depth_block_layer_index(n) or -1) >= threshold]
    assert selected == [
        "depth_decoder.layers.4.self_attn.q_proj.weight",
        "depth_decoder.layers.5.self_attn.q_proj.weight",
    ]


if __name__ == "__main__":
    import sys
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    sys.exit(1 if failed else 0)
