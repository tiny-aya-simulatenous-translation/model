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
