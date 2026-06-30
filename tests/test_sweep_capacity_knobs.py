"""Tests for the v0.3-SCALE capacity-sweep knobs.

Covers:
  - ``_parse_target_modules`` in scripts/train_hierarchical.py (the W&B sweep
    list-categorical parser) -- torch-free, extracted from source so CI stays light.
  - rsLoRA scaling (alpha/sqrt(r)) in src/model/lora_setup.py -- torch-gated.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]


def _extract_source(filename: str, name: str) -> str:
    """Return the source of a single top-level function or class by name, so we
    can exec it in isolation and never import the module's heavy top-level deps
    (peft/transformers/torch_xla) -- which makes the test robust to torch import
    state pollution from sibling tests in a shared pytest process."""
    src = (REPO / filename).read_text()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
            return ast.get_source_segment(src, node)
    raise AssertionError(f"{name} not found in {filename}")


def _load_parse_tm():
    ns: dict = {"json": __import__("json")}
    exec(_extract_source("scripts/train_hierarchical.py", "_parse_target_modules"), ns)
    return ns["_parse_target_modules"]


parse_tm = _load_parse_tm()


# ---- _parse_target_modules: must accept every way W&B may serialize a list ----

def test_parse_json_list():
    assert parse_tm('["q_proj","v_proj"]') == ["q_proj", "v_proj"]


def test_parse_python_repr_list():
    # W&B ${args} may pass a bracketed, unquoted list.
    assert parse_tm("[q_proj, v_proj, embed_tokens]") == ["q_proj", "v_proj", "embed_tokens"]


def test_parse_comma_separated():
    assert parse_tm("q_proj,v_proj,o_proj") == ["q_proj", "v_proj", "o_proj"]


def test_parse_space_separated():
    assert parse_tm("q_proj v_proj") == ["q_proj", "v_proj"]


def test_parse_single_bare_name():
    assert parse_tm("q_proj") == ["q_proj"]


def test_parse_actual_list_passthrough():
    assert parse_tm(["q_proj", "k_proj"]) == ["q_proj", "k_proj"]


def test_parse_mlp_full_set():
    raw = '["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]'
    assert parse_tm(raw) == [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ]


# ---- rsLoRA scaling on the custom text-embed adapter (torch-gated) ----

def test_lora_embedding_rslora_scaling():
    # torch only -- LoRAEmbedding is extracted standalone so we never import
    # lora_setup's peft/transformers deps (robust to torch import-state pollution).
    nn = pytest.importorskip("torch.nn")

    ns: dict = {"nn": nn}
    exec(_extract_source("src/model/lora_setup.py", "LoRAEmbedding"), ns)
    LoRAEmbedding = ns["LoRAEmbedding"]

    r, alpha = 16, 32
    vanilla = LoRAEmbedding(nn.Embedding(10, 8), r=r, alpha=alpha, use_rslora=False)
    rs = LoRAEmbedding(nn.Embedding(10, 8), r=r, alpha=alpha, use_rslora=True)
    assert vanilla.scaling == pytest.approx(alpha / r)            # 2.0
    assert rs.scaling == pytest.approx(alpha / (r ** 0.5))         # 8.0 -> bigger at high r
    # rsLoRA must not collapse as r grows: at r=64 the rslora scale >> vanilla.
    rs64 = LoRAEmbedding(nn.Embedding(10, 8), r=64, alpha=alpha, use_rslora=True)
    van64 = LoRAEmbedding(nn.Embedding(10, 8), r=64, alpha=alpha, use_rslora=False)
    assert rs64.scaling > van64.scaling
