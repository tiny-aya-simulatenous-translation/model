"""Tests for the W&B sweep promote helper + sweep-config consistency.

WHY THIS EXISTS
---------------
The sweep mechanism has two halves that must agree: the sweep grid
(``sweeps/sweep_stage2.yaml``) defines which flat params get swept, and the
promote helper (``scripts/promote_sweep_winner.py``) maps those flat names
onto the nested config sections. If they drift — a param added to the grid
with no mapping, or a mapping pointing at a section the config doesn't have —
a sweep silently runs the wrong recipe or the promote step no-ops.

These tests are torch-free (the training entrypoint can't be imported on a
CPU box), so they exercise the promote helper directly and cross-check it
against the actual sweep grid + TPU configs on disk.

Run: ``python -m pytest tests/test_sweep_promote.py -v``
"""

from __future__ import annotations

import importlib.util
import pathlib

import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
SWEEP_YAML = REPO / "sweeps" / "sweep_stage2.yaml"
SWEEP_V3_YAML = REPO / "sweeps" / "sweep_stage2_v3.yaml"
PROXY_CFG = REPO / "configs" / "tpu" / "stage2_tpu_v6e_proxy.yaml"
PROD_CFG = REPO / "configs" / "tpu" / "stage2_tpu_v6e_v2.yaml"
V3_CFG = REPO / "configs" / "tpu" / "stage2_tpu_v6e_v3.yaml"


def _load_promote():
    """Import scripts/promote_sweep_winner.py without a package install."""
    path = REPO / "scripts" / "promote_sweep_winner.py"
    spec = importlib.util.spec_from_file_location("promote_sweep_winner", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


promote = _load_promote()


# --------------------------------------------------------------------------
# resolve_updates: flat param -> (section, key, value)
# --------------------------------------------------------------------------

def test_resolve_updates_maps_each_param_to_its_section():
    hp = {
        "lr_lora": 2e-4, "lr_depth": 3e-4, "text_weight": 0.5,
        "warmup_steps": 300, "weight_decay": 1e-3, "max_steps": 15000,
        "val_every": 1000, "val_on_tpu": True, "lora_r": 32,
    }
    got = {(s, k): v for s, k, v in promote.resolve_updates(hp)}
    assert got[("optim", "lr_lora")] == repr(2e-4)
    assert got[("optim", "lr_depth")] == repr(3e-4)
    assert got[("loss", "text_weight")] == "0.5"
    assert got[("train", "warmup_steps")] == "300"
    assert got[("train", "weight_decay")] == repr(1e-3)
    assert got[("train", "max_steps")] == "15000"
    assert got[("logging", "val_every")] == "1000"
    assert got[("logging", "val_on_tpu")] == "true"   # bool -> yaml lowercase
    assert got[("lora", "r")] == "32"


def test_lora_alpha_is_mult_times_r():
    updates = dict(((s, k), v) for s, k, v in
                   promote.resolve_updates({"lora_r": 64, "lora_alpha_mult": 2}))
    assert updates[("lora", "r")] == "64"
    assert updates[("lora", "alpha")] == "128"   # 2 * 64


def test_alpha_mult_without_r_skips_alpha(capsys):
    updates = promote.resolve_updates({"lora_alpha_mult": 2})
    assert not any(k == "alpha" for _, k, _ in updates)
    assert "without lora_r" in capsys.readouterr().err


def test_unknown_params_ignored():
    # extra keys (e.g. wandb logs lr_projection in run.config) must not crash
    updates = promote.resolve_updates({"lr_lora": 1e-4, "lr_projection": 9.9})
    assert ("optim", "lr_lora", repr(1e-4)) in updates
    assert not any(k == "lr_projection" for _, k, _ in updates)


# --------------------------------------------------------------------------
# set_yaml_scalar_inplace: replace / insert / preserve comments
# --------------------------------------------------------------------------

SAMPLE = [
    "optim:\n",
    "  lr_lora: 1.5e-4          # sweep overrides\n",
    "  lr_depth: 2.5e-4\n",
    "train:\n",
    "  max_steps: 15000\n",
]


def test_replace_preserves_indent_and_inline_comment():
    out = promote.set_yaml_scalar_inplace(list(SAMPLE), "optim", "lr_lora", "9e-05")
    assert out[1] == "  lr_lora: 9e-05          # sweep overrides\n"


def test_insert_absent_key_under_section_header():
    out = promote.set_yaml_scalar_inplace(list(SAMPLE), "train", "weight_decay", "0.001")
    # inserted directly after the `train:` header line
    train_idx = out.index("train:\n")
    assert out[train_idx + 1] == "  weight_decay: 0.001\n"
    assert yaml.safe_load("".join(out))["train"]["weight_decay"] == 1e-3


def test_missing_section_raises():
    try:
        promote.set_yaml_scalar_inplace(list(SAMPLE), "nope", "k", "1")
    except KeyError:
        return
    raise AssertionError("expected KeyError for missing section")


def test_coerce_types():
    assert promote._coerce("true") is True
    assert promote._coerce("false") is False
    assert promote._coerce("300") == 300 and isinstance(promote._coerce("300"), int)
    assert promote._coerce("2e-4") == 2e-4 and isinstance(promote._coerce("2e-4"), float)
    assert promote._coerce("fsdpv2_lora") == "fsdpv2_lora"


# --------------------------------------------------------------------------
# Consistency: sweep grid <-> promote mapping <-> on-disk configs
# --------------------------------------------------------------------------

def test_every_swept_param_is_promotable():
    grid = yaml.safe_load(SWEEP_YAML.read_text())["parameters"]
    handled = set(promote.PARAM_MAP) | {"lora_alpha_mult"}
    unmapped = set(grid) - handled
    assert not unmapped, f"sweep params with no promote mapping: {unmapped}"


def test_mapping_sections_exist_in_configs():
    for cfg_path in (PROXY_CFG, PROD_CFG):
        cfg = yaml.safe_load(cfg_path.read_text())
        for _param, (section, _key) in promote.PARAM_MAP.items():
            assert section in cfg, f"{cfg_path.name} missing section '{section}'"


def test_sweep_command_points_at_existing_proxy_config():
    sweep = yaml.safe_load(SWEEP_YAML.read_text())
    cmd = sweep["command"]
    cfg_ref = cmd[cmd.index("--config") + 1]
    assert (REPO / cfg_ref).exists(), f"sweep --config path does not exist: {cfg_ref}"


# --------------------------------------------------------------------------
# Phase E: v0.3 regularization sweep (sweep_stage2_v3.yaml)
# --------------------------------------------------------------------------

def test_lora_dropout_maps_to_lora_dropout():
    assert promote.PARAM_MAP["lora_dropout"] == ("lora", "dropout")


def test_v3_sweep_every_param_is_promotable():
    grid = yaml.safe_load(SWEEP_V3_YAML.read_text())["parameters"]
    handled = set(promote.PARAM_MAP) | {"lora_alpha_mult"}
    unmapped = set(grid) - handled
    assert not unmapped, f"v3 sweep params with no promote mapping: {unmapped}"


def test_v3_sweep_optimises_composite():
    sweep = yaml.safe_load(SWEEP_V3_YAML.read_text())
    assert sweep["metric"]["name"] == "val/composite"
    assert sweep["metric"]["goal"] == "minimize"


def test_v3_sweep_command_points_at_v3_proxy_config():
    sweep = yaml.safe_load(SWEEP_V3_YAML.read_text())
    cmd = sweep["command"]
    cfg_ref = cmd[cmd.index("--config") + 1]
    assert (REPO / cfg_ref).exists(), f"v3 sweep --config missing: {cfg_ref}"
    assert cfg_ref.endswith("stage2_tpu_v6e_v3_proxy.yaml")


def test_v3_mapping_sections_and_dropout_key_exist():
    cfg = yaml.safe_load(V3_CFG.read_text())
    for _param, (section, _key) in promote.PARAM_MAP.items():
        assert section in cfg, f"v3 config missing section '{section}'"
    assert "dropout" in cfg["lora"], "v3 config lora.dropout must exist for the sweep"


if __name__ == "__main__":
    import sys
    import traceback
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            # crude capsys shim for standalone runs
            if "capsys" in fn.__code__.co_varnames:
                continue
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    sys.exit(1 if failed else 0)
