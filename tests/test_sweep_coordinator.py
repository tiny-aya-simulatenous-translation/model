"""Tests for the multi-host sweep coordinator's pure core.

enumerate_grid / build_trial_args / load_sweep_parameters have no third-party
deps (wandb/gsutil imports are lazy), so they import + run on a CPU box / CI.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]


def _load_coord():
    path = REPO / "scripts" / "tpu" / "sweep_coordinator.py"
    spec = importlib.util.spec_from_file_location("sweep_coordinator", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


coord = _load_coord()


# ---- enumerate_grid ----

def test_enumerate_grid_cartesian():
    params = {
        "a": {"values": [1, 2]},
        "b": {"value": "x"},
        "c": {"values": [True, False]},
    }
    grid = coord.enumerate_grid(params)
    assert len(grid) == 4  # 2 x 1 x 2
    assert {"a": 1, "b": "x", "c": True} in grid
    assert {"a": 2, "b": "x", "c": False} in grid


def test_enumerate_grid_rejects_distribution():
    params = {"lr": {"distribution": "log_uniform_values", "min": 1e-5, "max": 5e-4}}
    with pytest.raises(ValueError):
        coord.enumerate_grid(params)


# ---- build_trial_args ----

def test_build_trial_args_types():
    args = coord.build_trial_args(
        {"lora_r": 32, "lr_lora": 0.0002, "use_rslora": True,
         "target_modules": ["q_proj", "v_proj"]}
    )
    assert "--lora_r" in args and args[args.index("--lora_r") + 1] == "32"
    assert args[args.index("--use_rslora") + 1] == "true"
    # list -> COMPACT json (no spaces) so the host loop word-splits it as one token
    tm = args[args.index("--target_modules") + 1]
    assert tm == '["q_proj","v_proj"]'
    assert " " not in tm
    assert json.loads(tm) == ["q_proj", "v_proj"]


# ---- against the real Stage-1 grid YAML ----

def test_real_grid_yaml_enumerates_three_structural_trials():
    pytest.importorskip("yaml")
    y = REPO / "sweeps" / "sweep_stage2_scale_grid.yaml"
    params = coord.load_sweep_parameters(str(y))
    grid = coord.enumerate_grid(params)
    # only target_modules is multi-valued (3 options) -> exactly 3 trials
    assert len(grid) == 3
    tms = sorted(len(t["target_modules"]) for t in grid)
    assert tms == [3, 5, 8]  # q,v,embed  |  +k,o  |  +MLP
    for t in grid:
        assert t["use_rslora"] is True
        assert t["lora_r"] == 16
        # every trial builds a clean arg list
        a = coord.build_trial_args(t)
        assert "--target_modules" in a and "--use_rslora" in a
