"""Probe SPMD sharding strategies on the live TPU mesh.

WHY THIS EXISTS
---------------
Loading the real 5.17B composite onto a v5litepod-16 takes ~10
minutes (HF download, weight init, surgery). Trying three sharding
strategies on the real model would burn an hour just on setup. This
script swaps in a tiny stand-in model (a few Linear layers + a
frozen "backbone") and runs it through each strategy in a fresh
subprocess, so we can decide which strategy to commit to *before*
paying the full uv-sync + HF-download price.

What "subprocess per strategy" buys us
--------------------------------------
SPMD failures often kill the whole Python process (the XLA
partitioner asserts and aborts). Running each strategy in its own
``subprocess.run`` fork-and-execve insulates the parent: a crashed
``replicated`` run does not contaminate the ``fsdpv2`` run that
follows.

Usage on a TPU pod (worker 0 only -- the SPMD runtime is single-
process)::

    PJRT_DEVICE=TPU LD_LIBRARY_PATH=$LIBPYTHON_DIR \\
        uv run python scripts/tpu/probe_strategies.py \\
            --strategies replicated fsdpv2 fsdpv2_lora --steps 5

Output: a comparison table on stdout plus a JSON dump under
``/tmp/probe-results.json``. Use the ``compile_s``, ``median_step_s``,
and ``peak_hbm_gb`` columns to pick a strategy. Record the choice in
``.factory/memories.md`` under "TPU strategy decisions".
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

# --------------------------------------------------------------------------
# the tiny stand-in model
# --------------------------------------------------------------------------


class _FrozenBackbone(nn.Module):
    """Stand-in for TinyAyaBackbone: two big Linear layers, frozen."""

    def __init__(self, hidden: int = 1024, n_layers: int = 4):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(hidden, hidden, bias=False) for _ in range(n_layers)]
        )
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x


class _LoRA(nn.Module):
    def __init__(self, hidden: int = 1024, r: int = 16):
        super().__init__()
        self.down = nn.Linear(hidden, r, bias=False)
        self.up = nn.Linear(r, hidden, bias=False)

    def forward(self, x):
        return self.up(self.down(x))


class _ProbeModel(nn.Module):
    """Frozen backbone + LoRA + small projection. Mirrors our composite shape."""

    def __init__(self, hidden: int = 1024, vocab: int = 4096):
        super().__init__()
        self.backbone = _FrozenBackbone(hidden=hidden, n_layers=4)
        self.lora = _LoRA(hidden=hidden, r=16)
        self.head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, x):
        h = self.backbone(x)
        h = h + self.lora(h)
        logits = self.head(h)
        # mimic the (text_logits, audio_logits, hidden_states) tuple shape
        return logits, logits, h


# --------------------------------------------------------------------------
# strategy runner (runs inside a subprocess)
# --------------------------------------------------------------------------


def _run_single(strategy: str, steps: int, batch_per_chip: int, hidden: int) -> dict:
    os.environ.setdefault("XLA_DISABLE_FUNCTIONALIZATION", "0")
    os.environ["TPU_STRATEGY"] = strategy

    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.spmd as xs
    import torch_xla.runtime as xr
    from torch_xla.distributed.spmd import Mesh

    xr.use_spmd()
    device = xm.xla_device()
    n = xr.global_runtime_device_count()
    mesh = Mesh(device_ids=list(range(n)), mesh_shape=(n,), axis_names=("fsdp",))

    print(f"[probe:{strategy}] devices={n} torch_xla={torch_xla.__version__}")

    model = _ProbeModel(hidden=hidden).to(device)

    if strategy == "replicated":
        for p in model.parameters():
            xs.mark_sharding(p, mesh, (None,) * p.dim())
        wrapped = model
    elif strategy in ("fsdpv2", "fsdpv2_lora"):
        from torch_xla.experimental.spmd_fully_sharded_data_parallel import (
            SpmdFullyShardedDataParallel as FSDPv2,
        )

        def _shard_output(out, mesh):
            res = []
            for v in out:
                if isinstance(v, torch.Tensor) and v.dim() >= 1:
                    xs.mark_sharding(v, mesh, ("fsdp",) + (None,) * (v.dim() - 1))
                res.append(v)
            return tuple(res)

        def _policy(module, recurse, **kwargs):
            if recurse:
                return True
            if strategy == "fsdpv2_lora":
                return any(p.requires_grad for p in module.parameters(recurse=False))
            return isinstance(module, nn.Linear)

        wrapped = FSDPv2(model, mesh=mesh, auto_wrap_policy=_policy, shard_output=_shard_output)
    else:
        raise ValueError(f"unknown strategy {strategy}")

    optimizer = torch.optim.AdamW([p for p in wrapped.parameters() if p.requires_grad], lr=1e-4)

    B = batch_per_chip * n
    timings = []
    compile_time = None
    peak_gb = 0.0

    for step in range(steps):
        x = torch.randn(B, 32, hidden, device=device)
        xs.mark_sharding(x, mesh, ("fsdp", None, None))
        t0 = time.time()
        text_logits, audio_logits, hidden_states = wrapped(x)
        loss = text_logits.float().square().mean() + audio_logits.float().square().mean()
        loss.backward()
        xm.optimizer_step(optimizer)
        torch_xla.sync()
        dt = time.time() - t0
        if step == 0:
            compile_time = dt
        else:
            timings.append(dt)

        try:
            mem = xm.get_memory_info(device)
            peak_gb = max(peak_gb, mem.get("peak_bytes_used", 0) / 1e9)
        except Exception:
            pass

        print(
            f"[probe:{strategy}] step={step} dt={dt:.3f}s peak_hbm={peak_gb:.2f}GB "
            f"loss={loss.item():.4f}"
        )

    median_step = sorted(timings)[len(timings) // 2] if timings else float("nan")
    result = {
        "strategy": strategy,
        "ok": True,
        "compile_s": compile_time,
        "median_step_s": median_step,
        "peak_hbm_gb": peak_gb,
        "n_devices": n,
    }
    print(f"[probe:{strategy}] RESULT={json.dumps(result)}")
    return result


# --------------------------------------------------------------------------
# parent dispatcher
# --------------------------------------------------------------------------


def _spawn(strategy: str, steps: int, batch_per_chip: int, hidden: int) -> dict:
    """Run one strategy in a fresh subprocess so XLA crashes don't propagate."""
    cmd = [
        sys.executable,
        __file__,
        "--child",
        "--strategy",
        strategy,
        "--steps",
        str(steps),
        "--batch-per-chip",
        str(batch_per_chip),
        "--hidden",
        str(hidden),
    ]
    print(f"[probe] spawning: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    out = {"strategy": strategy, "ok": False, "returncode": proc.returncode}
    for line in proc.stdout.splitlines():
        if line.startswith(f"[probe:{strategy}] RESULT="):
            try:
                out = json.loads(line.split("RESULT=", 1)[1])
            except Exception:
                pass
            break
    if proc.returncode != 0:
        out["ok"] = False
        # capture last 20 lines of stderr for diagnosis
        out["stderr_tail"] = "\n".join(proc.stderr.splitlines()[-20:])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", action="store_true", help="internal: child process")
    ap.add_argument("--strategy", type=str, default=None)
    ap.add_argument(
        "--strategies",
        nargs="+",
        default=["replicated", "fsdpv2_lora", "fsdpv2"],
    )
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--batch-per-chip", type=int, default=1)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--out", type=str, default="/tmp/probe-results.json")
    args = ap.parse_args()

    if args.child:
        try:
            _run_single(args.strategy, args.steps, args.batch_per_chip, args.hidden)
            return 0
        except SystemExit:
            raise
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"[probe:{args.strategy}] CRASH: {e}")
            return 2

    results = []
    for strat in args.strategies:
        results.append(_spawn(strat, args.steps, args.batch_per_chip, args.hidden))

    Path(args.out).write_text(json.dumps(results, indent=2))
    print("\n========== STRATEGY COMPARISON ==========")
    print(f"{'strategy':<16}{'ok':<6}{'compile_s':<12}{'step_s':<10}{'peak_hbm_gb':<14}")
    for r in results:
        ok = "yes" if r.get("ok") else "no"
        print(
            f"{r.get('strategy', '?'):<16}{ok:<6}"
            f"{r.get('compile_s', float('nan')):<12.3f}"
            f"{r.get('median_step_s', float('nan')):<10.3f}"
            f"{r.get('peak_hbm_gb', float('nan')):<14.2f}"
        )
    print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
