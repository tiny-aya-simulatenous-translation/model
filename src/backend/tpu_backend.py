"""TPU backend with multiple SPMD sharding strategies.

WHY THIS EXISTS
---------------
On a GPU pod we use **DDP** -- one Python process per rank, NCCL
all-reduce on the backward pass. On a TPU pod we use **SPMD**: a
single logical program runs across every chip via PJRT, and the XLA
partitioner decides where each tensor lives. This file is the only
place in the codebase that should know about XLA-specific primitives
(``Mesh``, ``mark_sharding``, ``FSDPv2``, ``xm.optimizer_step``).

WHAT IS SPMD?
-------------
"Single Program Multiple Data". You write one ``forward`` as if for
a single device. You annotate inputs / parameters with
``mark_sharding(tensor, mesh, partition_spec)`` to tell the
partitioner how the tensor is split across the mesh. XLA inserts the
necessary ``all-gather`` / ``reduce-scatter`` / ``all-reduce``
operations during compile.

GPU-vs-TPU comparison
---------------------
=======================  ===============================  ==================================
Concept                  GPU (DDP)                        TPU (SPMD)
=======================  ===============================  ==================================
Process model            N processes (one per rank)       1 process driving N chips
Data parallelism         Each rank sees a different       ``mark_sharding(x, mesh, ("fsdp",
                         batch slice via                   None,...))`` tells the
                         ``DistributedSampler``           partitioner to split x on dim 0
Gradient sync            NCCL all-reduce on backward      Inserted by the XLA partitioner
                                                          inside the compiled graph
Mixed precision          ``torch.amp.autocast``           Cast model to ``torch.bfloat16``
                                                          once; activations follow
Memory probe             ``torch.cuda.memory_allocated``  ``xm.get_memory_info(device)``
=======================  ===============================  ==================================

Strategy selection (env: ``TPU_STRATEGY``)
------------------------------------------
``replicated``
    Each chip holds a full copy of the model. Inputs are sharded on
    the batch axis. XLA inserts cross-chip all-reduce on the
    backward pass automatically. Best for small trainable parameter
    counts (LoRA only) and for *avoiding* the SPMD partitioner --
    useful when the partitioner crashes on multi-output composites.

``fsdpv2``
    ``SpmdFullyShardedDataParallel`` wrapping the *whole* composite
    model. Shards weights, gradients, and optimiser state along the
    fsdp axis. Lowest per-chip footprint; highest comm cost.

``fsdpv2_lora``
    Wrap only modules that contain trainable LoRA parameters. The
    frozen backbone stays replicated. Compromise: shards optimiser
    state (the bulk of trainable-side memory) but keeps the heavy
    frozen weights cheap to materialise. Default for the canary.

``auto``
    Pick ``replicated`` if trainable_params < 500M else ``fsdpv2``.

Compile-time gotcha
-------------------
SPMD runs as a single process. If any chip OOMs or errors, the whole
job dies. Mitigated by frequent checkpoints and the spot-preemption
restart loop in ``scripts/tpu/startup_script.sh``. See
``.factory/memories.md`` for empirical compile / step / HBM
measurements per strategy.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn

from src.backend.base import BackendBase

#: Strategies that ``TPU_STRATEGY`` may take. ``auto`` is resolved
#: against the model's trainable-parameter count at wrap time.
_VALID_STRATEGIES: tuple[str, ...] = ("auto", "replicated", "fsdpv2", "fsdpv2_lora")


def _resolve_strategy(model: nn.Module | None) -> str:
    """Pick the SPMD strategy from the environment (and the model if needed).

    Parameters
    ----------
    model : nn.Module or None
        The unwrapped composite. Used to count trainable parameters
        when ``TPU_STRATEGY=auto``. ``None`` is treated as "no
        information" and falls back to ``replicated``.

    Returns
    -------
    str
        One of ``"replicated"``, ``"fsdpv2"``, ``"fsdpv2_lora"``.

    Notes
    -----
    The ``500M`` threshold for the auto path is a heuristic: at the
    canary scale (~274M trainable LoRA params) replicated comfortably
    fits; once the trainable count crosses ~500M the optimiser state
    alone (AdamW keeps fp32 m + v) becomes the dominant memory cost
    and FSDPv2's optimiser-state sharding starts to pay off.
    """
    raw = os.environ.get("TPU_STRATEGY", "auto").strip().lower()
    if raw not in _VALID_STRATEGIES:
        print(f"[tpu_backend] unknown TPU_STRATEGY={raw!r}; falling back to auto")
        raw = "auto"
    if raw != "auto":
        return raw
    if model is None:
        return "replicated"
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    chosen = "replicated" if trainable < 500_000_000 else "fsdpv2"
    print(f"[tpu_backend] auto-strategy: trainable={trainable / 1e6:.0f}M -> {chosen}")
    return chosen


class TPUBackend(BackendBase):
    """``BackendBase`` implementation for TPU pods using SPMD."""

    def __init__(self) -> None:
        self._device: torch.device | None = None
        self._mesh = None
        self._world_size_val: int | None = None
        self._strategy: str | None = None
        self._memory_warning_emitted = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init_distributed(self) -> None:
        """Initialise PJRT, enable SPMD, and build a 1-D mesh.

        GPU analogue: ``dist.init_process_group("nccl")`` followed by
        ``torch.cuda.set_device(local_rank)``.

        Notes
        -----
        TPU note: ``XLA_DISABLE_FUNCTIONALIZATION=0`` *must* stay set.
        With value ``1`` the SPMD partitioner crashes on multi-output
        composite models (pytorch/xla #8607). 2.9's default is already
        ``0`` but we set it defensively here in case the user's shell
        injected the broken value.

        We do NOT set ``XLA_USE_BF16`` -- removed in torch_xla 2.6 and
        silently no-ops in 2.9. The bf16 cast happens in
        :meth:`wrap_model` instead.
        """
        import torch_xla.core.xla_model as xm
        import torch_xla.runtime as xr
        from torch_xla.distributed.spmd import Mesh

        os.environ.setdefault("XLA_DISABLE_FUNCTIONALIZATION", "0")

        # GPU analogue: dist.init_process_group("nccl")
        xr.use_spmd()
        # GPU analogue: torch.device(f"cuda:{local_rank}")
        self._device = xm.xla_device()

        num_devices = xr.global_runtime_device_count()
        self._world_size_val = num_devices
        # 1-D mesh named "fsdp". The partition_spec for a sharded
        # tensor will match this axis name -- e.g., a per-batch shard
        # is ``("fsdp", None, None, ...)``.
        self._mesh = Mesh(
            device_ids=list(range(num_devices)),
            mesh_shape=(num_devices,),
            axis_names=("fsdp",),
        )
        print(
            f"[tpu_backend] SPMD initialized: {num_devices} devices, "
            f"mesh_shape={self._mesh.mesh_shape}, axis=fsdp"
        )
        self.diagnose("post-init")

    def get_device(self) -> torch.device:
        """Return the lazy XLA device handle.

        Notes
        -----
        TPU note: the XLA device is *logical* and *lazy*. Tensors
        moved to it accumulate in a graph; they do not become
        physical TPU tensors until the next ``torch_xla.sync()`` or
        ``xm.optimizer_step``. This is why the first training step
        is slow (compile + execute) and subsequent steps are fast
        (cached binary).
        """
        if self._device is None:
            import torch_xla.core.xla_model as xm

            self._device = xm.xla_device()
        return self._device

    # ------------------------------------------------------------------
    # Model wrapping
    # ------------------------------------------------------------------

    def wrap_model(self, model: nn.Module) -> nn.Module:
        """Cast model to bf16 and apply the chosen SPMD strategy.

        Parameters
        ----------
        model : nn.Module
            The composite, already on the XLA device.

        Returns
        -------
        nn.Module
            The wrapped model. On a single chip this is a no-op
            (the bf16 cast still happens). On a pod it is either the
            same module with sharding spec attached (replicated) or
            an ``SpmdFullyShardedDataParallel`` wrapper (fsdpv2*).

        Notes
        -----
        TPU note: bf16 cast happens *here*, not via env vars. The
        legacy ``XLA_USE_BF16=1`` was removed in torch_xla 2.6+ and
        silently no-ops; tensors stay in f32 and the model OOMs on
        v5e (16 GiB / chip). AdamW keeps optimiser moments in fp32
        even when params are bf16, so optimiser numerics stay clean.

        Do NOT call ``model.gradient_checkpointing_enable()`` here on
        torch 2.9 + torch_xla 2.9 -- the legacy HF checkpoint hook
        calls ``torch._get_device_module("xla")`` which raises. Use
        the ``xla_grad_checkpoint`` flag on ``TinyAyaMoshiComposite``
        instead.
        """
        # GPU analogue: torch.amp.autocast("cuda", dtype=torch.bfloat16)
        # but applied as a permanent cast rather than a per-op context.
        model = model.to(torch.bfloat16)
        print("[tpu_backend] cast model to bfloat16")

        if self.world_size() <= 1:
            print("[tpu_backend] single chip, skipping wrap_model")
            self._strategy = "single"
            return model

        self._strategy = _resolve_strategy(model)
        print(f"[tpu_backend] wrap_model strategy={self._strategy}")

        if self._strategy == "replicated":
            return self._wrap_replicated(model)
        if self._strategy == "fsdpv2":
            return self._wrap_fsdpv2(model, lora_only=False)
        if self._strategy == "fsdpv2_lora":
            return self._wrap_fsdpv2(model, lora_only=True)
        raise RuntimeError(f"unhandled strategy={self._strategy}")

    def _wrap_replicated(self, model: nn.Module) -> nn.Module:
        """Replicated weights, sharded data. Each chip holds the full model.

        We rely on XLA's all-reduce-on-backward via the SPMD
        partitioner: when inputs are sharded along the batch axis and
        the parameters are replicated, the partitioner inserts the
        appropriate cross-replica sums during gradient accumulation
        automatically. No FSDP wrapper is needed.

        Notes
        -----
        TPU note: this path is the safest fallback when the SPMD
        partitioner is misbehaving (it has fewer fused passes to go
        wrong). It's also the cheapest path memory-wise when the
        trainable count is small (LoRA-only).
        """
        import torch_xla.distributed.spmd as xs

        # ``named_parameters`` / ``named_buffers`` yield (name, tensor)
        # pairs; we only need the tensors for ``mark_sharding``. The
        # name is kept in the for-target purely for debug-print
        # readability when this loop is stepped through interactively.
        for _name, param in model.named_parameters():
            if param.requires_grad:
                # GPU analogue: nothing -- DDP doesn't need explicit
                # sharding hints because each rank already has a copy.
                xs.mark_sharding(param, self._mesh, (None,) * param.dim())
        for _name, buf in model.named_buffers():
            xs.mark_sharding(buf, self._mesh, (None,) * buf.dim())
        print(
            "[tpu_backend] replicated: marked all parameters and buffers as "
            "replicated; data will be sharded on batch dim by mark_sharding "
            "calls in the train loop"
        )
        return model

    def _wrap_fsdpv2(self, model: nn.Module, *, lora_only: bool) -> nn.Module:
        """Wrap ``model`` in ``SpmdFullyShardedDataParallel``.

        Parameters
        ----------
        model : nn.Module
            Composite on the XLA device, bf16 cast applied.
        lora_only : bool
            If True, only modules whose subtree contains trainable
            parameters are wrapped (LoRA-bearing
            ``CohereDecoderLayer`` instances). Frozen
            ``MoshiDecoderLayer`` instances stay replicated. If
            False, every transformer layer is wrapped.

        Returns
        -------
        nn.Module
            ``SpmdFullyShardedDataParallel(model, ...)``.
        """
        import torch_xla.distributed.spmd as xs
        from torch.nn import Embedding
        from torch_xla.experimental.spmd_fully_sharded_data_parallel import (
            SpmdFullyShardedDataParallel as FSDPv2,
        )

        def _shard_output(output, mesh):
            """Attach per-tensor sharding spec to a composite's tuple output.

            The composite returns ``(text_logits, audio_logits,
            hidden_states)``. All three need an explicit sharding
            spec or the SPMD partitioner sees them as replicated and
            asserts in ``spmd_partitioner_util.h``. We shard each on
            the ``fsdp`` (batch) axis; remaining dims are replicated.
            """
            sharded = []
            for v in output:
                if isinstance(v, torch.Tensor) and v.dim() >= 1:
                    spec = ("fsdp",) + (None,) * (v.dim() - 1)
                    xs.mark_sharding(v, mesh, spec)
                sharded.append(v)
            return tuple(sharded)

        # iter 24c: deliberately exclude ``Cohere2DecoderLayer`` from
        # the auto-wrap policy. iter 24a/24b added it (matching the
        # real HF class name) which created 36 per-layer FSDPv2
        # wraps and 36 per-layer bf16 reduce-scatters on the
        # backward pass. On v6e that triggers the documented bf16
        # reduce-scatter numerics bug (pytorch/xla #8591 + #8778):
        # audio_loss went NaN at step 24 in both 24a (flash-attn on)
        # and 24b (flash-attn off). FSDPv2 has no
        # ``fp32_reduce_scatter`` flag (only FSDPv1 does, see
        # pytorch/xla #3588 / #8056) so we fall back to ONE outer
        # reduce-scatter at the composite level, which iter 17-23
        # validated as bf16-stable. The cache_all_gather OOM is
        # instead mitigated by the per-layer backward optimization
        # barrier installed in train_hierarchical.py
        # (``_apply_fsdpv2_backward_barriers``); that hook uses
        # ``register_full_backward_hook`` which works regardless of
        # whether the layer is wrapped by FSDPv2.
        layer_type_names = (
            "CohereDecoderLayer",
            "MoshiDecoderLayer",
        )

        def _has_trainable(m: nn.Module) -> bool:
            """True if any parameter in ``m`` (recursive) requires grad."""
            return any(p.requires_grad for p in m.parameters(recurse=True))

        def _wrap_policy(module, recurse, **kwargs):
            """FSDPv2 auto-wrap policy.

            * Always recurse into children so the policy can decide
              for each transformer layer.
            * Never wrap raw ``Embedding`` modules (they are tiny and
              already sharded by the partitioner).
            * Wrap only Cohere / Moshi decoder layers.
            * In LoRA-only mode skip layers that have no trainable
              params (i.e., the frozen Moshi depth-decoder layers).
            """
            if recurse:
                return True
            if isinstance(module, Embedding):
                return False
            if type(module).__name__ not in layer_type_names:
                return False
            if lora_only and not _has_trainable(module):
                return False
            return True

        return FSDPv2(
            model,
            mesh=self._mesh,
            auto_wrap_policy=_wrap_policy,
            shard_output=_shard_output,
        )

    # ------------------------------------------------------------------
    # Per-step ops
    # ------------------------------------------------------------------

    def optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Optimizer step. Behaviour depends on FSDPv2 vs replicated.

        Per pytorch/xla FSDPv2 docs:
            "When stepping the optimizer, directly call optimizer.step and
             DO NOT call xm.optimizer_step. The latter reduces the gradient
             across ranks, which is not needed for FSDP (where the
             parameters are already sharded)."
            -- https://docs.pytorch.org/xla/master/perf/fsdp.html

        Calling xm.optimizer_step on a model wrapped with FSDPv2 issues
        a SECOND all-reduce on top of the FSDPv2 reduce-scatter, and the
        heterogeneous parameter groups (LoRA + full_ft + projection +
        depth + text_embed) cause the second collective to hang at a
        futex barrier (PyTorch/XLA #3424, HF transformers #41881).

        We therefore:
        - For replicated strategy: keep xm.optimizer_step (need the
          all-reduce + execution barrier).
        - For fsdpv2 / fsdpv2_lora: call optimizer.step() then
          xm.mark_step() to materialise.
        """
        import torch_xla.core.xla_model as xm

        if self._strategy in ("fsdpv2", "fsdpv2_lora"):
            optimizer.step()
            xm.mark_step()
        else:
            xm.optimizer_step(optimizer)

    def backward(self, loss: torch.Tensor) -> None:
        """``loss.backward()``. SPMD inserts the gradient reductions
        during compile, so there is nothing extra to do here."""
        loss.backward()

    def save_checkpoint(self, state: dict, path: str) -> None:
        """``xm.save`` -- writes once per master ordinal after a barrier."""
        import torch_xla.core.xla_model as xm

        xm.save(state, path)

    def load_checkpoint(self, path: str) -> dict:
        """Load checkpoint to host CPU; SPMD will re-shard on assign."""
        return torch.load(path, map_location="cpu", weights_only=False)

    def barrier(self) -> None:
        """``xm.rendezvous("barrier")``. GPU analogue: ``dist.barrier()``."""
        import torch_xla.core.xla_model as xm

        xm.rendezvous("barrier")

    def is_main_process(self) -> bool:
        """True only on the GLOBAL primary host (host 0 + master ordinal).

        On a multi-host TPU pod (e.g. v4-32 with 4 hosts) every host
        independently runs a Python process, and ``xm.is_master_ordinal``
        returns True on each one. That breaks "rank-0-only" patterns
        (wandb.init, file writes, console logs) -- you get N copies.

        We gate on ``xr.host_index() == 0`` AND ``xm.is_master_ordinal()``
        so the predicate is True on exactly one process across the
        entire pod.
        """
        import torch_xla.core.xla_model as xm
        import torch_xla.runtime as xr

        try:
            host_idx = xr.host_index()
        except Exception:
            host_idx = 0
        return host_idx == 0 and xm.is_master_ordinal()

    def host_index(self) -> int:
        """0-based host index across the TPU pod (0 on a single-host VM)."""
        import torch_xla.runtime as xr

        try:
            return xr.host_index()
        except Exception:
            return 0

    def world_size(self) -> int:
        """Number of TPU chips visible to this PJRT runtime."""
        if self._world_size_val is not None:
            return self._world_size_val
        import torch_xla.runtime as xr

        return xr.global_runtime_device_count()

    def reduce_mean(self, tensor: torch.Tensor) -> torch.Tensor:
        """Mean-reduce across chips with ``xm.mesh_reduce``."""
        import torch_xla.core.xla_model as xm

        return xm.mesh_reduce("reduce_mean", tensor, lambda x: sum(x) / len(x))

    def autocast_context(self, dtype=torch.bfloat16):
        """No-op context manager.

        Notes
        -----
        TPU note: the bf16 cast applied by :meth:`wrap_model` is
        permanent for the duration of training. Per-op autocast adds
        nothing but XLA tracing overhead, so we return
        ``nullcontext()``.
        """
        return nullcontext()

    def no_sync(self, model: nn.Module):
        """No-op context manager.

        Notes
        -----
        TPU note: SPMD already runs as a single process, so there is
        no per-rank gradient sync to skip during gradient
        accumulation. The XLA partitioner inserts the cross-chip
        reductions only at ``optimizer_step`` time, which is exactly
        when we want them.
        """
        return nullcontext()

    def _warn_memory_unavailable(self, reason: str) -> None:
        """Emit a one-time warning when TPU HBM telemetry cannot be read.

        Parameters
        ----------
        reason : str
            Human-readable reason included in the tmux log.

        Notes
        -----
        TPU note: under SPMD, ``xm.get_memory_info`` can report all
        zeros instead of raising. A single explicit warning prevents
        operators from mistaking missing HBM telemetry for true zero
        memory usage.
        """
        if self._memory_warning_emitted:
            return
        print(
            "[tpu_backend] WARNING: HBM telemetry unavailable; "
            f"{reason}. mem/* metrics will be -1.",
            flush=True,
        )
        self._memory_warning_emitted = True

    def _unknown_memory_info(self, reason: str) -> dict[str, float]:
        """Return a sentinel memory record for unavailable TPU HBM stats.

        Parameters
        ----------
        reason : str
            Human-readable reason logged once for the operator.

        Returns
        -------
        dict[str, float]
            Sentinel values. ``-1`` is intentionally not a valid HBM
            size, so W&B charts make missing telemetry obvious.
        """
        self._warn_memory_unavailable(reason)
        return {
            "allocated_gb": -1.0,
            "max_allocated_gb": -1.0,
            "limit_gb": -1.0,
            "hbm_available": 0.0,
        }

    def _find_tpu_info_binary(self) -> str | None:
        """Find the ``tpu-info`` CLI used as SPMD HBM fallback.

        Returns
        -------
        str or None
            Path to an executable ``tpu-info`` binary, if available.

        Notes
        -----
        TPU note: hot redeploy launches via ``sudo uv run python`` on
        fresh TPU VMs. Depending on how uv resolves Python, ``sys.prefix``
        may not be the virtualenv that owns console scripts. Check PATH,
        the active executable's sibling directory, and common TPU/uv
        locations before giving up.
        """
        candidates: list[str | None] = [
            os.environ.get("TPU_INFO_BIN"),
            shutil.which("tpu-info"),
            str(Path(sys.executable).parent / "tpu-info"),
            str(Path(sys.prefix) / "bin" / "tpu-info"),
            "/usr/local/bin/tpu-info",
            "/usr/bin/tpu-info",
            "/opt/conda/bin/tpu-info",
            "/root/.local/bin/tpu-info",
        ]
        seen: set[str] = set()
        for candidate in candidates:
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            path = Path(candidate)
            if path.is_file() and os.access(path, os.X_OK):
                return str(path)
        return None

    def _get_tpu_info_memory_info(self) -> dict[str, float]:
        """Read TPU HBM usage through ``tpu-info``.

        Returns
        -------
        dict[str, float]
            Parsed HBM usage for chip 0, or the standard unavailable
            sentinel when ``tpu-info`` cannot provide a value.
        """
        import subprocess as _sp

        tpu_info = self._find_tpu_info_binary()
        if not tpu_info:
            return self._unknown_memory_info("tpu-info binary was not found on PATH")

        try:
            out = _sp.run(
                [tpu_info, "--metric", "hbm_usage"],
                capture_output=True,
                text=True,
                timeout=10,
                env={**os.environ, "PJRT_DEVICE": "TPU"},
            )
        except Exception as exc:
            return self._unknown_memory_info(f"tpu-info failed with {exc!r}")

        output = f"{out.stdout}\n{out.stderr}".strip()
        if out.returncode != 0 or "N/A" in out.stdout:
            return self._unknown_memory_info(
                f"tpu-info returned no HBM usage (exit={out.returncode}, output={output[:160]!r})"
            )

        for line in out.stdout.splitlines():
            # tpu-info table cell for chip 0. The HBM unit (GiB) lives in
            # the column header on newer builds, so the cell is a bare
            # number ("| 0 | 12.34 |"); older builds embed it ("12.34 GiB
            # / 31.25 GiB"). Accept both, plus an optional "/ limit".
            match = re.match(
                r"^\|\s*0\s*\|\s*([0-9.]+)\s*(?:GiB)?"
                r"(?:\s*/\s*([0-9.]+)\s*(?:GiB)?)?\s*\|",
                line,
            )
            if not match:
                continue
            used_gb = float(match.group(1))
            limit_gb = float(match.group(2)) if match.group(2) else 31.246
            return {
                "allocated_gb": used_gb,
                "max_allocated_gb": used_gb,
                "limit_gb": limit_gb,
                "hbm_available": 1.0,
            }

        return self._unknown_memory_info(
            f"tpu-info returned no parseable hbm_usage output: {out.stdout[:160]!r}"
        )

    def get_memory_info(self) -> dict | None:
        """Return per-chip HBM usage in GB, or ``None`` on import failure.

        Notes
        -----
        TPU note: ``xm.get_memory_info(device)`` reports HBM stats
        for the calling chip only. To see all chips, use
        ``xr.global_runtime_device_count()`` and probe each in turn.
        For routine training we only print the master chip's usage,
        which is representative under SPMD because each chip holds
        roughly the same shard.

        SPMD limitation (pytorch/xla #9022, closed 2026-01 as
        deprioritized): ``xm.get_memory_info`` is NOT supported
        under SPMD -- it returns all-zeros rather than raising.
        The iter-18 ``mark_step + wait_device_ops`` workaround
        was insufficient because the limitation is in the PJRT
        computation client, not a timing issue.

        Iter 19 fix: when running under SPMD, fall back to the
        ``tpu-info`` CLI tool (``tpu-info --bytes_used``) which
        queries the TPU runtime directly and works under SPMD.
        When NOT under SPMD, use ``xm.get_memory_info`` as before.
        """
        try:
            import torch_xla.core.xla_model as xm
            import torch_xla.runtime as xr
        except Exception as exc:
            return self._unknown_memory_info(f"torch_xla memory imports failed with {exc!r}")

        try:
            is_spmd = xr.using_spmd()
        except AttributeError:
            return self._get_tpu_info_memory_info()
        except Exception as exc:
            info = self._get_tpu_info_memory_info()
            if info.get("hbm_available", 0.0) == 1.0:
                return info
            return self._unknown_memory_info(f"xr.using_spmd failed with {exc!r}")

        if is_spmd:
            return self._get_tpu_info_memory_info()

        try:
            mem = xm.get_memory_info(self.get_device())
        except Exception as exc:
            return self._unknown_memory_info(f"xm.get_memory_info failed with {exc!r}")

        allocated_gb = mem.get("bytes_used", 0) / 1e9
        peak_gb = mem.get("peak_bytes_used", 0) / 1e9
        limit_gb = mem.get("bytes_limit", 0) / 1e9
        if allocated_gb == 0 and peak_gb == 0 and limit_gb == 0:
            return self._get_tpu_info_memory_info()

        return {
            "allocated_gb": allocated_gb,
            "max_allocated_gb": peak_gb,
            "limit_gb": limit_gb,
            "hbm_available": 1.0,
        }

    def sync(self) -> None:
        """``torch_xla.sync()`` -- fence the lazy graph builder."""
        import torch_xla

        torch_xla.sync()

    # ------------------------------------------------------------------
    # SPMD-only methods (no-op on GPU)
    # ------------------------------------------------------------------

    def mark_sharding(self, tensor: torch.Tensor, partition_spec: tuple) -> None:
        """Tell the SPMD partitioner how ``tensor`` is sharded.

        GPU analogue: ``DistributedSampler`` putting different rows
        on different ranks. SPMD is more general -- it can shard any
        dimension of any tensor, not just the batch dim.
        """
        import torch_xla.distributed.spmd as xs

        xs.mark_sharding(tensor, self._mesh, partition_spec)

    def diagnose(self, tag: str = "diagnose") -> None:
        """Print mesh layout + per-chip HBM. Cheap; safe every N steps.

        Output line example::

            [tpu_backend][step=10] global=16 local=4 strategy=fsdpv2_lora
            hbm_used=7.84GB/limit=15.75GB peak=10.21GB

        ``global`` is the total chip count across all hosts; ``local``
        is the chips on *this* host (always 4 on v5e).
        """
        try:
            import torch_xla.runtime as xr

            mem = self.get_memory_info() or {}
            n_global = xr.global_runtime_device_count()
            n_local = xr.addressable_runtime_device_count()
            print(
                f"[tpu_backend][{tag}] global={n_global} local={n_local} "
                f"strategy={self._strategy} "
                f"hbm_used={mem.get('allocated_gb', 0):.2f}GB/"
                f"limit={mem.get('limit_gb', 0):.2f}GB "
                f"peak={mem.get('max_allocated_gb', 0):.2f}GB"
            )
        except Exception as e:
            print(f"[tpu_backend][{tag}] diagnose failed: {e}")
