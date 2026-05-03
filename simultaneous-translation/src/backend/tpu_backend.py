"""TPU backend with multiple SPMD sharding strategies.

Strategy selection is controlled by the TPU_STRATEGY environment variable:

  replicated  Each chip holds a full copy of the model. Inputs are sharded
              across the batch dimension via xs.mark_sharding. Gradients are
              implicitly all-reduced when XLA compiles the backward pass.
              Best for: small trainable parameter counts (LoRA), tight HBM.
              No FSDP partitioner involved -> avoids the SPMD partitioner
              "ShapeUtil::IsScalarWithElementType" crash on torch_xla 2.9.

  fsdpv2      SpmdFullyShardedDataParallel wrapping the whole composite
              model. Shards weights, grads, and optimizer state across the
              fsdp axis. Lowest per-chip memory footprint, but currently
              hits a partitioner crash on v5e + torch_xla 2.9 unless
              XLA_DISABLE_FUNCTIONALIZATION=0 (the default in 2.9 anyway).

  fsdpv2_lora Wrap only modules that contain trainable LoRA parameters with
              FSDPv2. The frozen backbone remains replicated. Compromise:
              shards optimizer state (the bulk of trainable-side memory)
              but keeps the heavy frozen weights cheap to materialize.

  auto        Pick replicated if trainable_params < 500M else fsdpv2.

Hardware-utilization diagnostics:

  diagnose()  Prints per-chip HBM (used / total), mesh layout, and the
              sharding spec of any tensor passed in. Use from the trainer
              after the first forward to confirm the partitioner sharded
              activations the way we expect.

SPMD trade-off note:
  SPMD runs as a single process across all TPU chips. If any chip OOMs or
  errors, the whole job dies. Mitigated by frequent async checkpoints and
  the spot-preemption restart loop in scripts/tpu/startup_script.sh.
"""
from __future__ import annotations

import os
from contextlib import nullcontext

import torch
import torch.nn as nn

from src.backend.base import BackendBase


_VALID_STRATEGIES = ("auto", "replicated", "fsdpv2", "fsdpv2_lora")


def _resolve_strategy(model: nn.Module | None) -> str:
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
    print(f"[tpu_backend] auto-strategy: trainable={trainable/1e6:.0f}M -> {chosen}")
    return chosen


class TPUBackend(BackendBase):
    def __init__(self):
        self._device = None
        self._mesh = None
        self._world_size_val = None
        self._strategy = None

    def init_distributed(self) -> None:
        import torch_xla.core.xla_model as xm
        import torch_xla.runtime as xr
        from torch_xla.distributed.spmd import Mesh

        # Functionalization must stay enabled on torch_xla 2.9 to avoid the
        # SPMD partitioner crash described in pytorch/xla#8607. We force it
        # here defensively even though 2.9's default is already on.
        os.environ.setdefault("XLA_DISABLE_FUNCTIONALIZATION", "0")
        # XLA_USE_BF16 was removed in torch_xla 2.6+. We cast the model to
        # bf16 explicitly inside wrap_model() instead. On v5e (16 GiB HBM)
        # the 5.17B composite model can't fit in f32 -- we OOM during XLA
        # compile if compute stays at f32.

        xr.use_spmd()
        self._device = xm.xla_device()

        num_devices = xr.global_runtime_device_count()
        self._world_size_val = num_devices
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
        if self._device is None:
            import torch_xla.core.xla_model as xm
            self._device = xm.xla_device()
        return self._device

    def wrap_model(self, model: nn.Module) -> nn.Module:
        # HF gradient_checkpointing_enable() is incompatible with torch 2.9 +
        # XLA (its internal _get_device_module does getattr(torch, "xla")
        # which raises). On v5e with the strategies below we have enough HBM
        # for batch_size=1 + bf16 weights without checkpointing for the
        # canary; if we OOM, switch on torch.utils.checkpoint manually.

        # Cast to bf16 to fit the model in 16 GiB/chip on v5e. Trainable
        # parameters stay in bf16 too -- we accept the precision loss for
        # LoRA fine-tuning (standard practice in HF PEFT). AdamW keeps its
        # moment buffers in f32 internally even when params are bf16, so
        # optimizer numerics stay clean.
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

        We rely on XLA's all-reduce-on-backward via SPMD partitioner: when
        the inputs are sharded along the batch axis and the parameters are
        replicated, the partitioner inserts the appropriate cross-replica
        sums during gradient accumulation automatically. No FSDP wrapper.
        """
        import torch_xla.distributed.spmd as xs

        for name, param in model.named_parameters():
            if param.requires_grad:
                xs.mark_sharding(param, self._mesh, (None,) * param.dim())
        for name, buf in model.named_buffers():
            xs.mark_sharding(buf, self._mesh, (None,) * buf.dim())
        print(
            "[tpu_backend] replicated: marked all parameters and buffers as "
            "replicated; data will be sharded on batch dim by mark_sharding "
            "calls in the train loop"
        )
        return model

    def _wrap_fsdpv2(self, model: nn.Module, *, lora_only: bool) -> nn.Module:
        from torch_xla.experimental.spmd_fully_sharded_data_parallel import (
            SpmdFullyShardedDataParallel as FSDPv2,
        )
        from torch.nn import Embedding
        import torch_xla.distributed.spmd as xs

        def _shard_output(output, mesh):
            """Composite returns (text_logits, audio_logits, hidden_states).

            All three need an explicit sharding spec or the SPMD partitioner
            sees them as replicated and asserts in spmd_partitioner_util.h.
            We shard each on the fsdp (batch) axis; remaining dims are
            replicated.
            """
            sharded = []
            for v in output:
                if isinstance(v, torch.Tensor) and v.dim() >= 1:
                    spec = ("fsdp",) + (None,) * (v.dim() - 1)
                    xs.mark_sharding(v, mesh, spec)
                sharded.append(v)
            return tuple(sharded)

        layer_type_names = ("CohereDecoderLayer", "MoshiDecoderLayer")

        def _has_trainable(m: nn.Module) -> bool:
            return any(p.requires_grad for p in m.parameters(recurse=True))

        def _wrap_policy(module, recurse, **kwargs):
            if recurse:
                return True
            if isinstance(module, Embedding):
                return False
            if type(module).__name__ not in layer_type_names:
                return False
            if lora_only and not _has_trainable(module):
                return False
            return True

        model = FSDPv2(
            model,
            mesh=self._mesh,
            auto_wrap_policy=_wrap_policy,
            shard_output=_shard_output,
        )
        return model

    def optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        import torch_xla.core.xla_model as xm
        xm.optimizer_step(optimizer)

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()

    def save_checkpoint(self, state: dict, path: str) -> None:
        import torch_xla.core.xla_model as xm
        xm.save(state, path)

    def load_checkpoint(self, path: str) -> dict:
        return torch.load(path, map_location="cpu", weights_only=False)

    def barrier(self) -> None:
        import torch_xla.core.xla_model as xm
        xm.rendezvous("barrier")

    def is_main_process(self) -> bool:
        import torch_xla.core.xla_model as xm
        return xm.is_master_ordinal()

    def world_size(self) -> int:
        if self._world_size_val is not None:
            return self._world_size_val
        import torch_xla.runtime as xr
        return xr.global_runtime_device_count()

    def reduce_mean(self, tensor: torch.Tensor) -> torch.Tensor:
        import torch_xla.core.xla_model as xm
        return xm.mesh_reduce("reduce_mean", tensor, lambda x: sum(x) / len(x))

    def autocast_context(self, dtype=torch.bfloat16):
        return nullcontext()

    def no_sync(self, model: nn.Module):
        return nullcontext()

    def get_memory_info(self) -> dict | None:
        """Returns per-chip HBM info from the XLA runtime, if available."""
        try:
            import torch_xla.core.xla_model as xm
            mem = xm.get_memory_info(self.get_device())
            return {
                "allocated_gb": mem.get("bytes_used", 0) / 1e9,
                "max_allocated_gb": mem.get("peak_bytes_used", 0) / 1e9,
                "limit_gb": mem.get("bytes_limit", 0) / 1e9,
            }
        except Exception:
            return None

    def sync(self) -> None:
        import torch_xla
        torch_xla.sync()

    def mark_sharding(self, tensor: torch.Tensor, partition_spec: tuple) -> None:
        """Mark a tensor for SPMD sharding across the mesh."""
        import torch_xla.distributed.spmd as xs
        xs.mark_sharding(tensor, self._mesh, partition_spec)

    def diagnose(self, tag: str = "diagnose") -> None:
        """Print mesh + per-chip HBM. Cheap; safe to call every N steps."""
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
