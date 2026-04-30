"""TPU backend with SPMD + FSDPv2 support.

SPMD Trade-off Note:
    SPMD runs as a single process across all TPU chips. Unlike DDP, there's
    no per-chip process isolation -- if one chip OOMs or errors, the entire
    job fails. Mitigated by: frequent async checkpoints, spot preemption
    resume, and conservative memory settings. If stability issues arise,
    consider falling back to multi-process xmp.spawn + DDP-style training
    via torch_xla.distributed.xla_backend.
"""
import os
from contextlib import nullcontext

import torch
import torch.nn as nn

from src.backend.base import BackendBase


class TPUBackend(BackendBase):
    def __init__(self):
        self._device = None
        self._mesh = None
        self._world_size_val = None

    def init_distributed(self) -> None:
        import torch_xla.core.xla_model as xm
        import torch_xla.runtime as xr
        import torch_xla.distributed.spmd as xs
        from torch_xla.distributed.spmd import Mesh

        xr.use_spmd()
        self._device = xm.xla_device()

        num_devices = xr.global_runtime_device_count()
        self._world_size_val = num_devices
        self._mesh = Mesh(
            device_ids=list(range(num_devices)),
            mesh_shape=(num_devices,),
            axis_names=("fsdp",),
        )
        print(f"TPU SPMD initialized: {num_devices} devices, mesh={self._mesh.mesh_shape}")

    def get_device(self) -> torch.device:
        if self._device is None:
            import torch_xla.core.xla_model as xm
            self._device = xm.xla_device()
        return self._device

    def wrap_model(self, model: nn.Module) -> nn.Module:
        from torch_xla.experimental.spmd_fully_sharded_data_parallel import (
            SpmdFullyShardedDataParallel as FSDPv2,
        )
        # Gradient checkpointing MUST be enabled BEFORE FSDPv2 wrapping
        # (torch_xla requirement -- otherwise infinite recursion)
        if hasattr(model, "backbone") and hasattr(model.backbone, "model"):
            model.backbone.model.gradient_checkpointing_enable()

        model = FSDPv2(model, mesh=self._mesh)
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
        return None

    def sync(self) -> None:
        import torch_xla
        torch_xla.sync()

    def mark_sharding(self, tensor: torch.Tensor, partition_spec: tuple) -> None:
        """Mark a tensor for SPMD sharding across the mesh."""
        import torch_xla.distributed.spmd as xs
        xs.mark_sharding(tensor, self._mesh, partition_spec)
