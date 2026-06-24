"""GPU/CUDA backend with optional DDP support.

WHY THIS EXISTS
---------------
The training loop runs on a workstation GPU during development. This
backend provides the GPU side of the abstraction declared in
``base.py`` -- standard PyTorch, no surprises:

* Single GPU: a thin wrapper that delegates to plain
  ``optimizer.step()``, ``loss.backward()``, etc.
* Multi-GPU: NCCL + ``DistributedDataParallel``, launched via
  ``torchrun``.

This file deliberately contains zero TPU code so that an engineer
debugging GPU runs can read it top-to-bottom without learning XLA.
"""

import os
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn as nn

from src.backend.base import BackendBase


class GPUBackend(BackendBase):
    """``BackendBase`` implementation for CUDA GPUs (single or DDP).

    Notes
    -----
    The constructor reads ``LOCAL_RANK`` and ``WORLD_SIZE`` from the
    environment, which ``torchrun`` sets automatically. Single-process
    runs land on rank 0 with world size 1 and the DDP path is a no-op.
    """

    def __init__(self):
        self._local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self._world_size = int(os.environ.get("WORLD_SIZE", 1))
        self._is_ddp = self._world_size > 1
        self._device: torch.device | None = None

    def init_distributed(self) -> None:
        """Initialise NCCL if running under ``torchrun``."""
        if self._is_ddp:
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(self._local_rank)
        self._device = torch.device(
            f"cuda:{self._local_rank}" if torch.cuda.is_available() else "cpu"
        )

    def get_device(self) -> torch.device:
        """Return ``cuda:<local_rank>`` or ``cpu`` if no CUDA is visible."""
        if self._device is None:
            self._device = torch.device(
                f"cuda:{self._local_rank}" if torch.cuda.is_available() else "cpu"
            )
        return self._device

    def wrap_model(self, model: nn.Module) -> nn.Module:
        """Wrap in DDP when ``WORLD_SIZE > 1``, else return as-is."""
        if self._is_ddp:
            return nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self._local_rank],
                output_device=self._local_rank,
                # find_unused_parameters=True covers heads that are
                # frozen at runtime (depth decoder internals); the
                # cost is a per-step graph trace which is acceptable.
                find_unused_parameters=True,
                broadcast_buffers=False,
            )
        return model

    def optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """``optimizer.step()`` -- gradients were already all-reduced by DDP."""
        optimizer.step()

    def backward(self, loss: torch.Tensor) -> None:
        """``loss.backward()``."""
        loss.backward()

    def save_checkpoint(self, state: dict, path: str) -> None:
        """Write checkpoint from rank 0 only."""
        if self.is_main_process():
            torch.save(state, path)

    def load_checkpoint(self, path: str) -> dict:
        """Load checkpoint to host CPU (``map_location="cpu"``)."""
        return torch.load(path, map_location="cpu", weights_only=False)

    def barrier(self) -> None:
        """``dist.barrier()`` under DDP, no-op otherwise."""
        if self._is_ddp:
            dist.barrier()

    def is_main_process(self) -> bool:
        """True on rank 0."""
        return self._local_rank == 0

    def world_size(self) -> int:
        """Number of GPU ranks (1 for single-process)."""
        return self._world_size

    def reduce_mean(self, tensor: torch.Tensor) -> torch.Tensor:
        """All-reduce-sum then divide by ``world_size`` to get the mean."""
        if self._is_ddp:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor /= self._world_size
        return tensor

    def autocast_context(self, dtype=torch.bfloat16):
        """Standard ``torch.amp.autocast`` (CUDA or CPU device type)."""
        if not torch.cuda.is_available():
            return torch.amp.autocast("cpu", dtype=dtype)
        return torch.amp.autocast("cuda", dtype=dtype)

    def no_sync(self, model: nn.Module):
        """Suppress DDP gradient all-reduce during gradient-accumulation
        micro-steps. No-op for single-GPU runs."""
        if self._is_ddp:
            return model.no_sync()
        return nullcontext()

    def get_memory_info(self) -> dict | None:
        """Return CUDA memory usage in GB, or ``None`` on CPU-only systems."""
        if torch.cuda.is_available():
            return {
                "allocated_gb": torch.cuda.memory_allocated() / 1e9,
                "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            }
        return None

    def sync(self) -> None:
        """No-op. CUDA is eager; nothing to flush."""
        pass
