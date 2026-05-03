"""Abstract base class for training backends."""
from abc import ABC, abstractmethod
from contextlib import nullcontext

import torch
import torch.nn as nn


class BackendBase(ABC):
    """Interface that hides GPU vs TPU differences."""

    @abstractmethod
    def init_distributed(self) -> None:
        """Initialize distributed training (DDP or SPMD)."""

    @abstractmethod
    def get_device(self) -> torch.device:
        """Return the device to place tensors on."""

    @abstractmethod
    def wrap_model(self, model: nn.Module) -> nn.Module:
        """Wrap model for distributed training (DDP or FSDPv2)."""

    @abstractmethod
    def optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Execute optimizer step with backend-specific sync."""

    @abstractmethod
    def backward(self, loss: torch.Tensor) -> None:
        """Run backward pass."""

    @abstractmethod
    def save_checkpoint(self, state: dict, path: str) -> None:
        """Save checkpoint (local or GCS)."""

    @abstractmethod
    def load_checkpoint(self, path: str) -> dict:
        """Load checkpoint (local or GCS)."""

    @abstractmethod
    def barrier(self) -> None:
        """Synchronization barrier across all processes/chips."""

    @abstractmethod
    def is_main_process(self) -> bool:
        """Whether this is the main process (for logging/saving)."""

    @abstractmethod
    def world_size(self) -> int:
        """Number of devices/processes."""

    @abstractmethod
    def reduce_mean(self, tensor: torch.Tensor) -> torch.Tensor:
        """All-reduce a tensor with mean across devices."""

    @abstractmethod
    def autocast_context(self, dtype=torch.bfloat16):
        """Return an autocast context manager (or nullcontext for TPU)."""

    @abstractmethod
    def no_sync(self, model: nn.Module):
        """Return a no_sync context manager for gradient accumulation."""

    @abstractmethod
    def get_memory_info(self) -> dict | None:
        """Return memory usage info, or None if unavailable."""

    @abstractmethod
    def sync(self) -> None:
        """Trigger execution/compilation (needed for lazy backends like XLA)."""

    def diagnose(self, tag: str = "diagnose") -> None:
        """Optional hardware-utilization diagnostic. No-op for backends that
        don't implement it."""
        return None

    def mark_sharding(self, tensor: torch.Tensor, partition_spec: tuple) -> None:
        """Optional SPMD sharding hint. No-op outside TPU SPMD."""
        return None
