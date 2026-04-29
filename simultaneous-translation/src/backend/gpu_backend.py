"""GPU/CUDA backend with optional DDP support."""
import os
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn as nn

from src.backend.base import BackendBase


class GPUBackend(BackendBase):
    def __init__(self):
        self._local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self._world_size = int(os.environ.get("WORLD_SIZE", 1))
        self._is_ddp = self._world_size > 1
        self._device = None

    def init_distributed(self) -> None:
        if self._is_ddp:
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(self._local_rank)
        self._device = torch.device(f"cuda:{self._local_rank}" if torch.cuda.is_available() else "cpu")

    def get_device(self) -> torch.device:
        if self._device is None:
            self._device = torch.device(f"cuda:{self._local_rank}" if torch.cuda.is_available() else "cpu")
        return self._device

    def wrap_model(self, model: nn.Module) -> nn.Module:
        if self._is_ddp:
            return nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self._local_rank],
                output_device=self._local_rank,
                find_unused_parameters=True,
                broadcast_buffers=False,
            )
        return model

    def optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        optimizer.step()

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()

    def save_checkpoint(self, state: dict, path: str) -> None:
        if self.is_main_process():
            torch.save(state, path)

    def load_checkpoint(self, path: str) -> dict:
        return torch.load(path, map_location="cpu", weights_only=False)

    def barrier(self) -> None:
        if self._is_ddp:
            dist.barrier()

    def is_main_process(self) -> bool:
        return self._local_rank == 0

    def world_size(self) -> int:
        return self._world_size

    def reduce_mean(self, tensor: torch.Tensor) -> torch.Tensor:
        if self._is_ddp:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor /= self._world_size
        return tensor

    def autocast_context(self, dtype=torch.bfloat16):
        if not torch.cuda.is_available():
            return torch.amp.autocast("cpu", dtype=dtype)
        return torch.amp.autocast("cuda", dtype=dtype)

    def no_sync(self, model: nn.Module):
        if self._is_ddp:
            return model.no_sync()
        return nullcontext()

    def get_memory_info(self) -> dict | None:
        if torch.cuda.is_available():
            return {
                "allocated_gb": torch.cuda.memory_allocated() / 1e9,
                "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            }
        return None

    def sync(self) -> None:
        pass
