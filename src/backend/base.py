"""Abstract base class for training backends.

WHY THIS EXISTS
---------------
Stage 2 trains on either GPUs (single-host or DDP) or TPU pods
(SPMD). The training loop in ``scripts/train_hierarchical.py`` calls
the same handful of methods regardless: ``init_distributed``,
``wrap_model``, ``optimizer_step``, ``barrier``, ``mark_sharding``,
... This module declares that interface as an abstract base class so
both backends can be type-checked and so the training script never
imports ``torch_xla`` directly (avoiding hard import failures on a
GPU dev box).

GPU vs TPU at a glance
----------------------
* On GPU we initialise NCCL via ``torch.distributed`` and wrap the
  model in ``DistributedDataParallel``.
* On TPU we initialise PJRT + an SPMD ``Mesh`` and either replicate
  the model or wrap it in ``SpmdFullyShardedDataParallel`` (FSDPv2).

Two methods are *only* meaningful on TPU and provide a no-op default
here so GPU code can call them blindly:

* ``mark_sharding(tensor, partition_spec)`` -- the SPMD sharding
  hint. GPU's analogue is the ``DistributedSampler`` putting different
  rows on different ranks; SPMD makes that decision at the partitioner
  level instead.
* ``diagnose(tag)`` -- a per-chip HBM probe. No GPU equivalent needed
  because the OOM error is per-process and immediately visible.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BackendBase(ABC):
    """Interface that hides GPU vs TPU differences.

    Concrete subclasses live in ``gpu_backend.py`` and
    ``tpu_backend.py``. The training script depends only on this
    abstract interface and never imports the concrete backends.
    """

    @abstractmethod
    def init_distributed(self) -> None:
        """Initialise distributed training.

        GPU analogue: ``torch.distributed.init_process_group`` + NCCL.

        TPU note: sets up PJRT, calls ``xr.use_spmd()``, and creates
        a 1-D ``Mesh`` over all chips named ``"fsdp"``.
        """

    @abstractmethod
    def get_device(self) -> torch.device:
        """Return the device that fresh tensors should be moved to.

        GPU returns ``cuda:<local_rank>``; TPU returns the lazy XLA
        device (``xla:0`` etc.).
        """

    @abstractmethod
    def wrap_model(self, model: nn.Module) -> nn.Module:
        """Wrap ``model`` for distributed training.

        GPU returns a ``DistributedDataParallel`` (or the bare model
        for single-GPU runs). TPU returns a model with SPMD sharding
        spec attached, optionally wrapped in FSDPv2.
        """

    @abstractmethod
    def optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Run ``optimizer.step`` with the right cross-device sync.

        GPU calls ``optimizer.step()`` directly; the gradient
        all-reduce already happened in ``loss.backward()`` under DDP.
        TPU calls ``xm.optimizer_step(optimizer)`` which fences XLA
        execution and triggers the SPMD-side gradient reduction.
        """

    @abstractmethod
    def backward(self, loss: torch.Tensor) -> None:
        """Run backward. Both backends call ``loss.backward()`` here."""

    @abstractmethod
    def save_checkpoint(self, state: dict, path: str) -> None:
        """Save a checkpoint dictionary.

        GPU writes once from rank 0. TPU uses ``xm.save`` which
        gathers shards from all chips before writing.
        """

    @abstractmethod
    def load_checkpoint(self, path: str) -> dict:
        """Load a checkpoint dictionary onto host CPU."""

    @abstractmethod
    def barrier(self) -> None:
        """Synchronisation barrier across all processes/chips.

        GPU uses ``dist.barrier()``. TPU uses ``xm.rendezvous``.
        """

    @abstractmethod
    def is_main_process(self) -> bool:
        """Whether the calling process should perform single-rank work
        (logging, checkpoint write, W&B init)."""

    @abstractmethod
    def world_size(self) -> int:
        """Number of GPU ranks or TPU chips participating."""

    @abstractmethod
    def reduce_mean(self, tensor: torch.Tensor) -> torch.Tensor:
        """All-reduce ``tensor`` with mean across devices."""

    @abstractmethod
    def autocast_context(self, dtype=torch.bfloat16):
        """Return an autocast context manager.

        GPU returns ``torch.amp.autocast("cuda", dtype=dtype)``.
        TPU returns ``nullcontext()`` -- the bf16 cast is applied
        once at ``wrap_model`` time and remains the dtype thereafter,
        so per-op autocast adds nothing.
        """

    @abstractmethod
    def no_sync(self, model: nn.Module):
        """Return a context manager that suppresses gradient sync.

        Used during gradient accumulation to skip cross-device
        all-reduce on every micro-step.
        """

    @abstractmethod
    def get_memory_info(self) -> dict | None:
        """Return device memory usage as a dict with at least
        ``allocated_gb`` and ``max_allocated_gb`` keys, or ``None``."""

    @abstractmethod
    def sync(self) -> None:
        """Trigger execution / compilation.

        GPU is a no-op (CUDA is eager). TPU calls ``torch_xla.sync()``
        which fences the lazy XLA graph builder.
        """

    # ------------------------------------------------------------------
    # Optional methods with no-op defaults. GPU code may call these
    # blindly without importing torch_xla.
    # ------------------------------------------------------------------

    def diagnose(self, tag: str = "diagnose") -> None:
        """Optional hardware-utilisation diagnostic.

        Default: no-op. ``TPUBackend`` overrides this to print mesh
        layout and per-chip HBM. GPU has no analogue because OOM is
        immediate and visible.
        """
        return None

    def mark_sharding(self, tensor: torch.Tensor, partition_spec: tuple) -> None:
        """Optional SPMD sharding hint.

        Default: no-op. ``TPUBackend`` overrides this with
        ``xs.mark_sharding`` which tells the SPMD partitioner how
        ``tensor`` should be split across the mesh.

        GPU analogue: ``DistributedSampler`` placing different batches
        on different ranks. SPMD is more general -- it works on any
        tensor dimension, not just the batch dim.
        """
        return None
