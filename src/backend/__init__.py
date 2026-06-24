"""Backend abstraction for GPU/TPU training.

WHY THIS EXISTS
---------------
Training code should not have to care whether it is running on a CUDA
GPU or a TPU pod. This module hides that decision behind two helpers:

* ``detect_backend()`` -- picks the right backend by checking
  ``DEVICE_BACKEND`` (explicit override) or by trying to import
  ``torch_xla`` (implicit detection).
* ``get_backend(backend_type)`` -- returns a concrete backend
  instance.

Both backend classes share a duck-typed interface declared in
``src/backend/base.py``: ``init_distributed``, ``wrap_model``,
``optimizer_step``, ``mark_sharding`` (TPU-only no-op on GPU), etc.
"""

import os


def detect_backend() -> str:
    """Detect the available backend.

    Returns
    -------
    str
        ``"tpu"`` if ``DEVICE_BACKEND=tpu`` or ``torch_xla`` imports
        cleanly; ``"gpu"`` otherwise.

    Notes
    -----
    TPU note: importing ``torch_xla.core.xla_model`` and calling
    ``xla_device()`` does *not* require a real TPU -- on a CPU box
    with the package installed, it returns a CPU-backed XLA device.
    The function therefore prefers the explicit ``DEVICE_BACKEND``
    override when set.
    """
    explicit = os.environ.get("DEVICE_BACKEND", "").lower()
    if explicit in ("tpu", "gpu"):
        return explicit

    try:
        import torch_xla.core.xla_model as xm

        # Touching ``xla_device()`` forces ``torch_xla`` to fully
        # import; the call result is intentionally unused.
        xm.xla_device()
        return "tpu"
    except Exception:
        return "gpu"


def get_backend(backend_type: str | None = None):
    """Return a concrete backend instance.

    Parameters
    ----------
    backend_type : {"tpu", "gpu", None}, default None
        Explicit selector. ``None`` triggers ``detect_backend()``.

    Returns
    -------
    BackendBase
        Either ``TPUBackend`` or ``GPUBackend``.
    """
    if backend_type is None:
        backend_type = detect_backend()

    if backend_type == "tpu":
        from src.backend.tpu_backend import TPUBackend

        return TPUBackend()
    from src.backend.gpu_backend import GPUBackend

    return GPUBackend()
