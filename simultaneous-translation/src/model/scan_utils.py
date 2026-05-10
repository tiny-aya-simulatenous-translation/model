"""XLA-aware ``scan_layers`` and gradient-checkpointing for HF transformers.

WHY THIS EXISTS
---------------
Two concrete problems that bite GPU engineers the moment their model
touches a TPU:

1. **XLA compile time blows up with unrolled transformer stacks.**
   On a GPU, PyTorch eagerly executes 36 ``CohereDecoderLayer`` calls
   in a Python ``for`` loop and the GPU just runs them. On a TPU, the
   first forward pass has to be *traced* into a single XLA HLO graph,
   compiled to TPU machine code, and only then executed. Tracing 36
   identical decoder layers produces 36 copies of the same subgraph
   in the HLO; the XLA optimiser has to walk through all of it. We
   measured 25+ minutes of compile on the 5.17B composite model
   (see PROGRESS 2026-05-03T14:30:00Z).

   ``torch_xla.experimental.scan_layers.scan_layers`` solves this by
   compiling **one** layer's HLO and running it ``N`` times via
   ``xla.while_loop``. The HLO graph stays roughly constant in size as
   the number of layers grows. This is the TPU analogue of GPU
   "kernel re-use": you pay the lowering cost once.

2. **Activation memory under SPMD is per-chip and unforgiving.**
   On a single-GPU GPU run, you can dial up batch size until the GPU
   OOMs, see the OOM, and lower it. On a TPU pod under SPMD, *every
   chip OOMing kills the whole job*; there is no "single chip" in the
   user-facing programming model. So we want gradient checkpointing
   on by default for the layer stacks. ``torch.utils.checkpoint`` with
   ``use_reentrant=False`` is XLA-compatible in torch 2.9 and works
   inside ``scan_layers``' loop body.

THE PROXY PATTERN
-----------------
HuggingFace's ``Cohere2Model.forward`` (and friends) iterates layers
with code like::

    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            ...
        )
        hidden_states = layer_outputs[0]

Re-implementing that whole ``forward`` to call ``scan_layers`` directly
would lock us to a specific HF transformers version. Instead, we swap
``self.layers`` (a ``nn.ModuleList``) for a ``_ScannedLayerStack``
proxy. The proxy implements ``__getitem__(slice(...))`` so that
``self.layers[:N]`` returns **a one-element list** containing a single
"fused" layer. HF's outer for-loop iterates exactly once; the fused
layer runs all ``N`` original layers inside ``scan_layers``.

GPU PATH IS A NO-OP
-------------------
On GPU/CPU, ``scan_layers`` is not imported (``PJRT_DEVICE`` is unset),
the proxy falls back to the manual loop, and gradient checkpointing
runs through ``torch.utils.checkpoint``. So the same composite model
class works on both backends without branching.

References
----------
- pytorch/xla docs on ``scan_layers``:
  https://github.com/pytorch/xla/blob/master/docs/source/learn/scan_layers.rst
- ``.factory/memories.md`` -> "scan_layers wrapper as a ModuleList proxy"
- ``.factory/PLAN.md`` -> Phase 1 + Phase 2
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Backend probes
# ---------------------------------------------------------------------------


def _is_tpu_runtime() -> bool:
    """Return True if we are executing on a real TPU host.

    Returns
    -------
    bool
        True iff ``PJRT_DEVICE`` is set to ``"TPU"``. The variable is
        injected automatically by ``torch_xla`` when ``libtpu`` is
        loadable, and explicitly by our ``startup_script.sh``. On a
        GPU/CPU box it is unset and we should *not* try to import any
        ``torch_xla`` modules (they pull in ``libtpu`` and explode).

    Notes
    -----
    TPU note: do not call ``torch_xla.runtime.is_tpu()`` here -- that
    function imports ``torch_xla`` first, which is what we are trying
    to avoid on GPU/CPU.
    """
    return os.environ.get("PJRT_DEVICE", "").upper() == "TPU"


def _import_scan_layers() -> Callable | None:
    """Best-effort import of ``torch_xla.experimental.scan_layers.scan_layers``.

    Returns
    -------
    callable or None
        The ``scan_layers`` callable if torch_xla is installed and
        importable; ``None`` otherwise. We deliberately swallow *all*
        import errors because:

        * ``ImportError`` -- torch_xla not installed (GPU dev box).
        * ``OSError`` -- libtpu not loadable on a TPU image with a
          mismatched runtime.
        * ``RuntimeError`` -- the experimental API was renamed in a
          torch_xla minor version we don't pin.

        Any of these collapse to "use the manual loop instead".
    """
    try:
        from torch_xla.experimental.scan_layers import scan_layers
    except Exception:
        return None
    return scan_layers


# ---------------------------------------------------------------------------
# Gradient checkpointing wrapper
# ---------------------------------------------------------------------------


def _xla_safe_checkpoint(fn: Callable, *args: Any, **kwargs: Any) -> Any:
    """Run ``fn(*args, **kwargs)`` under gradient checkpointing if possible.

    GPU analogue: ``torch.utils.checkpoint.checkpoint(fn, *args)`` with
    ``use_reentrant=False`` (the recommended modern variant since
    PyTorch 2.0).

    On torch 2.9 + torch_xla 2.9 the non-reentrant variant works on
    XLA: the saved-tensors hooks are device-agnostic and the
    recomputation pass re-traces the forward inside the backward. We
    explicitly set ``use_reentrant=False`` because the reentrant path
    relies on a CUDA-specific RNG state save/restore that asserts on
    XLA tensors.

    Parameters
    ----------
    fn : callable
        The function whose forward we want to checkpoint. Must take
        only tensor positional args plus arbitrary kwargs.
    *args : Any
        Positional args forwarded to ``fn``.
    **kwargs : Any
        Keyword args forwarded to ``fn``. ``torch.utils.checkpoint``
        supports kwargs since 2.0.

    Returns
    -------
    Any
        Whatever ``fn`` returns.

    Notes
    -----
    TPU note: if checkpointing fails with an unfamiliar
    ``AttributeError`` (e.g., the legacy ``torch.utils.checkpoint``
    path that calls ``torch._get_device_module("xla")``), we fall
    through to a direct call. The direct call still works -- you just
    lose the activation-memory savings.
    """
    try:
        from torch.utils.checkpoint import checkpoint as _cp

        return _cp(fn, *args, use_reentrant=False, **kwargs)
    except (AttributeError, TypeError, RuntimeError):
        return fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# Single-layer wrapper with optional checkpointing
# ---------------------------------------------------------------------------


class _KwargBoundLayer(nn.Module):
    """Adapter that closes over an HF decoder layer's loop-invariant kwargs.

    WHY THIS EXISTS
    ---------------
    ``torch_xla.experimental.scan_layers.scan_layers`` in torch_xla 2.9
    has the signature ``scan_layers(layers, input_data, partition_fn=...,
    is_layer_pure=...)``. It does **not** accept arbitrary ``**kwargs``
    -- and yet HuggingFace's ``Cohere2Model.forward`` passes
    ``attention_mask``, ``position_embeddings``, ``position_ids`` (and
    sometimes ``cache_position``) to every layer call.

    Without this wrapper, the bare ``scan_fn(layers, hidden, **kwargs)``
    raises ``TypeError("scan_layers() got an unexpected keyword argument
    'attention_mask'")`` and the manual-loop fallback unrolls 36+6
    layers into the HLO, making backward compile take 4h+ (observed
    2026-05-05 on a v4-32 spot canary). With this wrapper, every layer
    is reduced to a uniform ``forward(hidden) -> hidden`` signature
    that ``scan_layers`` accepts, the kwargs become loop-invariant
    side inputs to the scan body, and compile drops to single-digit
    minutes.

    GPU ANALOGUE
    ------------
    Same idea as ``functools.partial`` for an nn.Module: the closure
    captures the loop-invariant tensors so the inner step has a clean
    one-tensor-in-one-tensor-out signature.
    """

    def __init__(
        self,
        layer: nn.Module,
        args: tuple,
        kwargs: dict[str, Any],
    ) -> None:
        """Bind a layer with its loop-invariant args/kwargs.

        Parameters
        ----------
        layer : nn.Module
            The raw HF decoder layer (NOT a ``_ScanLayerWrapper``).
            We deliberately want the unchecked-pointed layer so that
            ``scan_layers`` can do its own remat caching via the
            ``is_layer_pure=True`` path; double-checkpointing inside
            scan stacks two saved-tensor regions per layer for no
            benefit.
        args : tuple
            Positional args after ``hidden_states`` (rare; HF usually
            uses kwargs only).
        kwargs : dict
            All keyword args the HF outer loop passes per layer call.
            These are stored as plain Python attributes (NOT registered
            as buffers/parameters) so ``scan_layers``'
            ``_ensure_same_structure`` walk over each layer's
            parameters/buffers stays unaffected by them.
        """
        super().__init__()
        # Store the underlying HF layer as a real submodule so its
        # parameters remain visible to the optimizer and to the SPMD
        # sharding pass.
        self.layer = layer
        # Plain attrs -- not registered. scan_layers treats them as
        # closed-over Python state, exactly as we want.
        self._bound_args = args
        self._bound_kwargs = kwargs

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the wrapped layer with the bound kwargs and strip the tuple."""
        out = self.layer(hidden_states, *self._bound_args, **self._bound_kwargs)
        # HF layers usually return (hidden, attn_weights, ...). scan_layers
        # requires a single-tensor carry, so drop everything but the carry.
        if isinstance(out, tuple):
            return out[0]
        return out


class _ScanLayerWrapper(nn.Module):
    """Adapter that gives an HF decoder layer a uniform ``(hidden) -> hidden``
    shape, optionally wrapped in gradient checkpointing.

    HF decoder layers return a tuple where the first element is the
    new hidden state. ``scan_layers`` and our manual loop both want
    the layer to return a single tensor (the "carry"). This wrapper
    extracts ``layer_outputs[0]`` if the underlying layer returns a
    tuple, else returns the value verbatim.
    """

    def __init__(self, layer: nn.Module, *, use_grad_checkpoint: bool) -> None:
        """Wrap a single HF decoder layer.

        Parameters
        ----------
        layer : nn.Module
            A ``CohereDecoderLayer``, ``MoshiDecoderLayer``, or
            anything else with a ``forward(hidden, **kwargs) ->
            tuple[Tensor, ...]`` signature.
        use_grad_checkpoint : bool
            If True, the layer's forward is wrapped in
            ``_xla_safe_checkpoint``. If False, the layer is called
            directly.
        """
        super().__init__()
        self.layer = layer
        self.use_grad_checkpoint = use_grad_checkpoint

    def forward(  # noqa: D401 - simple delegating forward
        self,
        hidden_states: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run the wrapped layer and return only its hidden-state output."""

        def _call(h: torch.Tensor) -> torch.Tensor:
            out = self.layer(h, *args, **kwargs)
            # HF layers usually return (hidden, attn_weights, ...). We
            # only forward the carry and discard the rest. This is fine
            # for training where output_attentions=False.
            if isinstance(out, tuple):
                return out[0]
            return out

        if self.use_grad_checkpoint:
            return _xla_safe_checkpoint(_call, hidden_states)
        return _call(hidden_states)


# ---------------------------------------------------------------------------
# Fused scan-layer that runs the whole stack in one shot
# ---------------------------------------------------------------------------


class _FusedScanLayer(nn.Module):
    """A single ``nn.Module`` that runs an entire layer stack at once.

    This is what HF's ``for layer in self.layers[:N]:`` pulls out of
    the ``_ScannedLayerStack`` proxy. From HF's perspective it is just
    "one layer". From our perspective it is the whole 36-deep stack
    fused via ``scan_layers`` (TPU) or a manual loop (GPU/CPU).
    """

    def __init__(self, stack: _ScannedLayerStack) -> None:
        """Bind the proxy stack so ``forward`` can reach the layers.

        Parameters
        ----------
        stack : _ScannedLayerStack
            The parent proxy holding the wrapped layers and the runtime
            choice between scan_layers and the fallback loop.
        """
        super().__init__()
        # We deliberately do not register ``stack`` as a submodule
        # because it is already the parent. Storing as a plain attr
        # avoids a parameter-discovery double-count when PyTorch walks
        # the module tree.
        self._stack = (stack,)

    @property
    def stack(self) -> _ScannedLayerStack:
        """Return the parent ``_ScannedLayerStack`` (smuggled in a tuple)."""
        return self._stack[0]

    def forward(  # noqa: D401 - explicit signature; not delegating
        self,
        hidden_states: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor]:
        """Run every wrapped layer in order and return ``(hidden_out,)``.

        HF's outer code expects ``layer_outputs[0]`` to be the new
        hidden state, so we always return a single-element tuple.

        Parameters
        ----------
        hidden_states : torch.Tensor
            The carry tensor (``[batch, time, hidden]``).
        *args, **kwargs :
            Loop-invariant arguments forwarded to every layer
            (``attention_mask``, ``position_ids``,
            ``position_embeddings``, ...). They must be identical for
            every layer or ``scan_layers`` will raise on shape
            mismatch.

        Returns
        -------
        tuple[torch.Tensor]
            ``(hidden_out,)`` -- HF's call site does
            ``hidden_states = layer_outputs[0]``.

        Notes
        -----
        TPU note: the first time this runs on TPU, XLA traces and
        compiles the inner layer body. With scan_layers that is *one*
        layer's HLO; without it (the GPU fallback path) it is N
        copies. The same code path works on both because the manual
        loop emits Python-level repetition that XLA still has to
        unroll into HLO.
        """
        layers = self.stack.layers_list
        scan_fn = self.stack.scan_fn

        if scan_fn is not None and _is_tpu_runtime():
            # torch_xla 2.9 signature:
            #     scan_layers(layers, input_data, partition_fn=None,
            #                 is_layer_pure=False)
            # It does NOT accept **kwargs, so we must bind the per-step
            # loop-invariants (attention_mask, position_embeddings, etc.)
            # into per-layer closures. We pass the *raw* HF layer (not
            # the checkpoint-wrapped one) so scan_layers can manage
            # remat itself; otherwise we would stack a
            # torch.utils.checkpoint region inside an xla.while_loop
            # body and double the saved-tensor work.
            #
            # Failure modes we have observed in production
            # (see .factory/memories.md 2026-05-05/06):
            #   1. ValueError "mismatched keys" -- the layer stack is
            #      not parameter-uniform (e.g., last K layers do full
            #      FT while first N-K layers do LoRA). scan_layers
            #      needs identical parameter structure across layers.
            #   2. AssertionError "FakeTensor" -- the is_layer_pure
            #      symbolic-trace path tripped on a non-faked op
            #      (typically a position-embedding gather).
            # When ANY scan attempt fails we permanently disable scan
            # for this stack so we don't pay the overhead of a doomed
            # retry on every forward call. The HLO produced by the
            # manual-loop fallback is the same regardless, so the
            # XLA persistent cache can warm up uninterrupted.
            bound_layers = nn.ModuleList([_KwargBoundLayer(w.layer, args, kwargs) for w in layers])
            try:
                # PyTorch issue #105485: is_layer_pure=True triggers a
                # FakeTensorMode trace through aten.index_select, which
                # asserts on the position-embedding gather inside the
                # decoder layer. We drop the flag (let scan_layers fall
                # through its non-pure path) so the trace stays real.
                output = scan_fn(bound_layers, hidden_states)
                return (output,)
            except Exception as err:  # pragma: no cover - depends on TPU runtime
                print(
                    f"[scan_utils] scan_layers raised "
                    f"{type(err).__name__}: {err!r}; "
                    "permanently disabling scan for this stack."
                )
                self.stack.scan_fn = None

        # Manual loop fallback. Each wrapped layer already implements
        # gradient checkpointing internally if ``use_grad_checkpoint``
        # was True at construction. Hit on (a) GPU/CPU runtime, (b)
        # scan_fn import failure, (c) any prior scan failure that
        # latched scan_fn to None.
        carry = hidden_states
        for layer in layers:
            carry = layer(carry, *args, **kwargs)
        return (carry,)


# ---------------------------------------------------------------------------
# ModuleList drop-in replacement
# ---------------------------------------------------------------------------


class _ScannedLayerStack(nn.Module):
    """Drop-in replacement for an HF transformer's ``self.layers``.

    Looks and acts like an ``nn.ModuleList`` for the bits HF cares
    about (``__len__``, ``__iter__``, ``__getitem__``), but rewrites
    the iteration protocol so the outer ``for`` loop in HF's forward
    yields exactly **one** fused layer that runs the whole stack via
    ``scan_layers``.

    Why not subclass ``nn.ModuleList``?
        ``nn.ModuleList.__getitem__(slice)`` returns a *new*
        ``ModuleList`` containing the slice, which would defeat the
        point. Subclassing it correctly is annoying. A plain
        ``nn.Module`` with the right magic methods is simpler and
        keeps parameter registration straightforward.

    Notes
    -----
    TPU note: HuggingFace also iterates ``self.layers`` to collect
    ``all_hidden_states`` when the caller passes
    ``output_hidden_states=True``. With this proxy in place that
    collection sees only the input and output of the whole stack
    (two snapshots), not 36 per-layer snapshots. For training that is
    fine; for probing hooks, disable the proxy by constructing the
    composite with ``use_scan_layers=False``.
    """

    def __init__(
        self,
        layers: nn.ModuleList,
        *,
        use_scan: bool,
        use_grad_checkpoint: bool,
    ) -> None:
        """Wrap an existing ModuleList of homogeneous transformer layers.

        Parameters
        ----------
        layers : nn.ModuleList
            The original layer stack from the HF model. Must contain
            instances with a uniform ``forward`` signature.
        use_scan : bool
            If True and we are on TPU, use ``scan_layers``. On
            GPU/CPU this is silently downgraded to the manual loop.
        use_grad_checkpoint : bool
            If True, each layer's forward is wrapped in
            ``torch.utils.checkpoint.checkpoint(use_reentrant=False)``.
            Recommended on TPU: HBM is precious.
        """
        super().__init__()
        self._n_original = len(layers)
        # Wrap each underlying layer so it returns a single tensor
        # (HF returns tuples) and so it carries the grad-checkpoint
        # flag. Wrapped layers stay registered so the optimizer can
        # see their parameters.
        self.layers_list = nn.ModuleList(
            [_ScanLayerWrapper(layer, use_grad_checkpoint=use_grad_checkpoint) for layer in layers]
        )
        self.use_scan = use_scan
        self.use_grad_checkpoint = use_grad_checkpoint
        # Resolve scan_fn at construction time. On GPU it stays None;
        # on TPU it is the real callable, or None if the experimental
        # API moved.
        self.scan_fn = _import_scan_layers() if use_scan else None

    # ------------------------------------------------------------------
    # ModuleList-shaped magic methods so HF iteration "just works".
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Pretend to have the original layer count.

        HF reads ``self.config.num_hidden_layers`` separately and uses
        it to slice ``self.layers``. Returning the original count
        avoids surprising any caller that compares lengths.
        """
        return self._n_original

    def __iter__(self):
        """Yield exactly one fused layer.

        HF's ``for decoder_layer in self.layers[: N]:`` will iterate
        once over the slice. Our slice always contains a single
        ``_FusedScanLayer`` regardless of ``N``; one iteration runs
        the full stack via scan.
        """
        yield _FusedScanLayer(self)

    def __getitem__(self, index: int | slice) -> Any:
        """Index by int returns the wrapped layer; slice returns a fused proxy.

        Slice semantics intentionally diverge from ``nn.ModuleList``:
        we always return ``[<single fused layer>]`` so that any HF
        code path of the form ``for layer in self.layers[:N]:``
        iterates exactly once.

        Parameters
        ----------
        index : int or slice
            Integer indexing returns the underlying wrapped layer
            (debug / hook compatibility). Slice indexing returns a
            single-element list as described above.

        Returns
        -------
        nn.Module or list[nn.Module]
            See above.
        """
        if isinstance(index, slice):
            return [_FusedScanLayer(self)]
        return self.layers_list[index]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def replace_layers_with_scan(
    parent: nn.Module,
    layer_attr: str,
    *,
    use_scan_layers: bool,
    use_grad_checkpoint: bool,
) -> bool:
    """Replace ``parent.<layer_attr>`` (a ``ModuleList``) with a scan proxy.

    GPU analogue: there is none -- on GPU you would just call
    ``model.gradient_checkpointing_enable()`` and let CUDA streams
    overlap layer execution. On TPU we need both the per-layer
    checkpoint *and* the loop fusion to keep compile time + HBM under
    control.

    Parameters
    ----------
    parent : nn.Module
        The HF model holding the layer stack as a child attribute.
        Common values:

        * ``backbone.model.model`` -- a ``Cohere2Model`` whose
          ``layers`` is a ``ModuleList[CohereDecoderLayer]``.
        * ``depth_decoder`` -- a ``MoshiDepthDecoder`` whose
          ``layers`` is a ``ModuleList[MoshiDecoderLayer]``.
    layer_attr : str
        Name of the attribute holding the layer stack. Almost always
        ``"layers"``.
    use_scan_layers : bool
        Forward to ``_ScannedLayerStack``. If False *and*
        ``use_grad_checkpoint`` is also False, this function returns
        ``False`` and does nothing -- no point paying for the proxy.
    use_grad_checkpoint : bool
        Forward to ``_ScannedLayerStack``. Recommended True on TPU.

    Returns
    -------
    bool
        ``True`` if the swap happened, ``False`` if the call was a
        no-op (flags off, attribute missing, or already patched).

    Examples
    --------
    >>> # Inside TinyAyaMoshiComposite.__init__:
    >>> replace_layers_with_scan(
    ...     self.backbone.model.model, "layers",
    ...     use_scan_layers=True, use_grad_checkpoint=True,
    ... )
    [scan_utils] patched Cohere2Model.layers (n_layers=36 scan=True ckpt=True)
    True

    Notes
    -----
    TPU note: idempotent. Calling twice on the same parent is safe; we
    detect ``_ScannedLayerStack`` and bail out the second time. This
    matters when training resumes from a checkpoint and the composite
    is reconstructed.
    """
    if not (use_scan_layers or use_grad_checkpoint):
        return False

    layers = getattr(parent, layer_attr, None)
    if layers is None:
        print(
            f"[scan_utils] {type(parent).__name__}.{layer_attr!r} missing; "
            "skipping scan/checkpoint patch."
        )
        return False

    if isinstance(layers, _ScannedLayerStack):
        # Already patched -- e.g., resume-from-checkpoint flow.
        return False

    if not isinstance(layers, nn.ModuleList):
        print(
            f"[scan_utils] {type(parent).__name__}.{layer_attr!r} is not "
            f"an nn.ModuleList (got {type(layers).__name__}); skipping."
        )
        return False

    new_stack = _ScannedLayerStack(
        layers,
        use_scan=use_scan_layers,
        use_grad_checkpoint=use_grad_checkpoint,
    )
    setattr(parent, layer_attr, new_stack)
    print(
        f"[scan_utils] patched {type(parent).__name__}.{layer_attr} "
        f"(n_layers={new_stack._n_original} "  # noqa: SLF001 - same module
        f"scan={use_scan_layers} ckpt={use_grad_checkpoint})"
    )
    return True


__all__ = [
    "replace_layers_with_scan",
]
