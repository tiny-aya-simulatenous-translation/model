"""Composite model: TinyAya backbone + projection + Moshi depth decoder.

WHY THIS EXISTS
---------------
This file builds the trainable model that lives behind ``apply_lora``
and the SPMD wrapping in ``src/backend/tpu_backend.py``. Conceptually,
each frame ``t`` of audio is generated as follows:

1. Sum the text-token embedding ``W_{t-1}`` and the
   codebook-0 audio embedding ``A_{t-1, 0}`` into a single vector
   (Moshi-style multimodal fusion).
2. Run that vector through the **Cohere backbone transformer** (36
   ``CohereDecoderLayer`` instances) to get a hidden state
   ``[B, T, 2048]`` and the next text-token logits.
3. Project the hidden state up to ``[B, T, 4096]`` to match the depth
   decoder's input width.
4. Run the **Moshi depth decoder** (6 ``MoshiDecoderLayer`` instances)
   autoregressively across the 8 codebook positions for each frame to
   produce ``[B, 8, T, 2048]`` audio logits.

TPU-specific concerns this file is responsible for
--------------------------------------------------
* **scan_layers + gradient checkpointing.** Both layer stacks are
  swapped for ``_ScannedLayerStack`` proxies via
  ``replace_layers_with_scan``. On TPU this collapses 36 + 6 unrolled
  decoder forwards into one HLO each, dropping compile time from 25+
  min to a few minutes. On GPU it falls back to the manual loop with
  ``torch.utils.checkpoint`` -- no behavioural change.

* **Chunked depth-decoder forward.** The depth decoder runs over
  ``num_codebooks`` autoregressive steps per frame; running all
  ``T`` frames at once would peak HBM. We chunk the time axis with
  ``depth_chunk_size`` and ``_xla_safe_checkpoint`` each chunk.

GPU/TPU parity
--------------
Nothing in this file imports ``torch_xla`` directly. All TPU-aware
behaviour is delegated to ``scan_utils.py``, which probes
``PJRT_DEVICE`` lazily. So the same ``TinyAyaMoshiComposite`` class
runs on a GPU dev box without modification.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

from .backbone import TinyAyaBackbone
from .depth_decoder import create_depth_decoder
from .scan_utils import replace_layers_with_scan
from .surgery import create_projection, extract_depth_decoder_state_dict


def _checkpoint(fn, *args, use_reentrant: bool = False):
    """XLA-aware gradient checkpoint shim.

    GPU analogue: ``torch.utils.checkpoint.checkpoint(fn, *args)`` --
    standard activation recomputation in the backward pass.

    Parameters
    ----------
    fn : callable
        The function to checkpoint.
    *args : Any
        Positional args forwarded to ``fn``.
    use_reentrant : bool, default False
        Pinned to False because the reentrant variant relies on
        CUDA-specific RNG state save/restore that asserts on XLA.

    Returns
    -------
    Any
        Whatever ``fn(*args)`` returns.

    Notes
    -----
    TPU note: certain torch_xla minor versions raise
    ``AttributeError`` from inside ``torch.utils.checkpoint`` when the
    XLA device module probe fails. We swallow that and fall through
    to a direct call. The activation memory savings are lost in that
    case but training still works.
    """
    try:
        return cp.checkpoint(fn, *args, use_reentrant=use_reentrant)
    except AttributeError:
        return fn(*args)


class TinyAyaMoshiComposite(nn.Module):
    """Composite model: Cohere backbone + projection + Moshi depth decoder.

    GPU analogue: a regular ``nn.Module`` you would feed to a normal
    PyTorch training loop. The TPU-specific bits are entirely in
    ``scan_utils`` (which this class calls) and in
    ``src/backend/tpu_backend.py`` (which wraps the *whole* composite).

    Parameters
    ----------
    num_codebooks : int, default 8
        Number of audio codebooks the depth decoder will produce per
        frame. Trained at 8 for the canary; the production run also
        uses 8.
    use_scan_layers : bool, default False
        If True, swap the backbone and depth-decoder layer stacks for
        ``_ScannedLayerStack`` proxies. Recommended True on TPU.
        On GPU/CPU this is a no-op (the proxy falls back to a manual
        loop). See ``.factory/PLAN.md`` Phase 1.
    xla_grad_checkpoint : bool, default False
        If True, each layer's forward is wrapped in
        ``torch.utils.checkpoint`` with ``use_reentrant=False``.
        Recommended True on TPU to keep per-chip HBM under 12 GB.
        See ``.factory/PLAN.md`` Phase 2.

    Notes
    -----
    TPU note: do NOT call ``HF`` ``model.gradient_checkpointing_enable()``
    on this composite under torch 2.9 + torch_xla 2.9 -- the legacy
    HF grad-checkpoint hook calls ``torch._get_device_module("xla")``
    which raises. Use the ``xla_grad_checkpoint`` flag instead, which
    routes through ``scan_utils`` and works on both backends.
    """

    def __init__(
        self,
        num_codebooks: int = 8,
        *,
        use_scan_layers: bool = False,
        xla_grad_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.num_codebooks = num_codebooks
        self.use_scan_layers = use_scan_layers
        self.xla_grad_checkpoint = xla_grad_checkpoint

        # ------------------------------------------------------------
        # Step 1: Cohere backbone (handles text + audio-0 embeddings
        # internally and computes text logits + last hidden state).
        # ------------------------------------------------------------
        print("=" * 60)
        print("Step 1: Loading TinyAya backbone")
        print("=" * 60)
        # num_codebooks=1 here means "no audio prediction heads beyond
        # codebook 0"; codebooks 1..N are produced by the depth
        # decoder, not by the backbone.
        self.backbone = TinyAyaBackbone(num_codebooks=1)

        # ------------------------------------------------------------
        # Step 2: Projection from backbone hidden_size (2048) to depth
        # decoder input_size (4096). This is a tiny Linear layer that
        # we always train.
        # ------------------------------------------------------------
        print("\n" + "=" * 60)
        print("Step 2: Creating projection layer")
        print("=" * 60)
        self.projection = create_projection(
            in_features=self.backbone.hidden_size,
            out_features=4096,
        )

        # ------------------------------------------------------------
        # Step 3: Depth decoder pulled out of Moshiko. The state dict
        # is loaded once on host CPU; ``create_depth_decoder`` extends
        # the codebook dimension if requested. The local reference is
        # deleted right after to free host RAM.
        # ------------------------------------------------------------
        print("\n" + "=" * 60)
        print("Step 3: Extracting depth decoder from Moshiko")
        print("=" * 60)
        depth_sd = extract_depth_decoder_state_dict()
        self.depth_decoder = create_depth_decoder(
            depth_sd,
            num_codebooks=num_codebooks,
        )
        del depth_sd

        # Convenience: text vocab + special tokens live BEFORE the
        # audio token range in the extended embedding table.
        self.audio_token_offset = self.backbone.audio_token_offset

        # ------------------------------------------------------------
        # TPU optimisation: swap the layer stacks for scan_layers
        # proxies. On GPU the swap still runs but ``scan_utils``
        # detects the absence of ``PJRT_DEVICE`` and falls back to a
        # manual loop, so this is safe to call unconditionally.
        # ------------------------------------------------------------
        if use_scan_layers or xla_grad_checkpoint:
            # Cohere backbone: AutoModelForCausalLM -> .model is the
            # base Cohere2Model, .model.layers is the ModuleList.
            replace_layers_with_scan(
                self.backbone.model.model,
                "layers",
                use_scan_layers=use_scan_layers,
                use_grad_checkpoint=xla_grad_checkpoint,
            )
            # MoshiDepthDecoder: the layers attribute lives directly
            # on the module.
            replace_layers_with_scan(
                self.depth_decoder,
                "layers",
                use_scan_layers=use_scan_layers,
                use_grad_checkpoint=xla_grad_checkpoint,
            )

    def forward(
        self,
        text_ids: torch.LongTensor,
        audio_codes: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        full_audio_codes: torch.LongTensor | None = None,
        depth_chunk_size: int = 16,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with hierarchical (frame-then-codebook) generation.

        GPU analogue: a vanilla ``nn.Module.forward``. The only thing
        TPU-special here is the chunked depth-decoder loop, which
        keeps activation memory bounded under SPMD.

        Parameters
        ----------
        text_ids : torch.LongTensor
            Interleaved text token IDs, shape ``[B, T]``. Values lie
            in ``[0, total_text_vocab)``.
        audio_codes : torch.LongTensor
            Codebook-0 audio codes, shape ``[B, T]``. Used as the
            backbone input together with ``text_ids``.
        attention_mask : torch.Tensor or None, default None
            Standard HF attention mask, shape ``[B, T]``. Ones mark
            valid positions, zeros mark padding.
        full_audio_codes : torch.LongTensor or None, default None
            Complete codebook tensor for teacher-forcing the depth
            decoder, shape ``[B, num_codebooks, T]``. If ``None``, the
            depth decoder generates from zeros (still useful for
            inference probes; loss won't match teacher-forced).
        depth_chunk_size : int, default 16
            Number of timesteps processed per depth-decoder chunk. We
            checkpoint each chunk independently to keep activation
            memory linear in ``depth_chunk_size`` rather than ``T``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            ``(text_logits, audio_logits, hidden_states)``.

            * ``text_logits`` -- ``[B, T, vocab]``
            * ``audio_logits`` -- ``[B, num_codebooks, T, 2048]``
            * ``hidden_states`` -- ``[B, T, 2048]`` (the backbone's
              last hidden state, used by some loss variants).

        Notes
        -----
        TPU note: returning a tuple (not a dict) is deliberate. Our
        ``_shard_output`` helper in ``tpu_backend.py`` walks tuple
        elements to apply ``xs.mark_sharding`` on each. A dict would
        also work but tuples sidestep the SPMD partitioner's
        replicated-output assertion.
        """
        b, t = text_ids.shape
        device = text_ids.device

        # --------------------------------------------------------------
        # Backbone forward: handles text+audio embedding sum, runs the
        # 36-layer transformer, returns last hidden state and text
        # logits. With scan_layers enabled, the 36 layers are fused
        # into one HLO behind the scenes.
        # --------------------------------------------------------------
        backbone_out = self.backbone(
            text_ids=text_ids,
            audio_codes=audio_codes,
            attention_mask=attention_mask,
        )
        hidden_states = backbone_out["hidden_states"]
        text_logits = backbone_out["text_logits"]

        # --------------------------------------------------------------
        # Project to depth-decoder input space (2048 -> 4096).
        # --------------------------------------------------------------
        projected = self.projection(hidden_states)

        # --------------------------------------------------------------
        # Depth decoder, chunked over time. Each chunk runs the 6
        # depth-decoder layers across the codebook axis. We checkpoint
        # the per-chunk forward to keep activation memory bounded.
        # --------------------------------------------------------------

        def _depth_forward(
            input_ids: torch.LongTensor, last_hidden_state: torch.Tensor
        ) -> torch.Tensor:
            """Wrapper around the depth decoder's logits-only output."""
            return self.depth_decoder(
                input_ids=input_ids,
                last_hidden_state=last_hidden_state,
                use_cache=False,
                return_dict=True,
            ).logits

        audio_logits_chunks = []
        for t_start in range(0, t, depth_chunk_size):
            t_end = min(t_start + depth_chunk_size, t)
            chunk_len = t_end - t_start

            # Gather chunk data.
            ctx_chunk = projected[:, t_start:t_end, :]  # [B, chunk, 4096]

            # Reshape to "batch": one row per (sample, frame) pair.
            # Each row is fed through the depth decoder along the
            # codebook axis.
            ctx_flat = ctx_chunk.reshape(b * chunk_len, 1, -1)
            ctx_expanded = ctx_flat.expand(b * chunk_len, self.num_codebooks, -1).contiguous()

            # Depth input layout per row:
            #   [text_id_padding, audio_cb0, audio_cb1, ..., audio_cb(N-2)]
            # We zero the text slot (training does not condition on
            # explicit text here -- the backbone hidden state already
            # carries that information) and teacher-force the audio
            # codebooks if available.
            depth_input = torch.zeros(
                b * chunk_len,
                self.num_codebooks,
                dtype=torch.long,
                device=device,
            )
            if full_audio_codes is not None:
                audio_chunk = full_audio_codes[:, :, t_start:t_end]  # [B, CB, chunk]
                # Permute to [B, chunk, CB] then flatten to [B*chunk, CB].
                audio_flat = audio_chunk.permute(0, 2, 1).reshape(b * chunk_len, -1)
                # Position 1..N-1 gets codebooks 0..N-2.
                depth_input[:, 1:] = audio_flat[:, : self.num_codebooks - 1]

            # Checkpointed depth forward. ``_checkpoint`` falls back
            # to a direct call on backends where checkpointing breaks.
            chunk_logits_flat = _checkpoint(
                _depth_forward,
                depth_input,
                ctx_expanded,
                use_reentrant=False,
            )

            # Restore the [B, chunk, num_codebooks, 2048] shape.
            chunk_logits = chunk_logits_flat.reshape(b, chunk_len, self.num_codebooks, -1)
            audio_logits_chunks.append(chunk_logits)

        # Concatenate along the time axis, then permute to the
        # expected [B, num_codebooks, T, 2048].
        audio_logits = torch.cat(audio_logits_chunks, dim=1).permute(0, 2, 1, 3)

        return (text_logits, audio_logits, hidden_states)
