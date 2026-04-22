"""Composite model: TinyAya backbone + projection + Moshi depth decoder.

Forward flow at each timestep t:
  1. Input: text_embed(W_{t-1}) + audio_embed(A_{t-1,0})  [summed]
  2. Backbone transformer -> hidden_states [B, T, 2048]
  3. Text prediction: LM head -> text_logits [B, T, vocab]
  4. Projection: hidden_states -> [B, T, 4096]
  5. Depth decoder: for each frame, autoregressively generate 8 codebooks
     -> audio_logits [B, 8, T, 2048]
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

from .backbone import TinyAyaBackbone
from .surgery import extract_depth_decoder_state_dict, create_projection
from .depth_decoder import create_depth_decoder


class TinyAyaMoshiComposite(nn.Module):
    """Full model: TinyAya backbone + projection + Moshi depth decoder."""

    def __init__(self, num_codebooks: int = 8):
        super().__init__()
        self.num_codebooks = num_codebooks

        # 1. Backbone (handles text_embed + audio_embed internally)
        print("=" * 60)
        print("Step 1: Loading TinyAya backbone")
        print("=" * 60)
        self.backbone = TinyAyaBackbone(num_codebooks=1)  # no audio_heads needed

        # 2. Projection
        print("\n" + "=" * 60)
        print("Step 2: Creating projection layer")
        print("=" * 60)
        self.projection = create_projection(
            in_features=self.backbone.hidden_size,  # 2048
            out_features=4096,
        )

        # 3. Depth decoder from Moshiko
        print("\n" + "=" * 60)
        print("Step 3: Extracting depth decoder from Moshiko")
        print("=" * 60)
        depth_sd = extract_depth_decoder_state_dict()
        self.depth_decoder = create_depth_decoder(
            depth_sd, num_codebooks=num_codebooks,
        )
        del depth_sd

        self.audio_token_offset = self.backbone.audio_token_offset

    def forward(
        self,
        text_ids: torch.LongTensor,
        audio_codes: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        full_audio_codes: torch.LongTensor | None = None,
        depth_chunk_size: int = 16,
    ) -> dict[str, torch.Tensor]:
        """Forward pass with depth decoder for hierarchical codebook generation.

        Args:
            text_ids: [B, T] text token IDs
            audio_codes: [B, T] codebook-0 codes (for backbone input)
            attention_mask: [B, T]
            full_audio_codes: [B, num_codebooks, T] all codebooks for teacher-forcing depth decoder
            depth_chunk_size: timesteps per checkpointed depth call
        """
        B, T = text_ids.shape
        device = text_ids.device

        # Step 1: Backbone forward (text + audio-0 summed embeddings)
        backbone_out = self.backbone(
            text_ids=text_ids,
            audio_codes=audio_codes,
            attention_mask=attention_mask,
        )
        hidden_states = backbone_out["hidden_states"]  # [B, T, 2048]
        text_logits = backbone_out["text_logits"]  # [B, T, vocab]

        # Step 2: Project to depth decoder input space
        projected = self.projection(hidden_states)  # [B, T, 4096]

        # Step 3: Depth decoder — chunked over time for memory efficiency
        def _depth_forward(input_ids, last_hidden_state):
            return self.depth_decoder(
                input_ids=input_ids,
                last_hidden_state=last_hidden_state,
                use_cache=False,
                return_dict=True,
            ).logits

        audio_logits_chunks = []
        for t_start in range(0, T, depth_chunk_size):
            t_end = min(t_start + depth_chunk_size, T)
            chunk_len = t_end - t_start

            # Gather chunk data
            ctx_chunk = projected[:, t_start:t_end, :]  # [B, chunk, 4096]
            text_chunk = text_ids[:, t_start:t_end]  # [B, chunk]

            # We need all codebooks for depth decoder input (teacher forced)
            # audio_codes is [B, T] (CB0 only) — for depth decoder we need
            # the full [B, num_codebooks, T] but we only have CB0 here.
            # For training, full codes are passed via the loss function.
            # Here we use zeros for CB1+ (they get teacher-forced in the loss).

            # Reshape to batch: [B*chunk, num_codebooks, ...]
            ctx_flat = ctx_chunk.reshape(B * chunk_len, 1, -1)
            ctx_expanded = ctx_flat.expand(
                B * chunk_len, self.num_codebooks, -1
            ).contiguous()

            # Depth input: [text_id, audio_cb0, ..., audio_cb(N-2)]
            # Depth input: position 0 = text (0 = padding in 32K space),
            # positions 1..N-1 = audio codebooks 0..N-2 (teacher-forced)
            depth_input = torch.zeros(
                B * chunk_len, self.num_codebooks,
                dtype=torch.long, device=device,
            )
            # Teacher-force audio codebooks into depth decoder if available
            if full_audio_codes is not None:
                audio_chunk = full_audio_codes[:, :, t_start:t_end]  # [B, CB, chunk]
                audio_flat = audio_chunk.permute(0, 2, 1).reshape(B * chunk_len, -1)  # [B*chunk, CB]
                # Position 1..N-1 gets codebooks 0..N-2
                depth_input[:, 1:] = audio_flat[:, :self.num_codebooks - 1]

            # Checkpointed depth forward
            chunk_logits_flat = cp.checkpoint(
                _depth_forward,
                depth_input,
                ctx_expanded,
                use_reentrant=False,
            )

            # [B*chunk, num_codebooks, 2048] -> [B, chunk, num_codebooks, 2048]
            chunk_logits = chunk_logits_flat.reshape(
                B, chunk_len, self.num_codebooks, -1
            )
            audio_logits_chunks.append(chunk_logits)

        # [B, T, num_codebooks, 2048] -> [B, num_codebooks, T, 2048]
        audio_logits = torch.cat(audio_logits_chunks, dim=1).permute(0, 2, 1, 3)

        return {
            "text_logits": text_logits,
            "audio_logits": audio_logits,  # [B, num_codebooks, T, 2048]
            "hidden_states": hidden_states,
        }
