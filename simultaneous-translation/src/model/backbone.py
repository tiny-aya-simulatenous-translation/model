"""TinyAya backbone with text embedding for text-audio interleaving.

At each frame t:
  input = text_embed(W_{t-1}) + audio_embed(A_{t-1})  [summed]
  hidden = backbone_transformer(input)
  text_logits = lm_head(hidden)         -> predict W_t
  audio_logits = audio_head(hidden)     -> predict A_t (codebook 0)
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class TinyAyaBackbone(nn.Module):
    """TinyAya (Cohere2) wrapped for speech-text multimodal modeling."""

    # Special token IDs (appended after TinyAya's 262144 vocab)
    TEXT_PADDING = 262144
    END_OF_TEXT_PADDING = 262145
    ZERO_PADDING = 262146
    IN_WORD_PADDING = 262147
    NUM_SPECIAL = 4

    def __init__(
        self,
        model_name: str = "CohereLabs/tiny-aya-base",
        audio_vocab_size: int = 2048,
        num_codebooks: int = 32,
        load_in_bf16: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.audio_vocab_size = audio_vocab_size
        self.num_codebooks = num_codebooks

        # Load backbone
        dtype = torch.bfloat16 if load_in_bf16 else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True,
        )

        self.text_vocab_size = self.model.config.vocab_size  # 262144
        self.total_text_vocab = self.text_vocab_size + self.NUM_SPECIAL  # 262148

        # Extend embedding table: [text 262144 | special 4 | audio 2048] = 264196
        self.audio_token_offset = self.total_text_vocab  # 262148
        target_vocab = self.total_text_vocab + audio_vocab_size  # 264196
        self.model.resize_token_embeddings(target_vocab)

        # Override HF's random init for new rows with mean of original embeddings
        with torch.no_grad():
            embed = self.model.get_input_embeddings().weight
            old_mean = embed.data[:self.text_vocab_size].mean(dim=0)
            embed.data[self.text_vocab_size:] = old_mean

        # Separate text embedding for summing with audio (Moshi-style)
        # Initialized from backbone's own embeddings -> warm start
        hidden = self.model.config.hidden_size
        self.text_embed = nn.Embedding(self.total_text_vocab, hidden)
        with torch.no_grad():
            src = self.model.get_input_embeddings().weight.data
            self.text_embed.weight.data[:self.text_vocab_size] = src[:self.text_vocab_size].clone()
            avg = self.text_embed.weight.data[:self.text_vocab_size].mean(dim=0)
            self.text_embed.weight.data[self.text_vocab_size:] = avg
            self.text_embed.weight.data[self.ZERO_PADDING] = 0.0

        # Audio prediction heads — one per codebook, all 32
        self.audio_heads = nn.ModuleList([
            nn.Linear(hidden, audio_vocab_size, bias=False)
            for _ in range(num_codebooks)
        ])
        self.hidden_size = hidden

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def forward(
        self,
        text_ids: torch.LongTensor,
        audio_codes: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass with summed text + audio embeddings.

        Args:
            text_ids: [B, T] interleaved text token IDs (0..262147)
            audio_codes: [B, T] Mimi codebook-0 codes (0..2047)
            attention_mask: [B, T]
        """
        # Audio embeddings from extended backbone table
        audio_token_ids = audio_codes + self.audio_token_offset  # shift into audio range
        audio_embeds = self.get_input_embeddings()(audio_token_ids)

        # Text embeddings from separate table
        text_embeds = self.text_embed(text_ids)

        # Sum at each frame (Moshi-style interleaving)
        combined = audio_embeds + text_embeds

        # Backbone transformer
        outputs = self.model(
            inputs_embeds=combined,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states[-1]

        # Predict next text token (full vocab LM head)
        text_logits = self.model.lm_head(hidden_states)

        # Predict next audio tokens — all 32 codebooks
        # Each head: [B, T, 2048], stack to [B, 32, T, 2048]
        audio_logits = torch.stack(
            [head(hidden_states) for head in self.audio_heads], dim=1
        )

        return {
            "text_logits": text_logits,
            "audio_logits": audio_logits,  # [B, 32, T, 2048]
            "hidden_states": hidden_states,
        }

    def gradient_checkpointing_enable(self):
        self.model.gradient_checkpointing_enable()
