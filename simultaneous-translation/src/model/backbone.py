"""TinyAya backbone with text embedding for text-audio interleaving.

WHY THIS EXISTS
---------------
This module wraps the Cohere2 ``tiny-aya-base`` LLM into a
multimodal backbone that, at each frame ``t``, sees the *sum* of
the text-token embedding ``W_{t-1}`` and the audio codebook-0
embedding ``A_{t-1, 0}``. The summed-embedding scheme is the
Moshi-style text-audio interleaving. Outputs:

* ``text_logits`` -- next-text-token prediction over the full
  Cohere vocab plus four special padding IDs.
* ``audio_logits`` -- next-audio-token prediction for *each* of the
  ``num_codebooks`` audio codebooks (codebook-0 only is used by the
  composite; codebooks 1..N are produced by the depth decoder).
* ``hidden_states`` -- the backbone's last hidden state, fed into
  the projection + depth decoder downstream.

TPU concerns this file is responsible for
-----------------------------------------
* ``use_cache=False`` is hardcoded in :meth:`forward`. Cohere2's
  sliding-window cache uses negative indexing in its position ids,
  which XLA cannot lower (see pytorch/xla #6234). Setting
  ``use_cache=True`` would *work on GPU* but break TPU compile.
* The bf16 cast for the *whole* composite happens in
  ``tpu_backend.wrap_model``; the ``load_in_bf16`` flag here only
  controls the dtype at HF load time, which saves host RAM.

GPU vs TPU
----------
Nothing in this file imports ``torch_xla`` directly. The same code
works on a CUDA box -- the TPU adaptations are confined to the
backend wrapper and to ``scan_utils.py``.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class TinyAyaBackbone(nn.Module):
    """TinyAya (Cohere2) wrapped for speech-text multimodal modelling.

    Parameters
    ----------
    model_name : str, default ``"CohereLabs/tiny-aya-base"``
        HuggingFace model ID for the backbone. ``trust_remote_code``
        is True because Cohere2 ships its modelling code in the repo.
    audio_vocab_size : int, default 2048
        Mimi codebook size. The extended embedding table appends this
        many rows after the four special padding tokens.
    num_codebooks : int, default 32
        How many ``audio_heads`` the backbone exposes. The composite
        only uses head 0; the rest are kept for compatibility with
        Stage 1 checkpoints.
    load_in_bf16 : bool, default True
        Save host RAM by loading the HF weights directly in bf16.
        The TPU backend later re-applies ``model.to(torch.bfloat16)``
        which is a cheap no-op when the source dtype already matches.
    """

    # Special token IDs appended after TinyAya's 262144 vocab. They
    # share the embedding table with regular text tokens so the
    # backbone can attend to them without a separate code path.
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
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
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
            old_mean = embed.data[: self.text_vocab_size].mean(dim=0)
            embed.data[self.text_vocab_size :] = old_mean

        # Separate text embedding for summing with audio (Moshi-style)
        # Initialized from backbone's own embeddings -> warm start
        hidden = self.model.config.hidden_size
        self.text_embed = nn.Embedding(self.total_text_vocab, hidden)
        with torch.no_grad():
            src = self.model.get_input_embeddings().weight.data
            self.text_embed.weight.data[: self.text_vocab_size] = src[
                : self.text_vocab_size
            ].clone()
            avg = self.text_embed.weight.data[: self.text_vocab_size].mean(dim=0)
            self.text_embed.weight.data[self.text_vocab_size :] = avg
            self.text_embed.weight.data[self.ZERO_PADDING] = 0.0

        # Audio prediction heads — one per codebook, all 32
        self.audio_heads = nn.ModuleList(
            [nn.Linear(hidden, audio_vocab_size, bias=False) for _ in range(num_codebooks)]
        )
        self.hidden_size = hidden

    def get_input_embeddings(self):
        """Return the extended HF input embedding (text + special + audio)."""
        return self.model.get_input_embeddings()

    def forward(
        self,
        text_ids: torch.LongTensor,
        audio_codes: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass with summed text + audio embeddings.

        Parameters
        ----------
        text_ids : torch.LongTensor
            ``[B, T]`` interleaved text token IDs in
            ``[0, total_text_vocab)``.
        audio_codes : torch.LongTensor
            ``[B, T]`` Mimi codebook-0 codes in
            ``[0, audio_vocab_size)``.
        attention_mask : torch.Tensor or None, default None
            ``[B, T]`` mask. Ones mark valid positions.

        Returns
        -------
        dict[str, torch.Tensor]
            ``{"text_logits": [B, T, vocab],``
            ``"audio_logits": [B, num_codebooks, T, 2048],``
            ``"hidden_states": [B, T, hidden]}``.

        Notes
        -----
        TPU note: ``use_cache=False`` is **mandatory**. Cohere2's
        sliding-window cache uses negative position indexing that XLA
        cannot lower. The flag is hardcoded here rather than exposed
        as a parameter so a future caller cannot accidentally re-enable
        it on TPU.
        """
        # Audio embeddings from the extended backbone table. We shift
        # raw codebook IDs into the audio range that follows the text
        # vocab + special tokens.
        audio_token_ids = audio_codes + self.audio_token_offset
        audio_embeds = self.get_input_embeddings()(audio_token_ids)

        # Text embeddings from the separate table (Moshi-style sum
        # fusion). Initialised from the backbone's own embeddings so
        # the model starts from a warm point.
        text_embeds = self.text_embed(text_ids)

        # Per-frame sum (Moshi-style interleaving). This stays a
        # tensor-add on GPU and TPU; XLA fuses it into the embedding
        # lookup at compile time.
        combined = audio_embeds + text_embeds

        outputs = self.model(
            inputs_embeds=combined,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            # use_cache=False is mandatory on TPU. See class docstring.
            use_cache=False,
        )
        hidden_states = outputs.hidden_states[-1]

        # Next-text-token prediction over the full extended vocab.
        text_logits = self.model.lm_head(hidden_states)

        # Per-codebook next-audio-token logits. Each head: [B, T, 2048];
        # stack into [B, num_codebooks, T, 2048].
        audio_logits = torch.stack([head(hidden_states) for head in self.audio_heads], dim=1)

        return {
            "text_logits": text_logits,
            "audio_logits": audio_logits,
            "hidden_states": hidden_states,
        }

    def gradient_checkpointing_enable(self):
        """Enable HF's per-layer grad checkpointing.

        Notes
        -----
        TPU note: do NOT call this on torch 2.9 + torch_xla 2.9 -- the
        legacy HF checkpoint hook calls
        ``torch._get_device_module("xla")`` which raises. The TPU
        path uses ``TinyAyaMoshiComposite(xla_grad_checkpoint=True)``
        instead; see ``src/model/scan_utils.py``.
        """
        self.model.gradient_checkpointing_enable()
