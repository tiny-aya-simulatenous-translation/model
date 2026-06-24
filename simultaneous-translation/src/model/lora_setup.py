"""LoRA application and parameter-group creation for the TinyAya backbone.

WHY THIS EXISTS
---------------
We don't fine-tune all 3.36B backbone parameters; we apply Low-Rank
Adaptation (LoRA) to attention and embedding projections, plus full
fine-tuning of the last two transformer blocks. This module owns the
PEFT integration and the parameter-group scheme that the training
loop hands to AdamW.

LoRA matters more on TPU than GPU
---------------------------------
The composite is 5.17B parameters. With full fine-tuning the
optimiser state alone (AdamW keeps fp32 m + v) is 8x the trainable
size, which OOMs even fsdpv2 on v5e. LoRA + frozen-base trims the
trainable count to ~274M (5.3%); the rest of the per-chip HBM budget
is then dominated by activations, which ``scan_utils`` controls.
"""

import re

import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model


class LoRAEmbedding(nn.Module):
    """Drop-in replacement for nn.Embedding that freezes the base table and
    learns a low-rank adapter on top (LoRA-style)."""

    def __init__(self, base_embed: nn.Embedding, r: int = 16, alpha: int = 32):
        super().__init__()
        self.base_embed = base_embed
        self.base_embed.weight.requires_grad = False
        num_embeddings, embedding_dim = base_embed.weight.shape
        self.lora_A = nn.Embedding(num_embeddings, r)
        self.lora_B = nn.Linear(r, embedding_dim, bias=False)
        self.scaling = alpha / r
        nn.init.normal_(self.lora_A.weight, std=1.0 / r)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.base_embed(x) + self.lora_B(self.lora_A(x)) * self.scaling

    @property
    def weight(self):
        return self.base_embed.weight


def apply_lora(
    backbone,
    r=16,
    lora_alpha=32,
    target_modules=None,
    num_full_ft_layers=0,
    lora_exclude_top=2,
):
    """Apply LoRA to the TinyAya backbone (config-driven; sweepable).

    Strategy:
    - LoRA (rank ``r``, scale ``lora_alpha``) on ``target_modules`` for layers
      ``0 .. N - max(lora_exclude_top, num_full_ft_layers)``. The excluded top
      layers are FROZEN, except the top ``num_full_ft_layers`` which are fully
      fine-tuned.
    - ``text_embed`` wrapped with ``LoRAEmbedding`` (frozen base + adapter).
    - ``audio_heads`` always trainable.

    Defaults reproduce the proven-finite surface: LoRA on 0..N-2, top-2 frozen
    (production ~122M trainable, ~26/31 GB HBM). NOTE: LoRA on ALL layers
    (``lora_exclude_top=0``) was observed to spike HBM to ~29.5/31 GB and drive
    a non-finite forward via the fsdpv2_lora wrapping of the heavy top blocks --
    keep top layers excluded until that is understood.

    ``num_full_ft_layers`` is an OPT-IN capacity lever (sweep candidate), OFF by
    default: each unfrozen block adds its full param + AdamW state (~78M +
    optimiser on a 36-layer/2048-hidden backbone), which strains HBM. The
    unfreeze is module-based (not param-name matching) and asserts it took
    effect, so it can never silently no-op.
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "embed_tokens"]
    target_modules = list(target_modules)

    num_layers = backbone.model.config.num_hidden_layers
    # LoRA on 0..N-excluded; the excluded top layers are frozen (or full-FT'd
    # below). layers_to_transform=None would mean ALL layers (see HBM caveat).
    excluded = max(lora_exclude_top, num_full_ft_layers)
    lora_layers = list(range(num_layers - excluded)) if excluded > 0 else None

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        layers_to_transform=lora_layers,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    backbone.model = get_peft_model(backbone.model, lora_config)

    # Full fine-tune the top ``num_full_ft_layers`` blocks. Operate on the
    # layer MODULE objects (robust to PEFT name-mangling), then assert a
    # non-zero count so an enabled unfreeze can't silently fail.
    unfrozen = 0
    if num_full_ft_layers > 0:
        targets = set(range(num_layers - num_full_ft_layers, num_layers))
        for mod_name, module in backbone.model.named_modules():
            m = re.fullmatch(r".*\.layers\.(\d+)", mod_name)
            if m and int(m.group(1)) in targets:
                for p in module.parameters(recurse=True):
                    if not p.requires_grad:
                        p.requires_grad = True
                        unfrozen += p.numel()
        assert unfrozen > 0, (
            f"num_full_ft_layers={num_full_ft_layers} but unfroze 0 params "
            f"(layer-module match failed for {num_layers}-layer backbone)"
        )

    # Wrap text_embed with LoRA adapter (frozen base + trainable low-rank)
    backbone.text_embed = LoRAEmbedding(backbone.text_embed, r=r, alpha=lora_alpha)

    # Ensure audio_heads are trainable
    for param in backbone.audio_heads.parameters():
        param.requires_grad = True

    print(
        f"[lora] r={r} alpha={lora_alpha} targets={target_modules} "
        f"lora_layers=0..{num_layers - excluded - 1} (exclude_top={excluded}) "
        f"num_full_ft_layers={num_full_ft_layers} (+{unfrozen / 1e6:.1f}M full-FT)"
    )
    backbone.model.print_trainable_parameters()
    return backbone


def register_embedding_grad_mask(backbone):
    """No-op — kept for backward compatibility with other scripts.

    With LoRA on embed_tokens, the base weights are frozen and only LoRA
    adapter parameters are trainable, so a gradient mask is unnecessary.
    """
    print("Embedding grad mask: skipped (base embeddings frozen, LoRA adapters handle updates)")


def get_parameter_groups(
    backbone,
    lr_lora=1e-4,
    lr_full_ft=5e-5,
    lr_audio_embed=5e-4,
    lr_text_embed=5e-4,
    lr_audio_head=5e-4,
):
    """Create optimizer parameter groups with per-component learning rates."""
    num_layers = (
        backbone.model.config.num_hidden_layers if hasattr(backbone.model, "config") else 36
    )
    ft_start = num_layers - 2

    groups = {
        "lora": {"params": [], "lr": lr_lora, "name": "lora"},
        "full_ft": {"params": [], "lr": lr_full_ft, "name": "full_ft"},
        "text_embed": {"params": [], "lr": lr_text_embed, "name": "text_embed"},
        "audio_head": {"params": [], "lr": lr_audio_head, "name": "audio_head"},
    }

    for name, param in backbone.named_parameters():
        if not param.requires_grad:
            continue

        if "audio_head" in name or "audio_heads" in name:
            groups["audio_head"]["params"].append(param)
        elif "text_embed" in name:
            groups["text_embed"]["params"].append(param)
        elif "lora_" in name or "lora_embedding" in name:
            groups["lora"]["params"].append(param)
        elif any(f"layers.{i}." in name for i in range(ft_start, num_layers)):
            groups["full_ft"]["params"].append(param)
        else:
            groups["lora"]["params"].append(param)

    result = [g for g in groups.values() if g["params"]]

    print("\n=== Parameter Groups ===")
    for g in result:
        n = sum(p.numel() for p in g["params"])
        print(f"  {g['name']}: {len(g['params'])} tensors, {n:,} params, lr={g['lr']}")

    return result
