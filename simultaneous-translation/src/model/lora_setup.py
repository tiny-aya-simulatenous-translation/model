"""LoRA application and parameter group creation for TinyAya backbone."""

import torch
from peft import LoraConfig, TaskType, get_peft_model


def apply_lora(backbone, r=16, lora_alpha=32, num_full_ft_layers=2):
    """Apply LoRA to TinyAya backbone.

    Strategy:
    - LoRA on q_proj, v_proj for layers 0..N-num_full_ft_layers
    - Full fine-tuning on last num_full_ft_layers layers
    - Audio embeddings always trainable (gradient mask freezes text rows)
    """
    num_layers = backbone.model.config.num_hidden_layers  # 36
    lora_layers = list(range(num_layers - num_full_ft_layers))  # [0..33]

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],
        layers_to_transform=lora_layers,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        modules_to_save=["embed_tokens"],
    )

    backbone.model = get_peft_model(backbone.model, lora_config)

    # Unfreeze last N layers
    for name, param in backbone.model.named_parameters():
        for layer_idx in range(num_layers - num_full_ft_layers, num_layers):
            if f"layers.{layer_idx}." in name:
                param.requires_grad = True
                break

    # Ensure audio_heads and text_embed are trainable
    for param in backbone.audio_heads.parameters():
        param.requires_grad = True
    for param in backbone.text_embed.parameters():
        param.requires_grad = True

    backbone.model.print_trainable_parameters()
    return backbone


def register_embedding_grad_mask(backbone):
    """Zero gradients for text token rows in backbone embeddings.

    Only audio token rows (262148+) should update. Text rows (0-262143) stay frozen.
    Special tokens (262144-262147) also get gradients zeroed in the backbone embed
    since they have their own text_embed table.
    """
    embed_weight = None
    for name, param in backbone.named_parameters():
        if "embed_tokens" in name and "weight" in name and param.requires_grad:
            embed_weight = param
            break

    if embed_weight is None:
        print("WARNING: Could not find trainable embed_tokens, skipping grad mask")
        return

    offset = backbone.audio_token_offset  # 262148

    def _mask_grad(grad):
        grad[:offset] = 0  # zero text + special token rows
        return grad

    embed_weight.register_hook(_mask_grad)
    print(f"Embedding grad mask: rows 0..{offset-1} masked, {offset}+ trainable")


def get_parameter_groups(backbone, lr_lora=1e-4, lr_full_ft=5e-5,
                         lr_audio_embed=5e-4, lr_text_embed=5e-4,
                         lr_audio_head=5e-4):
    """Create optimizer parameter groups with per-component learning rates."""
    num_layers = backbone.model.config.num_hidden_layers if hasattr(backbone.model, 'config') else 36
    ft_start = num_layers - 2

    groups = {
        "lora": {"params": [], "lr": lr_lora, "name": "lora"},
        "full_ft": {"params": [], "lr": lr_full_ft, "name": "full_ft"},
        "audio_embed": {"params": [], "lr": lr_audio_embed, "name": "audio_embed"},
        "text_embed": {"params": [], "lr": lr_text_embed, "name": "text_embed"},
        "audio_head": {"params": [], "lr": lr_audio_head, "name": "audio_head"},
    }

    for name, param in backbone.named_parameters():
        if not param.requires_grad:
            continue

        if "audio_head" in name or "audio_heads" in name:
            groups["audio_head"]["params"].append(param)
        elif "text_embed" in name and "model.model" not in name:
            groups["text_embed"]["params"].append(param)
        elif "lora_" in name:
            groups["lora"]["params"].append(param)
        elif "embed_tokens" in name:
            groups["audio_embed"]["params"].append(param)
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
