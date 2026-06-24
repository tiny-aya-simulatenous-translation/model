"""Debug per-codebook accuracy after overfit."""
import sys, torch
sys.path.insert(0, ".")

from src.model.composite import TinyAyaMoshiComposite
from src.model.lora_setup import apply_lora
from src.training.checkpointing import load_checkpoint
from src.data.dataset import StreamingTranslationDataset
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("CohereLabs/tiny-aya-base", trust_remote_code=True)
model = TinyAyaMoshiComposite(num_codebooks=8)
model.backbone = apply_lora(model.backbone, r=16, num_full_ft_layers=0)
load_checkpoint(model, None, None, "checkpoints/overfit_parallel/step_000300")
model = model.to("cuda").to(torch.bfloat16).eval()

ds = StreamingTranslationDataset(
    "/home/alperiox/training_data_full/splits/small/train_20.jsonl",
    tok, max_frames=300, encoded_dir="/home/alperiox/training_data_full/encoded"
)
s = ds[0]
T = s["text_ids"].shape[0]
src = s["source_length"]
print(f"T={T}, src={src}, tgt={T-src}", flush=True)

with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    try:
        out = model(
            text_ids=s["text_ids"].unsqueeze(0).to("cuda"),
            audio_codes=s["user_audio_codes"][0].unsqueeze(0).to("cuda"),
            model_audio_codes=s["model_audio_codes"][0].unsqueeze(0).to("cuda"),
            attention_mask=torch.ones(1, T, dtype=torch.long, device="cuda"),
            full_audio_codes=s["model_audio_codes"].unsqueeze(0).to("cuda")[:, :8, :],
            depth_chunk_size=16,
        )
    except Exception as e:
        print(f"ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    text_logits, audio_logits, hidden = out
    print(f"audio_logits shape: {audio_logits.shape}", flush=True)

    for cb in range(8):
        p = audio_logits[0, cb, src-1:-1, :].argmax(-1).cpu()
        t = s["model_audio_codes"][cb, src:src + p.shape[0]]
        acc = (p == t).float().mean().item()
        print(f"CB{cb} acc={acc*100:.1f}% pred={p[:5].tolist()} tgt={t[:5].tolist()}", flush=True)
