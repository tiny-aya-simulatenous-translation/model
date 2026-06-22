# Third-Party Notices

The TinyAya Stage 2 source code is licensed under Apache-2.0 (see `LICENSE`).
This project, however, **builds on third-party models, datasets, and
libraries that are governed by their own licenses**. Everyone using or
contributing to this project is responsible for complying with the terms
of each component listed below. The Apache-2.0 license of this repository
does **not** relicense any of them.

## Models / weights

| Component | Source | License | What you must know |
|-----------|--------|---------|--------------------|
| **Cohere `tiny-aya` backbone** | CohereLabs / Cohere | **Cohere model license** (NOT Apache-2.0) | This repo ships **LoRA adapter deltas only**, never the base weights. To run/train you must obtain the base model directly and comply with Cohere's license (including any non-commercial / acceptable-use terms). |
| **Moshi / Mimi (depth decoder + audio codec)** | [kyutai-labs/moshi](https://github.com/kyutai-labs/moshi) | MIT | Retain the MIT copyright/permission notice when redistributing derived weights. |
| **Whisper (alignment, data pipeline)** | OpenAI | MIT | Used for forced alignment in the data pipeline. |

## Datasets

| Dataset | Source | License | What you must know |
|---------|--------|---------|--------------------|
| **FLEURS** (source speech/text) | Google | **CC BY 4.0** | Attribution required for redistribution of derived data. |
| `tiny-aya-translate/*` encoded datasets | HuggingFace Hub | See each dataset card | Derived from the above; inherit upstream obligations. |

## Libraries (runtime dependencies)

| Library | License |
|---------|---------|
| PyTorch, PyTorch/XLA | BSD-3-Clause |
| HuggingFace Transformers, Tokenizers, Hub, Datasets, Accelerate, PEFT | Apache-2.0 |
| NumPy, SciPy, scikit-learn | BSD-3-Clause |
| librosa, soundfile, soxr | ISC / BSD / LGPL (see each project) |
| Weights & Biases (`wandb`) | MIT |
| `gcsfs`, `sacrebleu`, `pyyaml`, `tqdm` | Apache-2.0 / MIT / BSD |

Library versions are pinned in `simultaneous-translation/pyproject.toml` /
`uv.lock`; consult each upstream project for the authoritative license text.

## Compute acknowledgment

Cloud TPU resources were provided by Google's **TPU Research Cloud (TRC)**.
Research outputs acknowledge the TRC program per its terms (see `NOTICE`).

---

If you believe a component is missing or mis-attributed here, please open an
issue or PR so we can correct it.
