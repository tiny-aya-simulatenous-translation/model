---
name: tpu-doc-style
description: Apply the TinyAya Stage 2 TPU-for-GPU-engineers documentation style and run the project ruff lint workflow. Use when authoring or editing any Python under simultaneous-translation/, when the user asks for "TPU docs", "lint pass", or "PEP8 + docstrings". Pairs with simultaneous-translation/AGENTS.md "TPU code documentation style (mandatory)".
---

# tpu-doc-style

The single source of truth for *how* code in this repo is commented,
docstring'd, type-annotated, and lint-checked. The full convention
lives in `simultaneous-translation/AGENTS.md`; this skill is a
session-priming checklist plus the canonical lint command line.

## Audience contract (recap)

Reader knows: PyTorch, autograd, AMP, DDP, FSDP-on-GPU,
gradient checkpointing in concept.

Reader does **not** know: PJRT, SPMD, FSDPv2 (the SPMD variant),
`scan_layers`, `xm.optimizer_step`, HBM vs. host RAM, why
`XLA_USE_BF16` is deprecated, why `use_cache=True` breaks XLA
tracing.

## Required artefacts in every Python file

1. **Module docstring** opens with a `WHY THIS EXISTS` block (4-10
   lines) explaining the file's role, plus a one-paragraph plain-
   English explainer for any TPU concept introduced for the first
   time in the file.

2. **GPU-vs-TPU callouts** wherever a TPU primitive replaces a GPU
   one:
   ```python
   xm.optimizer_step(optimizer)         # GPU analogue: optimizer.step()
   xs.mark_sharding(x, mesh, ("fsdp",)) # GPU analogue: input.cuda(rank)
   model.to(torch.bfloat16)             # GPU analogue: torch.cuda.amp.autocast(...)
   ```

3. **NumPy-style docstrings** on every public function/method:
   `Args`, `Returns`, `Raises`, `Notes`. When TPU behaviour diverges
   from GPU, add a `Notes` section starting with `TPU note:`.

4. **Inline comments** explain *trade-offs and gotchas*, never
   restate code. Cross-reference `.factory/memories.md` whenever the
   inline comment summarises a recorded decision.

5. **Type hints** on all public APIs. PEP8 compliance enforced by
   `ruff` (config in `simultaneous-translation/pyproject.toml`).

6. **YAML configs** — block comments per section explaining what the
   knob does *and* what changes when running on TPU vs. GPU.

Pure-utility files with no TPU contact (e.g., text-tokenisation
helpers) only need the module docstring + NumPy docstrings on public
functions; the GPU-vs-TPU paragraph is skipped.

## Lint command line (canonical)

Run from `simultaneous-translation/`:

```bash
# auto-format
.venv/bin/python -m ruff format src/ scripts/

# lint with auto-fix where safe
.venv/bin/python -m ruff check --fix src/ scripts/

# verify zero remaining findings
.venv/bin/python -m ruff check src/ scripts/

# byte-compile every source file
find src -name '*.py' -not -path '*/__pycache__/*' -print0 \
  | xargs -0 -n1 python3 -m py_compile

# byte-compile entry-point scripts
python3 -m py_compile scripts/train_hierarchical.py \
                     scripts/eval_stage2.py \
                     scripts/make_splits.py \
                     scripts/tpu/probe_strategies.py

# parse all YAML configs
for y in configs/*.yaml; do
  python3 -c "import yaml; yaml.safe_load(open('$y'))"
done

# parse all shell scripts
bash -n scripts/tpu/*.sh
```

The first run installs ruff into the project venv if missing:

```bash
uv pip install --quiet 'ruff>=0.7'
```

## Success criteria

- `ruff format --check` reports no diffs.
- `ruff check` reports zero findings (the repo ignores `E501` so
  long-line warnings are not findings).
- `py_compile` succeeds on every `.py` file.
- `bash -n` succeeds on every `.sh` file.
- Every YAML parses with `yaml.safe_load`.

## Failure modes

- `ruff format` rewrites a file unexpectedly — read the diff,
  decide if the rewrite obscures the intent (rare; usually it's a
  net improvement). If unacceptable, narrow the rule with
  `# noqa: <code>` and a one-line justification.
- `ruff check` flags a long-line in a comment — `E501` is already
  ignored in `[tool.ruff.lint]`. If a *new* line-length issue
  surfaces, prefer to wrap the line; only suppress when the line
  is a literal URL or a verbatim copy of an external string.

## Composes with

- `verify` — the project's full verification suite still runs after
  this skill.
- `update-plan` — tick the PLAN.md "lint" / "docs" checkboxes once
  this skill reports clean.
