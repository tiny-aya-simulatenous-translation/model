# Contributing to TinyAya Stage 2

Thank you for your interest in contributing. This project trains a
multi-billion-parameter speech-to-speech translation model on Google
Cloud TPU TRC, so contributions span ML research, distributed
systems, infra automation, and documentation. All of them are
welcome.

## Table of Contents

- [Ways to contribute](#ways-to-contribute)
- [Before you start](#before-you-start)
- [Development environment](#development-environment)
- [The External Memory System](#the-external-memory-system)
- [Branching and commits](#branching-and-commits)
- [The `/verify` gate](#the-verify-gate)
- [Documentation conventions](#documentation-conventions)
- [Running tests](#running-tests)
- [Pull request process](#pull-request-process)
- [TPU-specific contributions](#tpu-specific-contributions)
- [Reporting issues](#reporting-issues)
- [Security](#security)
- [Code of conduct](#code-of-conduct)

---

## Ways to contribute

- **Bug reports** — open an issue with reproduction steps; check
  [`memories.md`](.factory/memories.md) first to make sure it's not
  a known gotcha.
- **Feature requests** — open an issue describing the use case
  before submitting a PR; we may have already considered it
  (search [`PROGRESS.md`](.factory/PROGRESS.md)).
- **Documentation** — every Python file under
  `simultaneous-translation/` follows the
  [TPU-for-GPU-engineers documentation style][docstyle]. Bringing
  older files up to that bar is one of the most-valued contributions.
- **Bug fixes** — pick an issue tagged `good first issue` or
  `help wanted`. If it touches the TPU path, please also add a
  diagnosis-table row in
  [`.factory/orchestration/playbook/diagnosis-table.md`][dxtable].
- **New TPU strategies / SPMD primitives** — bring evidence:
  per-strategy compile / step / HBM measurements on the same model
  + config (use `scripts/tpu/probe_strategies.py`).
- **Eval improvements** — better ASR-BLEU/DNSMOS pipelines or new
  metrics. Please benchmark against the current eval first.

[docstyle]: simultaneous-translation/AGENTS.md
[dxtable]: .factory/orchestration/playbook/diagnosis-table.md

---

## Before you start

1. **Read the memory files** in this order:
   1. [`.factory/PLAN.md`](.factory/PLAN.md) — current goal +
      Definition of Done + phased tasks.
   2. [`.factory/PROGRESS.md`](.factory/PROGRESS.md) — most recent
      entries (top of file).
   3. [`.factory/VERIFY.md`](.factory/VERIFY.md) — the bash blocks
      that prove "done".
   4. [`.factory/memories.md`](.factory/memories.md) — long-term
      decisions and gotchas.
2. **Skim the relevant `AGENTS.md`** — the root one applies
   monorepo-wide; subprojects override it under
   `simultaneous-translation/AGENTS.md` and
   `phase-3-data-generation-pipeline/AGENTS.md`.
3. **Open an issue** for non-trivial features before sending a PR.
   Sending a large PR without prior discussion may result in
   rejection if the direction conflicts with the active plan.

---

## Development environment

### Required tooling

- **Python 3.12** (TPU images ship 3.12; do not regress to 3.11).
- [`uv`](https://docs.astral.sh/uv/) — dependency manager. Each
  subproject has its own `pyproject.toml` and committed `uv.lock`.
- `gcloud` SDK — for live TPU work.
- `git` ≥ 2.40 — `git restore`, `git switch` are part of the
  documented workflow.
- A POSIX shell (`bash` 5+). On WSL2, watch out for CRLF line
  endings; run `dos2unix` if `bash -n` complains about `\r`.

### Local setup

```bash
git clone <repo-url> tinyaya-stage2-scale
cd tinyaya-stage2-scale

# Each subproject has its own uv-managed venv
cd simultaneous-translation && uv sync && cd ..
cd phase-3-data-generation-pipeline && uv sync && cd ..
```

### Workspaces

- **Local workstation** — `/home/<user>/Workspace/tinyaya-stage2-scale/`
  on WSL2 Ubuntu, has `.git`. Edit and commit here.
- **TPU VM** — `/opt/tinyaya/` is a transient snapshot, no `.git`,
  overwritten on every `hot_redeploy.sh`. **Do not commit from the
  TPU.** If you must, `git clone` to `~/work/` first.
- **`/mnt/c`** is the Windows bridge from inside WSL — slow. Keep
  build artifacts under `/home/...`.

---

## The External Memory System

This repo uses [Tip 10: Keep Context Fresh][tip10] — four append-only
files in `.factory/` that preserve project state across AI sessions
and human contributors. **Treat these like the README of your
project's brain.**

| File          | When to write                                              |
|---------------|------------------------------------------------------------|
| `PROGRESS.md` | After every non-trivial action: edits, exec, decisions, failures, verifications. Most recent at top. |
| `PLAN.md`     | When the active goal or Definition of Done changes. Use `/plan` to regenerate from current state. |
| `VERIFY.md`   | Add a new bash block whenever a check should run before "done" is declared. |
| `memories.md` | For long-term architecture decisions and gotchas. Mark superseded entries `SUPERSEDED`; do not delete. |

[tip10]: simultaneous-translation/docs/MEMORY-SYSTEM.md

### Quick capture

Start any chat message with one of these tags to capture without
breaking flow:

- `#progress <text>` — append a freeform `info` entry.
- `#plan <task>` — add a `[ ]` task to the active plan.
- `#decision <text>` — append a permanent decision to `memories.md`.
- `#verify <bash command>` — append a bash block to `VERIFY.md`.

Or use the slash commands `/progress`, `/plan`, `/remember`,
`/verify` (also `/curate` to deduplicate older memories).

### Lifecycle hooks

The hooks in `.factory/hooks/` automatically:

- Load the four memory files into context at the start of any
  non-trivial task.
- Append a `done | edit` entry to `PROGRESS.md` after each tool use.
- Run a secrets scrubber over every `PROGRESS.md` write.
- Trigger `/verify` before `Stop` and during `pre_compact`.

You generally don't need to invoke them manually.

---

## Branching and commits

### Branch model

- **Default branch:** `main`. Never push directly.
- **Feature work:** `feat/<short-slug>`.
- **Bug fixes:** `fix/<short-slug>`.
- **Docs:** `docs/<short-slug>`.
- **Refactors / chores:** `refactor/<short-slug>` or `chore/<…>`.

The current TPU work lives on `feat/tpu-support`.

### Commit conventions

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short summary>

<wrap body at 72 cols>

<optional footer>
```

Types: `feat`, `fix`, `docs`, `test`, `chore`, `refactor`, `perf`.
Scope is optional but useful (`tpu`, `data`, `train`, `eval`,
`infra`).

Example:

```
feat(tpu): collator pads to max_frames to eliminate per-batch recompile

Patch 11 of the TPU canary self-heal loop. Iter 6 reached step 2
then stalled in a per-batch HLO recompile loop because each batch's
actual frame count differed by 1-15 frames. The collator now pads
every TPU batch to cfg.data.max_frames (300 for canary, 300 for
full). XLA compiles one HLO and reuses it.

Validated by iter 7 reaching step 100 with sec/step settling to
3.41 from step 30 onwards.

See pytorch/xla user-guide "Avoiding Recompilations".

Co-authored-by: factory-droid[bot] <138933559+factory-droid[bot]@users.noreply.github.com>
```

### AI-pair-programming attribution

When AI agents (Factory.ai droids, Copilot, Claude, etc.) authored
or co-authored a change, add the co-author line:

```
Co-authored-by: factory-droid[bot] <138933559+factory-droid[bot]@users.noreply.github.com>
```

This is a non-negotiable repo norm — see the root
[`AGENTS.md`](AGENTS.md).

### Pre-commit hygiene

Before any `git commit`:

1. Run `/verify` (slash command) — the gate must be green.
2. Run `git diff --cached` and visually inspect for secrets,
   credentials, API keys, large binaries, or stale `print` /
   `breakpoint` calls.
3. The hook `.factory/hooks/_lib.py` runs a regex scrubber over
   tracked diffs. The patterns cover HF tokens, OpenAI keys, GitHub
   tokens, AWS access keys, and PEM private keys.

---

## The `/verify` gate

Every PR must show a green `/verify` run in its description. The
gate is the source of truth for "done":

```bash
/verify          # slash command (preferred)
# or:
bash .factory/skills/verify/SKILL.md
```

What it does: reads every fenced bash block in
[`.factory/VERIFY.md`](.factory/VERIFY.md) in order, executes them,
and reports a structured PASS/FAIL table. Sections:

- `monorepo` — settings JSON, hooks compile, AGENTS.md tiers
  exist, no merge markers, no obvious secrets.
- `simultaneous-translation` — every `*.py` compiles, every
  `*.yaml` parses, every `*.sh` is `bash -n` clean.
- `phase-3-data-generation-pipeline` — Python compile + CLI help.
- `TPU sharding` — only runs when `PJRT_DEVICE` is set; skipped
  on the workstation.
- `Orchestration artifacts` — spec / diagrams / playbook / poller
  all present and consistent.

### Adding a verify block

When your change introduces a new invariant, add a bash block:

```markdown
## simultaneous-translation

```bash
# my-new-invariant: <one-line description>
python3 simultaneous-translation/scripts/check_my_thing.py
```
```

…or use the `#verify <cmd>` quick-capture tag.

### Failing tests

Currently the `phase-3-data-generation-pipeline` CLI test in the
gate fails with `ModuleNotFoundError: No module named 'src.config'`.
This is a pre-existing condition unrelated to the TPU path; treat
it as expected baseline noise unless your PR touches that
subproject.

---

## Documentation conventions

### TPU-for-GPU-engineers style (mandatory)

Every Python file under `simultaneous-translation/` carries a
module docstring whose first section is titled `WHY THIS EXISTS`
and gives a 4-10 line plain-English description, plus `# GPU
analogue:` callouts whenever a TPU primitive replaces a GPU
equivalent. Functions get NumPy-style docstrings; behaviour that
diverges on TPU gets a `Notes` block starting with `TPU note:`.

```python
"""TPU backend with multiple SPMD sharding strategies.

WHY THIS EXISTS
---------------
On GPU we use DDP (one process per GPU, NCCL all-reduce on
backward). On a TPU pod we use **SPMD** -- one logical Python
program drives every chip via PJRT, and the XLA partitioner
decides where each tensor lives. This file picks the partitioner's
*sharding strategy* and is the only place that should know about
XLA-specific primitives like xs.mark_sharding, Mesh, or FSDPv2.
"""

def wrap_model(model: nn.Module) -> nn.Module:
    """Wrap `model` with the SPMD strategy chosen via TPU_STRATEGY.

    Args:
        model: Unwrapped composite model. Must be on the XLA device.

    Returns:
        The wrapped model. On a single chip this is a no-op; on a
        pod it is either a replicated mark_sharding'd model, or an
        FSDPv2-wrapped model.

    Notes:
        TPU note: bf16 cast happens *here*, not via env vars. The
        legacy XLA_USE_BF16=1 was removed in torch_xla 2.6 and
        silently no-ops in 2.9.
    """
    xs.mark_sharding(model, mesh, ("fsdp",))   # GPU analogue: input.cuda(rank) under DDP
    model = model.to(torch.bfloat16)           # GPU analogue: torch.cuda.amp.autocast(...)
    ...
```

Run `Skill("tpu-doc-style")` (or `/tpu-doc-style`) to load the full
checklist into a chat session.

### Inline comments

- Bad: `# loop over layers`.
- Good: `# scan_layers compiles ONE layer's HLO and runs it via
  xla.while; this replaces the 36-way unrolled HLO that was costing
  25+ min compile (see PROGRESS 2026-05-03T14:30:00Z).`

### YAML configs

Configs are read by both researchers and lifecycle hooks. Each
section gets a block comment explaining what the knob does *and*
what changes when running on TPU vs GPU. Cross-link to memories.md
where relevant.

### When you don't have to

Pure utility code with no TPU contact (text tokenisers, etc.) only
needs the module docstring and NumPy docstrings on public
functions; the GPU-vs-TPU callouts and the `WHY THIS EXISTS` TPU
paragraph are skipped.

### Ruff

Run before every commit:

```bash
cd simultaneous-translation
.venv/bin/python -m ruff format .
.venv/bin/python -m ruff check --fix .
```

The repo's ruff config (`pyproject.toml`) is the source of truth:
`target=py312`, `line-length=100`, `select=E,F,W,I,B,UP`,
`ignore=E501`.

---

## Running tests

Tests live next to code under each subproject. Common entry points:

```bash
# simultaneous-translation
cd simultaneous-translation
uv run python -m pytest tests/ -v             # if tests/ exists
uv run python -m py_compile $(git ls-files '*.py')   # quick lint

# phase-3-data-generation-pipeline
cd phase-3-data-generation-pipeline
uv run python -m pytest tests/ -v
```

For TPU-specific tests, set `PJRT_DEVICE=TPU` and run
`scripts/tpu/probe_strategies.py` against your strategy of choice.

---

## Pull request process

1. **Open an issue first** for non-trivial features.
2. **Branch off `main`** as `feat/<slug>` (or fix/, docs/, …).
3. **Make focused changes** — one logical change per PR. Refactors
   should be separate from feature work.
4. **Run `/verify`** locally; paste the output (or a screenshot)
   into the PR description.
5. **Update memory files** if your change is non-obvious:
   - New decision → `memories.md`.
   - Plan / DoD change → `PLAN.md`.
   - New invariant → add a bash block to `VERIFY.md`.
6. **Open a PR against `main`** (or the active feature branch if
   the maintainer specified one). Use the conventional-commit title
   format. Body should answer: *what changed, why, what's not in
   scope, how was it tested.*
7. **Respond to review** — push fixups onto the same branch; do
   not force-push during review unless asked. After approval, the
   maintainer will squash-merge.

### What we look for in review

- Code matches the [TPU-for-GPU-engineers style][docstyle].
- New TPU-side gotchas have a corresponding entry in
  `memories.md` with a regex signature added to
  `playbook/diagnosis-table.md` if appropriate.
- The `/verify` gate is green.
- No secrets, no large binaries, no `_artifacts/` content
  accidentally tracked.
- Imports are minimal; no new top-level dependencies without
  justification.
- Tests cover the non-trivial branches (or document why they
  can't, e.g. "requires a live TPU mesh").

---

## TPU-specific contributions

Working on the TPU path? A few extra rules:

- **Probe before merging.** Any change that affects the SPMD
  partitioner, `wrap_model`, or the strategy matrix must include
  fresh `probe_strategies.py` numbers in the PR description for
  whichever strategies you touched.
- **Use the diagnosis table.** If you fixed a TPU failure mode,
  add a row to
  [`.factory/orchestration/playbook/diagnosis-table.md`][dxtable]
  with: regex signature, root cause, patch summary, tier, and
  source link.
- **Memory-sensitive changes.** Document the per-chip HBM impact
  in the PR. Use `diagnose()` from the backend to capture before /
  after numbers; "I think it should fit" is not enough.
- **Don't re-enable `scan_layers` casually.** It's intentionally
  off — see `memories.md` 2026-05-05 entry on the
  `_ensure_same_structure` rejection. If you want to bring it back,
  open an issue with a reproduction first.
- **Hot redeploy first, QR re-create second.** If you can fix the
  problem with a code change, use `hot_redeploy.sh`; only ask for
  a fresh QR if the VM itself is corrupt (Tier 3 in the playbook).
- **One queued resource at a time.** Always delete an unused QR
  before requesting another; TRC quota is shared.

---

## Reporting issues

Please include:

- **Environment**: workstation / TPU profile (e.g.
  `v6e-8 spot europe-west4-a` for the current production path, or legacy
  `v4-32 spot us-central2-b`), Python version, `torch_xla`
  version, `wandb` version. State the worker count explicitly
  (1 worker on v6e-8 single-host, 4 on v4-32).
- **Reproduction**: minimal `launch_*.sh` invocation or a
  `train_hierarchical.py` command line.
- **Expected vs actual**: what should have happened, what did.
- **Logs**: relevant lines from `/tmp/train.log` (TPU side) or
  the chat tmux scrollback. Redact secrets.
- **What you've tried**: links to relevant `memories.md` /
  `PROGRESS.md` entries you already checked.

For TPU stalls, please include a py-spy native dump if possible:

```bash
sudo /tmp/py_spy-0.4.2.data/scripts/py-spy dump --pid <REAL_PID> --native
```

(Watch out — the `uv run` PID isn't the real Python PID. Use
`ps -e --forest -o pid,pcpu,etime,comm,args` to find the actual
training process.)

---

## Security

- **Never commit secrets.** `.env` is gitignored. The hook scrubber
  catches HF tokens, OpenAI keys, GitHub tokens, AWS access keys,
  and PEM private keys, but it's a safety net, not a primary
  defence.
- **Pre-commit hooks** — we recommend installing
  [`gitleaks`](https://github.com/gitleaks/gitleaks) or
  [`detect-secrets`](https://github.com/Yelp/detect-secrets)
  against the staged diff for an extra check.
- **TPU credentials** — the TPU VM authenticates via its attached
  service account. Never copy a personal `gcloud` token onto a
  TPU worker.
- **Reporting vulnerabilities** — for sensitive issues, do not
  open a public issue. Email the maintainer at the address in
  the project metadata.

---

## Code of conduct

Be kind, be precise, assume good faith. Disagreements about
technical direction are fine; personal attacks are not. We follow
the spirit of the [Contributor Covenant](https://www.contributor-covenant.org/).

---

## Thank you

Every contribution helps push the TinyAya project forward. Whether
it's a typo fix in this file or a new TPU sharding strategy that
shaves 20 % off compile time, we appreciate it. If you're not sure
where to start, look for issues tagged `good first issue` or open a
discussion thread.

When in doubt, read the memory files. They are the project's
nervous system.
