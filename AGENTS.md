# AGENTS.md — TinyAya Stage 2 (monorepo)

Briefing for AI coding agents working anywhere in this repository.
Subproject-specific norms live in `simultaneous-translation/AGENTS.md`
and `phase-3-data-generation-pipeline/AGENTS.md` and override anything
here for files inside those folders.

## Repo layout

```
tinyaya-stage2-scale/
|-- AGENTS.md                                # this file (monorepo norms)
|-- simultaneous-translation/                # training / model / eval / TPU
|   |-- AGENTS.md                            # subproject norms
|   `-- docs/                                # incl. MEMORY-SYSTEM.md
|-- phase-3-data-generation-pipeline/        # data encoding + alignment
|   `-- AGENTS.md                            # subproject norms
`-- .factory/                                # External Memory System
    |-- PROGRESS.md PLAN.md VERIFY.md memories.md
    |-- hooks/  skills/  droids/  commands/
    `-- settings.json
```

## Memory System

This repo uses the External Memory System (Tip 10: Keep Context Fresh).
Before any non-trivial task, read these files in order:

1. `.factory/PLAN.md` — current goal + checklist + Definition of Done.
2. `.factory/PROGRESS.md` — most recent entries (top of file).
3. `.factory/VERIFY.md` — commands that prove "done".
4. `.factory/memories.md` — long-term decisions and gotchas.

Lifecycle hooks load these automatically. See
`simultaneous-translation/docs/MEMORY-SYSTEM.md` for the architecture
and `MEMORY-USAGE.md` for the daily-driver flow.

## Branching, commits, PRs

- Feature work goes on `feat/<short-slug>`. Current TPU work is on
  `feat/tpu-support`.
- Default branch is `main`. Never push directly; PRs only.
- Commit messages: conventional commits prefix (`feat:`, `fix:`,
  `docs:`, `test:`, `chore:`, `refactor:`). Body wraps at 72 chars.
- Co-author line: `Co-authored-by: factory-droid[bot]
  <138933559+factory-droid[bot]@users.noreply.github.com>`.
- Before committing: run `/verify` (the slash command). Before
  pushing: also run `git diff --cached` and scan for secrets.

## Secret handling

- `.env` is gitignored. **Never** read its contents into chat or
  echo into logs.
- The .gitignore already protects `.env`, `.env.local`, `*.env.local`,
  checkpoints, wandb runs, audio scratch, and `_artifacts/`.
- The memory hooks (`.factory/hooks/_lib.py`) run a regex scrubber
  over every PROGRESS write. The patterns cover HF tokens, OpenAI
  keys, GitHub tokens, AWS access keys, and PEM private keys.
- Pre-commit hook recommended: `gitleaks` or
  `detect-secrets` against the staged diff.

## Build / runtime conventions (cross-cutting)

- Python 3.12 minimum (TPU images ship 3.12).
- Dependency manager: `uv` per subproject (`uv.lock` is committed).
- No system Python globally polluted; use the project venv.
- No emojis in code or commit messages.
- No new top-level dependencies without a PR description that
  justifies them.

## Workspaces

The repo lives at:

- Local workstation: `/home/cataluna84/Workspace/tinyaya-stage2-scale/`
  (WSL2 Ubuntu, has `.git`).
- TPU VM `/opt/tinyaya/`: a transient snapshot, no `.git`,
  overwritten by `hot_redeploy.sh`. **Do not commit from the TPU**;
  if you need to commit there, `git clone` to `~/work/` first.

## How agents should approach tasks

1. **Recall first** — read the memory files; never duplicate work
   already recorded in `memories.md`.
2. **Plan to PLAN.md** — if no plan exists, run `/plan` or invoke the
   `plan-driver` droid before editing code.
3. **Verify after** — run `/verify` before declaring "done"; the
   `Stop` hook will do this automatically anyway.
4. **Log decisions** — anything non-obvious that future you would
   thank you for: capture via `/remember` or `#decision`.
5. **Stay in scope** — respect the Out-of-scope list in PLAN.md.

## Cross-subproject changes

When work touches both `simultaneous-translation/` and
`phase-3-data-generation-pipeline/`, stay at the repo root so only
this AGENTS.md (the monorepo tier) loads. State the cross-cutting
nature explicitly in PLAN.md so reviewers see it coming.

## Known gotchas (monorepo-wide)

- `_artifacts/` is gitignored; useful for tarballs and transient
  outputs. Never put tracked artifacts there.
- WSL2 + Windows: scripts edited on Windows can have CRLF endings.
  If `bash -n` complains about `\r`, run `dos2unix` on the file.
- `/mnt/c` paths from inside WSL are slow; keep all build artifacts
  under `/home/...`.
- The TPU side has its own `.cache/` for HF; clean it manually if
  free disk drops below 5 GB.
