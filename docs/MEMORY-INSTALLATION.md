# Memory System — Installation

The memory system ships pre-installed with this repo: every file lives
under `.factory/`, the three `AGENTS.md` tiers are at their canonical
locations, and the docs are under `simultaneous-translation/docs/`. To
activate the **dynamic** parts (hooks, skills, droids), follow the steps
below once per workstation.

## Prerequisites

| Requirement | Check |
|-------------|-------|
| Factory CLI installed | `droid --version` |
| Python 3.9+ | `python3 --version` |
| `jq` for JSON-piping in hooks | `jq --version` |
| Repo cloned at a stable path | `git rev-parse --show-toplevel` |

If any of the above is missing, install before continuing. On WSL2/Ubuntu:
`sudo apt-get install -y jq python3`.

## Step 1 — Make hooks executable

```bash
cd "$(git rev-parse --show-toplevel)"
chmod +x .factory/hooks/*.py
```

If the hooks were checked out without the executable bit (common on
Windows shares / WSL `/mnt/c`), this step is mandatory. Without it the
hooks will silently fail and the running log will not auto-update.

## Step 2 — Verify settings.json is the snapshot Droid uses

```bash
cat .factory/settings.json
```

Expected: a JSON object with a `hooks` key listing six events
(`SessionStart`, `UserPromptSubmit`, `PostToolUse`, `Stop`, `PreCompact`,
`SessionEnd`).

Droid takes a hook snapshot on session start. If you edit
`settings.json` while a session is open, the changes apply only after a
restart. Run `/hooks` inside Droid to confirm registration.

## Step 3 — Sanity-check the four memory files

```bash
ls -la .factory/{PROGRESS,PLAN,VERIFY,memories}.md
```

Each must exist (even if seeded empty). The `SessionStart` hook reads
all four; if one is missing the hook logs a warning and continues with a
degraded context.

## Step 4 — First session smoke test

```bash
droid                      # launch from repo root
```

Inside Droid:

```
/recall                    # should print all four memory files in tables
/verify                    # should run VERIFY.md commands and report
```

If either errors, see the troubleshooting section of
`MEMORY-MAINTENANCE.md`.

## Step 5 — Confirm the three AGENTS.md tiers are loaded

Inside Droid, with cwd at the repo root, ask:

> Read AGENTS.md and tell me which file you loaded.

It should report the root `/AGENTS.md`. Then `cd` into a subproject and
ask the same question — Droid should now report **two** files (root +
subproject), with the subproject taking precedence.

## Optional — Personal cross-project memory

If you want preferences that follow you across all repos on this
machine, create `~/.factory/AGENTS.md` and `~/.factory/memories.md` from
your own template. Examples in
[Factory.ai docs](https://docs.factory.ai/guides/power-user/memory-management).

The repo's hooks never write to `~/.factory/`, so personal files stay
under your control.

## Optional — Plugin-ify for sharing

The whole `.factory/` tree can be packaged as a Factory plugin so other
repos in your org get the system with one command. See
`MEMORY-EXTENSIONS.md` for the conversion steps.

## Uninstall

The memory system is opt-in for runtime; to fully disable:

```bash
mv .factory/settings.json .factory/settings.json.disabled
```

The files stay readable, but no hooks fire. To remove entirely, delete
`.factory/`. The three `AGENTS.md` files remain useful even without the
runtime — they're plain documentation that any agent can use.

## Troubleshooting cheatsheet

| Symptom | First check |
|---------|-------------|
| `/verify` says "no such command" | `.factory/commands/verify.md` exists; restart `droid`. |
| PROGRESS.md never updates | Hooks not executable (`ls -la .factory/hooks`); chmod +x. |
| Hook timeout / hang | `droid --debug` shows the failing hook; bump `timeout` in `settings.json`. |
| Wrong AGENTS.md loaded | Check cwd: subproject AGENTS.md only loads when you're inside it. |
| Hooks print to terminal noisily | Set `suppressOutput: true` in the hook's JSON output. |
