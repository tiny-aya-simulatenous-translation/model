# Memory System — Daily Usage

Practical patterns for using the four-file system day to day. All
examples assume `droid` is launched from the repo root.

## The minimal daily loop

1. **Start a session** — `SessionStart` hook auto-loads PROGRESS, PLAN,
   VERIFY, and the right AGENTS.md tiers.
2. **State your goal** — if no PLAN.md exists for the goal, the
   `plan-driver` droid generates one.
3. **Work** — `PostToolUse` hook records each non-trivial change to
   PROGRESS.md.
4. **Verify** — `Stop` hook (or `/verify`) runs all VERIFY.md commands.
5. **Exit** — `SessionEnd` writes "Next steps" to PROGRESS.md.

You don't have to remember any of these. The hooks fire automatically.

## Quick capture: the `#` tags

Begin a message with one of these tags to send a single-line entry to
the right file with timestamp and current git rev:

| Tag | Destination | Example |
|-----|-------------|---------|
| `#progress` | `.factory/PROGRESS.md` | `#progress fsdpv2_lora compile finished in 8 min after scan_layers` |
| `#plan` | `.factory/PLAN.md` (new sub-task) | `#plan add scan_layers around CohereDecoderLayer` |
| `#decision` | `.factory/memories.md` (Decisions) | `#decision use bf16 cast not XLA_USE_BF16 (deprecated 2.6+)` |
| `#verify` | `.factory/VERIFY.md` (new check) | `#verify python3 simultaneous-translation/scripts/tpu/probe_strategies.py` |

Each tag is intercepted by `UserPromptSubmit` so the entry lands in the
right file without any other interaction.

## Slash commands (manual mode)

| Command | Argument | Effect |
|---------|----------|--------|
| `/remember <text>` | free text | Append to memories.md, dated. |
| `/progress <text>` | free text | Append to PROGRESS.md, dated. |
| `/plan` | none | Run plan-driver to (re)generate PLAN.md from current goal. |
| `/verify` | none | Run all VERIFY.md commands and report. |
| `/recall` | none | Print all four memory files in chat as tables. |
| `/curate` | none | Run memory-curator to dedupe + archive old entries. |

## Real-world examples (TPU training)

### Starting a fresh shift after a context loss

```
$ droid
> /recall
```

Droid prints PROGRESS, PLAN, VERIFY, memories. You see "Next steps" left
by yesterday's `SessionEnd`: "Implement scan_layers around
CohereDecoderLayer". Continue from there.

### Capturing a decision in flight

```
> #decision use fsdpv2_lora over fsdpv2 for first stable run because
  comm volume is lower with frozen MoshiDecoder layers
```

Droid acknowledges, the line lands in `memories.md`, and the session
continues without breaking flow.

### Verifying before declaring victory

```
> we should be done with the bf16 cast change
> /verify
```

The verify skill runs every command in VERIFY.md, reports a
pass/fail table, and (if all green) ticks the matching PLAN.md
checkbox. If anything fails, the failure is appended to PROGRESS.md
with the exact command and exit code so you can resume later.

### Adding a new check on the fly

```
> #verify python3 -c "import torch_xla; print(torch_xla.__version__)"
```

The new line is added to VERIFY.md and will run on every subsequent
`/verify`.

## Working across subprojects

The three AGENTS.md tiers cooperate. When you're working inside
`simultaneous-translation/`, Droid sees:

```
/AGENTS.md                                       (monorepo norms)
+ /simultaneous-translation/AGENTS.md            (TPU specifics, override)
```

When you're working inside `phase-3-data-generation-pipeline/`:

```
/AGENTS.md                                       (monorepo norms)
+ /phase-3-data-generation-pipeline/AGENTS.md    (data specifics, override)
```

If you need to do cross-cutting work, just stay at the repo root — only
the root AGENTS.md applies.

## Avoiding common mistakes

- **Don't edit PROGRESS.md by hand.** Hooks reformat on every write; manual
  entries that don't match the line format may get clobbered. Use
  `/progress` or `#progress` instead.
- **Do edit VERIFY.md by hand.** It is a curated list; treat it like a
  Makefile target.
- **Do edit memories.md by hand for big decisions.** The `/remember`
  command is for quick captures; longer architectural notes deserve a
  proper section.
- **Treat PLAN.md as ephemeral per goal.** When a goal completes, the
  curator archives old PLANs into memories.md as "completed milestones".

## When things drift

If PROGRESS.md grows past a few hundred lines, run `/curate`. If the
session feels like it's lost track, run `/recall` and re-anchor on the
real state. If `/verify` keeps failing, fix the underlying problem
**before** writing more code — VERIFY.md is the contract for "done".
