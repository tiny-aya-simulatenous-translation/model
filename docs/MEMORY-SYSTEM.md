# Memory System — Keep Context Fresh

This repository implements the **External Memory System** pattern (Tip 10
from Factory.ai's power-user playbook): a small set of living markdown
files that act as the long-term memory of every Droid session. The picture
behind it lives at `simultaneous-translation/docs/memory-system.png`.

The system answers a simple problem:

> Droid sessions are stateless. Without an external memory, every restart
> begins from zero — same questions, same explorations, same gotchas
> rediscovered. We pay that cost once and write the answer down.

## The four canonical files

```
EXTERNAL MEMORY SYSTEM
+-------------------+      +------------------------+
| AGENTS.md         | <--> | PROGRESS.md            |
| repo norms,       |      | running log: changes,  |
| commands, gotchas |      | failures, next steps   |
+-------------------+      +------------------------+
         ^                              ^
         |                              |
         v                              v
+-------------------+      +------------------------+
| PLAN.md           | <--> | VERIFY.md              |
| checklist plan +  |      | exact commands to      |
| definition of done|      | prove it works         |
+-------------------+      +------------------------+
                |
        updating | retrieving
                v
            droid CLI
```

| File | Role | Lifetime |
|------|------|----------|
| `AGENTS.md` (3 tiers) | Norms, commands, gotchas | Stable; edited by humans |
| `PROGRESS.md` | Append-only log of what changed and what failed | Hot; auto-managed by hooks |
| `PLAN.md` | Current goal + checklist + Definition of Done | One per active goal |
| `VERIFY.md` | Curated, ordered shell commands that prove "done" | Stable; edited by humans |

A fifth file, `memories.md`, holds **long-term decisions** (architecture,
trade-offs, domain knowledge) that don't fit the running log.

## The three AGENTS.md tiers

This repo is a monorepo. Factory.ai's discovery rule is: agents merge
every AGENTS.md from the file under edit up to the repo root, with the
**closest one winning** on conflicts. We exploit this with three tiers:

```
/AGENTS.md                                       # monorepo-wide
/simultaneous-translation/AGENTS.md              # TPU + training subproject
/phase-3-data-generation-pipeline/AGENTS.md      # data + audio subproject
```

When you edit `simultaneous-translation/src/backend/tpu_backend.py`,
Droid loads the root AGENTS.md **plus** the training subproject's
AGENTS.md. When you edit `phase-3-data-generation-pipeline/cli.py`,
Droid loads the root **plus** the data-pipeline AGENTS.md.

A personal `~/.factory/AGENTS.md` is opt-in and not part of the repo.

## Where each piece lives

```
tinyaya-stage2-scale/
|-- AGENTS.md                                # tier 1: monorepo
|-- simultaneous-translation/
|   |-- AGENTS.md                            # tier 2a: training
|   `-- docs/
|       `-- MEMORY-SYSTEM.md                 # this file
|-- phase-3-data-generation-pipeline/
|   `-- AGENTS.md                            # tier 2b: data
`-- .factory/
    |-- AGENTS.md                            # (intentionally absent;
    |                                        #  tier 1 lives at repo root)
    |-- PROGRESS.md                          # auto-managed log
    |-- PLAN.md                              # current goal
    |-- VERIFY.md                            # done-criteria commands
    |-- memories.md                          # long-term decisions
    |-- archive/                             # quarterly-pruned PROGRESS
    |-- hooks/*.py                           # 6 lifecycle hooks
    |-- skills/<name>/SKILL.md               # 6 skills
    |-- droids/<name>.md                     # 3 custom droids
    |-- commands/<name>.md                   # 6 slash commands
    `-- settings.json                        # hook + permission registration
```

## The lifecycle (autonomous loop)

```
SessionStart  -> read 4 files, inject as additionalContext
              v
UserPromptSubmit -> if message starts with #progress / #plan / #decision /
              v   #verify, route to the right file
              v
PostToolUse  -> after each Edit/Create/Execute, append a structured
              v entry to PROGRESS.md (with timestamp + git rev)
              v
PreCompact   -> dump current state into PROGRESS.md so compaction
              v never loses context
              v
Stop         -> run verifier (executes VERIFY.md commands); on pass,
              v tick PLAN.md checkboxes; on fail, log to PROGRESS.md
              v
SessionEnd   -> write a "Next steps" block to PROGRESS.md from
                unchecked PLAN items
```

The next session's `SessionStart` reads exactly what the previous
`SessionEnd` wrote, so the tape is unbroken.

## Layers, end to end

| Layer | Component | Purpose |
|-------|-----------|---------|
| 1 — Files | 4 memory files + 3 AGENTS.md | The canonical state |
| 2 — Hooks | 6 Python scripts | Deterministic glue, never optional |
| 3 — Skills | 6 SKILL.md folders | Reusable workflows the agent can invoke |
| 4 — Droids | 3 custom subagents | Specialised roles (planner, verifier, curator) |
| 5 — Commands | 6 user-invocable slash commands | Manual overrides for everything |
| 6 — Settings | `settings.json` | Wires hooks to events, declares permissions |

## Why all six layers and not just files?

Files alone are passive — they require the user to remember to update them.
Hooks turn passive files into a system that updates itself; skills give
the agent a deterministic interface to read/write them; custom droids
isolate roles so plan-driving doesn't bleed into verification; slash
commands give humans a fast manual path. Each layer covers a failure mode
of the layer below it.

## How to operate

| You want to... | Use |
|----------------|-----|
| Remember a permanent decision | `/remember <text>` |
| Note today's progress manually | `/progress <text>` or `#progress <text>` |
| Generate a checklist for a new goal | `/plan` |
| Run all verifications | `/verify` |
| See current memory state in chat | `/recall` |
| Prune + archive old entries | `/curate` |

See `MEMORY-USAGE.md` for daily-driver examples and
`MEMORY-MAINTENANCE.md` for the monthly-review checklist.

## Extension points

The system is designed to grow. Common extensions:

- Add another tier of AGENTS.md when a new subproject lands.
- Replace the file-based `archive-progress` skill with an MCP memory
  server (e.g. mem0) for vector recall over years of PROGRESS.
- Wire `verify` into CI so the same commands gate human PRs.

`MEMORY-EXTENSIONS.md` walks through each.
