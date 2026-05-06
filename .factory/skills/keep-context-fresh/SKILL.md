---
name: keep-context-fresh
description: Master entry-point for the External Memory System. Invoke at the start of any non-trivial task to load PROGRESS, PLAN, VERIFY, AGENTS.md, and memories.md, then propose the next action consistent with what's recorded. Use when the user asks "where were we", "what's the state", "load context", or starts a new goal.
---

# Keep Context Fresh

This is the master skill that orchestrates the External Memory System
("TIP 10: Keep Context Fresh"). It guarantees that every non-trivial
session begins from the same snapshot of state.

## When to use

- The user starts a new task and you have no recent memory of the goal.
- The user asks "where were we?", "load context", "recall the state".
- A new session begins after a `clear`, compact, or restart.
- Before producing a plan that touches multiple files or subprojects.

## Steps

1. Read all four canonical memory files:
   - `.factory/PROGRESS.md` (last 30 entries)
   - `.factory/PLAN.md` (in full)
   - `.factory/VERIFY.md` (in full)
   - `.factory/memories.md` (in full)

2. Read the relevant `AGENTS.md` tier(s) for the cwd:
   - Root `/AGENTS.md` always.
   - The closest subproject `AGENTS.md` if cwd is under it.

3. Render a concise state summary in chat using `<json-render>`:
   - Goal (from PLAN heading)
   - Top 5 unchecked PLAN items
   - Top 5 most recent PROGRESS entries (with status)
   - Open verification commands
   - One-line "What I'd do next" suggestion derived from PLAN + PROGRESS

4. Ask the user to confirm the goal before proceeding to actual edits.

## Inputs

- Optional `goal: <string>` override if the user names a different goal.

## Success criteria

- All four memory files were read in the current session (verifiable by
  Droid's tool log).
- The user has either confirmed the goal or asked for plan regeneration
  (in which case invoke the `plan-driver` custom droid).

## Failure modes

- If any memory file is missing, do **not** silently continue. Surface
  the gap and offer to recreate from `MEMORY-INSTALLATION.md`.
- If `AGENTS.md` is missing at root, refuse and instruct the user to
  install the memory system first.

## Composes with

- `recall-context` — programmatic recall, no chat output (used by hooks).
- `plan-driver` (custom droid) — for goal -> PLAN.md generation.
- `verify` — for running VERIFY.md commands.
