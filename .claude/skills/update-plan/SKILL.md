---
name: update-plan
description: Maintain .claude/PLAN.md - tick completed items, add new sub-tasks, edit Definition of Done. Idempotent. Use when work has progressed and the plan needs to reflect reality.
---

# Update Plan

Edits PLAN.md in place. The Stop hook calls this on every successful
verify; humans can call it manually with `/plan` (which delegates to
the `plan-driver` subagent for full regeneration) or by adding
`#plan <task>` quick-capture lines.

In the unified TPU control plane, `PLAN.md` owns only the active goal,
phase checklist, and Definition of Done. Put chronological run events in
`PROGRESS.md`, durable decisions in `memories.md`, and operating
procedures in `.claude/orchestration/`.

## Inputs

One of:

- `tick: <substring>` — find the first matching `- [ ]` line and
  promote it to `- [x]`.
- `add: <text>` — add a new `- [ ] <text>` under `## Tasks`.
- `dod_add: <text>` — add a new bullet under `## Definition of Done`.
- `regenerate: true` — delegate to `plan-driver` subagent.

## Steps

1. Read `.claude/PLAN.md`.
2. Apply the requested edit:
   - For `tick`: case-insensitive substring search; replace `- [ ]`
     with `- [x]` on the first match. If no match, fail loudly.
   - For `add`: append under `## Tasks`, before any `## Out of scope`.
   - For `dod_add`: append under `## Definition of Done`.
3. Write the file back atomically.
4. If all DoD items are checked, append a "milestone" entry to
   `memories.md` and a `done | plan` entry to `PROGRESS.md`.

## Idempotency

- Ticking an already-checked item is a no-op.
- Adding an item that already exists (substring match on the body) is
  a no-op.

## Success criteria

- The diff applied is minimal — single-line change for tick, single-line
  insert for add.
- File still parses as markdown (use `python3 -c "import re;
  re.compile('^- \\[[ x]\\]', re.M)"` mentally).

## Failure modes

- `## Tasks` heading missing -> refuse to add; ask the user to seed
  PLAN.md from the template.
- More than one match for `tick` -> tick the first one and warn.
