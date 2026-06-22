---
name: recall-context
description: Background knowledge skill. Auto-loads when the agent plans a multi-step task; reads memory files into context without rendering them in chat. Use when starting a substantive change.
---

# Recall Context

A silent, model-only counterpart to `keep-context-fresh`. The agent
loads it automatically when it judges a task non-trivial; it does not
appear in the slash-command palette.

## When the agent invokes this

- Before generating a multi-file plan or refactor.
- Before answering a question that requires knowing prior decisions
  (e.g. "why do we use bf16 cast?").
- Before producing a commit message that should align with the
  PROGRESS log.

## Steps

1. Read `.claude/PROGRESS.md`, last 50 lines.
2. Read `.claude/memories.md` in full.
3. Read `.claude/PLAN.md` in full.
4. Read root `AGENTS.md` and the nearest subproject `AGENTS.md`.
5. For TPU/orchestration/optimization tasks, read
   `.claude/orchestration/CONTROL_PLANE.md` and the relevant spec:
   `TPU_OPTIMIZATION_SPEC.md` for performance work, or `SPEC.md` for
   live run supervision and recovery.

6. Produce no chat output. Internal reasoning only. The agent should
   make subsequent decisions consistent with what it just read.

## Success criteria

- The next action references at least one memory entry by date or
  decision title (e.g. "per memories.md 2026-05-03 decision on bf16").
- TPU work also references the control plane or active orchestration
  spec in the plan.

## When NOT to invoke

- Trivial single-line edits.
- Questions answered entirely by code in the working tree.
- When the user explicitly asks for a fresh take ("ignore prior
  decisions").
