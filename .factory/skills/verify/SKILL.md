---
name: verify
description: Read .factory/VERIFY.md, execute every fenced bash block in order, and report pass/fail in a structured table. Use when the user asks "is it done", "/verify", or before declaring any milestone complete.
---

# Verify

Runs the curated verification commands and produces a clean
pass/fail report. The Stop hook invokes this automatically; humans can
invoke it via `/verify`.

## Steps

1. Parse `.factory/VERIFY.md`. Extract every fenced ```` ```bash ````
   block. Preserve order. Treat lines starting with `#` inside a block
   as comments.
2. For each block:
   - Run with `bash -eo pipefail -c <block>` from the repo root.
   - Capture stdout + stderr (last 400 chars).
   - Record exit code.
   - Honor a 20-second per-block timeout.
3. Render a `<json-render>` table with columns: `#`, `cmd`, `status`,
   `time (s)`, `exit`, `tail`.
4. If all pass:
   - Tick matching items in `.factory/PLAN.md` via the `update-plan`
     skill.
   - Append a `done | verify` entry to PROGRESS.md.
5. If any fail:
   - Append a `fail | verify` entry to PROGRESS.md with the failing
     command + tail.
   - List the failed commands at the bottom of the chat output.
   - Exit non-zero (so the Stop hook can decide whether to block).

## TPU-aware skipping

If `$PJRT_DEVICE` is unset, blocks that wrap themselves in
`[ -n "$PJRT_DEVICE" ] && ... || echo "skipped (no TPU)"` will report
"skipped" and count as passing. This lets the same VERIFY.md serve
both laptop and TPU contexts.

## Success criteria

- The table exhaustively reflects every block in VERIFY.md.
- A passing run leaves the working tree unchanged.
- A failing run leaves the working tree unchanged AND a new
  PROGRESS.md entry.

## Failure modes

- VERIFY.md missing -> point the user at `MEMORY-INSTALLATION.md`.
- A block runs longer than 60s total -> kill it and report `timeout`.
- A block prints to stderr but returns 0 -> still counts as pass; the
  stderr lands in the `tail` column.

## Composes with

- `update-plan` — called on full pass to tick PLAN items.
- `update-progress` — called regardless to log the verification.
