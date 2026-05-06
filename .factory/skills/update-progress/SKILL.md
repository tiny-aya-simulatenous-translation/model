---
name: update-progress
description: Append a structured entry to .factory/PROGRESS.md with timestamp, branch, sha, status, kind, and detail. Use when the user asks to log a change, capture a failure, or record a decision after the fact.
---

# Update Progress

Programmatic interface to the running log. Used by hooks and custom
droids; humans should prefer `/progress` or `#progress` triggers.

## Inputs

- `status`: one of `info | done | fail | block` (required)
- `kind`: one of `edit | exec | decide | plan | verify | session`
  (required)
- `summary`: a one-line summary, <= 280 chars (required)
- `detail`: optional multi-line block

## Steps

1. Read `.factory/PROGRESS.md` to find the first `\n---\n` after the
   format-spec block.
2. Compose the entry:
   ```
   ## <iso8601 utc> | <branch>@<short-sha> | <status> | <kind>
   <summary>

   <detail>
   ```
3. Insert the entry **after** the marker (newest entries first).
4. Run the same secret-scrub regex used by `_lib.py` before writing.
5. Confirm to the caller in one short line.

## Success criteria

- New entry is the topmost dated entry in PROGRESS.md.
- Branch and short-sha match `git rev-parse --abbrev-ref HEAD` and
  `git rev-parse --short HEAD` respectively.
- No secret patterns survived the scrub.

## Failure modes

- File missing -> create from template `MEMORY-SYSTEM.md` and retry.
- Locked / read-only -> surface the error; do not silently swallow.

## Composes with

- `update-plan` — typically runs together when a plan item completes.
- `archive-progress` — runs after this skill to keep the file lean.
