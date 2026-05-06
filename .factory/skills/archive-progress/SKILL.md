---
name: archive-progress
description: Move PROGRESS.md entries older than 90 days (or beyond 500 total lines) into .factory/archive/PROGRESS-YYYY-Qn.md. Use when PROGRESS.md grows unwieldy or as part of monthly maintenance.
---

# Archive Progress

Keeps `.factory/PROGRESS.md` lean by moving old entries to a
quarterly archive. Used by `/curate` and the `memory-curator` custom
droid.

## When to run

- Monthly, as part of `MEMORY-MAINTENANCE.md`.
- Whenever `PROGRESS.md` exceeds 500 lines.
- Before a major branch cut where you want a clean log.

## Steps

1. Read `.factory/PROGRESS.md`.
2. Parse the file into entry blocks: each block starts with
   `## YYYY-MM-DDTHH:MM:SSZ ...`. Headers and the `## Next steps`
   section are not entries.
3. Compute the cutoff:
   - `now - 90 days` (configurable via `MEMORY_ARCHIVE_DAYS` env).
4. For each entry older than cutoff, group by `YYYY-Qn` based on the
   timestamp.
5. For each group, append the entries (in chronological order) to
   `.factory/archive/PROGRESS-YYYY-Qn.md`. Create the file if absent.
6. Remove the archived entries from PROGRESS.md.
7. Add a single `info | session` entry to PROGRESS.md noting the
   archival count + destination filenames.

## Idempotency

- Archive files are append-only; rerunning the skill is a no-op for
  entries already moved.
- The skill never deletes from archive files.

## Success criteria

- `wc -l .factory/PROGRESS.md` decreases.
- All archive files start with a 1-line header `# PROGRESS archive
  YYYY Qn` if newly created.
- Total entry count (PROGRESS + all archives) is unchanged.

## Failure modes

- Permission denied on archive dir -> create then retry; report and
  stop on second failure.
- Malformed entry header -> skip, log a warning to PROGRESS.md.

## Composes with

- `update-progress` — used to record the archival event.
- `memory-curator` (custom droid) — runs this skill plus a dedupe pass.
