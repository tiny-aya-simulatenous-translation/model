---
description: Print the current memory state (PROGRESS, PLAN, VERIFY, memories, AGENTS.md) as structured tables.
---

Run the `keep-context-fresh` skill in interactive mode: read the four
memory files plus the relevant AGENTS.md tier(s), and render a clean
state summary as `<json-render>` tables and lists.

Output structure:

1. **Heading** — `Memory state` with current branch + short sha.
2. **Goal** — one-line summary from `.factory/PLAN.md`'s `## Goal`.
3. **DoD progress** — a `BarChart` (or numerical KeyValue) showing
   checked vs. unchecked Definition of Done items.
4. **Open PLAN tasks** — a Table of the top 10 unchecked items
   ordered by phase.
5. **Recent PROGRESS** — a Table of the 10 most-recent entries
   (timestamp, kind, status, one-line summary).
6. **Verify status** — a StatusLine pulled from the most recent
   `verify`-kind entry in PROGRESS.md.
7. **AGENTS.md tiers loaded** — a List of file paths, ordered by
   precedence (closest first).
8. **memories.md decision count** — a Metric.

Steps:

1. Read all four memory files plus root `AGENTS.md` and the closest
   subproject AGENTS.md (if cwd is inside one).
2. Compose the structured render.
3. Do not write any file.

Failure handling:

- If any of the four memory files is missing, render a Callout of
  type=warning and link to `MEMORY-INSTALLATION.md`.
- If PLAN.md has no `## Goal` heading, the Goal section reads
  "(none — run /plan to generate)".
