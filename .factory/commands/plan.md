---
description: Generate or regenerate .factory/PLAN.md from the current goal using the plan-driver custom droid.
argument-hint: <optional explicit goal>
---

Delegate to the `plan-driver` custom droid (see
`.factory/droids/plan-driver.md`) to produce or regenerate
`.factory/PLAN.md`.

Steps:

1. If `$ARGUMENTS` is provided, treat it as the explicit goal. If not,
   ask the user to state the goal in one sentence.
2. Read the current `.factory/PLAN.md` (if any) so the existing
   structure is preserved when refining.
3. Invoke the `plan-driver` droid via the Task tool with:
   - The user's goal
   - The current branch
   - A pointer to `.factory/PROGRESS.md` and `.factory/memories.md`
4. Wait for the droid's report.
5. Verify the new PLAN.md:
   - Has `## Goal`, `## Definition of Done`, `## Tasks`,
     `## Out of scope` headings.
   - Has at least 5 DoD items, each observable.
   - Has at least 3 phased Task subsections.
6. If the droid produced a non-conforming output, ask it to retry
   once with the constraints made explicit.
7. Confirm in one line: `plan: regenerated PLAN.md from goal "<...>"`.

Out of scope for this command: editing PROGRESS.md, memories.md,
VERIFY.md, or any AGENTS.md.
