---
description: Run every command in .factory/VERIFY.md and produce a structured pass/fail report.
---

Delegate to the `verifier` custom droid (see
`.factory/droids/verifier.md`).

Steps:

1. Confirm `.factory/VERIFY.md` exists and is non-empty.
2. Invoke the `verifier` droid via the Task tool. It will:
   - Run every fenced bash block in VERIFY.md, in order.
   - Skip TPU-specific blocks if `$PJRT_DEVICE` is unset.
   - Render a `<json-render>` table with results.
   - Append a `done | verify` or `fail | verify` entry to PROGRESS.md.
3. Render the droid's output verbatim.
4. If any failure, surface the first failing command at the top in a
   StatusLine: `verify: FAIL on <command>`.
5. If all pass, also tick matching items in `.factory/PLAN.md` via
   the `update-plan` skill.

This command **does not** block the session on failure by default. To
make it blocking, set `MEMORY_STOP_BLOCK_ON_FAIL=1` in your shell
before launching `droid`.

Out of scope: editing VERIFY.md, modifying source code, or
auto-fixing failures. The verifier reports; the user decides.
