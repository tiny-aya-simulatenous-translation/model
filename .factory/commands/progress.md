---
description: Append a manual entry to .factory/PROGRESS.md (without going through a hook).
argument-hint: <what to log>
---

Append `$ARGUMENTS` to `.factory/PROGRESS.md` as the newest entry.

Use the `update-progress` skill with:

- `status: info`
- `kind: session`
- `summary: $ARGUMENTS` (truncated to 280 chars)

Format the entry exactly as PROGRESS.md's header block specifies:

```
## <iso8601 utc> | <branch>@<short-sha> | info | session
$ARGUMENTS
```

Insert the entry **after** the first `\n---\n` separator so it
becomes the topmost dated entry.

Run the secret-scrub before writing.

If `$ARGUMENTS` is empty, prompt the user for what to log and stop.

Confirm in one line: `memory: appended to PROGRESS.md`.
