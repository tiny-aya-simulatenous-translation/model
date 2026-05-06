---
description: Save a long-term decision or piece of domain knowledge to .factory/memories.md.
argument-hint: <what to remember>
---

Append the user's note to `.factory/memories.md` under the
`## Architecture decisions` section, formatted as:

```markdown
### YYYY-MM-DD: <short title derived from the note>
**Decision:** $ARGUMENTS
**Captured-from:** /remember
```

Use today's UTC date. Choose a short title (<= 80 chars) by taking the
first sentence (or up to 80 chars) of `$ARGUMENTS`.

Steps:
1. Read `.factory/memories.md`.
2. Locate the `## Architecture decisions` heading.
3. Insert the new block immediately before the next `## ` heading
   (or at end of section if it's the last one).
4. Run the secret-scrub regex (`hf_[a-zA-Z0-9]{20,}` etc.) before
   writing; replace any match with `[REDACTED]`.
5. Write the file back.
6. Confirm in one short line: `memory: saved decision to memories.md`.

If `$ARGUMENTS` is empty, ask the user what to remember and stop.
