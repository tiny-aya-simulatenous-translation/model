# Memory System — Extensions

The base system covers the common case: four files, six hooks, six
skills, three droids, six commands. This doc walks through the
extension points when the base is no longer enough.

## 1. Add another AGENTS.md tier

When a new subproject lands (say, `inference-server/`), give it its own
AGENTS.md immediately:

```bash
touch inference-server/AGENTS.md
```

Seed it with the same six sections as the existing tier-2 files (see
`/simultaneous-translation/AGENTS.md` for the canonical layout). Droid
auto-discovers it on next session.

## 2. Promote `.factory/` into a Factory plugin

The whole tree (`.factory/hooks/`, `.factory/skills/`, etc.) can be
shipped as a Factory plugin so other repos in the org get the system
with `droid plugin install`. Conversion outline:

```
.factory/
+-> plugin/
    +-> plugin.json                # name, version, author
    +-> hooks/hooks.json           # hook events table
    +-> hooks/*.py                 # the six scripts (unchanged)
    +-> skills/<name>/SKILL.md     # six skills (unchanged)
    +-> commands/<name>.md         # six commands (unchanged)
    +-> droids/<name>.md           # three droids (unchanged)
```

Replace `$FACTORY_PROJECT_DIR` with `${DROID_PLUGIN_ROOT}` inside hook
commands. Publish via `droid plugin publish` per Factory docs.

## 3. Replace file-based archive with an MCP memory server

For long-lived projects, grep over thousands of PROGRESS lines becomes
slow. Drop in a vector-search MCP server (mem0, Chroma-MCP, etc.):

```jsonc
// .factory/settings.json (mcp section)
"mcpServers": {
  "memory": {
    "command": "uvx",
    "args": ["mem0-mcp", "--collection", "tinyaya"]
  }
}
```

Then update `archive-progress/SKILL.md` to push old entries into the
collection instead of (or in addition to) the file archive. The four
canonical `.md` files stay unchanged — MCP is purely a richer recall
backend.

## 4. Wire `/verify` into CI

The same VERIFY.md commands that gate Droid sessions can gate human
PRs. Minimum viable GitHub Action:

```yaml
# .github/workflows/verify.yml
name: verify
on: [push, pull_request]
jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.12' }
      - name: Run verifier
        run: bash .factory/skills/verify/run.sh
```

(`.factory/skills/verify/run.sh` is the headless equivalent of the
`/verify` slash command — extract every fenced block from VERIFY.md and
run it sequentially.)

## 5. Add a memory-aware custom slash command

Example: `/triage` — print only the unfinished items from PLAN.md and
the most recent five PROGRESS failures.

```markdown
---
name: triage
description: Show only failing or unfinished work to focus the next session.
---

Read .factory/PLAN.md and .factory/PROGRESS.md.

Output two short lists:
1. Unchecked items in PLAN.md (top 5 by recency).
2. Last 5 PROGRESS entries with status:fail.

Format as a markdown table.
```

Drop the file at `.factory/commands/triage.md`. Restart Droid; the
command is live.

## 6. Hook into Droid Computers / persistent VMs

When the project moves to a Droid Computer (a long-lived VM that
survives session restarts), the same `.factory/` tree works unchanged
on that VM. The only adjustment: ensure the VM has its own clone of
the repo (so `$FACTORY_PROJECT_DIR` resolves correctly) and that the
hooks have execute bit set (`chmod -R +x .factory/hooks/`).

## 7. Cross-machine memory sync

If you work across a laptop and a TPU host, commit the four `.md`
files (PROGRESS, PLAN, VERIFY, memories) regularly to a `memory/`
branch and rebase. The hooks already write to disk; git is the sync
mechanism.

A lightweight commit hook:

```json
{
  "hooks": {
    "Stop": [
      {
        "hooks": [{
          "type": "command",
          "command": "git add .factory/{PROGRESS,PLAN}.md && git commit -m 'memory: auto-sync' --allow-empty"
        }]
      }
    ]
  }
}
```

Use with care — auto-commits add noise to git history.

## 8. Telemetry and audit

For enterprise / regulated environments, every memory write can be
exported via OpenTelemetry. Add to each hook's tail:

```python
# at the end of session_end.py, post_tool_use.py, etc.
import os, requests
otlp = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
if otlp:
    requests.post(otlp + "/v1/logs", json={
        "body": "memory.write",
        "attributes": {"file": mem_file, "actor": session_id},
    }, timeout=2)
```

The settings.json policy block can then require OTEL export for any
write under `.factory/`.

## 9. Multi-author repositories

When more than one engineer commits memory updates, conflicts on
PROGRESS.md become common (everyone appends). Two strategies:

- **Rebase-on-pull** — keep PROGRESS.md as a single file but always
  pull --rebase. Auto-merge usually succeeds for append-only files.
- **Per-author log** — `.factory/PROGRESS-<email>.md`, one per author,
  with a curated `PROGRESS.md` that cites the others. The
  memory-curator skill knows how to walk the per-author files.

## 10. Different memory shapes per goal

The four-file model assumes one active goal at a time. For parallel
goals, give each its own subdirectory:

```
.factory/
+-> goals/
    +-> tpu-canary/PLAN.md
    +-> tpu-canary/PROGRESS.md
    +-> data-pipeline-v2/PLAN.md
    +-> data-pipeline-v2/PROGRESS.md
+-> VERIFY.md                       # shared across goals
+-> memories.md                     # shared across goals
```

Update the `recall-context` skill to honor a `FACTORY_GOAL=tpu-canary`
env var so different terminals scope to different goals.
