# Memory System — Maintenance

The memory system is mostly self-managing, but a few periodic tasks
keep it useful at scale.

## Monthly review checklist

```
[ ] Run /curate to dedupe and archive old PROGRESS entries
[ ] Skim memories.md; mark any "Decision" that was reversed
[ ] Verify VERIFY.md commands still pass on a clean checkout
[ ] Skim each AGENTS.md tier; remove stale gotchas
[ ] Confirm .factory/archive/ has the previous quarter file
```

## Archival policy

`PROGRESS.md` is append-only and grows. The `archive-progress` skill
moves entries into a quarterly archive when either condition is met:

- the file exceeds 500 lines
- entries older than 90 days are present

Archived files are named `.factory/archive/PROGRESS-YYYY-Qn.md` and are
**read-only** (Droid will refuse to edit them; recall is fine).

## Pruning memories.md

Decisions become obsolete when their reasoning no longer applies.
Rather than deleting, mark them as **superseded**:

```markdown
### 2024-09: Use replicated SPMD for first canary
**Status:** SUPERSEDED 2024-10 by fsdpv2_lora once OOM was confirmed
**Original decision:** Per-chip replication for fastest compile
**Why we changed:** Composite model exceeded HBM at batch=1
```

The curator preserves SUPERSEDED entries because the *reason for the
change* is itself valuable history.

## Hook health

Every two weeks, check that hooks are firing:

```bash
droid --debug                # then ask Droid to run any command
```

Look for lines like:

```
[DEBUG] Executing hooks for PostToolUse:Edit
[DEBUG] Found 1 hook commands to execute
[DEBUG] Hook command completed with status 0
```

If any hook reports a non-zero exit, inspect:

```bash
ls -la .factory/hooks/        # executable bit set?
python3 .factory/hooks/post_tool_use.py < /dev/null   # parses stdin?
```

## Settings drift detection

Droid snapshots `settings.json` at session start. If you edit it
mid-session, the changes are inert until restart. To verify the live
state:

- Run `/hooks` inside Droid; the rendered table is the live snapshot.
- Compare with `cat .factory/settings.json`. Discrepancies mean a
  stale session — restart.

## Disabling individual hooks (debugging)

To disable a single hook without removing it:

```bash
chmod -x .factory/hooks/post_tool_use.py
```

Droid will log "command not executable" but continue. Re-enable with
`chmod +x` and restart the session.

To disable all hooks at once:

```bash
mv .factory/settings.json .factory/settings.json.disabled
```

## Recovering from a corrupted memory file

PROGRESS.md is the only file that hooks rewrite frequently and is most
at risk. To recover:

```bash
git log --oneline .factory/PROGRESS.md   # find a clean revision
git checkout <sha> -- .factory/PROGRESS.md
```

If you committed memory updates regularly (recommended), this is
trivial. If not, the archive in `.factory/archive/` is the next
fallback.

## Verifying the system end-to-end

Run this trio after any config change:

```bash
# 1. Hook scripts parse
for h in .factory/hooks/*.py; do python3 -m py_compile "$h"; done

# 2. settings.json is valid JSON
python3 -c "import json; json.load(open('.factory/settings.json'))"

# 3. Each AGENTS.md is non-empty markdown
for a in AGENTS.md */AGENTS.md; do test -s "$a" || echo "EMPTY: $a"; done
```

The same trio is the first section of `.factory/VERIFY.md`, so
`/verify` covers it automatically.

## When to escalate

If `/verify` keeps failing for unrelated reasons, or if hooks start
producing systematic false positives, freeze the system:

1. `mv .factory/settings.json .factory/settings.json.disabled`
2. Open an issue with the contents of `droid --debug` output.
3. Resume work without hooks until the issue is identified — the
   memory files remain valuable as plain markdown.
