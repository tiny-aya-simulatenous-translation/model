# VERIFY — commands that prove "done"

> The `verify` skill (and `Stop` hook) reads each fenced bash block
> below in order, executes it, and considers the section passing if
> exit code is 0. Edit by hand or via `#verify <command>` quick-capture.

## monorepo

```bash
# settings.json is valid JSON
python3 -c "import json; json.load(open('.factory/settings.json'))"
```

```bash
# every hook script parses
for h in .factory/hooks/*.py; do python3 -m py_compile "$h" || exit 1; done
```

```bash
# every AGENTS.md tier exists and is non-empty
for a in AGENTS.md simultaneous-translation/AGENTS.md phase-3-data-generation-pipeline/AGENTS.md; do
  test -s "$a" || { echo "EMPTY: $a"; exit 1; }
done
```

```bash
# four core memory files exist
for f in PROGRESS.md PLAN.md VERIFY.md memories.md; do
  test -s ".factory/$f" || { echo "EMPTY: .factory/$f"; exit 1; }
done
```

```bash
# working tree is sane (no merge markers, no obvious leaks)
! grep -rE '^(<<<<<<< |======= |>>>>>>> )' \
    --include='*.py' --include='*.md' --include='*.sh' --include='*.json' \
    .factory simultaneous-translation phase-3-data-generation-pipeline AGENTS.md
```

## simultaneous-translation

```bash
# all shell scripts parse
bash -n simultaneous-translation/scripts/tpu/*.sh
```

```bash
# every Python source compiles
find simultaneous-translation/src -name '*.py' -print0 \
  | xargs -0 -n1 python3 -m py_compile
```

```bash
# entry-point scripts compile
python3 -m py_compile simultaneous-translation/scripts/train_hierarchical.py
python3 -m py_compile simultaneous-translation/scripts/eval_stage2.py
python3 -m py_compile simultaneous-translation/scripts/make_splits.py
python3 -m py_compile simultaneous-translation/scripts/tpu/probe_strategies.py
```

```bash
# every YAML config parses
for y in simultaneous-translation/configs/*.yaml; do
  python3 -c "import yaml,sys; yaml.safe_load(open('$y'))"
done
```

```bash
# secrets check on tracked + new files
! grep -rnE '(hf_[a-zA-Z0-9]{20,}|sk-[a-zA-Z0-9]{20,}|gh[ps]_[a-zA-Z0-9]{30,}|AKIA[A-Z0-9]{16})' \
    --exclude-dir='.venv' --exclude-dir='__pycache__' --exclude-dir='.mypy_cache' \
    --include='*.py' --include='*.sh' --include='*.yaml' --include='*.md' --include='*.json' \
    simultaneous-translation
```

## phase-3-data-generation-pipeline

```bash
# all Python source compiles
find phase-3-data-generation-pipeline -name '*.py' -not -path '*/__pycache__/*' -print0 \
  | xargs -0 -n1 python3 -m py_compile
```

```bash
# CLI entry point loads and prints help
python3 phase-3-data-generation-pipeline/cli.py --help > /dev/null \
  || (cd phase-3-data-generation-pipeline && PYTHONPATH=. python3 cli.py --help > /dev/null)
```

## TPU sharding (live mesh — only run on a TPU host)

> These commands require a connected TPU runtime. The `verify` skill
> skips them when `PJRT_DEVICE` is unset. Add new probes via
> `#verify <cmd>`.

```bash
# probe replicated strategy on tiny model
[ -n "$PJRT_DEVICE" ] && python3 simultaneous-translation/scripts/tpu/probe_strategies.py --strategy=replicated || echo "skipped (no TPU)"
```

```bash
# probe fsdpv2_lora strategy on tiny model
[ -n "$PJRT_DEVICE" ] && python3 simultaneous-translation/scripts/tpu/probe_strategies.py --strategy=fsdpv2_lora || echo "skipped (no TPU)"
```

```bash
# probe fsdpv2 strategy on tiny model
[ -n "$PJRT_DEVICE" ] && python3 simultaneous-translation/scripts/tpu/probe_strategies.py --strategy=fsdpv2 || echo "skipped (no TPU)"
```

## Orchestration artifacts (workstation-side)

> Validate the self-healing orchestrator scaffolding stays consistent.

```bash
# orchestrator spec + diagrams + playbook all parseable / non-empty
test -s .factory/orchestration/SPEC.md
test -s .factory/orchestration/README.md
for d in .factory/orchestration/diagrams/*.mmd; do test -s "$d"; done
for p in .factory/orchestration/playbook/*.md; do test -s "$p"; done
```

```bash
# orchestrator droids + skills exist
test -s .factory/droids/tpu-watchdog.md
test -s .factory/droids/tpu-diagnoser.md
test -s .factory/skills/tpu-orchestrate/SKILL.md
test -s .factory/skills/tpu-redeploy/SKILL.md
```

```bash
# poller + checkin helper compile
python3 -m py_compile _artifacts/orch_poll.py
python3 -m py_compile _artifacts/scheduled_checkin.py
```

```bash
# orch_state.json is valid JSON if present
test ! -e _artifacts/orch_state.json \
  || python3 -c "import json; json.load(open('_artifacts/orch_state.json'))"
```
