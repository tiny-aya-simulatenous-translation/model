# VERIFY — commands that prove "done"

> The `verify` skill (and `Stop` hook) reads each fenced bash block
> below in order, executes it, and considers the section passing if
> exit code is 0. Edit by hand or via `#verify <command>` quick-capture.
>
> Layout note: this repo is **flattened** — code lives at the repo root
> (`src/`, `scripts/`, `configs/{gpu,tpu}/`, `sweeps/`), not under a
> `simultaneous-translation/` subdir. TPU run-infra is `scripts/tpu/`.

## repo

```bash
# settings.json is valid JSON
python3 -c "import json; json.load(open('.claude/settings.json'))"
```

```bash
# every hook script parses
for h in .claude/hooks/*.py; do python3 -m py_compile "$h" || exit 1; done
```

```bash
# AGENTS.md exists and is non-empty
test -s AGENTS.md || { echo "EMPTY: AGENTS.md"; exit 1; }
```

```bash
# four core memory files exist
for f in PROGRESS.md PLAN.md VERIFY.md memories.md; do
  test -s ".claude/$f" || { echo "EMPTY: .claude/$f"; exit 1; }
done
```

```bash
# working tree is sane (no merge markers)
! grep -rE '^(<<<<<<< |======= |>>>>>>> )' \
    --include='*.py' --include='*.md' --include='*.sh' --include='*.json' \
    .claude src scripts configs sweeps AGENTS.md
```

## code (flattened root)

```bash
# all shell scripts parse
bash -n scripts/tpu/*.sh
```

```bash
# every Python source compiles
find src -name '*.py' -print0 | xargs -0 -n1 python3 -m py_compile
```

```bash
# entry-point scripts compile
for s in scripts/train_hierarchical.py scripts/make_splits.py \
         scripts/eval_checkpoint.py scripts/gen_parallel.py \
         scripts/tpu/probe_strategies.py; do
  test -f "$s" && python3 -m py_compile "$s"
done
```

```bash
# every YAML config parses (gpu/ + tpu/ + sweeps)
for y in configs/*/*.yaml sweeps/*.yaml; do
  python3 -c "import yaml,sys; yaml.safe_load(open('$y'))"
done
```

```bash
# the TPU↔GPU seam holds: torch_xla must NOT leak into shared model/data code
! grep -rn 'import torch_xla\|from torch_xla' src/model src/data
```

```bash
# torch-free unit tests (sweep promote helper + sweep/config consistency)
python3 tests/test_sweep_promote.py
```

```bash
# secrets check on tracked + new files
! grep -rnE '(hf_[a-zA-Z0-9]{20,}|sk-[a-zA-Z0-9]{20,}|gh[ps]_[a-zA-Z0-9]{30,}|AKIA[A-Z0-9]{16})' \
    --exclude-dir='.venv' --exclude-dir='__pycache__' --exclude-dir='.mypy_cache' \
    --include='*.py' --include='*.sh' --include='*.yaml' --include='*.md' --include='*.json' \
    src scripts configs sweeps
```

## TPU sharding (live mesh — only run on a TPU host)

> These commands require a connected TPU runtime. The `verify` skill
> skips them when `PJRT_DEVICE` is unset. Add new probes via
> `#verify <cmd>`.

```bash
# probe replicated strategy on tiny model
[ -n "$PJRT_DEVICE" ] && python3 scripts/tpu/probe_strategies.py --strategy=replicated || echo "skipped (no TPU)"
```

```bash
# probe fsdpv2_lora strategy on tiny model
[ -n "$PJRT_DEVICE" ] && python3 scripts/tpu/probe_strategies.py --strategy=fsdpv2_lora || echo "skipped (no TPU)"
```

```bash
# probe fsdpv2 strategy on tiny model
[ -n "$PJRT_DEVICE" ] && python3 scripts/tpu/probe_strategies.py --strategy=fsdpv2 || echo "skipped (no TPU)"
```

## Orchestration artifacts (workstation-side)

> Validate the self-healing orchestrator scaffolding stays consistent.

```bash
# orchestrator spec + diagrams + playbook all parseable / non-empty
test -s .claude/orchestration/SPEC.md
test -s .claude/orchestration/README.md
test -s .claude/orchestration/CONTROL_PLANE.md
test -s .claude/orchestration/TPU_OPTIMIZATION_SPEC.md
for d in .claude/orchestration/diagrams/*.mmd; do test -s "$d"; done
for p in .claude/orchestration/playbook/*.md; do test -s "$p"; done
```

```bash
# orchestrator agents + skills exist
test -s .claude/agents/tpu-watchdog.md
test -s .claude/agents/tpu-diagnoser.md
test -s .claude/skills/tpu-orchestrate/SKILL.md
test -s .claude/skills/tpu-redeploy/SKILL.md
test -s .claude/skills/keep-context-fresh/SKILL.md
test -s .claude/skills/recall-context/SKILL.md
```

```bash
# poller + checkin helper compile (workstation-local; skip if absent)
for f in _artifacts/orch_poll.py _artifacts/scheduled_checkin.py; do
  test -f "$f" && python3 -m py_compile "$f" || true
done
```

```bash
# orch_state.json is valid JSON if present
test ! -e _artifacts/orch_state.json \
  || python3 -c "import json; json.load(open('_artifacts/orch_state.json'))"
```

```bash
# optimization control-plane files are wired
for f in \
  .claude/orchestration/diagrams/06-optimization-flow.mmd \
  .claude/orchestration/diagrams/07-control-plane.mmd \
  .claude/orchestration/diagrams/08-memory-lifecycle.mmd \
  .claude/orchestration/playbook/optimization-experiment-matrix.md \
  .claude/orchestration/playbook/perf-metrics-schema.md; do
  test -s "$f" || { echo "EMPTY: $f"; exit 1; }
done
grep -q "TPU_OPTIMIZATION_SPEC.md" .claude/orchestration/README.md
grep -q "CONTROL_PLANE.md" .claude/skills/tpu-orchestrate/SKILL.md
grep -q "ORCHESTRATION_CONTROL_PLANE" .claude/hooks/session_start.py
```
