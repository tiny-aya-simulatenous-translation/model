#!/usr/bin/env python3
"""Stop hook: when Droid finishes responding, run the `verify` skill in
non-blocking mode. Pass results land in PROGRESS.md; a fail does NOT
block the stoppage by default (it just gets logged) -- override by
setting MEMORY_STOP_BLOCK_ON_FAIL=1 in the environment.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _lib import (  # noqa: E402
    PROJECT_DIR,
    VERIFY_FILE,
    PLAN_FILE,
    append_progress,
    emit,
    read_input,
)


CODE_FENCE = re.compile(r"```bash\s*\n(.*?)```", re.DOTALL)


def extract_commands() -> list[str]:
    if not VERIFY_FILE.exists():
        return []
    text = VERIFY_FILE.read_text(encoding="utf-8")
    return [m.group(1).strip() for m in CODE_FENCE.finditer(text) if m.group(1).strip()]


def run_command(cmd: str, timeout: int = 30) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            ["bash", "-eo", "pipefail", "-c", cmd],
            cwd=str(PROJECT_DIR),
            capture_output=True,
            timeout=timeout,
            text=True,
        )
        out = (proc.stdout + proc.stderr).strip()
        return proc.returncode, out[-400:]
    except subprocess.TimeoutExpired:
        return 124, f"timeout after {timeout}s"
    except Exception as exc:
        return 99, str(exc)[:400]


def tick_plan_for(cmd: str) -> None:
    if not PLAN_FILE.exists():
        return
    text = PLAN_FILE.read_text(encoding="utf-8")
    short = cmd.splitlines()[0][:60]
    if short and short.lower() in text.lower():
        text = text.replace("- [ ] " + short, "- [x] " + short, 1)
        PLAN_FILE.write_text(text, encoding="utf-8")


def main() -> None:
    data = read_input()
    if data.get("stop_hook_active"):
        return

    if os.environ.get("MEMORY_STOP_DISABLE_VERIFY") == "1":
        return

    commands = extract_commands()
    if not commands:
        return

    if len(commands) > 12:
        commands = commands[:12]

    failures: list[tuple[str, int, str]] = []
    passes: list[str] = []
    for cmd in commands:
        rc, out = run_command(cmd, timeout=20)
        first = cmd.splitlines()[0][:80]
        if rc == 0:
            passes.append(first)
            tick_plan_for(cmd)
        else:
            failures.append((first, rc, out))

    summary = (
        f"verify: {len(passes)} passed, {len(failures)} failed "
        f"out of {len(commands)} on Stop"
    )
    detail_lines = []
    for first, rc, out in failures:
        detail_lines.append(f"FAIL [{rc}] {first}")
        if out:
            detail_lines.append(f"    {out.splitlines()[-1]}")
    detail = "\n".join(detail_lines)

    append_progress(
        status="done" if not failures else "fail",
        kind="verify",
        summary=summary,
        detail=detail,
    )

    if failures and os.environ.get("MEMORY_STOP_BLOCK_ON_FAIL") == "1":
        emit({
            "decision": "block",
            "reason": (
                f"verify reported {len(failures)} failure(s). "
                "Fix them before stopping, or unset "
                "MEMORY_STOP_BLOCK_ON_FAIL to disable this gate."
            ),
        })
        return

    emit({"suppressOutput": True})


if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass
