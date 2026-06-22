#!/usr/bin/env python3
"""PostToolUse hook: append a structured PROGRESS entry whenever
Claude runs Write, Edit, MultiEdit, or a non-trivial Bash command.

Tool-name mapping from the Factory original:
  Create     -> Write      (file creation)
  ApplyPatch -> Edit       (file edit)
  Edit       -> Edit / MultiEdit
  Execute    -> Bash       (shell command; payload key `command`)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _lib import append_progress, emit, read_input  # noqa: E402


# Bash commands that are pure observation -- do not log to PROGRESS.
NOISY_EXECUTE_PATTERNS = (
    "git status",
    "git log",
    "git diff",
    "ls ",
    " ls",
    "pwd",
    "cat ",
    "head ",
    "tail ",
    "echo ",
    "which ",
    "rg ",
    "grep ",
    "find ",
    "wc ",
    "stat ",
    "du ",
    "df ",
)

INTERESTING_TOOLS = {"Write", "Edit", "MultiEdit", "Bash"}


def main() -> None:
    data = read_input()
    tool = data.get("tool_name", "")
    if tool not in INTERESTING_TOOLS:
        return

    inp = data.get("tool_input", {}) or {}
    resp = data.get("tool_response", {}) or {}

    if tool == "Bash":
        cmd = (inp.get("command") or "").strip()
        if not cmd:
            return
        first = cmd.splitlines()[0].strip()
        lower = first.lower()
        if any(p in lower for p in NOISY_EXECUTE_PATTERNS):
            return
        # Claude's Bash tool_response has no standard `success` key; treat
        # a present non-zero/interrupted signal as failure, else done.
        if isinstance(resp, dict):
            failed = resp.get("interrupted") is True or resp.get("success") is False
        else:
            failed = False
        status = "fail" if failed else "done"
        append_progress(
            status=status,
            kind="exec",
            summary=first[:280],
        )
        emit({"suppressOutput": True})
        return

    file_path = inp.get("file_path") or ""
    if not file_path:
        return

    if tool == "Write":
        action = "created"
    elif tool in {"Edit", "MultiEdit"}:
        action = "edited"
    else:
        return

    append_progress(
        status="done",
        kind="edit",
        summary=f"{action} `{file_path}`",
    )
    emit({"suppressOutput": True})


if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass
