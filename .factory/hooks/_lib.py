"""Shared helpers for memory-system hooks. Imported by sibling hook scripts."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(os.environ.get("FACTORY_PROJECT_DIR") or os.getcwd()).resolve()
FACTORY_DIR = PROJECT_DIR / ".factory"

PROGRESS_FILE = FACTORY_DIR / "PROGRESS.md"
PLAN_FILE = FACTORY_DIR / "PLAN.md"
VERIFY_FILE = FACTORY_DIR / "VERIFY.md"
MEMORIES_FILE = FACTORY_DIR / "memories.md"

SECRET_PATTERNS = [
    re.compile(r"hf_[a-zA-Z0-9]{20,}"),
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),
    re.compile(r"gh[ps]_[a-zA-Z0-9]{30,}"),
    re.compile(r"AKIA[A-Z0-9]{16}"),
    re.compile(r"BEGIN.*PRIVATE KEY"),
]


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_rev() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(PROJECT_DIR),
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
        )
        return out.strip() or "no-git"
    except Exception:
        return "no-git"


def git_branch() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(PROJECT_DIR),
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
        )
        return out.strip() or "detached"
    except Exception:
        return "no-git"


def scrub(text: str) -> str:
    for pat in SECRET_PATTERNS:
        text = pat.sub("[REDACTED]", text)
    return text


def read_input() -> dict:
    try:
        return json.load(sys.stdin)
    except Exception:
        return {}


def emit(payload: dict | None = None) -> None:
    if payload is not None:
        print(json.dumps(payload))


def append_progress(status: str, kind: str, summary: str, detail: str = "") -> None:
    if not PROGRESS_FILE.exists():
        return
    summary = scrub(summary).replace("\n", " ").strip()[:300]
    detail = scrub(detail).strip()
    if not summary:
        return
    header_marker = "\n---\n"
    new_block = (
        f"## {now_iso()} | {git_branch()}@{git_rev()} | {status} | {kind}\n"
        f"{summary}\n"
    )
    if detail:
        new_block += f"\n{detail}\n"
    new_block += "\n"

    content = PROGRESS_FILE.read_text(encoding="utf-8")
    insertion = content.find(header_marker)
    if insertion < 0:
        PROGRESS_FILE.write_text(content + "\n" + new_block, encoding="utf-8")
        return
    insertion_end = insertion + len(header_marker)
    PROGRESS_FILE.write_text(
        content[:insertion_end] + "\n" + new_block + content[insertion_end:],
        encoding="utf-8",
    )


def read_file_safe(path: Path, max_lines: int | None = None) -> str:
    if not path.exists():
        return f"<missing: {path}>"
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        return f"<unreadable: {path} ({exc})>"
    if max_lines is not None:
        lines = text.splitlines()
        if len(lines) > max_lines:
            text = "\n".join(lines[:max_lines]) + f"\n... <{len(lines)-max_lines} more lines>"
    return text


def find_relevant_subproject_agents(cwd: str) -> Path | None:
    cwd_path = Path(cwd or PROJECT_DIR).resolve()
    try:
        rel = cwd_path.relative_to(PROJECT_DIR)
    except ValueError:
        return None
    parts = list(rel.parts)
    while parts:
        candidate = PROJECT_DIR / Path(*parts) / "AGENTS.md"
        if candidate.exists() and candidate != PROJECT_DIR / "AGENTS.md":
            return candidate
        parts.pop()
    return None
