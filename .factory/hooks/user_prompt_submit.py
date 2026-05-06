#!/usr/bin/env python3
"""UserPromptSubmit hook: route '#progress', '#plan', '#decision',
'#verify' quick-capture prefixes to the right memory file.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _lib import (  # noqa: E402
    PROGRESS_FILE,
    PLAN_FILE,
    VERIFY_FILE,
    MEMORIES_FILE,
    emit,
    git_branch,
    git_rev,
    now_iso,
    read_input,
    scrub,
)


TRIGGERS = {
    "#progress": ("progress", PROGRESS_FILE),
    "#plan": ("plan", PLAN_FILE),
    "#decision": ("decision", MEMORIES_FILE),
    "#verify": ("verify", VERIFY_FILE),
}


def main() -> None:
    data = read_input()
    prompt = (data.get("prompt") or "").strip()
    if not prompt.startswith("#"):
        return

    first_token = prompt.split(None, 1)[0].lower()
    if first_token not in TRIGGERS:
        return

    kind, mem_file = TRIGGERS[first_token]
    body = prompt[len(first_token):].strip()
    if not body:
        return

    body = scrub(body)
    line = ""
    if kind == "progress":
        line = (
            f"\n## {now_iso()} | {git_branch()}@{git_rev()} | info | quick-capture\n"
            f"{body}\n"
        )
        _prepend_after_marker(mem_file, "\n---\n", line)
    elif kind == "plan":
        line = f"- [ ] {body}  <!-- captured {now_iso()} -->\n"
        _append_under_section(mem_file, "## Tasks", line)
    elif kind == "decision":
        ymd = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        block = (
            f"\n### {ymd}: {body[:80]}\n"
            f"**Decision:** {body}\n"
            f"**Captured-from:** quick-capture (`#decision`)\n"
        )
        _append_under_section(mem_file, "## Architecture decisions", block)
    elif kind == "verify":
        block = f"\n```bash\n{body}\n```\n"
        _append(mem_file, block)

    emit({
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": (
                f"Memory: captured `{kind}` entry to "
                f"{mem_file.relative_to(mem_file.parents[1])}."
            ),
        },
        "systemMessage": f"memory: saved `{kind}` to {mem_file.name}",
        "suppressOutput": True,
    })


def _prepend_after_marker(path: Path, marker: str, text: str) -> None:
    if not path.exists():
        return
    content = path.read_text(encoding="utf-8")
    idx = content.find(marker)
    if idx < 0:
        path.write_text(content + "\n" + text, encoding="utf-8")
        return
    end = idx + len(marker)
    path.write_text(content[:end] + text + content[end:], encoding="utf-8")


def _append_under_section(path: Path, section_header: str, text: str) -> None:
    if not path.exists():
        return
    content = path.read_text(encoding="utf-8")
    idx = content.find(section_header)
    if idx < 0:
        path.write_text(content + f"\n{section_header}\n{text}", encoding="utf-8")
        return
    next_section = content.find("\n## ", idx + len(section_header))
    if next_section < 0:
        path.write_text(content + text, encoding="utf-8")
        return
    path.write_text(content[:next_section] + text + content[next_section:], encoding="utf-8")


def _append(path: Path, text: str) -> None:
    if not path.exists():
        return
    with path.open("a", encoding="utf-8") as fh:
        fh.write(text)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass
