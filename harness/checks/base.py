"""Shared result type and text-recovery for artifact checks."""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class CheckResult:
    ok: bool
    reason: str = ""
    warnings: list[str] = field(default_factory=list)
    shape_count: int = 0
    has_title: bool = False


_FENCE = re.compile(r"```[a-zA-Z]*\s*\n(.*?)```", re.S)


def extract(text: str, root_tags: tuple[str, ...]) -> str:
    """Pull the artifact out of whatever the model wrapped it in.

    Models fence their output and chat around it constantly. Failing them for
    that measures prompt compliance, not the ability under test, so recover the
    payload first and judge the payload.
    """
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    for tag in root_tags:
        start = text.lower().find(f"<{tag}")
        if start == -1:
            continue
        end = text.lower().rfind(f"</{tag}>")
        if end != -1:
            return text[start:end + len(tag) + 3]
        return text[start:]
    return text.strip()
