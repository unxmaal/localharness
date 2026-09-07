"""Which two-stage image workflows can run here, and which are broken."""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

# mflux is a `uv tool`, so its version is not in this project's environment.
MFLUX_TOOL_ROOT = Path.home() / ".local/share/uv/tools/mflux"

_DIST_INFO = re.compile(r"^([A-Za-z0-9_.-]+?)-(\d[^-]*)\.dist-info$")


@lru_cache(maxsize=1)
def tool_versions(root: Path | None = None) -> dict[str, str]:
    """Installed mflux/mlx versions, read from the uv tool venv. {} if absent."""
    root = Path(root) if root is not None else MFLUX_TOOL_ROOT
    found: dict[str, str] = {}
    for site in sorted(root.glob("lib/python*/site-packages")):
        for child in site.iterdir():
            m = _DIST_INFO.match(child.name)
            if m:
                found.setdefault(m.group(1).replace("_", "-").lower(),
                                 m.group(2))
    return found


# Pinned to the versions each was MEASURED broken against, so an upstream fix
# re-enables the stage instead of the guard hiding it forever.
BROKEN_STAGES: dict[str, tuple[dict[str, str], str, int]] = {
    "upscale-seedvr2": (
        {"mflux": "0.19.1", "mlx": "0.32.2"},
        "mflux-upscale-seedvr2 passes an mx.array where mx.repeat wants an int "
        "for `repeats`, so it crashes on every image. Measured 0/3",
        27),
}

STAGE_ENTRY_POINTS: dict[str, str] = {
    "upscale-seedvr2": "mflux-upscale-seedvr2",
    "upscale-controlnet": "mflux-upscale-controlnet",
    "controlnet": "mflux-generate-controlnet",
}

ENTRY_POINT_STAGES: dict[str, str] = {v: k for k, v in STAGE_ENTRY_POINTS.items()}


def stage_unavailable(stage: str, versions: dict[str, str] | None = None) -> str:
    """Why `stage` must not run, or "" when it is free to try."""
    entry = BROKEN_STAGES.get(stage)
    if entry is None:
        return ""
    broken_at, why, issue = entry
    have = tool_versions() if versions is None else versions
    # An unknown version is not a broken one.
    for pkg, bad in broken_at.items():
        if have.get(pkg) != bad:
            return ""
    pinned = ", ".join(f"{p} {v}" for p, v in sorted(broken_at.items()))
    return (f"{stage} is broken against the installed stack ({pinned}): {why}. "
            f"See issue #{issue}; the refusal lifts on a version bump.")
