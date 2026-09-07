"""Which two-stage image workflows can actually run here, and which are broken.

A stage that is known broken must refuse BEFORE stage one runs. `upscale-seedvr2`
crashes inside mflux, but only after the base engine has spent a full generation
producing the image it is handed -- so the cost of finding out is a whole
diffusion run, every case, every time.

THE VERSION PIN IS THE WHOLE DESIGN. A flat "this stage is broken" flag would
still be refusing a year after mflux ships the fix, and nobody would ever find
out, because the guard removes the only thing that would have told them. So the
registry records the versions it was MEASURED broken against and refuses only
those. A newer stack is allowed through to fail or succeed on its own evidence.

Lives in `harness/` rather than beside the runner because `harness.discover`
reports availability and must not import from `evals/`.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

#: mflux is installed as a `uv tool`, so its version is not in this project's
#: environment. The dist-info directory in the tool's own venv is the only
#: place that knows.
MFLUX_TOOL_ROOT = Path.home() / ".local/share/uv/tools/mflux"

_DIST_INFO = re.compile(r"^([A-Za-z0-9_.-]+?)-(\d[^-]*)\.dist-info$")


@lru_cache(maxsize=1)
def tool_versions(root: Path | None = None) -> dict[str, str]:
    """What mflux and mlx are installed at, read from the uv tool venv.

    Returns an empty dict when mflux is not installed at all. An unknown
    version is NOT treated as broken: refusing to run because a lookup failed
    would turn a missing file into a phantom bug report.
    """
    root = Path(root) if root is not None else MFLUX_TOOL_ROOT
    found: dict[str, str] = {}
    for site in sorted(root.glob("lib/python*/site-packages")):
        for child in site.iterdir():
            m = _DIST_INFO.match(child.name)
            if m:
                # mlx_metal-0.32.2 and mlx-0.32.2 both normalise to a name we
                # can look up; last one wins and they agree in practice.
                found.setdefault(m.group(1).replace("_", "-").lower(),
                                 m.group(2))
    return found


#: stage -> (versions measured broken, why, issue). Every field is load-bearing:
#: the versions so a fix re-enables the stage, the reason so the refusal is
#: readable without opening GitHub, the issue so it is not re-discovered.
BROKEN_STAGES: dict[str, tuple[dict[str, str], str, int]] = {
    "upscale-seedvr2": (
        {"mflux": "0.19.1", "mlx": "0.32.2"},
        "mflux-upscale-seedvr2 passes an mx.array where mx.repeat wants an int "
        "for `repeats`, so it crashes on every image. Measured 0/3 on "
        "2026-09-07 against flux2-klein-4b at 3/3 on the same cases. The break "
        "is inside mflux, not here",
        27),
}


#: stage name -> the mflux entry point it drives. `lh discover` needs this to
#: tell a primitive that HAS a runner from one that does not: without it,
#: discover reported all nineteen as "no runner yet" months after three of them
#: got one, which is the exact staleness the tool exists to prevent.
#: Kept in step with `evals.runners.chain.STAGES` by a test.
STAGE_ENTRY_POINTS: dict[str, str] = {
    "upscale-seedvr2": "mflux-upscale-seedvr2",
    "upscale-controlnet": "mflux-upscale-controlnet",
    "controlnet": "mflux-generate-controlnet",
}

#: The reverse lookup, for walking a directory of entry points.
ENTRY_POINT_STAGES: dict[str, str] = {v: k for k, v in STAGE_ENTRY_POINTS.items()}


def stage_unavailable(stage: str, versions: dict[str, str] | None = None) -> str:
    """Why `stage` must not run, or "" when it is free to try.

    Truthy return is the refusal message, which is why it returns a string
    rather than a bool: a guard that says only "no" sends the reader looking
    for a bug in the wrong repository.
    """
    entry = BROKEN_STAGES.get(stage)
    if entry is None:
        return ""
    broken_at, why, issue = entry
    have = tool_versions() if versions is None else versions
    # An UNKNOWN version is not a broken one. If mflux cannot be found at all
    # the stage will fail with "not installed", which is the honest error.
    for pkg, bad in broken_at.items():
        if have.get(pkg) != bad:
            return ""
    pinned = ", ".join(f"{p} {v}" for p, v in sorted(broken_at.items()))
    return (f"{stage} is broken against the installed stack ({pinned}): {why}. "
            f"See issue #{issue}. This refusal is pinned to those exact "
            f"versions, so upgrading mflux re-enables the stage.")
