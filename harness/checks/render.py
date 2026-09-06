"""Rasterizing an SVG so the pixel checks apply to it too.

The structural checks read the markup: does it parse, is there an xmlns, how
many shapes were drawn. Every one of those can be true of a document that
renders as an empty rectangle -- white shapes on a white ground, a shape placed
outside the viewBox, a zero-size viewBox, everything hidden behind an opaque
rect. A model that emits well-formed SVG which draws nothing visible passes
every check the suite had. The only way to catch it is to draw it and look.

rsvg-convert is librsvg's CLI. It is an OPTIONAL dependency: when it is absent
the check warns and passes, because a missing tool that failed every candidate
at once would look exactly like a model regression.

HTML is not rasterized. It needs a browser engine, which is a much larger
dependency than one already-installed binary, and the self-containment checks
that matter most for a page are structural anyway.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from harness.checks.base import CheckResult

RASTERIZER = "rsvg-convert"
DEFAULT_WIDTH = 256
# A quarter of one percent of the canvas. Below that there is nothing a person
# would call a drawing; a real icon lands far above it.
MIN_INK = 0.0025


class RenderError(RuntimeError):
    """The document could not be drawn."""


def rasterize_svg(svg: str, out: str | Path, width: int = DEFAULT_WIDTH) -> Path:
    """Render `svg` (markup, not a path) to a PNG at `out`."""
    if shutil.which(RASTERIZER) is None:
        raise RenderError(
            f"{RASTERIZER} is not installed (brew install librsvg)")

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Through a file rather than stdin: rsvg-convert reports the offending line
    # number for a file, and that is the whole diagnostic for broken markup.
    with tempfile.NamedTemporaryFile("w", suffix=".svg", delete=False) as fh:
        fh.write(svg)
        src = fh.name
    try:
        proc = subprocess.run(
            [RASTERIZER, "--width", str(width), "--height", str(width),
             "--keep-aspect-ratio", "--background-color", "white",
             "-o", str(out), src],
            capture_output=True, text=True, timeout=30)
    finally:
        Path(src).unlink(missing_ok=True)

    if proc.returncode != 0 or not out.exists():
        raise RenderError(f"{RASTERIZER} failed: {proc.stderr.strip()[:300]}")
    return out


def ink(svg: str, width: int = DEFAULT_WIDTH) -> float:
    """Fraction of the canvas that is not the background.

    Rendered on white and compared against white, so "drew something" means
    "something a person would see" rather than "the DOM has nodes in it".

    The threshold is luminance below 250, not "differs from white": antialiasing
    and near-white fills would otherwise register as ink and a blank canvas
    would score above zero. Read from a histogram rather than per pixel, which
    is the same technique the image checker uses for standard deviation.
    """
    from PIL import Image

    with tempfile.TemporaryDirectory() as d:
        png = rasterize_svg(svg, Path(d) / "r.png", width=width)
        with Image.open(png) as im:
            histogram = im.convert("L").histogram()
    total = sum(histogram)
    return round(sum(histogram[:250]) / total, 4) if total else 0.0


def check(svg: str, min_ink: float = MIN_INK,
          rasterizer: str | None = RASTERIZER) -> CheckResult:
    """Does this SVG actually draw anything?"""
    if rasterizer is None or shutil.which(rasterizer) is None:
        return CheckResult(
            True, "",
            [f"not rasterized: {RASTERIZER} is not installed "
             f"(brew install librsvg)"])

    try:
        coverage = ink(svg)
    except RenderError as exc:
        return CheckResult(False, f"could not render: {exc}")

    result = CheckResult(coverage >= min_ink, "")
    result.metrics = {"ink": coverage}
    if not result.ok:
        result.reason = (
            f"renders blank: {coverage:.2%} of the canvas is marked. The markup "
            f"parses and has shapes in it, but nothing is visible.")
    return result
