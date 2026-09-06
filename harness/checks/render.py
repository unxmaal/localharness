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

HTML goes through headless Chrome, which is already on this machine, so it
needs no Playwright and no npm. Same failure being caught: a page can parse,
have a body, be perfectly self-contained, and render as a white rectangle.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from harness.checks.base import CheckResult

RASTERIZER = "rsvg-convert"
# Chromium first, since a `chromium` on PATH is the deliberate install; the .app
# is the one that happens to be there on any Mac.
CHROME_CANDIDATES = (
    "chromium",
    "chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
)
DEFAULT_WIDTH = 256
# A quarter of one percent of the canvas. Below that there is nothing a person
# would call a drawing; a real icon lands far above it.
MIN_INK = 0.0025


class RenderError(RuntimeError):
    """The document could not be drawn."""


def chrome_path() -> str | None:
    """The browser to render HTML with, or None if there is not one."""
    for candidate in CHROME_CANDIDATES:
        if candidate.startswith("/"):
            if Path(candidate).exists():
                return candidate
        elif shutil.which(candidate):
            return shutil.which(candidate)
    return None


def chrome_argv(src: Path, out: Path, width: int) -> list[str]:
    """Headless Chrome, rendering a LOCAL FILE.

    file:// and not a data: URL or a served page: a generated page may
    reference an external URL, and the checker must never be the thing that
    fetches it. An opaque white background is forced so a transparent body does
    not read as ink, and a virtual time budget lets layout and webfonts settle
    before the shot rather than capturing a half-painted frame.
    """
    return [
        chrome_path() or "chrome",
        "--headless=new",
        "--disable-gpu",
        "--hide-scrollbars",
        "--force-device-scale-factor=1",
        "--default-background-color=FFFFFFFF",
        "--virtual-time-budget=2000",
        f"--window-size={width},{int(width * 0.75)}",
        f"--screenshot={out}",
        f"file://{src}",
    ]


def rasterize_html(html: str, out: str | Path, width: int = 800) -> Path:
    """Render an HTML document to a PNG with headless Chrome."""
    if chrome_path() is None:
        raise RenderError("no Chrome or Chromium found to render HTML")

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "page.html"
        src.write_text(html)
        proc = subprocess.run(chrome_argv(src, out, width),
                              capture_output=True, text=True, timeout=60,
                              cwd=d)
    if not out.exists():
        raise RenderError(f"chrome produced no screenshot: "
                          f"{proc.stderr.strip()[:300]}")
    return out


def ink_html(html: str, width: int = 800) -> float:
    """Fraction of the rendered page that is not the background."""
    from PIL import Image

    with tempfile.TemporaryDirectory() as d:
        png = rasterize_html(html, Path(d) / "p.png", width=width)
        with Image.open(png) as im:
            histogram = im.convert("L").histogram()
    total = sum(histogram)
    return round(sum(histogram[:250]) / total, 4) if total else 0.0


def check_html(html: str, min_ink: float = 0.0005) -> CheckResult:
    """Does this page actually show anything?

    The threshold is lower than the SVG one: a page is mostly whitespace by
    design, and a heading on a white ground marks well under a percent of it.
    """
    if chrome_path() is None:
        return CheckResult(
            True, "",
            ["not rasterized: no Chrome or Chromium found to render HTML"])
    try:
        coverage = ink_html(html)
    except (RenderError, subprocess.TimeoutExpired) as exc:
        return CheckResult(False, f"could not render: {exc}")

    result = CheckResult(coverage >= min_ink, "")
    result.metrics = {"ink": coverage}
    if not result.ok:
        result.reason = (
            f"renders blank: {coverage:.3%} of the page is marked. The markup "
            f"parses and has content, but nothing is visible.")
    return result


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
