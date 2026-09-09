"""Turning a raster into vector paths, via vtracer."""
from __future__ import annotations

import re
import tempfile
from pathlib import Path

#: Below this fraction of marked canvas the trace produced nothing worth
#: keeping. A uniform image traces to a well-formed EMPTY document, which is
#: exactly the failure the language models had -- returning it as a success
#: would hand the caller a clean exit and no picture.
MIN_INK = 0.005


_SVG_TAG = re.compile(r"<svg\b[^>]*>", re.IGNORECASE)
_WIDTH = re.compile(r'\bwidth="(\d+(?:\.\d+)?)"', re.IGNORECASE)
_HEIGHT = re.compile(r'\bheight="(\d+(?:\.\d+)?)"', re.IGNORECASE)


def _ensure_viewbox(svg: str) -> str:
    """Add a viewBox derived from width/height when the tracer omitted one.

    vtracer emits width and height and no viewBox, so the file does not scale:
    dropped into a page at any other size it crops rather than fits. The svg
    checker warns about it on every traced file, which is a real defect rather
    than checker noise.
    """
    if "viewbox=" in svg.lower():
        return svg
    tag = _SVG_TAG.search(svg)
    if not tag:
        return svg
    w, h = _WIDTH.search(tag.group()), _HEIGHT.search(tag.group())
    if not (w and h):
        return svg
    box = f' viewBox="0 0 {w.group(1).rstrip(".0") or 0} {h.group(1).rstrip(".0") or 0}"'
    return svg.replace(tag.group(), tag.group()[:-1] + box + ">", 1)


class VectorError(RuntimeError):
    """The raster could not be turned into usable vector paths."""


# Issue #4. Measured 2026-09-07 on a traced 512px gear: resolution is the
# dominant lever on document size, path_precision a smaller one, and
# filter_speckle does nothing at all (8/32/64 were byte-identical).
#   512px pp3 26,310B ink 0.2763 | 128px pp1 8,202B ink 0.2808
TRACE_PRESETS = {
    "illustration": {"trace_at": None, "path_precision": 3},
    "icon": {"trace_at": 128, "path_precision": 1},
}


def trace(image: str | Path, *, colormode: str = "color",
          filter_speckle: int = 8, color_precision: int = 6,
          path_precision: int = 3, trace_at: int | None = None,
          preset: str = "") -> str:
    """Vectorize `image`, returning the SVG document as text."""
    if preset:
        if preset not in TRACE_PRESETS:
            raise VectorError(
                f"unknown trace preset {preset!r}; "
                f"known: {', '.join(sorted(TRACE_PRESETS))}")
        cfg = TRACE_PRESETS[preset]
        trace_at = cfg["trace_at"]
        path_precision = cfg["path_precision"]
    image = Path(image)
    if not image.exists():
        raise VectorError(f"no image at {image}")

    try:
        import vtracer
    except ImportError as exc:
        raise VectorError(f"vtracer is not installed: {exc}") from exc

    with tempfile.TemporaryDirectory() as d:
        if trace_at:
            from PIL import Image
            small = Path(d) / "small.png"
            with Image.open(image) as im:
                im.convert("RGB").resize((trace_at, trace_at),
                                         Image.LANCZOS).save(small)
            image = small
        out = Path(d) / "traced.svg"
        try:
            vtracer.convert_image_to_svg_py(
                str(image), str(out), colormode=colormode,
                filter_speckle=filter_speckle,
                color_precision=color_precision,
                path_precision=path_precision)
        except Exception as exc:  # noqa: BLE001
            raise VectorError(f"vtracer failed on {image.name}: {exc}") from exc
        if not out.exists():
            raise VectorError(f"vtracer wrote nothing for {image.name}")
        svg = _ensure_viewbox(out.read_text(encoding="utf-8"))

    from harness.checks import render
    try:
        drawn = render.ink(svg)
    except Exception:  # noqa: BLE001
        # A rasterizer problem is not the tracer's fault; let the document
        # through and let the checker downstream have the argument.
        return svg
    if drawn < MIN_INK:
        raise VectorError(
            f"traced {image.name} to a blank document (ink {drawn:.4f}); the "
            f"source image is probably uniform")
    return svg
