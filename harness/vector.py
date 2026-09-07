"""Turning a raster into vector paths.

THE SVG LANE'S ANSWER. Five language models were measured on it and all five
draw the same thing: valid markup that is not the picture. Asked for a cartoon
frog holding a coffee mug they emitted eighty near-identical `<path>` elements
and, once the sampling was fixed, a complete document of coloured blobs. Asked
for two concentric gears, two offset squares. That is not a model-size problem:
an LLM writes bezier coordinates it cannot see, with no spatial model tying one
shape to the next.

A diffusion model draws the frog in 54 seconds and this turns it into real
paths in 0.05. The module is deliberately thin, because the value is the
PIPELINE and vtracer is someone else's Rust that already works.

The tradeoff, stated: tracing produces MANY paths (a 512x512 cartoon runs to
~68KB) where a hand-authored icon would use a dozen. It is the right answer for
illustration and the wrong one for a 24x24 UI glyph.
"""
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


def trace(image: str | Path, *, colormode: str = "color",
          filter_speckle: int = 8, color_precision: int = 6,
          path_precision: int = 3) -> str:
    """Vectorize `image`, returning the SVG document as text.

    Defaults are tuned for flat illustration, which is what the image lane
    produces when asked for one: `filter_speckle` drops the single-pixel noise
    that diffusion leaves in flat areas, and it is the difference between a
    usable file and forty thousand paths.
    """
    image = Path(image)
    if not image.exists():
        raise VectorError(f"no image at {image}")

    try:
        import vtracer
    except ImportError as exc:
        raise VectorError(f"vtracer is not installed: {exc}") from exc

    with tempfile.TemporaryDirectory() as d:
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
        svg = _ensure_viewbox(out.read_text())

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
