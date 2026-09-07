"""Turning a raster into vector paths.

The svg lane's five language models all draw the same thing: valid markup that
is not the picture. Asked for a frog holding a coffee mug they produced
coloured blobs; asked for two concentric gears, two offset squares. A diffusion
model draws the frog in 54s and vtracer turns it into real paths in 0.05s.

This module is that second step. It is deliberately thin: the value is the
PIPELINE, and vtracer is someone else's Rust that already works.
"""
import shutil
from pathlib import Path

import pytest

from harness import vector

# The ink check shells out to rsvg-convert. Without the guard these fail on a
# machine that simply does not have it, which reads as a code defect.
needs_rsvg = pytest.mark.skipif(shutil.which("rsvg-convert") is None,
                                reason="needs rsvg-convert (brew install librsvg)")


def png(path, size=64):
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (size, size), "white")
    d = ImageDraw.Draw(im)
    d.ellipse((8, 8, size - 8, size - 8), fill="green", outline="black", width=3)
    im.save(path)
    return path


def test_a_raster_becomes_an_svg_document(tmp_path):
    out = vector.trace(png(tmp_path / "a.png"))
    assert out.lstrip().startswith(("<?xml", "<svg"))
    assert "</svg>" in out
    assert "<path" in out


@needs_rsvg
def test_the_svg_actually_draws_something(tmp_path):
    """A vectorizer that returns a well-formed empty document is the same
    failure the language models had."""
    from harness.checks import render
    assert render.ink(vector.trace(png(tmp_path / "a.png"))) > 0.05


@needs_rsvg
def test_a_blank_image_is_reported_rather_than_returned_empty(tmp_path):
    """A uniform image traces to nothing. Returning that as a success would
    hand the caller an empty document with a clean exit."""
    from PIL import Image
    p = tmp_path / "blank.png"
    Image.new("RGB", (64, 64), "white").save(p)
    with pytest.raises(vector.VectorError) as e:
        vector.trace(p)
    assert "blank" in str(e.value).lower() or "nothing" in str(e.value).lower()


def test_a_missing_file_fails_before_the_tracer(tmp_path):
    with pytest.raises(vector.VectorError) as e:
        vector.trace(tmp_path / "nope.png")
    assert "nope.png" in str(e.value)


def test_tracing_is_fast_enough_to_be_free(tmp_path):
    """54s of diffusion plus 0.05s of tracing is a pipeline; plus 30s would be
    a different decision."""
    import time
    t0 = time.time()
    vector.trace(png(tmp_path / "a.png", size=256))
    assert time.time() - t0 < 5.0


def test_the_traced_svg_carries_a_viewbox(tmp_path):
    """vtracer emits width and height and no viewBox, so the file does not
    scale: dropped into a page at any other size it crops instead of fitting.
    The svg checker warns about it on every single traced file."""
    out = vector.trace(png(tmp_path / "a.png", size=128))
    assert 'viewBox="0 0 128 128"' in out


def test_an_existing_viewbox_is_left_alone(tmp_path):
    """Only add what is missing; never rewrite a tracer that got it right."""
    doc = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 9 9"><path/></svg>'
    assert vector._ensure_viewbox(doc) == doc
