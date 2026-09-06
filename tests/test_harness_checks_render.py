"""Rasterizing an SVG, so the pixel checks apply to it too.

The structural checks read the markup: does it parse, does it declare an
xmlns, how many shapes did it draw. All of that can be true of a document that
renders as an empty rectangle -- white shapes on a white ground, shapes placed
outside the viewBox, a zero-size viewBox, everything hidden behind an opaque
rect. The only way to catch that is to draw it and look.

rsvg-convert is librsvg's CLI and is already installed here.
"""
import shutil

import pytest

from harness.checks import render

pytestmark = pytest.mark.skipif(shutil.which("rsvg-convert") is None,
                                reason="needs rsvg-convert (brew install librsvg)")

CIRCLE = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
          '<circle cx="12" cy="12" r="10" fill="black"/></svg>')
EMPTY = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"></svg>'
WHITE_ON_WHITE = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
                  '<circle cx="12" cy="12" r="10" fill="white"/></svg>')
OFF_CANVAS = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
              '<circle cx="9000" cy="9000" r="10" fill="black"/></svg>')
BROKEN = '<svg><circle</svg>'


def test_rasterizes_to_a_real_png(tmp_path):
    out = render.rasterize_svg(CIRCLE, tmp_path / "a.png", width=128)
    from PIL import Image
    with Image.open(out) as im:
        assert im.size == (128, 128)


def test_ink_coverage_of_a_drawn_shape_is_substantial():
    """A circle of radius 10 in a 24-unit box is most of the canvas."""
    assert render.ink(CIRCLE) > 0.4


def test_an_svg_with_no_shapes_has_no_ink():
    assert render.ink(EMPTY) == 0.0


def test_white_on_white_parses_and_counts_shapes_but_has_no_ink():
    """The failure the structural checks cannot see: one shape, valid markup,
    renders as an empty rectangle."""
    assert render.ink(WHITE_ON_WHITE) == 0.0


def test_a_shape_outside_the_viewbox_has_no_ink():
    assert render.ink(OFF_CANVAS) == 0.0


def test_unrenderable_markup_raises_rather_than_reading_as_blank(tmp_path):
    with pytest.raises(render.RenderError):
        render.rasterize_svg(BROKEN, tmp_path / "a.png")


def test_a_missing_rasterizer_is_reported_as_such(tmp_path, monkeypatch):
    """Not as an SVG that draws nothing. That would fail every candidate at
    once and look like a model regression."""
    monkeypatch.setattr(render.shutil, "which", lambda _: None)
    with pytest.raises(render.RenderError, match="rsvg-convert"):
        render.rasterize_svg(CIRCLE, tmp_path / "a.png")


# ---- the check ------------------------------------------------------------

def test_check_passes_a_drawn_svg():
    r = render.check(CIRCLE)
    assert r.ok
    assert r.metrics["ink"] > 0.4


def test_check_fails_an_svg_that_renders_blank():
    r = render.check(WHITE_ON_WHITE)
    assert not r.ok
    assert "renders blank" in r.reason.lower()
    assert r.metrics["ink"] == 0.0


def test_check_treats_an_unavailable_rasterizer_as_a_skip_not_a_failure():
    """A missing optional tool must not silently fail every candidate."""
    r = render.check(CIRCLE, rasterizer=None)
    assert r.ok
    assert r.warnings
    assert r.metrics == {}
