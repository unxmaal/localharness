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


# ---- HTML, via headless Chrome ---------------------------------------------

html_only = pytest.mark.skipif(render.chrome_path() is None,
                               reason="needs Google Chrome or Chromium")

PAGE = ("<!doctype html><html><head><title>t</title><style>"
        "body{background:#fff;color:#111;font:48px system-ui;margin:40px}"
        "</style></head><body><h1>Coffee Roaster</h1>"
        "<p>Beans since 1994.</p></body></html>")
BLANK_PAGE = ("<!doctype html><html><head><title>t</title><style>"
              "body{background:#fff;color:#fff}</style></head>"
              "<body><h1>Invisible</h1></body></html>")
EMPTY_PAGE = "<!doctype html><html><head><title>t</title></head><body></body></html>"


@html_only
def test_rasterizes_a_page_to_a_png(tmp_path):
    out = render.rasterize_html(PAGE, tmp_path / "p.png", width=800)
    from PIL import Image
    with Image.open(out) as im:
        assert im.size[0] == 800


@html_only
def test_a_page_with_visible_text_has_ink():
    assert render.ink_html(PAGE) > 0.001


@html_only
def test_white_on_white_renders_blank():
    """The same failure SVG has: valid markup, real content, nothing visible."""
    assert render.ink_html(BLANK_PAGE) == 0.0


@html_only
def test_an_empty_body_renders_blank():
    assert render.ink_html(EMPTY_PAGE) == 0.0


@html_only
def test_check_html_passes_a_real_page():
    r = render.check_html(PAGE)
    assert r.ok, r.reason
    assert r.metrics["ink"] > 0


@html_only
def test_check_html_fails_a_page_that_renders_nothing():
    r = render.check_html(BLANK_PAGE)
    assert not r.ok
    assert "blank" in r.reason.lower()


def test_check_html_without_chrome_warns_and_passes(monkeypatch):
    """A missing browser must not fail every candidate at once."""
    monkeypatch.setattr(render, "chrome_path", lambda: None)
    r = render.check_html(PAGE)
    assert r.ok
    assert r.warnings and "chrome" in r.warnings[0].lower()
    assert r.metrics == {}


@html_only
def test_the_page_is_loaded_from_a_file_never_from_a_url(tmp_path):
    """A generated page that references an external URL must not cause the
    checker to fetch it. Rendering happens offline, from disk."""
    argv = render.chrome_argv(tmp_path / "in.html", tmp_path / "out.png", 800)
    assert any(a.startswith("file://") for a in argv)
    assert "--disable-gpu" in argv or "--headless" in " ".join(argv)
