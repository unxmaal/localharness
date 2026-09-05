"""Objective quality checks for generated artifacts.

The point of the eval suite is choosing between candidates, which needs a metric
that is not a human squinting at output. For SVG and HTML the strongest cheap
signal is simply: does it parse, and does it draw anything. A model that emits
prose around its code, or unclosed tags, or an empty canvas, is disqualified
before anyone argues about aesthetics.
"""
import pytest

from evals.checks import html as html_check
from evals.checks import svg as svg_check

GOOD_SVG = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
  <circle cx="12" cy="12" r="10" fill="#e63946"/>
</svg>'''


# ---- SVG -------------------------------------------------------------------

def test_valid_svg_passes():
    r = svg_check.check(GOOD_SVG)
    assert r.ok
    assert r.shape_count == 1


def test_svg_wrapped_in_markdown_fence_is_recovered():
    """Models fence their output constantly. Punishing that measures prompt
    compliance, not SVG ability, so the fence is stripped before judging."""
    r = svg_check.check("```svg\n" + GOOD_SVG + "\n```")
    assert r.ok


def test_svg_with_prose_around_it_is_recovered():
    r = svg_check.check("Here you go!\n\n" + GOOD_SVG + "\n\nHope that helps.")
    assert r.ok


def test_malformed_xml_fails():
    r = svg_check.check('<svg><circle cx="1"</svg>')
    assert not r.ok
    assert "parse" in r.reason.lower()


def test_non_svg_root_fails():
    r = svg_check.check('<html><body>nope</body></html>')
    assert not r.ok


def test_empty_canvas_fails():
    """Parses fine and draws nothing. Worthless, and a real model failure."""
    r = svg_check.check('<svg xmlns="http://www.w3.org/2000/svg"></svg>')
    assert not r.ok
    assert r.shape_count == 0


def test_missing_viewbox_is_flagged_but_not_fatal():
    r = svg_check.check('<svg xmlns="http://www.w3.org/2000/svg">'
                        '<rect width="10" height="10"/></svg>')
    assert r.ok
    assert "viewBox" in " ".join(r.warnings)


def test_embedded_raster_is_flagged():
    """A PNG in a data: URI is a raster pretending to be a vector."""
    r = svg_check.check(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 1">'
        '<image href="data:image/png;base64,iVBORw0KGgo="/></svg>')
    assert any("raster" in w for w in r.warnings)


# ---- HTML ------------------------------------------------------------------

GOOD_HTML = '''<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>T</title></head><body><h1>Hello</h1><p>Text</p></body></html>'''


def test_valid_html_passes():
    r = html_check.check(GOOD_HTML)
    assert r.ok
    assert r.has_title


def test_html_fence_is_recovered():
    r = html_check.check("```html\n" + GOOD_HTML + "\n```")
    assert r.ok


def test_html_without_body_content_fails():
    r = html_check.check('<!doctype html><html><head><title>T</title></head>'
                         '<body></body></html>')
    assert not r.ok


def test_unclosed_tag_is_flagged():
    r = html_check.check('<!doctype html><html><body><div><p>x</body></html>')
    assert not r.ok or r.warnings


def test_missing_doctype_is_flagged_but_not_fatal():
    r = html_check.check('<html><body><h1>x</h1></body></html>')
    assert r.ok
    assert any("doctype" in w.lower() for w in r.warnings)


def test_external_network_dependency_is_flagged():
    """A page that only renders with a CDN up is not self-contained."""
    r = html_check.check(
        '<!doctype html><html><head><title>T</title>'
        '<script src="https://cdn.example.com/x.js"></script></head>'
        '<body><h1>x</h1></body></html>')
    assert any("external" in w.lower() for w in r.warnings)
