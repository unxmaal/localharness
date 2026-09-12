"""Rasterizing an SVG, so the pixel checks apply to it too.

The structural checks read the markup: does it parse, does it declare an
xmlns, how many shapes did it draw. All of that can be true of a document that
renders as an empty rectangle -- white shapes on a white ground, shapes placed
outside the viewBox, a zero-size viewBox, everything hidden behind an opaque
rect. The only way to catch that is to draw it and look.

rsvg-convert is librsvg's CLI and is already installed here.
"""
import sys
import shutil
from pathlib import Path

import pytest

from harness.checks import render

pytestmark = pytest.mark.skipif(render.rasterizer_path() is None,
                                reason="needs rsvg-convert (brew install librsvg / pacman -S mingw-w64-x86_64-librsvg)")

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
    # The CANDIDATE LIST is what models a machine without it. Stubbing
    # shutil.which only models one that is not on PATH, and the resolver also
    # looks in known locations -- so the rasterizer was still found and this
    # test passed for the wrong reason on any machine that has one installed
    # off PATH.
    monkeypatch.setattr(render, "RASTERIZER_CANDIDATES",
                        ("definitely-not-a-rasterizer-xyz",))
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


# ---- Chrome must not touch the user's browser ------------------------------

def test_chrome_uses_an_isolated_profile(tmp_path):
    """Without --user-data-dir, headless Chrome runs against the DEFAULT
    profile: it contends for the lock with a live browser, writes into the
    real profile, bounces the dock, and renders the page through whatever
    extensions happen to be installed -- so the result is not reproducible
    either. It was spotted as a bouncing dock icon."""
    argv = render.chrome_argv(tmp_path / "p.html", tmp_path / "o.png", 800,
                              profile=tmp_path / "profile")
    joined = " ".join(argv)
    assert f"--user-data-dir={tmp_path / 'profile'}" in joined
    assert "--no-first-run" in argv
    assert "--no-default-browser-check" in argv
    assert "--disable-extensions" in argv


def test_a_render_creates_and_cleans_up_its_own_profile(tmp_path, monkeypatch):
    """The caller should not have to know chrome needs a scratch directory."""
    seen = {}

    def fake_shoot(argv, out, cwd, timeout=60.0):
        seen["argv"] = argv
        assert any(a.startswith("--user-data-dir=") for a in argv)
        Path(out).write_bytes(b"\x89PNG\r\n\x1a\n")
        return ""

    monkeypatch.setattr(render, "_shoot", fake_shoot)
    monkeypatch.setattr(render, "chrome_path", lambda: "/fake/chrome")
    render.rasterize_html("<html><body>hi</body></html>", tmp_path / "shot.png")
    prof = next(a for a in seen["argv"] if a.startswith("--user-data-dir="))
    assert not Path(prof.split("=", 1)[1]).exists(), "scratch profile left behind"


def test_the_shot_does_not_wait_for_chrome_to_exit(tmp_path):
    """Issue #29: with its own profile chrome writes the PNG and then hangs, so
    waiting on the process is waiting for the timeout."""
    out = tmp_path / "shot.png"
    # The stand-in is python, not `/bin/sh`: there is no /bin/sh to exec on
    # Windows, and a path interpolated into a shell string arrives with
    # backslashes, which sh reads as escapes. !r lets the child receive the
    # path exactly as written on either platform.
    argv = [sys.executable, "-c",
            f"import pathlib, time; "
            f"pathlib.Path({str(out)!r}).write_bytes(b'x'); "
            f"time.sleep(30)"]
    import time as _t
    start = _t.monotonic()
    render._shoot(argv, out, str(tmp_path), timeout=20.0)
    assert _t.monotonic() - start < 10, "waited for a process that never exits"
    assert out.exists()


def test_a_chromium_family_browser_is_found_where_this_os_puts_it():
    """Chrome is not on PATH on Windows and there is no /Applications there.

    Edge IS Chromium and ships with every Windows install, which is the same
    role `/Applications/Google Chrome.app` plays on a Mac: not the deliberate
    install, but the one that happens to be there. Without it the html lane
    has no rasterizer on a stock Windows box and every page goes unrendered.
    """
    found = render.chrome_path()
    assert found, "no Chromium-family browser found on this machine"
    assert Path(found).exists(), found


def test_the_sandbox_is_dropped_only_where_the_kernel_refuses_it(monkeypatch,
                                                                 tmp_path):
    """Ubuntu 23.10 and later deny the unprivileged user namespace Chrome's
    renderer sandbox needs, and Chrome then dies with "No usable sandbox!" and
    writes no screenshot: the html and ink lanes go unmeasured on the whole
    machine. A kernel that allows the namespace keeps the sandbox."""
    src, out = tmp_path / "a.html", tmp_path / "a.png"
    monkeypatch.setattr(render, "_sandbox_is_unusable", lambda: False)
    assert "--no-sandbox" not in render.chrome_argv(src, out, 256)
    monkeypatch.setattr(render, "_sandbox_is_unusable", lambda: True)
    assert "--no-sandbox" in render.chrome_argv(src, out, 256)


def test_the_apparmor_switch_is_read_rather_than_guessed(tmp_path, monkeypatch):
    """kernel.apparmor_restrict_unprivileged_userns is the switch, and it is 1
    on a stock Ubuntu 24.04 and absent everywhere else this runs."""
    import builtins
    real = builtins.open

    def fake(path, *a, **k):
        if str(path).endswith("apparmor_restrict_unprivileged_userns"):
            return real(tmp_path / "switch", *a, **k)
        return real(path, *a, **k)

    (tmp_path / "switch").write_text("1\n", encoding="utf-8")
    monkeypatch.setattr(builtins, "open", fake)
    render._sandbox_is_unusable.cache_clear()
    assert render._sandbox_is_unusable() is True
    (tmp_path / "switch").write_text("0\n", encoding="utf-8")
    render._sandbox_is_unusable.cache_clear()
    assert render._sandbox_is_unusable() is False
    render._sandbox_is_unusable.cache_clear()


def test_a_snap_browser_is_not_a_rasterizer(monkeypatch):
    """`chromium` on Ubuntu is a snap, and this list asks for it first. A
    confined browser cannot read the page it is handed, which is written to a
    temporary directory: it starts, writes no screenshot, and the check times
    out after a minute while a browser is plainly installed."""
    monkeypatch.setattr(render.shutil, "which",
                        lambda name: "/snap/bin/chromium"
                        if name == "chromium" else None)
    assert render._first_available(("chromium",)) is None
    monkeypatch.setattr(render.shutil, "which",
                        lambda name: "/usr/bin/google-chrome-stable"
                        if name == "google-chrome-stable" else None)
    assert render._first_available(
        ("chromium", "google-chrome-stable")) == "/usr/bin/google-chrome-stable"


def test_the_renderer_makes_no_requests_of_its_own(tmp_path):
    """This module refuses to fetch a generated page's external URLs, and then
    chrome fetched its own: a Google Cloud Messaging registration was the last
    thing Chromium printed before the Linux runner's render timed out."""
    argv = render.chrome_argv(tmp_path / "a.html", tmp_path / "a.png", 256)
    assert "--disable-background-networking" in argv
    assert "--disable-component-update" in argv


def test_the_browser_can_be_named(monkeypatch, tmp_path):
    """A machine with three chromiums installed needs a way to say which one,
    without editing a candidate list in this repo."""
    named = tmp_path / "my-chrome"
    named.write_text("", encoding="utf-8")
    monkeypatch.setenv(render.CHROME_ENV, str(named))
    assert render.chrome_path() == str(named)
    monkeypatch.setenv(render.CHROME_ENV, str(tmp_path / "absent"))
    assert render.chrome_path() != str(tmp_path / "absent")


def test_the_isolated_profile_can_be_turned_off(monkeypatch):
    """It is not free: with one, chrome writes the screenshot and never exits,
    and on some builds no screenshot arrives at all."""
    assert render._isolate_profile() is True
    monkeypatch.setenv(render.PROFILE_ENV, "0")
    assert render._isolate_profile() is False


def test_a_browser_that_cannot_take_a_profile_is_retried_without_one(
        monkeypatch, tmp_path):
    """Measured on a Linux runner holding two browsers of the same version:
    Chromium 152.0.7977.0 writes no screenshot when handed its own
    --user-data-dir, and Chrome 152.0.7977.82 is fine either way. Dropping the
    profile everywhere would put every render back in the user's own Chrome
    profile, which is what issue #29 was about."""
    render._PROFILE_HANGS.discard("/fake/chromium")
    monkeypatch.setattr(render, "chrome_path", lambda: "/fake/chromium")
    seen = []

    def fake_shoot(argv, out, cwd, timeout=60.0):
        had_profile = any(a.startswith("--user-data-dir") for a in argv)
        seen.append(had_profile)
        if not had_profile:
            Path(out).write_bytes(b"\x89PNG\r\n\x1a\n")
        return "chrome said nothing useful"

    monkeypatch.setattr(render, "_shoot", fake_shoot)
    render.rasterize_html("<p>x</p>", tmp_path / "a.png")
    assert seen == [True, False], seen
    assert "/fake/chromium" in render._PROFILE_HANGS

    # And it is remembered: the next render does not pay the timeout again.
    seen.clear()
    render.rasterize_html("<p>x</p>", tmp_path / "b.png")
    assert seen == [False], seen
    render._PROFILE_HANGS.discard("/fake/chromium")
