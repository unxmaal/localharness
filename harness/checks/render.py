"""Rasterizing an SVG so the pixel checks apply to it too.

The structural checks read the markup: does it parse, is there an xmlns, how
many shapes were drawn. Every one of those can be true of a document that
renders as an empty rectangle -- white shapes on a white ground, a shape placed
outside the viewBox, a zero-size viewBox, everything hidden behind an opaque
rect. A model that emits well-formed SVG which draws nothing visible passes
every check the suite had. The only way to catch it is to draw it and look.

rsvg-convert is librsvg's CLI. It is an OPTIONAL dependency: when it is absent
the check warns and passes, because a missing tool that failed every candidate
at once would look exactly like a model regression. It is found by name on PATH
or by location -- see RASTERIZER_CANDIDATES for why the second is needed.

HTML goes through headless Chrome, which is already on this machine, so it
needs no Playwright and no npm. Same failure being caught: a page can parse,
have a body, be perfectly self-contained, and render as a white rectangle.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from functools import lru_cache
from pathlib import Path

from harness.checks.base import CheckResult

RASTERIZER = "rsvg-convert"
# Same shape as CHROME_CANDIDATES: the name on PATH is the deliberate install,
# then the places it lands when it was not put on PATH at all. librsvg is not
# in winget; the route to it on Windows is MSYS2, and pacman does not add
# mingw64\\bin to PATH -- so a machine that HAS rsvg-convert still answered
# "not installed" and skipped the whole ink lane. Naming the location is the
# alternative to putting all of MSYS2 on PATH, where its gcc and python would
# shadow the real ones.
RASTERIZER_CANDIDATES = (
    "rsvg-convert",
    r"C:\msys64\mingw64\bin\rsvg-convert.exe",
    r"C:\msys64\ucrt64\bin\rsvg-convert.exe",
    r"C:\Program Files\MSYS2\mingw64\bin\rsvg-convert.exe",
)
# Chromium first, since a `chromium` on PATH is the deliberate install; the
# rest are the ones that happen to be there. Chrome is NOT on PATH on Windows,
# so the deliberate install has to be named by location too -- and Edge is
# Chromium and ships with every Windows install, which makes it that machine's
# equivalent of the .app on any Mac. Without it a stock Windows box has no
# rasterizer and the whole html lane goes unmeasured.
CHROME_CANDIDATES = (
    "chromium",
    "chrome",
    # Linux puts the deliberate install on PATH under its package name. The
    # bare "chromium" above already covers the apt and snap builds, but
    # Google's own package installs as google-chrome-stable and matches
    # nothing else here, so a machine that HAS Chrome answered "not installed"
    # and the whole html lane went quiet.
    "google-chrome-stable",
    "google-chrome",
    "chromium-browser",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
)
DEFAULT_WIDTH = 256
# A quarter of one percent of the canvas. Below that there is nothing a person
# would call a drawing; a real icon lands far above it.
MIN_INK = 0.0025


class RenderError(RuntimeError):
    """The document could not be drawn."""


def _first_available(candidates) -> str | None:
    """The first candidate that exists, by PATH lookup or by location.

    A bare name is asked of PATH; anything already absolute is asked whether it
    is there. isabs, not startswith("/"): an absolute Windows path begins with
    a drive letter, and handing one to shutil.which asks PATHEXT about a file
    that is already fully named.
    """
    for candidate in candidates:
        if os.path.isabs(candidate):
            if Path(candidate).exists():
                return candidate
        else:
            found = shutil.which(candidate)
            if found:
                return found
    return None


def chrome_path() -> str | None:
    """The browser to render HTML with, or None if there is not one."""
    return _first_available(CHROME_CANDIDATES)


def rasterizer_path() -> str | None:
    """The SVG rasterizer, or None. Still OPTIONAL -- see the module docstring."""
    return _first_available(RASTERIZER_CANDIDATES)


@lru_cache(maxsize=1)
def _sandbox_is_unusable() -> bool:
    """True where the kernel refuses the user namespace Chrome's sandbox needs.

    Ubuntu 23.10 introduced kernel.apparmor_restrict_unprivileged_userns, on by
    default, and 24.04 kept it. Nothing else this project runs on sets it.
    """
    try:
        with open("/proc/sys/kernel/apparmor_restrict_unprivileged_userns",
                  encoding="utf-8") as fh:
            return fh.read().strip() == "1"
    except OSError:
        return False


def chrome_argv(src: Path, out: Path, width: int,
                profile: Path | None = None) -> list[str]:
    """Headless Chrome, rendering a LOCAL FILE in an ISOLATED PROFILE.

    file:// and not a data: URL or a served page: a generated page may
    reference an external URL, and the checker must never be the thing that
    fetches it. An opaque white background is forced so a transparent body does
    not read as ink, and a virtual time budget lets layout and webfonts settle
    before the shot rather than capturing a half-painted frame.

    An isolated `--user-data-dir` keeps the render reproducible and off the
    user's default profile. Issue #29.
    """
    argv = [
        chrome_path() or "chrome",
        "--headless=new",
        "--disable-gpu",
        "--hide-scrollbars",
        "--force-device-scale-factor=1",
        "--default-background-color=FFFFFFFF",
        "--virtual-time-budget=2000",
        f"--window-size={width},{int(width * 0.75)}",
        f"--screenshot={out}",
    ]
    if _sandbox_is_unusable():
        # Chrome's renderer sandbox needs an unprivileged user namespace, and
        # Ubuntu 23.10 and later deny one by AppArmor default. Without this it
        # dies with "No usable sandbox!" and produces no screenshot at all, so
        # the entire html and ink lanes go unmeasured on the machine.
        #
        # ASKED RATHER THAN ASSUMED: a machine whose kernel allows the
        # namespace keeps the sandbox. The pages rendered here are markup this
        # project generated, opened as file:// with no network fetch, but that
        # is a reason to accept the risk where it is forced, not to take it
        # everywhere.
        argv.append("--no-sandbox")
    if profile is not None:
        # A fresh profile otherwise spends its first run on setup work and
        # first-run prompts, which is time added to every single check.
        argv += [f"--user-data-dir={profile}",
                 "--no-first-run",
                 "--no-default-browser-check",
                 "--disable-extensions"]
    argv.append(f"file://{src}")
    return argv


def rasterize_html(html: str, out: str | Path, width: int = 800) -> Path:
    """Render an HTML document to a PNG with headless Chrome."""
    if chrome_path() is None:
        raise RenderError("no Chrome or Chromium found to render HTML")

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    # mkdtemp with an explicit removal, not TemporaryDirectory: its cleanup
    # raises on the first refusal, and on Windows the browser is still letting
    # go of the profile at that moment.
    d = tempfile.mkdtemp()
    try:
        src = Path(d) / "page.html"
        src.write_text(html, encoding="utf-8")
        profile = Path(d) / "chrome-profile"
        stderr = _shoot(chrome_argv(src, out, width, profile=profile), out, d)
    finally:
        _remove_tree(Path(d))
    if not out.exists():
        raise RenderError(f"chrome produced no screenshot: {stderr[:300]}")
    return out


def _kill_tree(proc) -> None:
    """Reap chrome's renderer and GPU processes, which outlive the parent.

    POSIX allows unlinking a file another process still holds, so the scratch
    profile can be removed with them running. Windows refuses it with
    WinError 32, and the temporary directory then fails to clean up.
    """
    if sys.platform != "win32":
        return
    try:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=15)
    except (OSError, subprocess.SubprocessError):
        pass


def _remove_tree(path: Path, attempts: int = 20, delay: float = 0.1) -> None:
    """Remove a directory a just-killed browser may still be holding open.

    The handles go within a moment of the tree dying, so this retries rather
    than giving up on the first refusal. Ignoring the error outright would
    leave a chrome profile in the temporary directory on every render.
    """
    for attempt in range(attempts):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except OSError:
            if attempt == attempts - 1:
                shutil.rmtree(path, ignore_errors=True)
                return
            time.sleep(delay)


def _shoot(argv: list[str], out: Path, cwd: str, timeout: float = 60.0) -> str:
    """Run chrome and stop once the PNG is written.

    With its own --user-data-dir chrome writes the screenshot and then does not
    exit, so waiting for the process is waiting for the timeout. Issue #29.
    """
    err = tempfile.TemporaryFile("w+")
    # A file, not a pipe: chrome's grandchildren inherit the handle and keep a
    # pipe open long after we terminate it, so reading one would block.
    proc = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=err, cwd=cwd)
    deadline = time.monotonic() + timeout
    size = -1
    exited = False
    try:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                exited = True
                break
            now = out.stat().st_size if out.exists() else -1
            if now > 0 and now == size:      # written and stable
                break
            size = now
            time.sleep(0.1)
        if not exited:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        _kill_tree(proc)
        err.seek(0)
        return err.read().strip()
    finally:
        err.close()


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
    binary = rasterizer_path()
    if binary is None:
        raise RenderError(
            f"{RASTERIZER} is not installed "
            f"(brew install librsvg / pacman -S mingw-w64-x86_64-librsvg)")

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Through a file rather than stdin: rsvg-convert reports the offending line
    # number for a file, and that is the whole diagnostic for broken markup.
    with tempfile.NamedTemporaryFile("w", suffix=".svg", delete=False) as fh:
        fh.write(svg)
        src = fh.name
    try:
        proc = subprocess.run(
            [binary, "--width", str(width), "--height", str(width),
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
    # None still disables the check outright. Otherwise resolve it the same way
    # rasterize_svg does -- gating this on PATH alone while the rasterizer runs
    # from a known location made every SVG pass unmeasured, which is the one
    # failure mode this check exists to prevent.
    resolved = None if rasterizer is None else _first_available(
        RASTERIZER_CANDIDATES if rasterizer == RASTERIZER else (rasterizer,))
    if resolved is None:
        return CheckResult(
            True, "",
            [f"not rasterized: {RASTERIZER} is not installed "
             f"(brew install librsvg / pacman -S mingw-w64-x86_64-librsvg)"])

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
