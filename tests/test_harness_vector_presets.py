"""Issue #4: a traced icon was 28KB with nothing reporting the size."""
import argparse
import shutil
from pathlib import Path

import pytest

from evals.run import TRACE_PREFIXES, kind_of, modality_of
from harness.checks import render
from harness import vector

# The ink check shells out to rsvg-convert. Without the guard these fail on a
# machine that simply does not have it, which reads as a code defect.
needs_rsvg = pytest.mark.skipif(render.rasterizer_path() is None,
                                reason="needs rsvg-convert (brew install librsvg)")
from harness.vector import TRACE_PRESETS, VectorError, trace


def gear(path, size=512):
    """Detailed on purpose. A smooth two-circle icon traces to ~1KB either way,
    so it cannot show the difference real diffusion output does."""
    import random
    from PIL import Image, ImageDraw
    rng = random.Random(7)
    im = Image.new("RGB", (size, size), "white")
    d = ImageDraw.Draw(im)
    d.ellipse((size * .1, size * .1, size * .9, size * .9), fill="black")
    for _ in range(60):
        x, y = rng.randrange(size), rng.randrange(size)
        r = rng.randrange(6, 22)
        d.ellipse((x - r, y - r, x + r, y + r),
                  fill=(rng.randrange(256), rng.randrange(256),
                        rng.randrange(256)))
    im.save(path)
    return path


@pytest.fixture
def src(tmp_path):
    return gear(tmp_path / "gear.png")


def test_the_icon_preset_is_substantially_smaller(src):
    big = trace(src, preset="illustration")
    small = trace(src, preset="icon")
    assert len(small.encode()) < len(big.encode()) * 0.7


@needs_rsvg
def test_the_icon_preset_still_draws_the_picture(src):
    """Smaller is easy if you are allowed to draw nothing."""
    from harness.checks import render
    assert render.ink(trace(src, preset="icon")) > vector.MIN_INK * 2


def test_an_unknown_preset_names_the_known_ones(src):
    with pytest.raises(VectorError) as exc:
        trace(src, preset="tiny")
    assert "illustration" in str(exc.value) and "icon" in str(exc.value)


def test_every_preset_declares_both_knobs():
    for name, cfg in TRACE_PRESETS.items():
        assert set(cfg) == {"trace_at", "path_precision"}, name


def test_both_trace_prefixes_resolve_to_the_svg_lane():
    for prefix in TRACE_PREFIXES:
        candidate = f"{prefix}:mflux:flux2-klein-4b"
        assert kind_of(candidate) == prefix
        assert modality_of(candidate) == "svg"


def test_the_prefixes_map_onto_real_presets():
    assert set(TRACE_PREFIXES.values()) <= set(TRACE_PRESETS)


def test_size_is_reported_as_a_metric_and_never_ranked_on():
    """A traced illustration is legitimately large and a glyph legitimately
    small, so ranking on it would put the blank document first."""
    from evals.core import direction_of
    assert direction_of("svg_bytes") == "neutral"


def test_the_svg_check_reports_the_document_size():
    from evals.core import Case, CHECKERS
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10" '
           'width="10" height="10"><rect width="10" height="10" '
           'fill="black"/></svg>')
    r = CHECKERS["svg"](svg, Case(id="x", modality="svg", prompt="p"))
    assert r.metrics["svg_bytes"] == len(svg.encode())


def test_the_cli_offers_every_preset_the_eval_can_rank():
    """Issue #38. The eval could prove the icon preset was 3.2x smaller and
    `lh` had no way to ask for one."""
    import argparse

    from harness import cli
    parser = cli.build_parser() if hasattr(cli, "build_parser") else None
    if parser is None:
        import pytest
        pytest.skip("no build_parser to introspect")
    choices = None
    for action in parser._subparsers._group_actions[0].choices["svg"]._actions:
        if action.dest == "method":
            choices = set(action.choices)
    assert choices is not None
    assert set(TRACE_PRESETS) <= choices | {"illustration"}
    assert "icon" in choices


def test_the_cli_icon_method_uses_the_icon_preset(tmp_path, monkeypatch):
    from harness import cli
    seen = {}

    def fake_generate(engine, prompt, png, params):
        gear(png)
        return 0

    monkeypatch.setattr(cli, "_generate", fake_generate)
    real = vector.trace
    monkeypatch.setattr(vector, "trace",
                        lambda p, **kw: seen.setdefault("preset",
                                                        kw.get("preset")) or real(p, **kw))
    a = argparse.Namespace(method="icon", prompt="a gear", engine="x",
                           output=str(tmp_path / "o.svg"), width=64, height=64,
                           seed=1, json=False)
    cli.cmd_svg(a)
    assert seen["preset"] == "icon"
