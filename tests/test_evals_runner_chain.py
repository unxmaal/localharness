"""Two-stage image workflows: generate, then do something to the result.

This is the vocabulary ComfyUI is wanted for, and mflux ships all of it
natively in MLX -- controlnet, depth, fill, redux, kontext, in-context, two
upscalers. Nineteen primitives, and until now none of them could be a candidate
because every runner here assumed one command produces the artifact.

`upscale:` is the first, and the shape is deliberately general: a base engine
produces an image, a second command consumes it. Adding `controlnet:` later
should be a stage definition, not another runner.
"""
from pathlib import Path

import pytest

from evals.core import Case
from evals.runners.chain import STAGE_SCALE, STAGES, ChainRunner
from harness.engines import Engine


def png(path, size=64):
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (size, size), "white")
    ImageDraw.Draw(im).ellipse((4, 4, size - 4, size - 4), fill="green")
    im.save(path)
    return path


def fake_engine(**over):
    import sys
    kwargs = dict(
        name="fake", spec="fake:1",
        argv=lambda p, out, params: [
            sys.executable, "-c",
            "import sys;from PIL import Image,ImageDraw;"
            "im=Image.new('RGB',(64,64),'white');"
            "ImageDraw.Draw(im).ellipse((4,4,60,60),fill='green');"
            "im.save(sys.argv[1])", str(out)],
        modality="image", output_suffix=".png", timeout=60.0)
    kwargs.update(over)
    return Engine(**kwargs)


def case():
    return Case(id="fox", modality="image", prompt="a fox",
                params={"width": 64, "height": 64})


def test_the_second_stage_receives_the_first_stage_output(tmp_path, monkeypatch):
    seen = {}

    def fake_stage(src, dst, params):
        seen["src"] = src
        png(dst, 128)
        return ["true"]

    monkeypatch.setitem(STAGES, "probe", fake_stage)
    # A stage that changes the resolution must SAY so, or the size check
    # rightly rejects its output. That is the contract, not a workaround.
    monkeypatch.setitem(STAGE_SCALE, "probe", 2)
    r = ChainRunner(fake_engine(), "probe", tmp_path).run(case())
    assert r.passed, r.detail
    assert Path(seen["src"]).exists(), "stage two was not given a real image"


def test_the_intermediate_is_kept(tmp_path, monkeypatch):
    """When the final image is wrong the first question is whether the first
    stage was already wrong, and deleting it throws away the only way to tell."""
    monkeypatch.setitem(STAGES, "probe",
                        lambda src, dst, params: png(dst, 128) and ["true"])
    monkeypatch.setitem(STAGE_SCALE, "probe", 2)
    ChainRunner(fake_engine(), "probe", tmp_path).run(case())
    assert list(Path(tmp_path).glob("*stage1*.png")), "intermediate deleted"


def test_a_failing_first_stage_never_runs_the_second(tmp_path, monkeypatch):
    import sys
    ran = []
    monkeypatch.setitem(STAGES, "probe",
                        lambda src, dst, params: ran.append(1) or ["true"])
    broken = fake_engine(argv=lambda p, out, params: [sys.executable, "-c",
                                                      "import sys; sys.exit(3)"])
    r = ChainRunner(broken, "probe", tmp_path).run(case())
    assert not r.passed
    assert not ran, "the second stage ran on a missing image"


def test_the_candidate_name_shows_both_stages(tmp_path):
    r = ChainRunner(fake_engine(), "upscale-seedvr2", tmp_path)
    assert "upscale-seedvr2" in r.candidate and "fake" in r.candidate


def test_the_row_reports_the_cost_of_both_stages(tmp_path, monkeypatch):
    """A two-stage workflow costs both, and a table without that makes it look
    as cheap as a single generation."""
    monkeypatch.setitem(STAGES, "probe",
                        lambda src, dst, params: png(dst, 128) and ["true"])
    monkeypatch.setitem(STAGE_SCALE, "probe", 2)
    r = ChainRunner(fake_engine(), "probe", tmp_path).run(case())
    assert r.metrics["stages"] == 2


def test_an_unknown_stage_is_refused_when_the_runner_is_built(tmp_path):
    with pytest.raises(ValueError) as e:
        ChainRunner(fake_engine(), "nope", tmp_path)
    assert "nope" in str(e.value)


def test_the_upscaler_stage_builds_a_real_command(tmp_path):
    argv = STAGES["upscale-seedvr2"](tmp_path / "in.png", tmp_path / "out.png",
                                     {"width": 512})
    assert any("upscale-seedvr2" in a for a in argv)
    assert "--image-path" in argv and "--output" in argv
    # On 32GB this is the difference between running and not.
    assert "--low-ram" in argv or "--vae-tiling" in argv


def test_a_stage_that_does_not_declare_a_scale_must_preserve_the_size(tmp_path,
                                                                     monkeypatch):
    """The contract, stated as a test. A stage that silently changes the
    resolution is indistinguishable from a broken one, and the size check is
    the only thing standing between the two."""
    monkeypatch.setitem(STAGES, "quiet",
                        lambda src, dst, params: png(dst, 128) and ["true"])
    r = ChainRunner(fake_engine(), "quiet", tmp_path).run(case())
    assert not r.passed
    assert "64" in r.detail and "128" in r.detail
