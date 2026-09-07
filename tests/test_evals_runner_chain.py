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
from evals.runners import chain
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

    def fake_stage(src, dst, params, prompt):
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
                        lambda src, dst, params, prompt: png(dst, 128) and ["true"])
    monkeypatch.setitem(STAGE_SCALE, "probe", 2)
    ChainRunner(fake_engine(), "probe", tmp_path).run(case())
    assert list(Path(tmp_path).glob("*stage1*.png")), "intermediate deleted"


def test_a_failing_first_stage_never_runs_the_second(tmp_path, monkeypatch):
    import sys
    ran = []
    monkeypatch.setitem(STAGES, "probe",
                        lambda src, dst, params, prompt: ran.append(1) or ["true"])
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
                        lambda src, dst, params, prompt: png(dst, 128) and ["true"])
    monkeypatch.setitem(STAGE_SCALE, "probe", 2)
    r = ChainRunner(fake_engine(), "probe", tmp_path).run(case())
    assert r.metrics["stages"] == 2


def test_an_unknown_stage_is_refused_when_the_runner_is_built(tmp_path):
    with pytest.raises(ValueError) as e:
        ChainRunner(fake_engine(), "nope", tmp_path)
    assert "nope" in str(e.value)


def test_the_upscaler_stage_builds_a_real_command(tmp_path):
    argv = STAGES["upscale-seedvr2"](tmp_path / "in.png", tmp_path / "out.png",
                                     {"width": 512}, "a fox")
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
                        lambda src, dst, params, prompt: png(dst, 128) and ["true"])
    r = ChainRunner(fake_engine(), "quiet", tmp_path).run(case())
    assert not r.passed
    assert "64" in r.detail and "128" in r.detail


# ---- conditioning primitives ----------------------------------------------
# The reason people reach for ComfyUI. Stage one generates a base image, stage
# two treats it as a control image -- which is the img2img/ControlNet pattern
# without needing a case to supply an input from outside.

def test_a_stage_can_receive_the_prompt():
    """generate-controlnet conditions on an image AND a prompt, so a stage
    that only sees the previous file cannot express it."""
    argv = STAGES["controlnet"](Path("/in.png"), Path("/out.png"),
                                {"width": 512, "height": 512}, "a red fox")
    assert "--controlnet-image-path" in argv and "/in.png" in argv
    assert "--prompt" in argv and "a red fox" in argv


def test_the_controlnet_upscaler_sets_its_own_size():
    argv = STAGES["upscale-controlnet"](Path("/in.png"), Path("/out.png"),
                                        {"width": 512, "height": 512}, "x")
    assert "--controlnet-image-path" in argv
    # It takes explicit width/height rather than a scale factor, so the stage
    # must state the size it intends and STAGE_SCALE must agree.
    assert "1024" in argv
    assert STAGE_SCALE["upscale-controlnet"] == 2


def test_conditioning_preserves_the_requested_size():
    """A controlnet regeneration is the same size as its input; only the
    upscalers scale."""
    assert STAGE_SCALE.get("controlnet", 1) == 1


def test_every_stage_declares_a_scale():
    """A stage whose scale is unknown fails the size check for reasons nobody
    can debug."""
    for name in STAGES:
        assert name in STAGE_SCALE, f"{name} declares no scale"


def test_every_stage_takes_the_same_four_arguments():
    """A stage is a small contract, and it stays small only if it is checked."""
    import inspect
    for name, fn in STAGES.items():
        params = list(inspect.signature(fn).parameters)
        assert params == ["src", "dst", "params", "prompt"], f"{name}: {params}"


# ---- stage two is not exempt from the memory guard -------------------------
# A stage loads its OWN model, often a bigger one than the base engine: the
# controlnet stages pull FLUX.1-dev, 31GB on disk. ChainRunner launched them
# with no check at all, which is the exact shape that took this machine down
# once already -- and worse here, because stage one's weights may still be
# resident when stage two starts.

def test_a_stage_too_large_for_this_machine_is_refused(tmp_path, monkeypatch):
    monkeypatch.setitem(STAGES, "huge",
                        lambda src, dst, params, prompt: png(dst, 128) and ["true"])
    monkeypatch.setitem(STAGE_SCALE, "huge", 2)
    monkeypatch.setitem(chain.STAGE_MODELS, "huge", ("some/enormous-model", 4))
    monkeypatch.setattr(chain.memory, "check_model",
                        lambda repo, **kw: (False, "needs 40.0 GB but only 8.0 GB"))
    r = ChainRunner(fake_engine(), "huge", tmp_path).run(case())
    assert not r.passed
    assert "40.0 GB" in r.detail


def test_the_refusal_happens_before_stage_one_is_wasted(tmp_path, monkeypatch):
    """Fifty seconds of diffusion, then a refusal, is fifty seconds thrown
    away. Check what the whole workflow needs before starting any of it."""
    ran = []
    monkeypatch.setitem(STAGES, "huge",
                        lambda src, dst, params, prompt: ran.append(1) or ["true"])
    monkeypatch.setitem(STAGE_SCALE, "huge", 1)
    monkeypatch.setitem(chain.STAGE_MODELS, "huge", ("some/enormous-model", 4))
    monkeypatch.setattr(chain.memory, "check_model",
                        lambda repo, **kw: (False, "too big"))
    engine_ran = []
    eng = fake_engine()
    original = eng.argv
    r = ChainRunner(eng, "huge", tmp_path).run(case())
    assert not r.passed
    assert not ran, "stage two ran despite the refusal"
    assert not list(Path(tmp_path).glob("*stage1*.png")), \
        "stage one ran before the guard checked"


def test_a_stage_with_no_declared_model_is_allowed(tmp_path, monkeypatch):
    """Not every stage loads weights, and refusing what cannot be sized would
    make the guard the thing that breaks the workflow."""
    monkeypatch.setitem(STAGES, "light",
                        lambda src, dst, params, prompt: png(dst, 64) and ["true"])
    monkeypatch.setitem(STAGE_SCALE, "light", 1)
    r = ChainRunner(fake_engine(), "light", tmp_path).run(case())
    assert r.passed, r.detail
