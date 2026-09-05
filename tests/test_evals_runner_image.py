"""The image runner: measures an external generator as a subprocess.

Unlike the text runner there is no completion to parse. The contract is the
process: does it exit clean, how long did it take, how much memory did it peak
at, and is the file it left behind a real image.
"""
import sys
from pathlib import Path

import pytest

from evals.core import Case
from evals.runners.image import ImageRunner, Engine

HERE = Path(__file__).parent


def case(**over):
    base = dict(id="fox", modality="image", prompt="a red fox in snow",
                assertions={"width": 64, "height": 64})
    base.update(over)
    return Case(**base)


def fake_engine(script: str) -> Engine:
    """An engine whose command is a python one-liner we control."""
    return Engine(
        name="fake",
        argv=lambda c, out, a: [sys.executable, "-c", script, str(out)],
        output_suffix=".png")


WRITE_GOOD = (
    "import sys;from PIL import Image;import random;"
    "im=Image.new('RGB',(64,64));"
    "im.putdata([(random.randint(0,255),)*3 for _ in range(64*64)]);"
    "im.save(sys.argv[1])")
WRITE_BLANK = ("import sys;from PIL import Image;"
               "Image.new('RGB',(64,64),(128,128,128)).save(sys.argv[1])")
WRITE_NOTHING = "import sys"
CRASH = "import sys; sys.exit(3)"


def test_successful_generation_passes(tmp_path):
    r = ImageRunner(fake_engine(WRITE_GOOD), tmp_path).run(case())
    assert r.passed, r.detail
    assert r.candidate == "fake"
    assert r.seconds > 0
    assert r.artifact and Path(r.artifact).exists()


def test_peak_memory_is_recorded(tmp_path):
    r = ImageRunner(fake_engine(WRITE_GOOD), tmp_path).run(case())
    assert r.peak_kb > 0, "a subprocess runner that reports no memory is useless"


def test_blank_output_fails_even_though_the_command_succeeded(tmp_path):
    """Exit 0 plus a grey square is the failure mode worth catching."""
    r = ImageRunner(fake_engine(WRITE_BLANK), tmp_path).run(case())
    assert not r.passed
    assert "uniform" in r.detail.lower()


def test_missing_output_fails(tmp_path):
    r = ImageRunner(fake_engine(WRITE_NOTHING), tmp_path).run(case())
    assert not r.passed
    assert "no output" in r.detail.lower()


def test_nonzero_exit_fails_with_the_exit_code(tmp_path):
    r = ImageRunner(fake_engine(CRASH), tmp_path).run(case())
    assert not r.passed
    assert "3" in r.detail


def test_wrong_dimensions_fail(tmp_path):
    r = ImageRunner(fake_engine(WRITE_GOOD), tmp_path).run(
        case(assertions={"width": 512, "height": 512}))
    assert not r.passed
    assert "512" in r.detail


def test_timeout_is_a_failed_row_not_a_hang(tmp_path):
    slow = fake_engine("import time; time.sleep(30)")
    r = ImageRunner(slow, tmp_path, timeout=1.0).run(case())
    assert not r.passed
    assert "timed out" in r.detail.lower()


def test_missing_binary_is_a_clear_failure(tmp_path):
    eng = Engine(name="absent",
                 argv=lambda c, out, a: ["definitely-not-a-real-binary-xyz"],
                 output_suffix=".png")
    r = ImageRunner(eng, tmp_path).run(case())
    assert not r.passed
    assert "not installed" in r.detail.lower() or "not found" in r.detail.lower()


def test_artifacts_are_named_per_case_so_runs_do_not_collide(tmp_path):
    runner = ImageRunner(fake_engine(WRITE_GOOD), tmp_path)
    a = runner.run(case(id="one"))
    b = runner.run(case(id="two"))
    assert Path(a.artifact).name != Path(b.artifact).name
    assert Path(a.artifact).exists() and Path(b.artifact).exists()


# ---- mflux engine spec parsing --------------------------------------------

from evals.runners.image import mflux_engine  # noqa: E402


def argv_for(spec, **assertions):
    from evals.core import Case
    eng = mflux_engine(spec)
    c = Case(id="t", modality="image", prompt="a fox",
             assertions=assertions or {})
    return eng, eng.argv(c, Path("/tmp/o.png"), c.assertions)


def test_plain_model_uses_its_own_entry_point():
    """`z-image-turbo` ships as mflux-generate-z-image-turbo."""
    eng, argv = argv_for("z-image-turbo")
    assert argv[0] == "mflux-generate-z-image-turbo"
    assert "--model" not in argv
    assert eng.name == "mflux/z-image-turbo"


def test_entrypoint_slash_model_passes_model_as_a_flag():
    """flux2-klein-4b is a --model of the flux2 entry point, not a binary.

    Getting this wrong invents mflux-generate-flux2-klein-4b, which does not
    exist, and the run fails as 'not installed' rather than as a bad spec.
    """
    eng, argv = argv_for("flux2/flux2-klein-4b")
    assert argv[0] == "mflux-generate-flux2"
    assert argv[argv.index("--model") + 1] == "flux2-klein-4b"
    assert eng.name == "mflux/flux2-klein-4b"


def test_steps_and_dimensions_are_forwarded():
    _, argv = argv_for("z-image-turbo", width=512, height=512, steps=8, seed=42)
    assert argv[argv.index("--width") + 1] == "512"
    assert argv[argv.index("--steps") + 1] == "8"
    assert argv[argv.index("--seed") + 1] == "42"


def test_case_steps_override_the_candidate_default():
    eng = mflux_engine("z-image-turbo", steps=4)
    from evals.core import Case
    c = Case(id="t", modality="image", prompt="x", assertions={"steps": 20})
    argv = eng.argv(c, Path("/tmp/o.png"), c.assertions)
    assert argv[argv.index("--steps") + 1] == "20"


def test_prompt_is_passed_as_one_argument_not_split():
    _, argv = argv_for("z-image-turbo")
    assert "a fox" in argv
