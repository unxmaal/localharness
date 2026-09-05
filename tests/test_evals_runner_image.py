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
