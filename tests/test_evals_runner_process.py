"""The process runner: measures an external generator as a subprocess.

Unlike the completion runner there is nothing to parse. The contract is the
process: does it exit clean, how long did it take, how much memory did it peak
at, and is the file it left behind real.

Engine command lines are tested in test_harness_engines.py, because engines are
shared with the CLI and are not part of the eval suite.
"""
import sys
from pathlib import Path

from evals.core import Case
from evals.runners.process import ProcessRunner
from harness.engines import Engine


def case(**over):
    base = dict(id="fox", modality="image", prompt="a red fox in snow",
                params={"width": 64, "height": 64})
    base.update(over)
    return Case(**base)


def fake_engine(script: str, **over) -> Engine:
    """An engine whose command is a python one-liner we control."""
    kwargs = dict(name="fake", spec="fake:1",
                  argv=lambda p, out, params: [sys.executable, "-c", script,
                                               str(out)],
                  modality="image", output_suffix=".png", timeout=60.0)
    kwargs.update(over)
    return Engine(**kwargs)


WRITE_GOOD = (
    "import sys;from PIL import Image;import random;"
    "im=Image.new('RGB',(64,64));"
    "im.putdata([(random.randint(0,255),)*3 for _ in range(64*64)]);"
    "im.save(sys.argv[1])")
WRITE_BLANK = ("import sys;from PIL import Image;"
               "Image.new('RGB',(64,64),(128,128,128)).save(sys.argv[1])")
WRITE_NOTHING = "import sys"
CRASH = "import sys; sys.stderr.write('out of memory\\n'); sys.exit(3)"


def test_successful_generation_passes(tmp_path):
    r = ProcessRunner(fake_engine(WRITE_GOOD), tmp_path).run(case())
    assert r.passed, r.detail
    assert r.candidate == "fake"
    assert r.seconds > 0
    assert r.artifact and Path(r.artifact).exists()


def test_peak_memory_is_recorded(tmp_path):
    r = ProcessRunner(fake_engine(WRITE_GOOD), tmp_path).run(case())
    assert r.peak_kb > 0, "a subprocess runner that reports no memory is useless"


def test_blank_output_fails_even_though_the_command_succeeded(tmp_path):
    """Exit 0 plus a grey square is the failure mode worth catching."""
    r = ProcessRunner(fake_engine(WRITE_BLANK), tmp_path).run(case())
    assert not r.passed
    assert "uniform" in r.detail.lower()


def test_missing_output_fails(tmp_path):
    r = ProcessRunner(fake_engine(WRITE_NOTHING), tmp_path).run(case())
    assert not r.passed
    assert "no output" in r.detail.lower()


def test_nonzero_exit_fails_with_the_exit_code_and_the_last_stderr_line(tmp_path):
    r = ProcessRunner(fake_engine(CRASH), tmp_path).run(case())
    assert not r.passed
    assert "3" in r.detail
    assert "out of memory" in r.detail


def test_wrong_dimensions_fail(tmp_path):
    r = ProcessRunner(fake_engine(WRITE_GOOD), tmp_path).run(
        case(params={"width": 512, "height": 512}))
    assert not r.passed
    assert "512" in r.detail


def test_timeout_is_a_failed_row_not_a_hang(tmp_path):
    slow = fake_engine("import time; time.sleep(30)")
    r = ProcessRunner(slow, tmp_path, timeout=1.0).run(case())
    assert not r.passed
    assert "timed out" in r.detail.lower()


def test_the_engine_supplies_its_own_timeout(tmp_path):
    """Video is hours and images are a minute; one shared default is wrong for
    both."""
    runner = ProcessRunner(fake_engine(WRITE_GOOD, timeout=4321.0), tmp_path)
    assert runner.timeout == 4321.0


def test_missing_binary_is_a_clear_failure(tmp_path):
    eng = fake_engine("", argv=lambda p, out, params: ["not-a-real-binary-xyz"])
    r = ProcessRunner(eng, tmp_path).run(case())
    assert not r.passed
    assert "not installed" in r.detail.lower()


def test_a_bad_parameter_combination_is_a_row_not_a_traceback(tmp_path):
    def argv(p, out, params):
        raise ValueError("pass --frames or --seconds, not both")
    r = ProcessRunner(fake_engine("", argv=argv), tmp_path).run(case())
    assert not r.passed
    assert "not both" in r.detail


def test_artifacts_are_named_per_case_so_runs_do_not_collide(tmp_path):
    runner = ProcessRunner(fake_engine(WRITE_GOOD), tmp_path)
    a = runner.run(case(id="one"))
    b = runner.run(case(id="two"))
    assert Path(a.artifact).name != Path(b.artifact).name
    assert Path(a.artifact).exists() and Path(b.artifact).exists()


def test_a_slash_in_the_candidate_name_does_not_become_a_directory(tmp_path):
    """mflux/z-image-turbo-q8 is one candidate, not a path."""
    runner = ProcessRunner(fake_engine(WRITE_GOOD, name="mflux/z-image-turbo-q8"),
                           tmp_path)
    r = runner.run(case())
    assert r.passed, r.detail
    assert Path(r.artifact).parent == tmp_path
