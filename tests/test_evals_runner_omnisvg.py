"""The svg lane's third method: a model that emits draw commands as tokens.

Every test here builds a fake checkout in tmp_path. The real one is 16 GiB of
weights and a venv of its own, and a test that needs those asserts this Mac
rather than the runner.
"""
import json
import os
import sys
from pathlib import Path

import pytest

from evals.core import Case
from evals.runners import omnisvg
from evals.runners.omnisvg import MODELS, OmniSVGRunner


def case(prompt="a settings gear icon"):
    return Case(id="icon-gear", modality="svg", prompt=prompt, params={})


SVG = ('<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200">'
       '<path d="M10 10 L90 90" stroke="black"/></svg>')

#: Stands in for inference.py: same flags, and it records what it was given
#: somewhere that outlives the runner's temporary directory.
WRITES_SVG = f'''
import json, os, pathlib, sys
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
out = pathlib.Path(args["--output"])
out.mkdir(parents=True, exist_ok=True)
(out / "0001_gear.svg").write_text({SVG!r}, encoding="utf-8")
record = os.environ.get("OMNISVG_TEST_RECORD")
if record:
    pathlib.Path(record).write_text(json.dumps(
        {{"argv": sys.argv, "prompt": pathlib.Path(args["--input"]).read_text(encoding="utf-8")}}))
'''

WRITES_NOTHING = '''
import pathlib, sys
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
pathlib.Path(args["--output"]).mkdir(parents=True, exist_ok=True)
'''


def checkout(tmp_path, body):
    """A directory shaped like an OmniSVG checkout, running `body` for
    inference.py."""
    root = tmp_path / "omnisvg"
    (root / ".venv" / "bin").mkdir(parents=True)
    (root / "inference.py").write_text(body, encoding="utf-8")
    interpreter = root / ".venv" / "bin" / "python"
    try:
        interpreter.symlink_to(sys.executable)
    except OSError:
        # An unprivileged Windows process cannot create a symlink (WinError
        # 1314). A hardlink is the same file and answers the same question.
        os.link(sys.executable, interpreter)
    # A symlinked interpreter resolves back to the venv it came from and reads
    # that venv's pyvenv.cfg. A hardlink has no path back, so python looks
    # beside itself, finds nothing and exits 106 "No pyvenv.cfg file". Writing
    # one makes the fake checkout a real venv under either kind of link.
    (root / ".venv" / "pyvenv.cfg").write_text(
        f"""home = {sys.base_prefix}
include-system-site-packages = false
""",
        encoding="utf-8")
    return root


@pytest.fixture
def cached(monkeypatch, tmp_path):
    """Both halves of the model present, without touching the HF cache."""
    weights = tmp_path / "weights"
    weights.mkdir()
    monkeypatch.setattr(omnisvg, "cached", lambda repo, name: weights)
    return weights


def test_an_unknown_size_is_refused_by_name():
    with pytest.raises(ValueError) as exc:
        OmniSVGRunner("2B")
    assert "2B" in str(exc.value)
    assert "4B" in str(exc.value)


def test_zero_candidates_is_refused():
    with pytest.raises(ValueError):
        OmniSVGRunner("4B", candidates=0)


def test_the_svg_it_wrote_is_the_artifact(tmp_path, cached):
    root = checkout(tmp_path, WRITES_SVG)
    r = OmniSVGRunner("4B", root=root).run(case())
    assert r.passed, r.detail
    assert r.artifact == SVG
    assert r.candidate == "omnisvg:4B"


def test_the_prompt_and_the_flags_reach_the_script(tmp_path, cached, monkeypatch):
    record = tmp_path / "record.json"
    monkeypatch.setenv("OMNISVG_TEST_RECORD", str(record))
    root = checkout(tmp_path, WRITES_SVG)

    r = OmniSVGRunner("4B", candidates=3, root=root).run(case("a gear icon"))
    assert r.passed, r.detail

    seen = json.loads(record.read_text(encoding="utf-8"))
    assert seen["prompt"].strip() == "a gear icon"
    argv = seen["argv"]
    assert argv[argv.index("--num-candidates") + 1] == "3"
    assert argv[argv.index("--model-size") + 1] == "4B"
    assert argv[argv.index("--task") + 1] == "text-to-svg"
    # Local paths, not repo ids: HF_HUB_OFFLINE is 1 everywhere here, and
    # upstream would try to download anything that does not look like a path.
    assert Path(argv[argv.index("--model-path") + 1]).is_absolute()
    assert Path(argv[argv.index("--weight-path") + 1]).is_absolute()


def test_each_case_gets_its_own_output_directory(tmp_path, cached):
    # Upstream names the file from the first 50 characters of the prompt, so a
    # shared directory would let one case be scored on another's answer.
    root = checkout(tmp_path, WRITES_SVG)
    runner = OmniSVGRunner("4B", root=root)
    first = runner.run(case("one"))
    second = runner.run(case("two"))
    assert first.passed and second.passed


def test_missing_weights_name_the_repos_and_the_script(tmp_path, monkeypatch):
    root = checkout(tmp_path, WRITES_SVG)
    monkeypatch.setattr(omnisvg, "cached", lambda repo, name: None)
    r = OmniSVGRunner("4B", root=root).run(case())
    assert not r.passed
    assert MODELS["4B"][0] in r.detail
    assert MODELS["4B"][1] in r.detail
    assert "setup-omnisvg.sh" in r.detail


def test_a_missing_checkout_is_a_failed_row_not_a_traceback(tmp_path, cached):
    r = OmniSVGRunner("4B", root=tmp_path / "nowhere").run(case())
    assert not r.passed
    assert "setup-omnisvg.sh" in r.detail


def test_a_missing_interpreter_is_named(tmp_path, cached):
    root = checkout(tmp_path, WRITES_SVG)
    (root / ".venv" / "bin" / "python").unlink()
    r = OmniSVGRunner("4B", root=root).run(case())
    assert not r.passed
    assert "interpreter" in r.detail


def test_exit_zero_with_no_svg_is_a_failure(tmp_path, cached):
    # Upstream returns 0 when every sampled candidate renders empty, so a clean
    # exit is not evidence that anything was produced.
    root = checkout(tmp_path, WRITES_NOTHING)
    r = OmniSVGRunner("4B", root=root).run(case())
    assert not r.passed
    assert "no SVG" in r.detail


def test_a_crash_carries_the_childs_stderr(tmp_path, cached):
    root = checkout(tmp_path, 'import sys; sys.exit("mps out of memory")')
    r = OmniSVGRunner("4B", root=root).run(case())
    assert not r.passed
    assert "mps out of memory" in r.detail


def test_the_interpreter_can_be_overridden(tmp_path, monkeypatch):
    monkeypatch.setenv("OMNISVG_PYTHON", "/usr/bin/true")
    assert omnisvg.interpreter(tmp_path) == Path("/usr/bin/true")


def test_home_follows_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("OMNISVG_HOME", str(tmp_path / "elsewhere"))
    assert omnisvg.home() == tmp_path / "elsewhere"
