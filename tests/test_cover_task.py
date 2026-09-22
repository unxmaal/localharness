"""The music lane's style-transfer task. Issue #275.

`cover` has been in ACE-Step's TASK_TYPES_TURBO all along -- the config this
project runs -- and was never passed. What kept it unbuilt was not the engine
but the verdict: there is no per-clip style-similarity metric (#272), so the
case is decided by a person through lanes.HUMAN_JUDGED and `lh judge`.
"""
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import engines, lanes  # noqa: E402

CASE = (Path(__file__).resolve().parents[1]
        / "evals" / "cases" / "music" / "cover.yaml")


@pytest.fixture(autouse=True)
def root(monkeypatch, tmp_path):
    """Pin the checkout and the interpreter: reading the real ones would make
    this a test of the developer's disk (RULE #249)."""
    monkeypatch.setenv("ACESTEP_ROOT", str(tmp_path))
    monkeypatch.setattr(engines, "acestep_python", lambda r: "python")
    return tmp_path


def argv(params):
    e = engines.resolve("acestep:acestep-v15-turbo")
    return [str(x) for x in e.argv("a prompt", Path("/tmp/o.wav"), params)]


def test_the_engine_still_defaults_to_text2music():
    """The three existing music cases must not silently become covers."""
    got = argv({"duration": 30})
    assert "--task" not in got
    assert "--ref" not in got


def test_a_cover_passes_its_task_and_reference():
    got = argv({"task": "cover", "ref": "r.wav", "cover_strength": 0.2})
    assert got[got.index("--task") + 1] == "cover"
    assert got[got.index("--cover-strength") + 1] == "0.2"


def test_a_bare_reference_resolves_under_the_project_home():
    """A case cannot carry an absolute path that is right on two machines, and
    the reference is generated rather than committed."""
    got = argv({"task": "cover", "ref": "cover-reference.wav"})
    ref = Path(got[got.index("--ref") + 1])
    assert ref.is_absolute() and ref.name == "cover-reference.wav"
    assert ref.parent.name == "refs"


def test_an_absolute_reference_is_left_alone():
    got = argv({"task": "cover", "ref": "/elsewhere/x.wav"})
    assert got[got.index("--ref") + 1] == "/elsewhere/x.wav"


def test_cover_without_a_reference_is_refused_before_the_model_loads():
    """It restyles a reference; without one there is nothing to take a style
    from, and finding that out after a 90-second load is worse."""
    import subprocess
    out = subprocess.run(
        [sys.executable, "scripts/acestep_generate.py", "--out", "/tmp/x.wav",
         "--caption", "c", "--task", "cover"],
        capture_output=True, text=True,
        cwd=Path(__file__).resolve().parents[1])
    assert out.returncode == 2, out.stderr[-300:]
    assert "cover needs --ref" in out.stderr


# --- the case ------------------------------------------------------------

def test_the_case_asks_for_a_cover_and_names_its_reference():
    c = yaml.safe_load(CASE.read_text(encoding="utf-8"))
    assert c["params"]["task"] == "cover"
    assert c["params"]["ref"]
    assert c["modality"] == "music"


def test_the_cover_strength_is_set_for_style_transfer():
    """ACE-Step's own field defaults to 1.0, which reproduces the reference.
    Its docs say to set this SMALL for style transfer, so a case that leaves
    it alone is asking for a copy rather than a cover."""
    c = yaml.safe_load(CASE.read_text(encoding="utf-8"))
    assert 0 < c["params"]["cover_strength"] <= 0.5


def test_the_case_still_asserts_what_a_program_can_see():
    """The similarity half has no metric and is judged by a person. The parts
    a checker CAN see are asserted anyway rather than thrown away with it: a
    cover of the wrong length has failed whatever it sounds like."""
    c = yaml.safe_load(CASE.read_text(encoding="utf-8"))
    assert c["assert"]["duration_s"] == c["params"]["duration"]
    assert c["assert"]["expect_vocals"] is False


def test_the_lane_that_owns_this_case_is_judged_by_a_person():
    assert lanes.human_judged("music")


def test_the_reference_is_not_committed():
    """A 5 MB wav in a public repo, and somebody else's recording if it were
    not generated. scripts/make-cover-ref.sh writes it from a fixed seed."""
    repo = Path(__file__).resolve().parents[1]
    assert not list((repo / "evals" / "cases" / "music").glob("*.wav"))
    assert (repo / "scripts" / "make-cover-ref.sh").is_file()
    assert "refs/" in (repo / ".gitignore").read_text(encoding="utf-8")
