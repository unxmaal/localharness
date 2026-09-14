"""Issue #148 phase 3 and issue #172: the judge reads the store, and the
control that authorises it runs more than once.
"""
import pytest

from harness import cli, judge
from harness import memory_store as ms
from harness.memory_store import Seen


@pytest.fixture
def db(tmp_path):
    conn = ms.connect(tmp_path / "d.db")
    yield conn
    conn.close()


def queued(db, name, why="a thing", source="recap", times=1):
    for i in range(times):
        ms.record(db, Seen(name=name, source=source, url=f"https://x/{i}",
                           why=why, resolved=name, lane="image",
                           registry=ms.HUGGINGFACE))
    ms.decide(db, name, "queued", tier="inspect", detail="fits: 3.0 GiB")


# --- what the judge can reach --------------------------------------------

def test_the_judge_can_reach_what_the_source_tier_queued(db):
    """THE RUNG. _judge_fits() only ever saw the Fit objects from its own
    process, so everything the store held was unreachable."""
    queued(db, "org/model")
    got = ms.judgeable(db)
    assert [r["name"] for r in got] == ["org/model"]
    assert got[0]["inspected"] == "fits: 3.0 GiB"
    assert got[0]["why"] == "a thing"


def test_a_candidate_a_judge_already_read_is_not_offered_again(db):
    queued(db, "org/model")
    ms.decide(db, "org/model", "queued", tier="judge", score=7,
              rubric="novelty@5", judge="q3-4b")
    assert ms.judgeable(db) == []


def test_a_candidate_the_source_tier_declined_is_not_judged(db):
    """A thing that cannot run here is answered. Paying a model call to rank it
    buys a number for a candidate nothing will screen."""
    ms.record(db, Seen(name="org/big", source="recap", resolved="org/big"))
    ms.decide(db, "org/big", "declined", tier="inspect", detail="too-big")
    assert ms.judgeable(db) == []


def test_the_most_corroborated_candidate_is_judged_first(db):
    """Same ordering as pending(): a thing that keeps coming back is a
    different signal from a thing that trended once."""
    queued(db, "org/once", times=1)
    queued(db, "org/thrice", times=3)
    assert [r["name"] for r in ms.judgeable(db)] == ["org/thrice", "org/once"]


def test_ranked_returns_what_a_judge_scored_best_first(db):
    for name, score in (("org/a", 4), ("org/b", 9), ("org/c", 6)):
        queued(db, name)
        ms.decide(db, name, "queued", tier="judge", score=score,
                  rubric="novelty@5", judge="q3-4b")
    assert [r["name"] for r in ms.ranked(db)] == ["org/b", "org/c", "org/a"]


# --- the control, which is the gate --------------------------------------

def _args(**kw):
    base = {"judge": True, "from_store": True, "no_control": False, "runs": 3,
            "top": 25, "shard": "", "gateway": "", "json": False}
    base.update(kw)
    return type("A", (), base)()


def test_nothing_is_scored_when_the_rubric_does_not_separate(tmp_path,
                                                             monkeypatch):
    """THE REFUSAL IS THE FEATURE. A score from a rubric that does not
    discriminate is a number, not a ranking, and a tier running unattended on a
    schedule has nobody present to doubt it."""
    from harness import paths
    monkeypatch.setattr(paths, "home", lambda: tmp_path)
    conn = ms.connect()
    queued(conn, "org/model")
    conn.close()

    scored = []
    monkeypatch.setattr(judge, "control_repeated", lambda **kw: {
        "runs": 3, "gaps": [2, -1, 3], "spread": 4, "separates": False,
        "separated_in": 2, "rubric": "novelty@5", "model": "q3-4b", "rows": []})
    monkeypatch.setattr(judge, "score",
                        lambda *a, **kw: scored.append(a) or (7, "why"))

    assert cli._report_judge_store(_args()) == 1
    assert scored == [], "it scored despite the control failing"
    conn = ms.connect()
    assert ms.judgeable(conn), "the candidate must stay unjudged"
    conn.close()


def test_a_separating_control_lets_the_scores_through(tmp_path, monkeypatch):
    """The positive half. A gate that never opens is not a gate."""
    from harness import paths
    monkeypatch.setattr(paths, "home", lambda: tmp_path)
    conn = ms.connect()
    queued(conn, "org/model")
    conn.close()

    monkeypatch.setattr(judge, "control_repeated", lambda **kw: {
        "runs": 3, "gaps": [2, 2, 2], "spread": 0, "separates": True,
        "separated_in": 3, "rubric": "novelty@5", "model": "q3-4b", "rows": []})
    monkeypatch.setattr(judge, "score", lambda *a, **kw: (8, "it composes"))

    assert cli._report_judge_store(_args()) == 0
    conn = ms.connect()
    assert ms.judgeable(conn) == [], "the candidate should now be judged"
    row = ms.ranked(conn)[0]
    conn.close()
    assert row["score"] == 8
    # THE RUBRIC AND THE JUDGE RIDE WITH THE SCORE. Two runs under different
    # rubrics are different exams, and a score with neither recorded cannot be
    # told apart from one that is.
    assert row["rubric"] and row["judge"]


def test_the_judge_is_shown_what_the_source_tier_found(tmp_path, monkeypatch):
    """Issue #69: the judge scored a generic description 3/10 while the source
    tier had already proved the thing ran here. The tier with more evidence
    lost to the tier with less."""
    from harness import paths
    monkeypatch.setattr(paths, "home", lambda: tmp_path)
    conn = ms.connect()
    queued(conn, "org/model")
    conn.close()

    seen = []
    monkeypatch.setattr(judge, "control_repeated", lambda **kw: {
        "runs": 1, "gaps": [2], "spread": 0, "separates": True,
        "separated_in": 1, "rubric": "r", "model": "m", "rows": []})
    monkeypatch.setattr(judge, "score",
                        lambda item, *a, **kw: (seen.append(item), (5, "ok"))[1])
    cli._report_judge_store(_args())
    assert "fits: 3.0 GiB" in seen[0]
    assert "org/model" in seen[0]


def test_the_control_command_runs_the_repeated_form(monkeypatch):
    """Issue #172. control_repeated() was built because one run is one draw
    from a sampling judge, was tested, and nothing called it: this command went
    to the single-shot form, so the sentence authorising every score in the
    project came from one sample."""
    called = []
    monkeypatch.setattr(judge, "control_repeated", lambda **kw: called.append(kw) or {
        "runs": kw.get("runs", 0), "gaps": [2], "spread": 0, "separates": True,
        "separated_in": 1, "rubric": "r", "model": "m", "rows": []})
    monkeypatch.setattr(judge, "control", lambda *a, **kw: pytest.fail(
        "the single-run control must not be what a published verdict rests on"))
    assert cli._report_control(_args(control=True, runs=4, json=False)) == 0
    assert called[0]["runs"] == 4


# --- the rubric's identity, and the gateway it needs ----------------------

def test_editing_a_rubric_in_place_changes_what_is_recorded():
    """A version is a NAME, and a name survives every edit to the thing it
    names. Rewriting what_scores_high without touching `version` produces a
    second exam under the first one's identity -- the defect cases_digest
    exists to refuse, in the other instrument this project ranks with. It
    matters more now that a pod scores a queue with nobody watching."""
    a = judge.load()
    b = judge.Rubric(**{**a.__dict__, "prompt": a.prompt + "\nAlso reward blue."})
    assert a.identity == b.identity, "the readable name is deliberately stable"
    assert a.stamp != b.stamp, "what is recorded must not be"
    assert a.digest in a.stamp


def test_two_runs_of_the_same_rubric_stamp_the_same():
    """The negative half: a digest that changed for no reason would make every
    run incomparable with every other, which is the same failure as one that
    never changes."""
    assert judge.load().stamp == judge.load().stamp


def test_the_rubric_names_a_model_every_gateway_actually_serves():
    """EXTERNAL STATE WITH NO ASSERTION ON IT. The rubric names an alias; the
    gateway configs define them; nothing joined the two. The judge tier now
    runs in a pod on a schedule, where that mismatch surfaces as a failed Job
    at 4am rather than as a message to whoever edited the rubric."""
    import yaml as _yaml
    from pathlib import Path
    repo = Path(__file__).resolve().parent.parent
    model = judge.load().model
    configs = sorted(repo.glob("gateway/config*.yaml"))
    assert configs, "no gateway config found; this test checks nothing"
    for path in configs:
        raw = _yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        aliases = {m.get("model_name") for m in (raw.get("model_list") or [])}
        assert model in aliases, (
            f"the rubric judges with {model!r}, which {path.name} does not "
            f"serve. Either the config is missing the alias or the rubric "
            f"names one that machine cannot run")


def test_an_empty_queue_costs_no_model_calls(tmp_path, monkeypatch):
    """A Job on a schedule meets an empty queue most weeks. Running the control
    first would pay a model call per control item to prove a rubric separates
    and then score nothing."""
    from harness import paths
    monkeypatch.setattr(paths, "home", lambda: tmp_path)
    ms.connect().close()          # a store with nothing queued

    monkeypatch.setattr(judge, "control_repeated", lambda **kw: pytest.fail(
        "the control ran with nothing to judge"))
    monkeypatch.setattr(judge, "score", lambda *a, **kw: pytest.fail(
        "it scored with nothing to judge"))
    assert cli._report_judge_store(_args()) == 0
