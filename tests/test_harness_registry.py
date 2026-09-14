"""Issue #167: a name that does not say which registry holds it.

The sweep proposes HuggingFace model ids. The tier that consumed them cloned
from GitHub. Both halves worked; 227 of 235 candidates 404ed.
"""
import sqlite3

import pytest

from harness import fetching, inspect as ins
from harness import memory_store as ms
from harness.memory_store import Seen


@pytest.fixture
def db(tmp_path):
    conn = ms.connect(tmp_path / "d.db")
    yield conn
    conn.close()


def see(db, name, registry="", **kw):
    return ms.record(db, Seen(name=name, source=kw.pop("source", "recap"),
                              registry=registry, **kw))


# --- the column ----------------------------------------------------------

def test_a_proposal_records_which_registry_answers_for_it(db):
    see(db, "org/model", registry=ms.HUGGINGFACE, resolved="org/model")
    see(db, "org/tool", registry=ms.GITHUB, resolved="org/tool")
    got = {r["name"]: r["registry"]
           for r in db.execute("SELECT name, registry FROM proposals")}
    assert got == {"org/model": ms.HUGGINGFACE, "org/tool": ms.GITHUB}


def test_a_later_sighting_fills_an_empty_registry_and_never_overwrites_one(db):
    """Same rule as `lane`: evidence arrives late, but a recorded answer is not
    revised by a source that knows less."""
    see(db, "org/thing", registry="", url="https://x/1")
    see(db, "org/thing", registry=ms.HUGGINGFACE, url="https://x/2")
    see(db, "org/thing", registry=ms.GITHUB, url="https://x/3")
    row = db.execute("SELECT registry FROM proposals").fetchone()
    assert row["registry"] == ms.HUGGINGFACE


def test_pending_asks_one_registry_at_a_time(db):
    see(db, "org/model", registry=ms.HUGGINGFACE, resolved="org/model")
    see(db, "org/tool", registry=ms.GITHUB, resolved="org/tool")
    see(db, "org/mystery", registry="", resolved="org/mystery")
    assert ms.pending(db, registry=ms.HUGGINGFACE) == ["org/model"]
    assert ms.pending(db, registry=ms.GITHUB) == ["org/tool"]
    # NO FILTER RETURNS EVERYTHING, the unknown included. That is right for a
    # report and wrong for resolving, which is why the caller must say.
    assert set(ms.pending(db)) == {"org/model", "org/tool", "org/mystery"}
    # AND THE UNKNOWN ONES CAN BE ASKED FOR ON THEIR OWN: they are work waiting
    # on one question, not work nobody can do.
    assert ms.pending(db, registry="") == ["org/mystery"]


def test_by_registry_counts_the_names_nothing_can_resolve(db):
    see(db, "org/model", registry=ms.HUGGINGFACE, resolved="org/model")
    see(db, "a/b", registry="", resolved="a/b")
    see(db, "c/d", registry="", resolved="c/d")
    assert ms.by_registry(db)[""] == 2
    assert ms.by_registry(db)[ms.HUGGINGFACE] == 1


def test_a_settled_proposal_is_not_pending_in_its_registry(db):
    see(db, "org/model", registry=ms.HUGGINGFACE, resolved="org/model")
    ms.decide(db, "org/model", "declined", tier="inspect")
    assert ms.pending(db, registry=ms.HUGGINGFACE) == []


# --- the migration, which is where the 227 already-stored names live ------

def _v2_store(path):
    """A store as the code before this issue wrote it: no registry column."""
    ddl = "\n".join(line for line in ms._DDL.splitlines()
                    if "registry" not in line)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(ddl)
    conn.execute("INSERT INTO meta VALUES ('schema', '2')")
    return conn


def _insert(conn, name, kind, url):
    pid = conn.execute(
        "INSERT INTO proposals (name, kind, lane, resolved, first_seen, "
        "last_seen) VALUES (?,?,'','x',1.0,1.0)", (name, kind)).lastrowid
    conn.execute("INSERT INTO sightings (proposal_id, source, url, seen_at) "
                 "VALUES (?,'recap',?,1.0)", (pid, url))
    conn.commit()


def test_an_existing_store_is_backfilled_from_evidence_it_already_holds(tmp_path):
    path = tmp_path / "old.db"
    old = _v2_store(path)
    _insert(old, "org/model", "repo", "https://huggingface.co/org/model")
    _insert(old, "org/tool", "repo", "https://github.com/org/tool")
    _insert(old, "org/weights", "weights", "")
    _insert(old, "org/linked", "tool", "")
    old.close()

    conn = ms.connect(path)
    got = {r["name"]: r["registry"]
           for r in conn.execute("SELECT name, registry FROM proposals")}
    conn.close()
    # THE URL BEATS THE KIND, and it has to: the crowd tier wrote kind='repo'
    # for a GitHub repo and the sweep wrote the same kind for a HuggingFace
    # one, so kind alone would send half of them to the wrong API.
    assert got["org/model"] == ms.HUGGINGFACE
    assert got["org/tool"] == ms.GITHUB
    assert got["org/weights"] == ms.HUGGINGFACE
    assert got["org/linked"] == ms.GITHUB


def test_a_name_with_no_evidence_is_left_unknown_rather_than_guessed(tmp_path):
    """Guessing turns "nobody knows" into a wrong answer nothing revisits."""
    path = tmp_path / "old.db"
    old = _v2_store(path)
    _insert(old, "some-name", "candidate", "https://reddit.com/r/x/1")
    old.close()
    conn = ms.connect(path)
    row = conn.execute("SELECT registry FROM proposals").fetchone()
    conn.close()
    assert row["registry"] == ""


# --- reading a model card, which is the half that did not exist ----------

CARD = {"siblings": [{"size": 1_000_000}, {"size": 2_000_000}],
        "pipeline_tag": "text-to-image",
        "tags": ["mlx", "diffusion"],
        "lastModified": "2026-09-01T00:00:00.000Z"}


def test_a_model_card_yields_a_verdict_with_nothing_cloned():
    fit = ins.inspect_model("org/model", data=CARD)
    assert fit.verdict == "fits"
    assert fit.registry == ms.HUGGINGFACE
    assert fit.weights["org/model"] == 3_000_000
    assert fit.lanes["org/model"] == "image"
    assert fit.mlx is True


def test_a_model_is_not_refused_for_having_no_entry_point():
    """The rule that asks what there is to call belongs to source trees. A
    weights repo has nothing to call by construction, and applying it here
    refuses every model in the registry for not being a program."""
    fit = ins.inspect_model("org/model", data=CARD)
    assert fit.entry_points == []
    assert fit.verdict != "no-entry-point"


def test_a_source_tree_with_no_entry_point_is_still_refused():
    """The negative half: the rule must still fire where it belongs."""
    fit = ins.decide(ins.Fit(repo="org/tool", registry=ms.GITHUB))
    assert fit.verdict == "no-entry-point"


def test_a_card_listing_no_blob_sizes_is_unsized_rather_than_zero():
    fit = ins.inspect_model("org/model", data={"siblings": [], "tags": []})
    assert fit.unsized == ["org/model"]
    assert fit.verdict == "unknown"


def test_a_404_and_an_outage_are_different_answers():
    """A 404 settles a candidate. An unreachable registry must settle nothing,
    or one bad afternoon marks a sweep's worth of real models as broken."""
    def gone(url):
        raise RuntimeError("https://x: HTTP 404")

    def down(url):
        raise RuntimeError("https://x: timed out")

    with pytest.raises(ins.Gone):
        ins.hf_model("org/model", fetch=gone)
    with pytest.raises(ins.InspectError) as caught:
        ins.hf_model("org/model", fetch=down)
    assert not isinstance(caught.value, ins.Gone)


def test_a_gated_model_is_not_reported_as_missing():
    """401 means a licence click, not absence. Settling it would throw away
    exactly the candidates that need one."""
    def gated(url):
        raise RuntimeError("https://x: HTTP 401")

    with pytest.raises(ins.InspectError) as caught:
        ins.hf_model("org/model", fetch=gated)
    assert not isinstance(caught.value, ins.Gone)


# --- the reader audit: who else acts on a name ---------------------------

def test_the_download_queue_refuses_a_github_name(db):
    """The inverse of this issue, already paid for once: repo names were handed
    to snapshot_download and every one of them 401'd."""
    for name, registry in (("org/model", ms.HUGGINGFACE),
                           ("org/tool", ms.GITHUB)):
        see(db, name, registry=registry, kind="weights", resolved=name,
            lane="image")
        ms.decide(db, name, "queued", tier="inspect", detail="bytes=10")
    got = [r["name"] for r in fetching.queued(db, needs_lane=False)]
    assert got == ["org/model"]


# --- the seam: two halves that each work, and the agreement between them --

def test_the_inspect_tier_asks_each_registry_only_about_its_own_names(
        tmp_path, monkeypatch):
    """THE DEFECT ITSELF. The sweep's half wrote HuggingFace ids correctly and
    the inspect half read GitHub correctly; nothing owned the agreement, so
    every swept name went to the wrong API."""
    from harness import cli, github, paths

    monkeypatch.setattr(paths, "home", lambda: tmp_path)
    conn = ms.connect()
    see(conn, "org/model", registry=ms.HUGGINGFACE, resolved="org/model")
    see(conn, "org/tool", registry=ms.GITHUB, resolved="org/tool")
    see(conn, "org/mystery", registry="", resolved="org/mystery")
    conn.close()

    asked_github, asked_hf = [], []

    class Client:
        stale: list = []
        spent = 0

        def __init__(self, **kw):
            pass

        def repo(self, name):
            asked_github.append(name)
            return {"size": 10, "description": "a tool"}

    def fake_inspect(repo, work, meta=None, **kw):
        return ins.Fit(repo=repo, registry=ms.GITHUB, verdict="unknown")

    def fake_inspect_model(model_id, **kw):
        asked_hf.append(model_id)
        return ins.Fit(repo=model_id, registry=ms.HUGGINGFACE, verdict="unknown")

    def fake_hf_model(model_id, fetch=None):
        asked_hf.append(model_id)
        raise ins.Gone(f"{model_id}: no such model")

    monkeypatch.setattr(github, "Client", Client)
    monkeypatch.setattr(ins, "inspect", fake_inspect)
    monkeypatch.setattr(ins, "inspect_model", fake_inspect_model)
    monkeypatch.setattr(ins, "hf_model", fake_hf_model)

    args = type("A", (), {"repos": [], "from_store": True, "top": 10,
                          "budget": 10, "shard": "", "judge": False,
                          "json": False})()
    assert cli._report_inspect(args) == 0
    # The name each registry was asked about is the name it owns. The third,
    # which the store cannot route, is RESOLVED: asked of HuggingFace, and then
    # of GitHub when HuggingFace says it has no such model.
    assert asked_github == ["org/tool", "org/mystery"]
    assert asked_hf == ["org/model", "org/mystery"]
    conn = ms.connect()
    got = conn.execute("SELECT registry FROM proposals WHERE name = 'org/mystery'"
                       ).fetchone()["registry"]
    conn.close()
    assert got == ms.GITHUB, "the answer is written down, so nothing asks twice"


# --- resolving a name nothing wrote a registry down for -------------------

class _Repos:
    """A github client that knows a fixed set of repos."""

    def __init__(self, known=(), reachable=True):
        self.known, self.reachable = set(known), reachable

    def repo(self, name):
        from harness import github
        if not self.reachable:
            raise github.GitHubError("the API could not be reached")
        if name not in self.known:
            raise github.NotFound(f"gh api repos/{name}: Not Found")
        return {"size": 10, "description": "a tool"}


def _card(*_a, **_kw):
    return {"siblings": [{"size": 1}], "tags": []}


def _no_model(name, fetch=None):
    raise ins.Gone(f"{name}: no such model")


def _registry_down(name, fetch=None):
    raise ins.InspectError(f"{name}: timed out")


def test_huggingface_answers_first_where_both_would():
    """feeds.candidates() already runs its passes in this order, on the
    argument that a model is the thing the eval can actually run."""
    from harness.cli import resolve_registry
    got, card = resolve_registry("org/thing", _Repos(["org/thing"]), model=_card)
    assert got == ms.HUGGINGFACE
    assert card is not None, "the card it already fetched, so nothing asks twice"


def test_a_name_only_github_has_resolves_to_github():
    from harness.cli import resolve_registry
    got, meta = resolve_registry("org/tool", _Repos(["org/tool"]), model=_no_model)
    assert got == ms.GITHUB
    assert meta is not None


def test_a_name_neither_registry_has_is_a_definite_answer():
    from harness.cli import resolve_registry
    with pytest.raises(ins.Gone):
        resolve_registry("org/nothing", _Repos(), model=_no_model)


def test_an_unreachable_registry_settles_nothing():
    """THE NEGATIVE HALF, and the expensive one to get wrong: a 404 is now
    terminal, so an outage that reads as "neither has it" would settle a
    sweep's worth of real models as missing."""
    from harness.cli import resolve_registry
    got, _ = resolve_registry("org/thing", _Repos(reachable=False),
                              model=_registry_down)
    assert got == ""
    # And a HuggingFace outage beside a GitHub 404 is not "neither" either.
    got, _ = resolve_registry("org/thing", _Repos(), model=_registry_down)
    assert got == ""
