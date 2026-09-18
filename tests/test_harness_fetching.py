"""Issue #62: the durable download queue."""
import pytest

from harness import fetching as f
from harness import memory_store as ms


@pytest.fixture
def db(tmp_path):
    conn = ms.connect(tmp_path / "d.db")
    yield conn
    conn.close()


def seen(db, name, kind="weights", lane="stt"):
    ms.record(db, ms.Seen(name=name, source="inspect", resolved=name, kind=kind,
                          lane=lane))


# ---- what may be downloaded ------------------------------------------------

def test_an_unknown_size_is_refused_rather_than_attempted():
    """The point of inspecting before fetching is not to find out how big
    something is by downloading it."""
    assert not f.plan("a/b", 0).ok
    assert not f.plan("a/b", -1).ok


def test_a_download_that_would_fill_the_volume_is_refused():
    got = f.plan("a/b", 20 * f.GIB, free=60 * f.GIB)
    assert not got.ok and "floor" in got.why


def test_a_download_over_the_cap_is_refused_however_much_disk_there_is():
    assert not f.plan("a/b", 500 * f.GIB, free=5000 * f.GIB).ok


def test_a_download_that_fits_is_allowed():
    assert f.plan("a/b", 20 * f.GIB, free=900 * f.GIB).ok


# ---- the queue -------------------------------------------------------------

def test_only_the_inspect_tier_can_queue_a_download(db):
    """The JUDGE tier also writes `queued`, and it judges a DESCRIPTION: it
    queued a 122B model on a 32 GB machine. Nothing is downloaded on prose."""
    seen(db, "org/judged")
    seen(db, "org/inspected")
    ms.decide(db, "org/judged", "queued", tier="judge", score=9)
    ms.decide(db, "org/inspected", "queued", tier="inspect")
    assert [r["name"] for r in f.queued(db)] == ["org/inspected"]


def test_a_terminal_verdict_leaves_the_queue(db):
    seen(db, "org/a")
    ms.decide(db, "org/a", "queued", tier="inspect")
    ms.decide(db, "org/a", "declined", tier="fetch", detail="too big")
    assert f.queued(db) == []


def test_a_weight_inherits_the_score_of_the_repo_that_named_it(db):
    """The judge scores REPOS and the queue holds WEIGHTS, so a weight has no
    score of its own. Sorting on its own score put every row at 0 and "largest
    first" was whatever order the query returned."""
    for repo, weight, score in [("org/dull", "org/low", 4),
                                ("org/sharp", "org/high", 9)]:
        seen(db, repo, kind="repo")
        seen(db, weight)
        ms.link(db, repo, weight, "needs")
        ms.decide(db, repo, "queued", tier="inspect", score=score)
        ms.decide(db, weight, "queued", tier="inspect")
    assert [r["name"] for r in f.queued(db)] == ["org/high", "org/low"]


# ---- running it ------------------------------------------------------------

def test_one_at_a_time_by_default(db):
    calls = []
    for name in ["org/a", "org/b"]:
        seen(db, name)
        ms.decide(db, name, "queued", tier="inspect")
    f.run(db, {"org/a": 2 * f.GIB, "org/b": 2 * f.GIB},
          snapshot=lambda repo_id: calls.append(repo_id) or "/tmp/x",
          free=900 * f.GIB)
    assert len(calls) == 1


def test_something_too_big_is_declined_permanently(db):
    """Terminal, so the same oversized model is not re-queued every sweep."""
    seen(db, "org/huge")
    ms.decide(db, "org/huge", "queued", tier="inspect")
    f.run(db, {"org/huge": 500 * f.GIB}, snapshot=lambda **kw: "/x",
          free=900 * f.GIB)
    assert "org/huge" in ms.settled(db)


def test_a_failed_download_does_not_condemn_the_candidate(db):
    """A network error says nothing about the model."""
    seen(db, "org/a")
    ms.decide(db, "org/a", "queued", tier="inspect")

    def boom(repo_id):
        raise RuntimeError("connection reset")

    got = f.run(db, {"org/a": 2 * f.GIB}, snapshot=boom, free=900 * f.GIB)
    assert got[0]["ok"] is False
    assert "org/a" not in ms.settled(db)


def test_a_finished_download_records_where_it_landed(db):
    seen(db, "org/a")
    ms.decide(db, "org/a", "queued", tier="inspect")
    f.run(db, {"org/a": 2 * f.GIB}, snapshot=lambda repo_id: "/Volumes/FAST/hf/a",
          free=900 * f.GIB)
    row = db.execute("SELECT run_path FROM verdicts WHERE tier='fetch'").fetchone()
    assert row["run_path"] == "/Volumes/FAST/hf/a"


# ---- a repo is not a weight ------------------------------------------------

def test_a_github_repo_is_not_downloadable(db):
    """The inspect tier queued the repos it read, and this worker calls
    snapshot_download, which wants a HuggingFace model id. Every queued name
    401'd. A repo is something to install and screen; a weight is something to
    fetch."""
    seen(db, "Blaizzy/nativ", kind="repo")
    seen(db, "mlx-community/parakeet", kind="weights")
    ms.decide(db, "Blaizzy/nativ", "queued", tier="inspect")
    ms.decide(db, "mlx-community/parakeet", "queued", tier="inspect")
    assert [r["name"] for r in f.queued(db)] == ["mlx-community/parakeet"]


def test_something_already_in_the_cache_is_not_queued(db, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    (tmp_path / "hub" / "models--org--have").mkdir(parents=True)
    for name in ["org/have", "org/want"]:
        seen(db, name)
        ms.decide(db, name, "queued", tier="inspect")
    assert [r["name"] for r in f.queued(db)] == ["org/want"]


def test_the_size_is_read_from_the_store_not_asked_for_again():
    """The registry rate-limits, and a size already measured is a fact."""
    assert f.size_of({"detail": "fits: bytes=1234 MLX-native"}) == 1234
    assert f.size_of({"detail": "no size here"}) == 0
    assert f.size_of({}) == 0


def test_fetching_lifts_the_offline_guard_and_puts_it_back(monkeypatch):
    """HF_HUB_OFFLINE=1 everywhere else stops an eval silently re-downloading
    a model mid-run. Fetching is the one operation whose job is to go online."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    seen_value = {}

    def snapshot(repo_id):
        import os
        seen_value["during"] = os.environ.get("HF_HUB_OFFLINE")
        return "/tmp/x"

    f.download("org/x", snapshot=snapshot)
    import os
    assert seen_value["during"] == "0"
    assert os.environ["HF_HUB_OFFLINE"] == "1"


def test_a_refusal_does_not_consume_the_download_budget(db):
    """One unsized entry at the head of the queue consumed the whole budget,
    so nothing was ever fetched."""
    calls = []
    for name, score in [("org/unsized", 9), ("org/real", 5)]:
        seen(db, name)
        ms.decide(db, name, "queued", tier="inspect", score=score,
                  detail="" if name == "org/unsized" else "bytes=2000000000")
    f.run(db, {"org/real": 2 * f.GIB}, limit=1,
          snapshot=lambda repo_id: calls.append(repo_id) or "/tmp/x",
          free=900 * f.GIB)
    assert calls == ["org/real"]


# ---- issue #81: a weight no lane can test is never fetched -----------------

def test_a_weight_with_no_lane_is_not_downloaded(db):
    """silero-vad and MossFormer2 both downloaded cleanly in a manual run and
    neither can be screened: there is no VAD lane and no denoising lane. An
    automated loop would keep doing that forever."""
    calls = []
    seen(db, "org/measurable", lane="stt")
    seen(db, "org/orphan", lane="")
    for n in ("org/measurable", "org/orphan"):
        ms.decide(db, n, "queued", tier="inspect")
    f.run(db, {"org/measurable": 2 * f.GIB, "org/orphan": 2 * f.GIB}, limit=5,
          snapshot=lambda repo_id: calls.append(repo_id) or "/tmp/x",
          free=900 * f.GIB)
    assert calls == ["org/measurable"]


def test_a_laneless_weight_is_still_listed_not_hidden(db):
    """Not a verdict on the model. The eval suite has no case for it, and
    building one is sometimes the work -- language ID is issue #2."""
    seen(db, "org/orphan", lane="")
    ms.decide(db, "org/orphan", "queued", tier="inspect")
    assert [r["name"] for r in f.queued(db)] == ["org/orphan"]


def test_a_laneless_weight_is_not_declined(db):
    """Terminal would mean answered, and it is not: it is waiting on a lane."""
    seen(db, "org/orphan", lane="")
    ms.decide(db, "org/orphan", "queued", tier="inspect")
    f.run(db, {"org/orphan": 2 * f.GIB}, snapshot=lambda **kw: "/x",
          free=900 * f.GIB)
    assert "org/orphan" not in ms.settled(db)


# ---- what a model needs beyond itself (issue #196) -------------------------

def cached(root, model_id, config=None):
    """A repo in the shape huggingface_hub leaves behind."""
    import json

    d = (root / "hub" / f"models--{model_id.replace('/', '--')}"
         / "snapshots" / "abc123")
    d.mkdir(parents=True, exist_ok=True)
    if config is not None:
        (d / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return d


def test_a_config_naming_another_repo_declares_a_requirement(tmp_path):
    """Marvis-AI's 8-bit MLX repo is complete and names its tokenizer in a
    different repo. The load fails offline; the directory check cannot see it."""
    cached(tmp_path, "org/model-8bit",
           {"text_tokenizer": "org/model-base", "model_type": "llama"})
    assert f.requires("org/model-8bit", tmp_path) == ["org/model-base"]


def test_provenance_is_not_a_requirement(tmp_path):
    """`_name_or_path` records where a config came from. microsoft/
    wavlm-base-plus-sv names microsoft/wavlm-base-plus, which is NOT on this
    machine, and the model loads anyway -- so treating it as a dependency marks
    a working model unready. The first real repo scanned proved this."""
    cached(tmp_path, "org/derived", {"_name_or_path": "org/parent"})
    assert f.requires("org/derived", tmp_path) == []


def test_a_value_that_is_not_repo_shaped_is_not_a_repo(tmp_path):
    """The negative half. A matcher that claims everything claims nothing."""
    cached(tmp_path, "org/m", {"text_tokenizer": "bfloat16",
                               "codec_model": "/absolute/path",
                               "vocoder": "has spaces/in it"})
    assert f.requires("org/m", tmp_path) == []


def test_a_model_naming_itself_is_not_its_own_dependency(tmp_path):
    cached(tmp_path, "org/m", {"text_tokenizer": "org/m"})
    assert f.requires("org/m", tmp_path) == []


def test_a_model_with_no_config_requires_nothing(tmp_path):
    """Most repos have no config.json worth reading, and a missing one is not
    an error -- it is the common case."""
    cached(tmp_path, "org/plain")
    assert f.requires("org/plain", tmp_path) == []


def test_missing_reports_the_dependency_when_the_model_is_present(tmp_path):
    cached(tmp_path, "org/model-8bit", {"text_tokenizer": "org/model-base"})
    assert f.missing("org/model-8bit", tmp_path) == ["org/model-base"]


def test_missing_reports_the_model_itself_when_nothing_is_there(tmp_path):
    assert f.missing("org/absent", tmp_path) == ["org/absent"]


def test_missing_is_empty_when_the_whole_closure_is_present(tmp_path):
    cached(tmp_path, "org/model-8bit", {"text_tokenizer": "org/model-base"})
    cached(tmp_path, "org/model-base")
    assert f.missing("org/model-8bit", tmp_path) == []


def test_fetching_a_model_also_fetches_what_its_config_names(db, tmp_path, monkeypatch):
    """The other half of #196. Reporting "needs org/base" is no use if the one
    command that can go online will not fetch it: queued() reads proposals, and
    a tokenizer repo is never proposed by anybody."""
    ms.record(db, ms.Seen(name="org/model-8bit", source="inspect", lane="tts",
                          kind="weights", resolved="org/model-8bit"))
    ms.decide(db, "org/model-8bit", "queued", tier="inspect")

    got = []
    monkeypatch.setattr(f, "download", lambda repo, snapshot=None: got.append(repo) or "/x")
    monkeypatch.setattr(f, "requires", lambda name, root=None: ["org/base"])
    monkeypatch.setattr(f, "have", lambda name, root=None: False)
    monkeypatch.setattr(f, "plan",
                        lambda name, *a, **k: f.Plan(name, 1, True, "ok"))

    f.run(db, {"org/model-8bit": 1}, limit=1)
    assert got == ["org/model-8bit", "org/base"]


def test_a_dependency_already_present_is_not_refetched(db, monkeypatch):
    """The negative half: re-downloading what is already cached is the cost the
    offline guard exists to avoid."""
    ms.record(db, ms.Seen(name="org/model-8bit", source="inspect", lane="tts",
                          kind="weights", resolved="org/model-8bit"))
    ms.decide(db, "org/model-8bit", "queued", tier="inspect")

    got = []
    monkeypatch.setattr(f, "download", lambda repo, snapshot=None: got.append(repo) or "/x")
    monkeypatch.setattr(f, "requires", lambda name, root=None: ["org/base"])
    # Only the DEPENDENCY is cached. Patching have() true for everything would
    # also take the model itself out of the queue, and the test would pass for
    # the wrong reason.
    monkeypatch.setattr(f, "have", lambda name, root=None: name == "org/base")
    monkeypatch.setattr(f, "plan",
                        lambda name, *a, **k: f.Plan(name, 1, True, "ok"))

    f.run(db, {"org/model-8bit": 1}, limit=1)
    assert got == ["org/model-8bit"], "the cached dependency was refetched"
