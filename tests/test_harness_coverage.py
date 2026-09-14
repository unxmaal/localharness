"""Issue #99: what discovery never saw.

Precision measures the quality of what is caught. Nothing measured reach, and
those are different numbers.
"""
import json

import pytest

from harness import coverage
from harness import memory_store as ms
from harness.memory_store import Seen


@pytest.fixture
def db(tmp_path):
    conn = ms.connect(tmp_path / "d.db")
    yield conn
    conn.close()


def seen(db, name, source="reddit-sd-week", at=None):
    ms.record(db, Seen(name=name, source=source, url=f"https://x/{name}",
                       resolved=name), at=at)


def runs(tmp_path, name, modality, summary, when=None):
    import os
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    f = d / "results.json"
    f.write_text(json.dumps({"receipt": {"modality": modality,
                                         "tier": "measure"},
                             "summary": summary}), encoding="utf-8")
    if when:
        os.utime(f, (when, when))
    return tmp_path


# --- what counts as a find -----------------------------------------------

def test_a_source_that_surfaced_something_we_run_is_credited(db, tmp_path,
                                                             monkeypatch):
    monkeypatch.setattr(coverage, "adopted",
                        lambda *a, **kw: {"org/thing": {"measured"}})
    seen(db, "org/thing", source="reddit-sd-week")
    got = coverage.report(db, tmp_path)
    assert [e["name"] for e in got["found"]] == ["org/thing"]
    assert got["by_source"] == {"reddit-sd-week": 1}


def test_a_thing_no_source_ever_produced_is_a_hole_with_a_name(db, tmp_path,
                                                               monkeypatch):
    monkeypatch.setattr(coverage, "adopted",
                        lambda *a, **kw: {"org/unseen": {"served"}})
    seen(db, "somebody/else")
    got = coverage.report(db, tmp_path)
    assert [h["name"] for h in got["holes"]] == ["org/unseen"]
    assert got["found"] == []


def test_our_own_tiers_are_not_credited_with_finding_anything(db, tmp_path,
                                                              monkeypatch):
    """`inspect` records what it read and `installed` is the seed list of what
    we already run. Crediting either with the find would be circular."""
    monkeypatch.setattr(coverage, "adopted",
                        lambda *a, **kw: {"org/thing": {"served"}})
    seen(db, "org/thing", source="inspect")
    got = coverage.report(db, tmp_path)
    assert got["found"] == []
    assert got["holes"][0]["why"].startswith("only recorded by a tier")


def test_a_source_that_agreed_afterwards_did_not_lead_us_to_it(db, tmp_path,
                                                               monkeypatch):
    """A thing found a month after it was already running did not lead us to
    it. That is the difference between a source that works and one that
    eventually agrees."""
    monkeypatch.setattr(coverage, "adopted",
                        lambda *a, **kw: {"org/thing": {"measured"}})
    runs(tmp_path, "r", "image", {"org/thing": {"pass_rate": 1.0}},
         when=1_000_000.0)
    seen(db, "org/thing", source="reddit-sd-week", at=2_000_000.0)
    got = coverage.report(db, tmp_path)
    assert got["found"] == []
    assert [e["name"] for e in got["late"]] == ["org/thing"]
    assert got["late"][0]["days_late"] > 10


# --- the two sides spell things differently ------------------------------

def test_a_receipt_key_and_a_registry_id_are_the_same_thing(db, tmp_path,
                                                            monkeypatch):
    monkeypatch.setattr(coverage, "adopted",
                        lambda *a, **kw: {"parakeet-tdt-0.6b-v2": {"measured"}})
    seen(db, "mlx-community/parakeet-tdt-0.6b-v2", source="reddit-localllama-week")
    assert len(coverage.report(db, tmp_path)["found"]) == 1


def test_a_publisher_is_not_a_model(db, tmp_path, monkeypatch):
    """Matching on an owner segment would call every model under one org the
    same model."""
    monkeypatch.setattr(coverage, "adopted",
                        lambda *a, **kw: {"mlx-community/some-model": {"served"}})
    for other in ("mlx-community/a", "mlx-community/b", "mlx-community/c"):
        seen(db, other)
    got = coverage.report(db, tmp_path)
    assert [h["name"] for h in got["holes"]] == ["mlx-community/some-model"]


def test_a_base_model_is_not_the_requantisation_we_run(db, tmp_path,
                                                       monkeypatch):
    """`Qwen/Qwen2.5-7B` and `mlx-community/Qwen2.5-7B-Instruct-4bit` are a
    base and a requant of it, which are not the same thing to download or to
    run. Calling that a find credits a source with a surface it did not."""
    monkeypatch.setattr(coverage, "adopted", lambda *a, **kw: {
        "mlx-community/qwen2.5-7b-instruct-4bit": {"served"}})
    seen(db, "Qwen/Qwen2.5-7B")
    assert coverage.report(db, tmp_path)["found"] == []


# --- the unit is the model, not the key ----------------------------------

def test_a_voice_is_ours_to_pick_and_no_source_proposes_one():
    assert coverage.model_of("Kokoro-82M-bf16/af_sky", speech=True) == \
        "Kokoro-82M-bf16"
    assert coverage.model_of("Chatterbox-MLX/fleurs-fr-male-1", speech=True) == \
        "Chatterbox-MLX"


def test_an_engine_in_front_is_not_the_model():
    assert coverage.model_of("mflux/flux2-klein-4b-q8") == "flux2-klein-4b-q8"
    assert coverage.model_of("trace/mflux/flux2-klein-4b-q8") == \
        "flux2-klein-4b-q8"


def test_a_registry_id_keeps_its_org():
    """`mlx-community/X` IS how a registry names X, so the prefix stays."""
    assert coverage.model_of("mlx-community/parakeet-tdt-0.6b-v2") == \
        "mlx-community/parakeet-tdt-0.6b-v2"


def test_one_model_measured_across_voices_is_one_adopted_thing():
    """Counting five Kokoro voices as five adopted things overstates the holes
    fivefold."""
    got = coverage._collapse_variants({
        "kokoro-82m-bf16/af_sky": {"measured"},
        "kokoro-82m-bf16/am_adam": {"measured"},
        "kokoro-82m-bf16/bm_george": {"measured"}})
    assert list(got) == ["kokoro-82m-bf16"]


def test_an_owner_with_several_models_is_never_folded_into_one():
    got = coverage._collapse_variants(
        {"mlx-community/a": {"served"}, "mlx-community/b": {"served"}},
        orgs={"mlx-community"})
    assert set(got) == {"mlx-community/a", "mlx-community/b"}


# --- the report has to distinguish two opposite problems -----------------

def test_a_source_producing_plenty_nobody_ran_is_not_a_source_producing_nothing(
        db, tmp_path, monkeypatch):
    """Those want opposite fixes: one needs a better source, the other needs
    the ladder below it to turn."""
    monkeypatch.setattr(coverage, "adopted",
                        lambda *a, **kw: {"org/unseen": {"served"}})
    for i in range(4):
        seen(db, f"someone/proposal-{i}", source="reddit-sd-recap")
    got = coverage.report(db, tmp_path)
    assert got["proposed"] == {"reddit-sd-recap": 4}
    assert got["by_source"] == {}
