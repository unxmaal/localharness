"""What can this machine do that nobody has measured?

Every discovery pass in this project was a person or an agent grepping and
writing prose. That is stale the moment anything is installed, and it has missed
things for months: Qwen2.5 served four lanes without ever being compared to
anything, three gateway aliases had zero measurements between them, and the
image lane has used two of the ~20 generate entry points mflux ships.

A number in a document is wrong by the next commit. A command is right whenever
it is run.
"""
import json

import pytest

from harness import discover
from harness import machine as _mach
from harness.memory import Accelerator as _Accelerator


def test_a_capability_knows_whether_it_has_been_measured():
    c = discover.Capability(kind="model", name="q3-4b", lane="text",
                            source="gateway", how="--candidates q3-4b")
    assert c.measured is False


def test_gateway_aliases_are_read_from_the_config(tmp_path):
    (tmp_path / "config.yaml").write_text(
        "model_list:\n"
        "  - model_name: local-mid\n"
        "    litellm_params: {model: openai/x}\n"
        "  - model_name: q3-4b\n"
        "    litellm_params: {model: openai/y}\n", encoding="utf-8")
    names = {c.name for c in discover.gateway_aliases(tmp_path / "config.yaml")}
    assert names == {"local-mid", "q3-4b"}


def test_a_missing_gateway_config_is_not_an_error(tmp_path):
    """Discovery must degrade, never explode: it runs on machines that do not
    have everything."""
    assert discover.gateway_aliases(tmp_path / "nope.yaml") == []


def test_engines_come_from_the_installed_entry_points(tmp_path):
    b = tmp_path / "bin"
    b.mkdir()
    for n in ("mflux-generate-flux2", "mflux-generate-qwen", "mflux-concept",
              "python", "hf"):
        (b / n).touch()
    found = {c.name: c for c in discover.image_engines(b)}
    assert found["mflux-generate-flux2"].kind == "engine"
    assert found["mflux-generate-qwen"].kind == "engine"
    # `concept` produces an image from a reference rather than a prompt, so it
    # is a workflow primitive rather than an engine. It used to be dropped
    # entirely, which hid it.
    assert found["mflux-concept"].kind == "workflow"
    assert "python" not in found


def test_external_tools_report_whether_they_are_present(monkeypatch):
    monkeypatch.setattr(discover.shutil, "which",
                        lambda n: "/opt/homebrew/bin/" + n if n == "vtracer" else None)
    tools = {c.name: c for c in discover.external_tools()}
    assert tools["vtracer"].present is True
    assert tools["whisperkit-cli"].present is False


def test_measured_names_come_from_the_run_receipts(tmp_path, monkeypatch):
    run = tmp_path / "runs" / "20260907-0000-svg"
    run.mkdir(parents=True)
    (run / "results.json").write_text(json.dumps({
        "receipt": {"modality": "svg"},
        "summary": {"local-large": {"total": 3}, "trace/mflux/x": {"total": 3}}}), encoding="utf-8")
    monkeypatch.setattr(discover.paths, "runs", lambda: tmp_path / "runs")
    assert discover.measured() == {"local-large", "trace/mflux/x"}


def test_a_corrupt_results_file_does_not_stop_discovery(tmp_path, monkeypatch):
    run = tmp_path / "runs" / "bad"
    run.mkdir(parents=True)
    (run / "results.json").write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(discover.paths, "runs", lambda: tmp_path / "runs")
    assert discover.measured() == set()


def test_the_gap_is_what_exists_and_was_never_run(monkeypatch):
    monkeypatch.setattr(discover, "capabilities", lambda: [
        discover.Capability("model", "measured-one", "text", "gateway", "x"),
        discover.Capability("model", "never-run", "text", "gateway", "y"),
    ])
    monkeypatch.setattr(discover, "measured", lambda: {"measured-one"})
    gap = discover.gaps()
    assert [c.name for c in gap] == ["never-run"]


def test_a_capability_carries_the_command_that_would_measure_it():
    """A gap nobody knows how to close is a complaint, not a finding."""
    for c in discover.capabilities():
        assert c.how, f"{c.name} does not say how to measure it"


def test_receipts_are_found_at_any_depth(tmp_path, monkeypatch):
    """Runs nest: an archived batch is runs/legacy-logs/ev-extract/results.json,
    three levels down. Iterating only the top level found 12 names where 31
    receipts existed, and under-reporting sends someone to re-run work that was
    already done -- the dangerous direction for this tool to be wrong in."""
    deep = tmp_path / "runs" / "legacy" / "batch" / "ev-extract"
    deep.mkdir(parents=True)
    (deep / "results.json").write_text(json.dumps(
        {"summary": {"parakeet-tdt-0.6b-v2": {"total": 40}}}), encoding="utf-8")
    shallow = tmp_path / "runs" / "recent"
    shallow.mkdir(parents=True)
    (shallow / "results.json").write_text(json.dumps(
        {"summary": {"local-large": {"total": 3}}}), encoding="utf-8")
    monkeypatch.setattr(discover.paths, "runs", lambda: tmp_path / "runs")
    assert discover.measured() == {"parakeet-tdt-0.6b-v2", "local-large"}


# ---- looking outward -------------------------------------------------------
# Everything this project got wrong about candidate selection was the world
# moving while nothing here noticed: Qwen2.5 served four lanes for four months
# while Qwen3 shipped, and ComfyUI was dismissed in a paragraph.
#
# This asks REGISTRIES, not a language model. A model would produce plausible
# names for things that do not exist, and this repo has a rule about not
# asserting specifics from pretrained memory. An API answer is a fact with a
# URL and a date on it.

def test_external_candidates_come_from_a_registry_not_a_guess(monkeypatch):
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append((url, params))
        class R:
            status_code = 200
            def json(self):
                return [{"id": "mlx-community/Some-New-Model-4bit",
                         "downloads": 9999, "lastModified": "2026-09-01"}]
            def raise_for_status(self): pass
        return R()

    monkeypatch.setattr(discover.httpx, "get", fake_get)
    out = discover.external("text", limit=5)
    assert calls, "nothing was queried"
    assert "huggingface.co" in calls[0][0]
    assert out[0].name == "mlx-community/Some-New-Model-4bit"
    assert "huggingface.co" in out[0].source, "a claim must be checkable"


def test_an_external_candidate_says_how_to_measure_it(monkeypatch):
    monkeypatch.setattr(discover, "_hf_models",
                        lambda q, limit: [{"id": "mlx-community/X",
                                           "downloads": 1, "lastModified": "2026-09-01"}])
    for c in discover.external("stt", limit=3):
        assert c.how and "candidates" in c.how


def test_what_is_already_measured_is_not_proposed(monkeypatch):
    monkeypatch.setattr(discover, "_hf_models",
                        lambda q, limit: [
                            {"id": "mlx-community/parakeet-tdt-0.6b-v2",
                             "downloads": 1, "lastModified": "2026-01-01"},
                            {"id": "mlx-community/brand-new",
                             "downloads": 1, "lastModified": "2026-09-01"}])
    monkeypatch.setattr(discover, "measured",
                        lambda: {"parakeet-tdt-0.6b-v2"})
    names = [c.name for c in discover.external("stt", limit=5)]
    assert "mlx-community/brand-new" in names
    assert not any("parakeet-tdt-0.6b-v2" in n for n in names)


def test_being_offline_is_a_message_not_a_crash(monkeypatch):
    def boom(*a, **k):
        raise discover.httpx.ConnectError("no network")
    monkeypatch.setattr(discover.httpx, "get", boom)
    out = discover.external("text", limit=3)
    assert out == [], "offline discovery returns nothing rather than raising"


def test_results_are_dated_so_staleness_is_visible(monkeypatch):
    monkeypatch.setattr(discover, "_hf_models",
                        lambda q, limit: [{"id": "mlx-community/X",
                                           "downloads": 1,
                                           "lastModified": "2026-09-01"}])
    c = discover.external("text", limit=1)[0]
    assert "2026-09-01" in c.note, "an undated proposal cannot be judged stale"


def test_an_unknown_lane_is_refused_rather_than_queried_blindly():
    with pytest.raises(ValueError) as e:
        discover.external("telepathy")
    assert "telepathy" in str(e.value)


def test_measured_matching_does_not_fire_on_a_substring(monkeypatch):
    """A capability named `X` was treated as measured because some candidate
    row was called `Chatterbox-Multilingual-MLX-v2-Q8` and "X" appears inside
    "MLX". Substring matching silently marks things done that never ran, which
    is the direction that hides work rather than duplicating it."""
    monkeypatch.setattr(discover, "measured",
                        lambda: {"Chatterbox-Multilingual-MLX-v2-Q8/ref-1"})
    caps = [discover.Capability("model", "mlx-community/X", "tts", "s", "how")]
    assert discover.gaps(caps), "X was wrongly considered measured"


def test_a_real_match_still_counts(monkeypatch):
    monkeypatch.setattr(discover, "measured", lambda: {"parakeet-tdt-0.6b-v2"})
    caps = [discover.Capability("model", "mlx-community/parakeet-tdt-0.6b-v2",
                                "stt", "s", "how")]
    assert not discover.gaps(caps)


def test_a_workflow_prefix_still_matches_its_model(monkeypatch):
    """`trace/mflux/flux2-klein-4b-q8` in a results table means the engine
    behind it has been run."""
    monkeypatch.setattr(discover, "measured",
                        lambda: {"trace/mflux/flux2-klein-4b-q8"})
    caps = [discover.Capability("engine", "flux2-klein-4b-q8", "image", "s", "how")]
    assert not discover.gaps(caps)


def test_the_registry_is_asked_for_the_fields_we_print(monkeypatch):
    """The list endpoint omits lastModified unless asked, so every proposal
    printed "last modified ;" with a blank date -- and an undated proposal is
    exactly what this was built to avoid."""
    seen = {}

    def fake_get(url, params=None, timeout=None):
        seen.update(params or {})
        class R:
            status_code = 200
            def json(self): return []
            def raise_for_status(self): pass
        return R()

    monkeypatch.setattr(discover.httpx, "get", fake_get)
    discover._hf_models("q", 3)
    assert seen.get("full") or seen.get("expand"), "did not ask for the detail"


def test_a_missing_date_says_unknown_rather_than_nothing(monkeypatch):
    monkeypatch.setattr(discover, "_hf_models",
                        lambda q, limit: [{"id": "mlx-community/X",
                                           "downloads": 5}])
    monkeypatch.setattr(discover, "measured", lambda: set())
    assert "unknown" in discover.external("text", limit=1)[0].note.lower()


# ---- workflow primitives must not be hidden --------------------------------
# The first version dropped every entry point needing an input image, because
# they cannot answer a plain-prompt image case. That silently hid the most
# important gap in the project: mflux ships controlnet, depth, fill, redux,
# in-context, kontext and two upscalers -- the whole ComfyUI-style workflow
# vocabulary, natively in MLX -- and NONE of them has been measured.

def test_workflow_primitives_are_reported_not_dropped(tmp_path):
    b = tmp_path / "bin"
    b.mkdir()
    for n in ("mflux-generate-flux2", "mflux-generate-controlnet",
              "mflux-upscale-seedvr2", "mflux-generate-depth", "mflux-info"):
        (b / n).touch()
    found = {c.name: c for c in discover.image_engines(b)}
    assert "mflux-generate-flux2" in found
    assert found["mflux-generate-flux2"].kind == "engine"
    # Present, and marked as a different KIND so nobody drops one into the
    # plain image lane and reports a failure that means nothing.
    assert found["mflux-generate-controlnet"].kind == "workflow"
    assert found["mflux-upscale-seedvr2"].kind == "workflow"
    assert "mflux-info" not in found


def test_a_workflow_primitive_says_what_it_needs(tmp_path):
    b = tmp_path / "bin"
    b.mkdir()
    (b / "mflux-generate-controlnet").touch()
    c = discover.image_engines(b)[0]
    assert "input" in c.note.lower() or "runner" in c.note.lower()


def test_methods_lists_the_workflows_that_exist_not_a_stale_pair():
    """It used to hardcode two and cite #18 as the reason. #18 is closed."""
    from harness.stages import STAGE_ENTRY_POINTS
    names = {c.name for c in discover.methods()}
    assert {"llm", "trace", "repair"} <= names
    assert set(STAGE_ENTRY_POINTS) <= names


def test_a_broken_stage_carries_its_reason_and_is_not_a_gap():
    """BROKEN is a third state. `present` is about installation and `measured`
    is about history; upscale-seedvr2 is installed, HAS been measured (0/3,
    because it crashes), and must not read as either fine or absent.

    Needs mflux: the refusal is pinned to the versions it was measured broken
    against, so with mflux absent there is no version to match and nothing is
    blocked. That is the guard behaving correctly, not a failure.
    """
    from harness.stages import tool_versions
    tool_versions.cache_clear()
    if not tool_versions().get("mflux"):
        pytest.skip("mflux is not installed here")
    ms = {c.name: c for c in discover.methods()}
    seed = ms["upscale-seedvr2"]
    assert seed.blocked and "#27" in seed.blocked
    assert seed.present, "it is installed; it just cannot run"
    assert not ms["controlnet"].blocked


def test_an_implemented_primitive_is_a_candidate_not_a_gap(tmp_path):
    """Three of the nineteen have a runner; discover must stop calling them
    'no runner yet' or it reports the repo as it was months ago."""
    for n in ("mflux-generate-controlnet", "mflux-upscale-controlnet",
              "mflux-generate-depth"):
        (tmp_path / n).write_text("", encoding="utf-8")
    got = {c.name: c for c in discover.image_engines(tmp_path)}
    assert "--candidates controlnet:" in got["mflux-generate-controlnet"].how
    assert "no runner yet" not in got["mflux-upscale-controlnet"].how
    # Still a genuine gap, and it must keep reading as one.
    assert "no runner yet" in got["mflux-generate-depth"].how


def test_a_declined_tool_is_recorded_where_people_look():
    """Issue #20. The ComfyUI decision lived only in PLAN.md, so it kept being
    rediscovered. Same job the BROKEN state does for a broken stage."""
    # Pinned to the machine the argument was made for. Unpinned, this asserted
    # whatever the local machine is told, and the decision rests on mflux
    # shipping the workflows natively -- which is no argument on a box that has
    # neither mflux nor MLX.
    caps = {c.name: c for c in discover.not_adopted(_APPLE)}
    comfy = caps["ComfyUI"]
    assert comfy.kind == "decision"
    assert not comfy.present
    assert "#20" in comfy.blocked
    assert "mflux" in comfy.blocked


def test_declined_tools_appear_in_the_capability_list():
    assert "ComfyUI" in {c.name for c in discover.not_adopted(_APPLE)}


def test_a_declined_tool_is_not_a_gap():
    """It is not something you close by running a command."""
    gaps = {c.name for c in discover.gaps()}
    assert "ComfyUI" not in gaps


_APPLE = _mach.Machine(frozenset({"mlx", "cpu"}),
                       _Accelerator("unified", 32.0, 25.0, "Mac16,1"))
