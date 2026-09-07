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
        "    litellm_params: {model: openai/y}\n")
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
    names = {c.name for c in discover.image_engines(b)}
    assert "mflux-generate-flux2" in names
    assert "mflux-generate-qwen" in names
    # Not a generator; listing it would invite someone to evaluate it.
    assert "mflux-concept" not in names
    assert "python" not in names


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
        "summary": {"local-large": {"total": 3}, "trace/mflux/x": {"total": 3}}}))
    monkeypatch.setattr(discover.paths, "runs", lambda: tmp_path / "runs")
    assert discover.measured() == {"local-large", "trace/mflux/x"}


def test_a_corrupt_results_file_does_not_stop_discovery(tmp_path, monkeypatch):
    run = tmp_path / "runs" / "bad"
    run.mkdir(parents=True)
    (run / "results.json").write_text("{not json")
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
        {"summary": {"parakeet-tdt-0.6b-v2": {"total": 40}}}))
    shallow = tmp_path / "runs" / "recent"
    shallow.mkdir(parents=True)
    (shallow / "results.json").write_text(json.dumps(
        {"summary": {"local-large": {"total": 3}}}))
    monkeypatch.setattr(discover.paths, "runs", lambda: tmp_path / "runs")
    assert discover.measured() == {"parakeet-tdt-0.6b-v2", "local-large"}
