"""Models are reported by their real ids, never only by a gateway alias. #670."""
import pytest
import yaml

from harness import gateway, models

MAC = gateway.REPO / "gateway" / "config.yaml"
CUDA = gateway.REPO / "gateway" / "config.cuda.yaml"

#: The nicknames this change renamed; each stays one release as a deprecated alias.
OLD_MAC = {"local-small": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
           "local-mid": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
           "local-large": "mlx-community/Qwen2.5-7B-Instruct-4bit",
           "q3-1.7b": "mlx-community/Qwen3-1.7B-4bit",
           "q3-4b": "mlx-community/Qwen3-4B-Instruct-2507-4bit",
           "q3-8b": "mlx-community/Qwen3-8B-4bit",
           "q3-14b": "mlx-community/Qwen3-14B-4bit",
           "q3-30b": "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit",
           "q3-coder": "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",
           "eval-4b": "Qwen3-4B-Instruct-2507-Q4_K_M",
           "eval-imajev-4b": "imajev-4b-Q8_0",
           "eval-7b": "Qwen2.5-7B-Instruct-Q4_K_M",
           "eval-12b": "google_gemma-3-12b-it-Q4_K_M"}


def _entries(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))["model_list"]


@pytest.mark.parametrize("path", [MAC, CUDA], ids=["mac", "cuda"])
def test_every_model_is_named_by_the_id_it_sends_upstream(path):
    for e in _entries(path):
        name = e["model_name"]
        if e.get(models.DEPRECATED_KEY) or name in models.STABLE_ALIASES:
            continue
        assert name == gateway.strip_provider(e["litellm_params"]["model"]), name


@pytest.mark.parametrize("path", [MAC, CUDA], ids=["mac", "cuda"])
def test_a_deprecated_alias_serves_exactly_what_its_new_name_serves(path):
    by = {e["model_name"]: e for e in _entries(path)}
    olds = [e for e in by.values() if e.get(models.DEPRECATED_KEY)]
    assert olds
    for e in olds:
        new = by[e[models.DEPRECATED_KEY]]
        assert not new.get(models.DEPRECATED_KEY), e["model_name"]
        assert e["litellm_params"] == new["litellm_params"], e["model_name"]


def test_every_old_nickname_is_kept_one_release_and_points_at_its_real_id():
    assert models.deprecated(MAC) == OLD_MAC


@pytest.mark.parametrize("path", [MAC, CUDA], ids=["mac", "cuda"])
def test_every_alias_in_the_config_resolves_to_a_real_id(path):
    names = {e["model_name"] for e in _entries(path)}
    for e in _entries(path):
        got = models.resolve(e["model_name"], config=path)
        assert got and got == gateway.strip_provider(e["litellm_params"]["model"])
        assert got not in models.deprecated(path)
        if got in names:
            assert models.resolve(got, config=path) == got


def test_a_name_the_gateway_does_not_know_is_its_own_id():
    assert models.resolve("mlx-community/Fresh-9B-4bit", config=MAC) == "mlx-community/Fresh-9B-4bit"
    assert models.resolve("llamacpp:Ornith-35B-Q4_K_M", config=MAC) == "llamacpp:Ornith-35B-Q4_K_M"


def test_the_reference_alias_resolves_to_the_frontier_model():
    assert models.resolve("cloud-opus", config=MAC) == "claude-opus-5-5"


def test_a_lane_alias_resolves_through_the_served_config(tmp_path):
    base = tmp_path / "config.yaml"
    base.write_text(MAC.read_text(encoding="utf-8"), encoding="utf-8")
    gateway.write_served(base, defaults={"claims": "Qwen2.5-7B-Instruct-Q4_K_M"})
    assert models.resolve("sohot-claims", config=base) == "Qwen2.5-7B-Instruct-Q4_K_M"
    assert models.resolve("sohot-code", config=base) == ""


def test_an_alias_is_any_gateway_name_that_is_not_its_own_upstream():
    assert models.is_alias("eval-7b", config=MAC)
    assert models.is_alias("cloud-opus", config=MAC)
    assert models.is_alias("sohot-code", config=MAC)
    assert not models.is_alias("Qwen2.5-7B-Instruct-Q4_K_M", config=MAC)
    assert not models.is_alias("mlx-community/Qwen3-4B-Instruct-2507-4bit", config=MAC)
    assert not models.is_alias("llamacpp:x", config=MAC)


@pytest.mark.parametrize("key,resolved,want", [
    ("Qwen2.5-7B-Instruct-Q4_K_M", {}, ("Qwen2.5-7B-Instruct-Q4_K_M", "")),
    ("sohot-code", {"sohot-code": "Ornith-35B-Q4_K_M"}, ("Ornith-35B-Q4_K_M", "sohot-code")),
    ("sohot-code", {}, ("", "sohot-code")),
    ("eval-7b", {}, ("Qwen2.5-7B-Instruct-Q4_K_M", "eval-7b")),
    ("best-of:3:q3-4b", {}, ("mlx-community/Qwen3-4B-Instruct-2507-4bit", "q3-4b")),
    ("mflux:z-image-turbo", {}, ("mflux:z-image-turbo", "")),
])
def test_a_row_names_the_model_and_the_alias_it_was_served_as(key, resolved, want):
    got = models.label(key, resolved, config=MAC)
    assert (got["model"], got["served_as"]) == want


def test_a_stored_row_labelled_only_by_a_lane_alias_is_flagged_unknown():
    assert models.display("sohot-web", {}, config=MAC) == "sohot-web (model unknown)"
    assert (models.display("sohot-web", {"sohot-web": "mlx-community/Qwen3-4B-Instruct-2507-4bit"},
                           config=MAC)
            == "mlx-community/Qwen3-4B-Instruct-2507-4bit (served as sohot-web)")
    assert models.display("eval-7b", {}, config=MAC) == "Qwen2.5-7B-Instruct-Q4_K_M (served as eval-7b)"


def test_a_deprecated_name_warns_and_names_its_replacement():
    why = models.deprecation("eval-7b", config=MAC)
    assert "eval-7b" in why and "Qwen2.5-7B-Instruct-Q4_K_M" in why and "deprecated" in why
    assert models.deprecation("Qwen2.5-7B-Instruct-Q4_K_M", config=MAC) == ""
    assert models.deprecation("repair:q3-4b", config=MAC).startswith("q3-4b ")


def test_no_typed_default_is_an_alias():
    from harness import cli
    for name in ("SVG", "WEB", "CODE", "EXTRACT", "DECIDE", "CLAIMS"):
        spec = getattr(cli, f"DEFAULT_{name}_MODEL")
        assert not models.is_alias(spec, config=MAC), spec
        assert models.resolve(spec, config=MAC) == spec


def test_a_lane_command_warns_on_a_deprecated_model(capsys):
    from harness import cli
    assert cli.lane_model("code", "q3-4b") == "q3-4b"
    assert "deprecated" in capsys.readouterr().err


def test_the_receipt_records_what_each_alias_resolved_to():
    from evals import run
    got = run.resolved_models({"eval-7b": "eval-7b", "Qwen2.5-7B-Instruct-Q4_K_M":
                               "Qwen2.5-7B-Instruct-Q4_K_M", "img": "mflux:z-image-turbo"},
                              config=MAC)
    assert got == {"eval-7b": "Qwen2.5-7B-Instruct-Q4_K_M",
                   "Qwen2.5-7B-Instruct-Q4_K_M": "Qwen2.5-7B-Instruct-Q4_K_M"}


def test_the_receipt_round_trips_its_resolved_models():
    from evals.core import Receipt
    r = Receipt(modality="code", case_ids=("a",), repeat=1, sampling={}, gateway="",
                resolved={"sohot-code": "Ornith-35B-Q4_K_M"})
    assert Receipt.from_dict(r.as_dict()).resolved == {"sohot-code": "Ornith-35B-Q4_K_M"}


def test_the_run_summary_names_the_model_behind_each_alias(capsys, monkeypatch):
    from evals import run
    monkeypatch.setenv("GATEWAY_CONFIG", str(MAC))
    summary = {"eval-7b": {"passed": 1, "total": 1, "pass_rate": 1.0, "median_s": 1.0, "metrics": {}},
               "sohot-code": {"passed": 0, "total": 1, "pass_rate": 0.0, "median_s": 1.0, "metrics": {}}}
    run.report(summary, resolved={"sohot-code": "Ornith-35B-Q4_K_M"})
    out = capsys.readouterr().out
    assert "eval-7b = Qwen2.5-7B-Instruct-Q4_K_M (served as eval-7b)" in out
    assert "sohot-code = Ornith-35B-Q4_K_M (served as sohot-code)" in out


def test_a_machine_serving_another_build_of_a_default_finds_it_by_in_place_of():
    assert (models.build_here("mlx-community/Qwen2.5-7B-Instruct-4bit", config=CUDA)
            == "qwen2.5-7b-instruct-q4_k_m")
    assert models.build_here("mlx-community/Qwen2.5-7B-Instruct-4bit", config=MAC) == \
        "mlx-community/Qwen2.5-7B-Instruct-4bit"
    assert models.build_here("llamacpp:x", config=CUDA) == "llamacpp:x"


def test_the_typed_text_defaults_are_this_machines_builds(monkeypatch):
    from harness import winners
    monkeypatch.setenv("GATEWAY_CONFIG", str(CUDA))
    got = winners.typed()
    names = {e["model_name"] for e in _entries(CUDA)}
    for lane in ("svg", "web", "code", "extract", "decide"):
        assert got[lane] in names, (lane, got[lane])


def _run(conn, path, key, spec, when):
    from harness import runs
    runs.record(conn, path, {"generated": when, "receipt": {"modality": "code", "tier": "measure"},
                             "environment": {}, "specs": {key: spec},
                             "rows": [{"case_id": "c", "candidate": key, "passed": 1, "seconds": 1.0}]})


def test_a_renamed_default_keeps_its_old_names_measured_row_until_it_has_its_own():
    from harness import candidates
    from harness import memory_store as ms
    conn = ms.connect()
    new_spec = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
    _run(conn, "runs/old", "q3-4b", "q3-4b", "2026-10-01T10:00:00")
    old = candidates.ensure(conn, "q3-4b", lane="code")
    assert candidates.served(conn, new_spec, lane="code", config=MAC) == old
    _run(conn, "runs/new", new_spec, new_spec, "2026-10-02T10:00:00")
    new = candidates.ensure(conn, new_spec, lane="code")
    assert new != old
    assert candidates.served(conn, new_spec, lane="code", config=MAC) == new


def test_a_method_over_a_gateway_model_downloads_nothing_as_its_nickname_did(monkeypatch):
    from harness import methods
    monkeypatch.setenv("GATEWAY_CONFIG", str(MAC))
    assert methods.weights_of("plan:mlx-community/Qwen3-4B-Instruct-2507-4bit") == ""
    assert methods.weights_of("plan:org/Unknown-7B") == "org/Unknown-7B"


def test_the_local_report_names_what_a_lane_serves_by_its_id(monkeypatch):
    from harness import report
    monkeypatch.setenv("GATEWAY_CONFIG", str(MAC))
    state = {"generated": 0.0, "machine": {"runtimes": [], "accelerator": "x", "kind": "unified",
                                           "total_gb": 1.0, "available_gb": 1.0},
             "funnel": [], "lanes": [{"lane": "claims", "wanted": True, "serves": "eval-7b",
                                      "adopted": False, "measured": "", "pass_rate": None,
                                      "median_s": None, "metrics": {}, "run": "",
                                      "age_days": None, "unverified": True, "stale": False}],
             "queue": {"waiting": 0, "rankable": 0, "by_lane": {}, "top": [],
                       "wanted_with_none": []}, "sources": []}
    page = report.render(state, before={})
    assert "Qwen2.5-7B-Instruct-Q4_K_M (served as eval-7b)" in page
