"""Issue #53/#148: the rung between a ranked queue and a measurement."""
import json

import pytest

from harness import screen
from harness import memory_store as ms
from harness.memory_store import Seen


def rows(*specs):
    return [{"name": n, "lane": lane} for n, lane in specs]


# --- what it would do, and what stands in the way ------------------------

def test_a_candidate_whose_weights_are_absent_is_waiting_not_failed():
    """A SCREEN NEVER DOWNLOADS. Fetching is its own step with its own disk
    budget, and a tier that quietly pulls gigabytes because something ranked
    well is how a laptop fills up overnight."""
    # The name's tail must be a family mflux serves. `org/m` was not, so the
    # row is now correctly no-runner and this test asserted `waiting` on a
    # candidate the screen could never have run.
    got = screen.plan(rows(("org/z-image-turbo-4bit", "image")),
                      missing=lambda n: [n])[0]
    assert got["state"] == screen.WAITING
    assert "lh fetch" in got["why_not"]


def test_a_lane_with_no_runner_is_named_rather_than_dropped():
    """A queue that silently drops what it cannot run looks like a queue that
    ran out."""
    got = screen.plan(rows(("org/v", "video")), missing=lambda n: [])[0]
    assert got["state"] == screen.NO_RUNNER
    assert "video" in got["why_not"]


def test_a_candidate_with_no_lane_says_so_in_its_own_terms():
    got = screen.plan(rows(("org/x", "")), missing=lambda n: [])[0]
    assert got["state"] == screen.NO_RUNNER
    assert "no lane" in got["why_not"]


def test_a_ready_candidate_gets_the_spec_its_lane_actually_takes():
    """The specs are the ones evals.run already parses. A second spelling of
    the candidate language is a defect this project has filed twice."""
    from evals.run import kind_of
    for lane, expect in (("image", "process"), ("stt", "stt"), ("tts", "tts"),
                         ("code", "gateway")):
        spec = screen.candidate_for(lane, "org/m")
        assert spec, lane
        assert kind_of(spec) == expect, (lane, spec)


def test_the_command_is_one_case_one_repeat_and_no_metrics():
    """A screen asks whether it ran. A quality metric here would invite ranking
    a screen against a measurement, which is two different exams."""
    row = screen.plan(rows(("org/m", "image")), missing=lambda n: [])[0]
    argv = screen.argv(row)
    assert "--screen" in argv
    assert argv[argv.index("--repeat") + 1] == "1"
    assert "--adherence" not in argv


# --- turning a run into a verdict ----------------------------------------

def test_a_screen_that_did_not_run_is_broken_and_terminal():
    got, why = screen.outcome(1, None)
    assert got == "broken"
    assert got in ms.TERMINAL, "a thing that does not run is answered"


def test_a_screen_that_ran_and_passed_nothing_is_broken():
    """seedvr2 crashed 0/3 and local-small never closed a tag 0/9. Exiting zero
    is not the same as working."""
    assert screen.outcome(0, {"c": {"total": 3, "passed": 0}})[0] == "broken"


def test_a_screen_that_passed_is_screened_and_not_terminal():
    got, _ = screen.outcome(0, {"c": {"total": 2, "passed": 2}})
    assert got == "screened"
    assert got not in ms.TERMINAL, "it ran; every measurement is still ahead"


def test_screened_is_a_verdict_the_store_recognises(tmp_path):
    """It has been in the vocabulary since the store was built and nothing has
    ever written one."""
    conn = ms.connect(tmp_path / "d.db")
    ms.record(conn, Seen(name="org/m", source="recap", resolved="org/m"))
    ms.decide(conn, "org/m", "screened", tier=ms.SCREEN, detail="1 passed")
    row = conn.execute("SELECT outcome, tier FROM verdicts").fetchone()
    conn.close()
    assert (row["outcome"], row["tier"]) == ("screened", "screen")


# ---- ready means loadable, not merely present (issue #196) -----------------

def test_a_model_whose_config_names_an_absent_repo_is_not_ready():
    """Marvis-AI's 8-bit MLX repo is complete -- every symlink resolving, no
    .incomplete files -- and names a tokenizer in a DIFFERENT repo that is not
    on disk. `ready` on that cost a real run to discover."""
    got = screen.plan(rows(("org/m", "tts")),
                      missing=lambda n: ["org/tokenizer"])[0]
    assert got["state"] == screen.WAITING
    assert "org/tokenizer" in got["why_not"], "say WHAT to fetch"


def test_a_model_missing_only_itself_reads_as_a_plain_fetch():
    """The ordinary case must keep its ordinary wording, or every
    waiting-on-fetch row starts shouting about dependencies."""
    got = screen.plan(rows(("org/m", "tts")), missing=lambda n: [n])[0]
    assert got["state"] == screen.WAITING
    assert got["why_not"] == "weights are not on disk; lh fetch --run"


def test_a_model_with_everything_present_is_ready():
    """The negative half: a closure check that never passes screens nothing."""
    got = screen.plan(rows(("org/m", "tts")), missing=lambda n: [])[0]
    assert got["state"] == screen.READY


def test_a_lane_with_no_runner_is_not_asked_about_weights():
    """Ordering matters: a missing runner is the answer regardless of disk, and
    asking the cache first would touch the filesystem for every unrunnable row."""
    asked = []

    def missing(n):
        asked.append(n)
        return []

    got = screen.plan(rows(("org/x", "")), missing=missing)[0]
    assert got["state"] == screen.NO_RUNNER
    assert asked == [], "no runner, so the cache was never consulted"


# ---- an adapter is not a candidate (2026-09-18) ---------------------------

def test_a_lora_is_refused_before_anything_downloads():
    """Every candidate in the image queue was a style LoRA. `mflux:org/style`
    downloads gigabytes and then fails, recording a verdict that says nothing
    about the thing."""
    assert screen.is_attachment("text-to-image; diffusers; tagged lora") == "lora"
    assert screen.candidate_for("image", "org/style",
                                "tagged text-to-image, lora") == ""


def test_a_comfyui_node_pack_is_refused():
    assert screen.is_attachment("served by minimax-h3; tagged comfyui") == "comfyui"


def test_a_real_model_is_still_a_candidate():
    """The negative half. A filter that refuses everything screens nothing."""
    assert screen.is_attachment("task text-to-image; served by mlx; "
                                "tagged mlx, mflux, text-to-image") == ""
    assert screen.candidate_for("image", "org/base",
                                "tagged mlx, mflux") == "mflux:org/base"


def test_a_candidate_with_no_description_is_not_assumed_to_be_an_adapter():
    """Most rows carry no description. Treating silence as an adapter would
    empty the queue."""
    assert screen.is_attachment("") == ""
    assert screen.candidate_for("image", "org/x") == "mflux:org/x"


def test_an_enumerated_exception_survives():
    """`lora-ready` describes a base model that ACCEPTS adapters. Matching the
    substring blindly would refuse exactly the models worth having."""
    assert screen.is_attachment("a lora-ready base model") == ""


def test_the_state_says_why_rather_than_no_runner():
    """`no runner for the image lane` is false and sends the reader looking for
    a missing runner. The lane has one; this is not a model."""
    got = screen.plan([{"name": "org/style", "lane": "image",
                        "description": "tagged lora"}],
                      missing=lambda n: [])[0]
    assert got["state"] == screen.NO_RUNNER
    assert "attaches to a model" in got["why_not"]


# ---- the screen must report what it measured (2026-09-18) -----------------

def test_a_passing_screen_is_screened_not_broken():
    """The summary spells it `passed`. outcome() read `pass`, so every run
    scored 0 and a candidate that passed every case was recorded `broken` --
    which is TERMINAL. The tier reported the opposite of what it measured."""
    got, why = screen.outcome(0, {"m": {"total": 1, "passed": 1,
                                        "pass_rate": 1.0}})
    assert got == "screened"
    assert "1 case" in why


def test_a_failing_screen_is_still_broken():
    """The negative half. A tier that never says broken screens nothing out."""
    assert screen.outcome(0, {"m": {"total": 3, "passed": 0}})[0] == "broken"


def test_a_refused_request_is_not_the_candidates_failure():
    """An HTTP 400 from the gateway's alias table stopped Qwen3-8B-4bit before
    a token was generated, and it was recorded `broken`, which is terminal."""
    got, why = screen.outcome(1, None,
                              "gateway returned HTTP 400: Invalid model name")
    assert got == "queued", "not terminal: the candidate never ran"
    assert "says nothing about the candidate" in why


def test_a_real_failure_is_not_excused_as_a_refusal():
    """The other half. Treating every non-zero exit as a harness problem would
    mean nothing is ever screened out."""
    assert screen.outcome(1, None, "Traceback: ValueError")[0] == "broken"


def test_an_alias_goes_through_the_gateway(tmp_path):
    cfg = tmp_path / "g.yaml"
    cfg.write_text("model_list:\n  - model_name: q3-4b\n    litellm_params:\n"
                   "      model: openai/org/x\n      api_base: http://up/v1\n",
                   encoding="utf-8")
    assert screen.routed_gateway("q3-4b", cfg) == ""


def test_a_repo_id_goes_to_the_upstream(tmp_path):
    """LiteLLM validates the model name against its aliases and answers 400 for
    anything else. mlx_lm.server treats it as a live repo id and swaps to it,
    so a discovered candidate has to reach the upstream directly."""
    cfg = tmp_path / "g.yaml"
    cfg.write_text("model_list:\n  - model_name: q3-4b\n    litellm_params:\n"
                   "      model: openai/org/x\n      api_base: http://up/v1\n",
                   encoding="utf-8")
    # The `--gateway` FORM, without the /v1 the config carries: evals.run
    # appends its own path, and passing the config's base made every request
    # /v1/v1/chat/completions and 404. Stripping used to happen in argv() and
    # nowhere else, so the measure tier got the wrong one. #223.
    assert screen.routed_gateway("org/discovered", cfg) == "http://up"
    assert screen.gateway_routes(cfg)[1] == "http://up/v1", (
        "the config's own base keeps its /v1; only the argument drops it")


def test_a_missing_config_routes_nowhere_rather_than_guessing(tmp_path):
    assert screen.routed_gateway("org/x", tmp_path / "absent.yaml") == ""


def test_the_fixture_key_is_the_one_the_run_writes():
    """The tests above used `pass` and so did outcome(), so they agreed with
    each other and with nothing else. A hand-made fixture that never meets the
    real writer confirms whatever the code already does."""
    from evals.core import Result, summarize

    got = summarize([Result("case", "cand", True, 1.0, 0, "")])
    assert "passed" in got["cand"], "outcome() reads this key"
    assert "pass" not in got["cand"], "and there is no bare `pass` to read"
