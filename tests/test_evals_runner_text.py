"""The text runner: any candidate reachable through the local gateway.

Covers SVG and web generation, which are LLM jobs rather than separate engines.
Mocked at the HTTP boundary so the suite's own tests never need a model.
"""
import httpx
import pytest
import respx

from evals.core import Case
from evals.runners.text import TextRunner

GATEWAY = "http://127.0.0.1:4000"
ENDPOINT = f"{GATEWAY}/v1/chat/completions"
SVG = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 2 2">'
       '<circle cx="1" cy="1" r="1"/></svg>')


def case(**over):
    base = dict(id="circle", modality="svg", prompt="Draw a red circle.",
                assertions={"min_shapes": 1})
    base.update(over)
    return Case(**base)


def reply(content):
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


@respx.mock
def test_runs_a_case_and_scores_it():
    respx.post(ENDPOINT).mock(return_value=reply(SVG))
    r = TextRunner(GATEWAY, "local-mid").run(case())
    assert r.passed
    assert r.candidate == "local-mid"
    assert r.case_id == "circle"
    assert r.seconds >= 0
    assert r.artifact == SVG


@respx.mock
def test_sends_the_candidate_as_the_model():
    route = respx.post(ENDPOINT).mock(return_value=reply(SVG))
    TextRunner(GATEWAY, "local-summarize").run(case())
    import json
    assert json.loads(route.calls.last.request.content)["model"] == "local-summarize"


@respx.mock
def test_a_bad_artifact_is_a_failed_row_not_an_exception():
    """One dud in a 50-case run must not abort the run."""
    respx.post(ENDPOINT).mock(return_value=reply("I cannot draw."))
    r = TextRunner(GATEWAY, "x").run(case())
    assert r.passed is False
    assert r.detail


@respx.mock
def test_gateway_down_is_a_failed_row_not_an_exception():
    respx.post(ENDPOINT).mock(side_effect=httpx.ConnectError("refused"))
    r = TextRunner(GATEWAY, "x").run(case())
    assert r.passed is False
    assert "unreachable" in r.detail.lower()


@respx.mock
def test_timeout_is_a_failed_row():
    respx.post(ENDPOINT).mock(side_effect=httpx.ReadTimeout("slow"))
    r = TextRunner(GATEWAY, "x").run(case())
    assert r.passed is False
    assert "timeout" in r.detail.lower() or "timed out" in r.detail.lower()


@respx.mock
def test_empty_completion_is_a_failed_row():
    respx.post(ENDPOINT).mock(return_value=httpx.Response(200, json={"choices": []}))
    r = TextRunner(GATEWAY, "x").run(case())
    assert r.passed is False


@respx.mock
def test_system_prompt_steers_toward_bare_output():
    """Models fence and chat by default; the prompt should ask them not to,
    even though the checker recovers it anyway."""
    route = respx.post(ENDPOINT).mock(return_value=reply(SVG))
    TextRunner(GATEWAY, "x").run(case())
    import json
    msgs = json.loads(route.calls.last.request.content)["messages"]
    assert msgs[0]["role"] == "system"
    assert "svg" in msgs[0]["content"].lower()
