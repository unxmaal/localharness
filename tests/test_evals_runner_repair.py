"""Generate, check, repair: the workflow the checkers were always able to drive.

Every checker in this repo is an automated verifier -- the eval EXECUTES
generated code, rasterizes SVG, renders HTML in Chrome, transcribes speech --
and not one of them has ever been fed back into generation. They score and
nothing else.

A repair loop is the cheapest workflow to add because the verifier is already
written and already trusted, and it is a WORKFLOW rather than a model: the same
model, asked again with the checker's complaint attached.
"""
import httpx
import pytest
import respx

from evals.core import Case
from evals.runners.repair import RepairRunner

GW = "http://127.0.0.1:4000"


def reply(text):
    return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})


BAD = "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 9 9'></svg>"
GOOD = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 9 9'>"
        "<circle cx='4' cy='4' r='3'/><rect x='0' y='0' width='2' height='2'/></svg>")


def case():
    return Case(id="gear", modality="svg", prompt="a gear",
                assertions={"min_shapes": 2})


@respx.mock
def test_a_first_attempt_that_passes_is_not_retried():
    """The loop must cost nothing when the model gets it right."""
    route = respx.post(f"{GW}/v1/chat/completions").mock(return_value=reply(GOOD))
    r = RepairRunner(GW, "local-large", attempts=3).run(case())
    assert r.passed
    assert len(route.calls) == 1


@respx.mock
def test_a_failure_is_retried_with_the_checker_s_complaint():
    route = respx.post(f"{GW}/v1/chat/completions").mock(
        side_effect=[reply(BAD), reply(GOOD)])
    r = RepairRunner(GW, "local-large", attempts=3).run(case())
    assert r.passed, r.detail
    assert len(route.calls) == 2
    second = route.calls[1].request.read().decode()
    # The complaint has to reach the model, or the second attempt is just
    # another sample.
    assert "draws nothing" in second.lower() or "shape" in second.lower()


@respx.mock
def test_the_previous_attempt_is_shown_to_the_model():
    """Asking again without the failed artifact is a re-roll, not a repair."""
    route = respx.post(f"{GW}/v1/chat/completions").mock(
        side_effect=[reply(BAD), reply(GOOD)])
    RepairRunner(GW, "local-large", attempts=2).run(case())
    assert "<svg" in route.calls[1].request.read().decode()


@respx.mock
def test_it_gives_up_after_the_attempt_budget():
    route = respx.post(f"{GW}/v1/chat/completions").mock(return_value=reply(BAD))
    r = RepairRunner(GW, "local-large", attempts=3).run(case())
    assert not r.passed
    assert len(route.calls) == 3


@respx.mock
def test_the_row_reports_what_the_workflow_cost():
    """A two-pass workflow costs twice, and a table without that column makes
    it look free next to a single-shot candidate."""
    respx.post(f"{GW}/v1/chat/completions").mock(
        side_effect=[reply(BAD), reply(GOOD)])
    r = RepairRunner(GW, "local-large", attempts=3).run(case())
    assert r.metrics["attempts"] == 2


@respx.mock
def test_the_candidate_name_says_it_is_a_workflow():
    """`local-large` and `repair/local-large` are different products and must
    not share a row."""
    assert RepairRunner(GW, "local-large").candidate == "repair/local-large"


@respx.mock
def test_the_loop_reports_the_tokens_it_spent_across_all_attempts():
    """Without this the repair row shows "-" for throughput next to a
    single-shot row that shows a number, and the two cannot be compared on the
    axis where the loop is actually expensive."""
    respx.post(f"{GW}/v1/chat/completions").mock(side_effect=[
        httpx.Response(200, json={"choices": [{"message": {"content": BAD}}],
                                  "usage": {"completion_tokens": 40}}),
        httpx.Response(200, json={"choices": [{"message": {"content": GOOD}}],
                                  "usage": {"completion_tokens": 60}}),
    ])
    r = RepairRunner(GW, "local-large", attempts=3).run(case())
    assert r.passed
    # 40 + 60: what the WORKFLOW cost, not what the last attempt cost.
    assert r.metrics["completion_tokens"] == 100
    assert r.metrics["tokens_per_s"] > 0
