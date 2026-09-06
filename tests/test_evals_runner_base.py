"""The shared runner spine.

Timing, failure-to-Result, and scoring were copied between the text and process
runners, and the copies drifted: the process runner called the image checker
directly and so bypassed every shared assertion. This is the one place that
turns "what happened" into a comparable row.
"""
import time

import pytest

from evals.core import Case, Result
from evals.runners.base import BaseRunner, RunnerError

GOOD_SVG = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 2 2">'
            '<circle cx="1" cy="1" r="1"/></svg>')


class Stub(BaseRunner):
    """A runner whose only job is to hand back what it was constructed with."""

    def __init__(self, candidate="stub", artifact=GOOD_SVG, peak_kb=1234,
                 raises=None, delay=0.0):
        self.candidate = candidate
        self._artifact = artifact
        self._peak = peak_kb
        self._raises = raises
        self._delay = delay

    def generate(self, case):
        if self._delay:
            time.sleep(self._delay)
        if self._raises:
            raise self._raises
        return self._artifact, self._peak


def svg_case(**over):
    base = dict(id="c", modality="svg", prompt="a circle",
                assertions={"min_shapes": 1})
    base.update(over)
    return Case(**base)


def test_a_good_run_is_a_passing_row():
    r = Stub().run(svg_case())
    assert isinstance(r, Result)
    assert r.passed and r.detail == ""
    assert r.case_id == "c"


def test_every_row_carries_the_candidate_name():
    assert Stub(candidate="local-mid").run(svg_case()).candidate == "local-mid"


def test_timing_is_measured_by_the_base_not_by_each_runner():
    r = Stub(delay=0.2).run(svg_case())
    assert r.seconds >= 0.2


def test_peak_memory_survives_from_generate_to_the_row():
    assert Stub(peak_kb=9999).run(svg_case()).peak_kb == 9999


def test_shared_assertions_apply_whatever_the_runner_is():
    """The bug this exists to prevent: a runner that scores its own output
    quietly loses min_shapes/must_contain."""
    r = Stub().run(svg_case(assertions={"min_shapes": 5}))
    assert not r.passed and "5" in r.detail


def test_a_runner_error_is_a_failed_row_not_an_exception():
    """One dud must never abort a fifty-case run: every failure is a row."""
    r = Stub(raises=RunnerError("gateway unreachable")).run(svg_case())
    assert not r.passed
    assert r.detail == "gateway unreachable"
    assert r.candidate == "stub" and r.case_id == "c"


def test_a_failed_row_still_reports_the_time_it_burned():
    r = Stub(delay=0.15, raises=RunnerError("boom")).run(svg_case())
    assert r.seconds >= 0.15


def test_a_runner_error_can_carry_the_memory_it_measured_before_failing():
    """A crash after the model loaded is exactly when peak matters."""
    r = Stub(raises=RunnerError("exit 137", peak_kb=31_000_000)).run(svg_case())
    assert r.peak_kb == 31_000_000


def test_an_unexpected_exception_is_not_swallowed():
    """A bug in a runner must surface, not become a quiet FAIL row that reads
    like the candidate's fault."""
    with pytest.raises(ZeroDivisionError):
        Stub(raises=ZeroDivisionError()).run(svg_case())


def test_the_artifact_is_recorded_on_the_row():
    assert Stub().run(svg_case()).artifact == GOOD_SVG


def test_generate_must_be_implemented():
    class Half(BaseRunner):
        candidate = "half"
    with pytest.raises(NotImplementedError):
        Half().run(svg_case())
