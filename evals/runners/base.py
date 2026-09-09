"""The shared runner spine.

A runner's only real job is `generate`: turn a case into an artifact and say
what it cost. Everything after that -- timing, scoring, building the row -- is
identical for every modality, and must stay identical or the comparison is
between two rulers rather than two candidates.

That is not hypothetical. Timing, failure handling and Result construction were
copied between the text and process runners, and the copies drifted: the
process runner called the image checker directly and so lost every shared
assertion for image cases.
"""
from __future__ import annotations

import time

from evals.core import Case, Result, score


class RunnerError(RuntimeError):
    """A generation that failed for a reason worth putting in a row.

    Anything NOT raised as a RunnerError propagates: a bug in a runner must
    surface as a traceback, not as a quiet FAIL row that reads like the
    candidate's fault.
    """

    def __init__(self, detail: str, peak_kb: int = 0):
        super().__init__(detail)
        self.detail = detail
        # A crash after the model loaded is exactly when peak memory matters.
        self.peak_kb = peak_kb


class BaseRunner:
    def extra_metrics(self) -> dict:
        """Numbers the RUNNER measured, merged into the row alongside the
        checker's. Empty for runners that measure nothing extra."""
        return {}

    #: Name this runner reports in every row. Set by the subclass.
    candidate: str = ""

    def generate(self, case: Case):
        """Return (artifact, peak_kb). Raise RunnerError on a real failure.

        `artifact` is completion text for a text modality and the path to the
        produced file for a binary one; score() dispatches on the modality.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement generate()")

    def score_kwargs(self) -> dict:
        """Extra arguments for the checker.

        A checker for a binary artifact may need a model of its own -- speech
        needs a transcriber -- and it should use the same server the runner
        was pointed at rather than a module default.
        """
        return {}

    def run(self, case: Case) -> Result:
        # perf_counter, not monotonic: `monotonic` is GetTickCount64 on
        # Windows and quantises to 15.6ms, which measured a 0.15s run as
        # 0.14 in 75 of 200 tries. This number is reported as a result and
        # divided into token counts, so its resolution is the measurement.
        started = time.perf_counter()
        try:
            artifact, peak_kb = self.generate(case)
        except RunnerError as exc:
            return Result(case.id, self.candidate, False,
                          round(time.perf_counter() - started, 3),
                          exc.peak_kb, exc.detail)
        elapsed = time.perf_counter() - started

        row = score(case, artifact, **self.score_kwargs())
        row.candidate = self.candidate
        row.seconds = round(elapsed, 3)
        # A runner may have measured something the checker cannot see,
        # such as tokens/sec from the server's usage block.
        row.metrics = {**row.metrics, **self.extra_metrics()}
        row.peak_kb = peak_kb
        row.artifact = artifact if isinstance(artifact, str) else str(artifact)
        return row
