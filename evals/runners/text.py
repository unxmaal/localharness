"""Runner for candidates reachable through the local gateway.

SVG and web generation are language-model jobs, not separate engines, so both
run through here. The candidate is a gateway alias, which is the whole point of
the gateway: swapping what is under test costs a string.

The request itself, and the system prompts steering it, live in
harness.completion and are shared with the CLI. If the eval steered the model
differently from the product, it would be measuring something that does not
ship.
"""
from __future__ import annotations

from harness import completion

from evals.core import Case
from evals.runners.base import BaseRunner, RunnerError


class CompletionRunner(BaseRunner):
    def __init__(self, gateway: str, candidate: str, timeout: float = 180.0):
        self.gateway = gateway.rstrip("/")
        self.candidate = candidate
        self.timeout = timeout

    def generate(self, case: Case):
        try:
            text = completion.complete(case.prompt, model=self.candidate,
                                       gateway=self.gateway,
                                       modality=case.modality,
                                       timeout=self.timeout)
        except completion.CompletionError as exc:
            # One dud must never abort a fifty-case run: every failure is a row.
            raise RunnerError(str(exc)) from exc
        # Peak memory is not observable through an HTTP boundary; the server
        # holds the model. Reporting 0 is honest, and summarize() takes a max.
        return text, 0
