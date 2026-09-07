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
        import time

        started = time.monotonic()
        try:
            text, usage = completion.complete_with_usage(
                case.prompt, model=self.candidate, gateway=self.gateway,
                modality=case.modality, context=case.context,
                timeout=self.timeout)
        except completion.CompletionError as exc:
            # One dud must never abort a fifty-case run: every failure is a row.
            raise RunnerError(str(exc)) from exc
        # THROUGHPUT, not just latency. Two candidates can share a median while
        # one of them wrote three times as much, and a median alone cannot tell
        # a terse model from a fast one. Absent when the server reports no
        # usage -- MISSING rather than zero, because a zero would rank as the
        # slowest candidate rather than as an unknown.
        elapsed = time.monotonic() - started
        out = int(usage.get("completion_tokens") or 0)
        self.last_metrics = {}
        if out and elapsed > 0:
            self.last_metrics = {
                "completion_tokens": out,
                "tokens_per_s": round(out / elapsed, 1),
            }
        # Peak memory is not observable through an HTTP boundary; the server
        # holds the model. Reporting 0 is honest, and summarize() takes a max.
        return text, 0

    def extra_metrics(self) -> dict:
        return getattr(self, "last_metrics", {})
