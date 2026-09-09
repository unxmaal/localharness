"""Generate, check, repair -- a WORKFLOW candidate rather than a model.

Every checker in this repo is an automated verifier. The eval executes
generated code, rasterizes SVG, renders HTML in a browser, transcribes speech.
Not one of them has ever been fed back into generation: they score, and that is
all they do.

This closes that loop. Same model, asked again with the checker's own complaint
and its previous attempt attached. It is the cheapest workflow to add because
the verifier already exists and is already trusted, and it is the shape of
thing the suite could never express before -- `local-large` and
`repair/local-large` are different PRODUCTS and must not share a row.

WHAT IT COSTS IS PART OF THE RESULT. A three-attempt repair is up to three
generations, and a table that does not say so makes it look free beside a
single-shot candidate. The row reports `attempts`.
"""
from __future__ import annotations

from harness import completion

from evals.core import Case, score
from evals.runners.base import BaseRunner, RunnerError

#: How the complaint is handed back. Deliberately blunt: a small model given a
#: polite hint tends to produce another polite variation of the same mistake.
REPAIR_TEMPLATE = """Your previous attempt was rejected.

WHAT YOU PRODUCED:
{artifact}

WHY IT WAS REJECTED:
{reason}

Produce a corrected version. Output only the artifact, nothing else."""


class RepairRunner(BaseRunner):
    def __init__(self, gateway: str, model: str, attempts: int = 3,
                 timeout: float = 180.0):
        self.gateway = gateway.rstrip("/")
        self.model = model
        # More than three rarely helps and multiplies the cost; the point is to
        # measure whether the loop helps at all, not to grind.
        self.attempts = max(1, attempts)
        self.timeout = timeout
        self.candidate = f"repair/{model}"
        self.last_metrics: dict = {}

    def generate(self, case: Case):
        import time

        # perf_counter, not monotonic: `monotonic` is GetTickCount64 on
        # Windows and quantises to 15.6ms, which measured a 0.15s run as
        # 0.14 in 75 of 200 tries. This number is reported as a result and
        # divided into token counts, so its resolution is the measurement.
        started = time.perf_counter()
        spent = 0
        artifact = ""
        reason = ""
        for attempt in range(1, self.attempts + 1):
            prompt = case.prompt if attempt == 1 else REPAIR_TEMPLATE.format(
                artifact=artifact[:4000], reason=reason)
            try:
                raw, usage = completion.complete_with_usage(
                    prompt, model=self.model, gateway=self.gateway,
                    modality=case.modality,
                    context=case.context if attempt == 1 else "",
                    timeout=self.timeout)
            except completion.CompletionError as exc:
                raise RunnerError(str(exc)) from exc

            # ACROSS ALL ATTEMPTS. What the workflow cost, not what the last
            # attempt cost -- otherwise a three-attempt repair reports the same
            # token count as a single-shot candidate and looks free.
            spent += int(usage.get("completion_tokens") or 0)
            artifact = completion.artifact(raw, case.modality)
            self.last_metrics = self._metrics(attempt, spent, started)
            # Score with the SAME checker the eval will use, so the loop is
            # repairing against the real verdict and not a proxy for it.
            verdict = score(case, artifact)
            if verdict.passed:
                return artifact, 0
            reason = verdict.detail or "it did not meet the requirements"

        # Out of attempts: hand back the last try and let the eval judge it,
        # rather than inventing a failure the checker did not produce.
        return artifact, 0

    def _metrics(self, attempts: int, spent: int, started: float) -> dict:
        import time

        out: dict = {"attempts": attempts}
        elapsed = time.perf_counter() - started
        if spent and elapsed > 0:
            out["completion_tokens"] = spent
            out["tokens_per_s"] = round(spent / elapsed, 1)
        return out

    def extra_metrics(self) -> dict:
        return dict(self.last_metrics)
