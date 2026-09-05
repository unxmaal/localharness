"""Runner for candidates reachable through the local gateway.

SVG and web generation are language-model jobs, not separate engines, so both
run through here. The candidate is a gateway alias, which is the whole point of
the gateway: swapping what is under test costs a string.
"""
from __future__ import annotations

import time

import httpx

from evals.core import Case, Result, score

# Per-modality steering. The checkers recover fenced or chatty output anyway,
# but asking for the bare artifact makes the comparison about the artifact
# rather than about who ignores instructions most politely.
SYSTEM = {
    "svg": ("You output a single SVG document and nothing else. No prose, no "
            "markdown fences. Include an xmlns and a viewBox. Use vector "
            "shapes only, never an embedded raster image."),
    "web": ("You output a single complete HTML document and nothing else. No "
            "prose, no markdown fences. Start with <!doctype html>. The page "
            "must be self-contained: inline all CSS and JS, and do not "
            "reference any external URL."),
}


class TextRunner:
    def __init__(self, gateway: str, candidate: str, timeout: float = 180.0):
        self.gateway = gateway.rstrip("/")
        self.candidate = candidate
        self.timeout = timeout

    def run(self, case: Case) -> Result:
        system = SYSTEM.get(case.modality, "Answer directly and concisely.")
        payload = {
            "model": self.candidate,
            "temperature": 0.2,
            "max_tokens": 4000,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": case.prompt}],
        }

        started = time.monotonic()
        try:
            r = httpx.post(f"{self.gateway}/v1/chat/completions",
                           json=payload, timeout=self.timeout,
                           headers={"Authorization": "Bearer sk-local"})
            r.raise_for_status()
            choices = r.json().get("choices") or []
            artifact = choices[0]["message"]["content"] if choices else ""
        except httpx.TimeoutException:
            # One dud must never abort a fifty-case run: every failure is a row.
            return self._fail(case, started, f"timed out after {self.timeout}s")
        except httpx.HTTPError as exc:
            return self._fail(case, started, f"gateway unreachable: {exc}")
        except (KeyError, IndexError, ValueError) as exc:
            return self._fail(case, started, f"malformed response: {exc}")

        elapsed = time.monotonic() - started
        if not artifact.strip():
            return self._fail(case, started, "empty completion")

        scored = score(case, artifact)
        scored.candidate = self.candidate
        scored.seconds = round(elapsed, 3)
        scored.artifact = artifact
        return scored

    def _fail(self, case: Case, started: float, detail: str) -> Result:
        return Result(case.id, self.candidate, False,
                      round(time.monotonic() - started, 3), 0, detail)
