"""Text generation through the local gateway, and recovering the artifact.

SVG and web pages are language-model jobs, not separate engines, so both go
through here. The candidate is a gateway alias, which is the point of the
gateway: swapping what is under test costs a string.

The system prompts live in this package and not in the eval suite. If the eval
steers the model differently from the CLI, it is measuring a product that does
not ship.
"""
from __future__ import annotations

import httpx

from harness.checks.base import extract

DEFAULT_GATEWAY = "http://127.0.0.1:4000"

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
    "code": ("You output working code and nothing else. No prose, no "
             "explanation, no example usage. A single markdown fence is "
             "acceptable. Include any imports the code needs. Do not write "
             "tests; do not print anything."),
    # The lane exists so a small model can be handed a log or a file listing
    # instead of spending a large model's context on it. Its value is a short
    # exact answer that the caller can use without parsing, so the prompt is
    # blunt about that.
    "extract": ("You answer with the fact asked for and nothing else. No "
                "preamble, no explanation, no restating the question, no "
                "prose around the answer. If the answer is a number, a "
                "filename or a line, give exactly that. If the material does "
                "not contain the answer, reply: NOT FOUND"),
}
NEUTRAL_SYSTEM = "Answer directly and concisely."

DEFAULT_TEMPERATURE = 0.2

# Sampling, per modality, because the lanes want opposite things.
#
# The SVG lane's loudest failure is DEGENERATE REPETITION: the model emits a
# plausible <path>, and the highest-probability continuation is another one
# just like it, until the token budget runs out mid-attribute and leaves an
# unclosed document. `lh svg "a cartoon frog holding a coffee mug"` produced
# eighty near-identical paths and no frog.
#
# A repetition penalty is the standard lever for that. Its value here is NOT
# yet established: hand-run single samples suggested temperature was the
# culprit instead, and three-sample repeats contradicted that outright. This
# repo has a commit called "--repeat: one sample per prompt ranks noise" and
# an eval suite built for exactly this question, so the honest state is that
# the knob exists, is measurable, and has not been measured. Compare with:
#   uv run python -m evals.run --modality svg --repeat 5 --candidates local-large
# against a run overriding sampling to turn it off.
#
# `extract` and `code` are deliberately absent. Extract pulls one fact out of
# a log and wants to be as close to deterministic as the sampler allows;
# making it stochastic to fix a drawing problem would be a plain downgrade.
SAMPLING = {
    "svg": {"temperature": 0.4, "repetition_penalty": 1.1},
    "web": {"temperature": 0.4, "repetition_penalty": 1.1},
}

# Root tags worth recovering, per modality.
ROOT_TAGS = {"svg": ("svg",), "web": ("html", "!doctype"), "code": ()}

_START_HINT = "is the gateway up? ./scripts/serve-gateway.sh"


class CompletionError(RuntimeError):
    """The gateway did not return usable text."""


def user_message(prompt: str, context: str = "") -> str:
    """The material first, the instruction last.

    A small model that reads a long log and THEN the question does better than
    one that reads the question, works through forty lines, and has to
    remember what it was looking for.
    """
    if not context:
        return prompt
    return f"{context.rstrip()}\n\n---\n\n{prompt}"


def complete(prompt: str, model: str, gateway: str = DEFAULT_GATEWAY,
             modality: str = "", context: str = "", timeout: float = 180.0,
             temperature: float | None = None, max_tokens: int = 4000,
             sampling: dict | None = None) -> str:
    """Ask `model` for a completion. Returns the raw text, unrecovered.

    `sampling` overrides the per-modality defaults, so the eval can run the
    same case with a knob on and off and let the numbers decide.
    """
    knobs = dict(SAMPLING.get(modality, {}))
    if temperature is not None:
        knobs["temperature"] = temperature
    if sampling:
        knobs.update(sampling)
    knobs.setdefault("temperature", DEFAULT_TEMPERATURE)

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": SYSTEM.get(modality, NEUTRAL_SYSTEM)},
            {"role": "user", "content": user_message(prompt, context)},
        ],
        **knobs,
    }
    try:
        r = httpx.post(f"{gateway.rstrip('/')}/v1/chat/completions",
                       json=payload, timeout=timeout,
                       headers={"Authorization": "Bearer sk-local"})
        r.raise_for_status()
        choices = r.json().get("choices") or []
        text = choices[0]["message"]["content"]
    except httpx.TimeoutException as exc:
        raise CompletionError(f"timed out after {timeout}s") from exc
    except httpx.HTTPStatusError as exc:
        # LiteLLM explains itself in the body, not the status line.
        raise CompletionError(
            f"gateway returned HTTP {exc.response.status_code}: "
            f"{exc.response.text[:400]}") from exc
    except httpx.HTTPError as exc:
        raise CompletionError(
            f"gateway unreachable at {gateway}: {exc} ({_START_HINT})") from exc
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise CompletionError(f"malformed response: {exc}") from exc

    if not text or not text.strip():
        raise CompletionError("empty completion")
    return text



def artifact(text: str, modality: str) -> str:
    """Recovered artifact for a known modality; the raw text otherwise.

    `code` maps to an empty tag tuple rather than being absent: extract() with
    no root tags still unwraps a markdown fence, which is exactly what code
    needs and what a bare .strip() would leave in.
    """
    if modality not in ROOT_TAGS:
        return text.strip()
    return extract(text, ROOT_TAGS[modality])
