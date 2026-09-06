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
}
NEUTRAL_SYSTEM = "Answer directly and concisely."

# Root tags worth recovering, per modality.
ROOT_TAGS = {"svg": ("svg",), "web": ("html", "!doctype")}

_START_HINT = "is the gateway up? ./scripts/serve-gateway.sh"


class CompletionError(RuntimeError):
    """The gateway did not return usable text."""


def complete(prompt: str, model: str, gateway: str = DEFAULT_GATEWAY,
             modality: str = "", timeout: float = 180.0,
             temperature: float = 0.2, max_tokens: int = 4000) -> str:
    """Ask `model` for a completion. Returns the raw text, unrecovered."""
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": SYSTEM.get(modality, NEUTRAL_SYSTEM)},
            {"role": "user", "content": prompt},
        ],
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
    """Recovered artifact for a known modality; the raw text otherwise."""
    tags = ROOT_TAGS.get(modality)
    return extract(text, tags) if tags else text.strip()
