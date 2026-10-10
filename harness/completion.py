"""Text generation through the local gateway, and recovering the artifact.

SVG and web pages are language-model jobs, not separate engines, so both go
through here. The candidate is a gateway alias, which is the point of the
gateway: swapping what is under test costs a string.

The system prompts live in this package and not in the eval suite. If the eval
steers the model differently from the CLI, it is measuring a product that does
not ship.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import httpx

from harness import reasons
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
    # A typed decision: one letter per field, as JSON, so the letter tokens
    # carry the probabilities. #423.
    # A case carries its own system prompt; this one serves a delegated request. #654.
    "claims": ("You read a numbered chat transcript and list durable technical claims as JSON: "
               "c is a list of [user, claim, refs], refs being the line numbers that support it."),
    "decide": ("You answer with a single JSON object and nothing else. No "
               "prose, no markdown fences, no explanation. Each key is a field "
               "name and each value is the letter of one allowed choice."),
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
#   uv run python -m evals.run --modality svg --repeat 5 --candidates mlx-community/Qwen2.5-7B-Instruct-4bit
# against a run overriding sampling to turn it off.
#
# `extract` and `code` are deliberately absent. Extract pulls one fact out of
# a log and wants to be as close to deterministic as the sampler allows;
# making it stochastic to fix a drawing problem would be a plain downgrade.
SAMPLING = {
    "svg": {"temperature": 0.4, "repetition_penalty": 1.1},
    "web": {"temperature": 0.4, "repetition_penalty": 1.1},
    # The answer is the argmax; the distribution comes from the logprobs.
    "decide": {"temperature": 0.0},
    "claims": {"temperature": 0.0},
}

#: Chat-template kwargs per modality; decide reads one pass of option logprobs, so no reasoning. #311.
TEMPLATE = {"decide": {"enable_thinking": False}}

#: OpenAI's ceiling is 20 and mlx_lm.server's is 11. #423.
TOP_LOGPROBS = 10

# Root tags worth recovering, per modality.
ROOT_TAGS = {"svg": ("svg",), "web": ("html", "!doctype"), "code": ()}

_START_HINT = "is the gateway up? ./scripts/serve-gateway.sh"


#: The token budget and request timeout this harness chooses. #406.
MAX_TOKENS = 4000
TIMEOUT_S = 180.0
LOAD_TIMEOUT_S = 1800.0

#: Eval reply budget per lane; code serves a reasoning model, sized at Qwen3's documented output length. #628.
BUDGET = {"svg": MAX_TOKENS, "web": MAX_TOKENS, "code": 32768, "extract": MAX_TOKENS,
          "decide": MAX_TOKENS, "agent": 8000, "claims": 400}
#: The slowest decode a budget's timeout allows for, tokens per second. #628.
MIN_DECODE_TOK_S = 25.0


#: Budgets a laddered eval retries a cut-off reply at, smallest first. #668.
LADDER = {"svg": (4000, 16000), "web": (4000, 16000), "code": (4000, 16000, 32000, 65536),
          "extract": (4000, 16000), "decide": (4000, 16000), "agent": (8000, 32000),
          "claims": (400, 1600)}


def ladder(lane: str) -> tuple:
    """The default budget ladder for `lane`; () where no text is generated. #668."""
    return tuple(LADDER.get(lane, ()))


def budget(lane: str) -> int:
    """The reply budget an eval of `lane` runs at unless the run names one; 0 where no text is generated. #628."""
    return BUDGET.get(lane, 0)


def timeout_for(max_tokens: int) -> float:
    """A request timeout long enough to spend `max_tokens` at MIN_DECODE_TOK_S. #628."""
    return max(TIMEOUT_S, float(max_tokens) / MIN_DECODE_TOK_S)


class CompletionError(RuntimeError):
    """The gateway did not return usable text."""

    def __init__(self, detail: str, failure_class: str = "",
                 limit: tuple = ()):
        super().__init__(detail)
        # Set where the cause is known; "" leaves it to reasons.classify. #408.
        self.failure_class = failure_class
        self.limit = limit


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
             **kw) -> str:
    """Ask `model` for a completion. Returns the raw text, unrecovered."""
    return complete_with_usage(prompt, model, gateway, **kw)[0]


def complete_with_usage(prompt: str, model: str, gateway: str = DEFAULT_GATEWAY,
             modality: str = "", context: str = "", timeout: float = TIMEOUT_S,
             temperature: float | None = None, max_tokens: int = MAX_TOKENS,
             sampling: dict | None = None,
             template: dict | None = None) -> tuple[str, dict]:
    """As `complete()`, but also returns the server's token `usage`.

    The gateway reports usage on every completion and this was thrown away, so
    the text lanes measured latency and never throughput -- two candidates can
    share a median while one of them wrote three times as much. Absent usage is
    an empty dict, never an error: mlx_lm.server has answered without the block.
    """
    text, usage, _ = complete_with_logprobs(
        prompt, model, gateway, modality=modality, context=context,
        timeout=timeout, temperature=temperature, max_tokens=max_tokens,
        sampling=sampling, template=template)
    return text, usage


def complete_with_logprobs(prompt: str, model: str,
                           gateway: str = DEFAULT_GATEWAY, modality: str = "",
                           context: str = "", timeout: float = TIMEOUT_S,
                           temperature: float | None = None,
                           max_tokens: int = MAX_TOKENS,
                           sampling: dict | None = None,
                           template: dict | None = None,
                           top_logprobs: int = 0) -> tuple[str, dict, list]:
    """As complete_with_usage, plus the per-token logprobs when asked for and
    the server returns them (OpenAI shape: content[i].top_logprobs); else []."""
    got = complete_full(prompt, model, gateway, modality=modality,
                        context=context, timeout=timeout,
                        temperature=temperature, max_tokens=max_tokens,
                        sampling=sampling, template=template,
                        top_logprobs=top_logprobs)
    return got.text, got.usage, got.tokens


@dataclass
class Completion:
    text: str
    usage: dict = field(default_factory=dict)
    tokens: list = field(default_factory=list)
    #: ttft_s, first_reasoning_s, prefill_s; None where not observed. #468.
    timing: dict = field(default_factory=dict)
    #: The model the server says answered, which an alias hides; "" if unsaid.
    model: str = ""
    #: The whole assistant message, tool_calls included; chat() only. #474.
    message: dict = field(default_factory=dict)
    #: The server's finish_reason; "length" means the budget cut the reply off. #628.
    finish: str = ""


def complete_full(prompt: str, model: str, gateway: str = DEFAULT_GATEWAY,
                  modality: str = "", context: str = "",
                  timeout: float = TIMEOUT_S,
                  temperature: float | None = None,
                  max_tokens: int = MAX_TOKENS, sampling: dict | None = None,
                  template: dict | None = None, top_logprobs: int = 0,
                  stream: bool = False, system: str | None = None,
                  response_format: dict | None = None) -> Completion:
    """One completion with its timing. `stream` asks for SSE so the first
    content token can be timed; a server that answers whole leaves ttft_s None."""
    knobs = dict(SAMPLING.get(modality, {}))
    if temperature is not None:
        knobs["temperature"] = temperature
    if sampling:
        knobs.update(sampling)
    knobs.setdefault("temperature", DEFAULT_TEMPERATURE)

    system = system or SYSTEM.get(modality, NEUTRAL_SYSTEM)
    user = user_message(prompt, context)
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        **knobs,
    }
    template = template or TEMPLATE.get(modality)
    if template:
        payload["chat_template_kwargs"] = dict(template)
    if response_format:
        payload["response_format"] = dict(response_format)
    if top_logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = int(top_logprobs)
    # LiteLLM drops logprobs from a streamed reply, and the logprobs are the
    # product here, so a logprobs request is never streamed (RULE #426).
    if stream and not top_logprobs:
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
    try:
        r = _post(gateway, payload, timeout)
        if (400 <= getattr(r, "status_code", 200) < 500
                and _refuses_system_role(getattr(r, "text", ""))):
            # The model's chat template has no system role (Gemma 2, Mistral
            # v0.3): same words, one user turn. #305.
            payload["messages"] = [{"role": "user",
                                    "content": f"{system}\n\n{user}"}]
            r = _post(gateway, payload, timeout)
        for drop in ("stream_options", "stream"):
            # A server that refuses a streaming knob is asked again without it.
            if (drop in payload and 400 <= getattr(r, "status_code", 200) < 500
                    and "stream" in (getattr(r, "text", "") or "").lower()):
                payload.pop(drop)
                payload.pop("stream_options", None)
                r = _post(gateway, payload, timeout)
        r.raise_for_status()
        body = r.json()
        choices = body.get("choices") or []
        message = choices[0]["message"]
        served = str(body.get("model") or "")
        text = message.get("content")
        finish = str(choices[0].get("finish_reason") or "")
        usage = body.get("usage") or {}
        tokens = ((choices[0].get("logprobs") or {}).get("content") or []) \
            if top_logprobs else []
        timing = {"ttft_s": None, "first_reasoning_s": None,
                  **(getattr(r, "timing", None) or {}),
                  "prefill_s": prefill_s(body.get("timings"))}
    except httpx.TimeoutException as exc:
        raise CompletionError(f"timed out after {timeout}s", reasons.TIMEOUT,
                              ("timeout_s", timeout)) from exc
    except httpx.HTTPStatusError as exc:
        from harness import gateway_key
        status = exc.response.status_code
        if gateway_key.refused(status, exc.response.text, gateway_key.key()):
            raise CompletionError(
                f"gateway at {gateway} refused the key (HTTP {status}): "
                f"{gateway_key.HINT}", reasons.HARNESS_ERROR) from exc
        # LiteLLM explains itself in the body, not the status line.
        raise CompletionError(
            f"gateway returned HTTP {exc.response.status_code}: "
            f"{exc.response.text[:400]}") from exc
    except httpx.HTTPError as exc:
        raise CompletionError(
            f"gateway unreachable at {gateway}: {exc} ({_START_HINT})",
            reasons.REFUSED_BY_GATEWAY) from exc
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise CompletionError(f"malformed response: {exc}", reasons.CRASHED) from exc

    if text is None or not text.strip():
        # A THINKING MODEL that spent its whole budget reasoning returns null
        # content and a populated reasoning_content. Qwen3-8B and Qwen3-14B do
        # this; Qwen3-4B-Instruct-2507 and Qwen2.5 do not. Saying so beats a
        # TypeError from three frames away, or "empty completion" for a model
        # that in fact produced 4000 tokens of thought.
        reasoning = message.get("reasoning_content") or ""
        if reasoning:
            spent = reasoning_tokens(usage)
            raise CompletionError(
                f"{model} returned no answer: it spent the whole "
                f"{max_tokens}-token budget on reasoning ("
                + (f"{spent} tokens of reasoning, " if spent else "")
                + f"{len(reasoning)} characters). This is a hybrid "
                f"thinking model; use a non-thinking one for this lane, or "
                f"raise max_tokens.", reasons.TOKEN_BUDGET_EXHAUSTED,
                ("max_tokens", max_tokens))
        raise CompletionError("empty completion", reasons.CONTENT_FAILED)
    return Completion(text, usage,
                      list(tokens) if isinstance(tokens, list) else [], timing,
                      served, finish=finish)


def reasoning_tokens(usage) -> int | None:
    """Tokens the server says went to reasoning; all completion tokens when it gave no split."""
    usage = usage or {}
    details = usage.get("completion_tokens_details") or {}
    got = details.get("reasoning_tokens") or usage.get("completion_tokens")
    return int(got) if got else None


def prefill_s(timings) -> float | None:
    """llama-server's own prompt-processing time, in seconds; None elsewhere."""
    try:
        ms = (timings or {}).get("prompt_ms")
        return None if ms is None else round(float(ms) / 1000.0, 4)
    except (AttributeError, TypeError, ValueError):
        return None



#: What a chat template says when it has no system role. #305.
SYSTEM_ROLE_REFUSALS = ("system role not supported",
                        "conversation roles must alternate")


def _refuses_system_role(body: str) -> bool:
    low = (body or "").lower()
    return any(p in low for p in SYSTEM_ROLE_REFUSALS)


def _post(gateway: str, payload: dict, timeout: float):
    url = f"{gateway.rstrip('/')}/v1/chat/completions"
    from harness import gateway_key
    headers = gateway_key.headers()
    if not payload.get("stream"):
        return httpx.post(url, json=payload, timeout=timeout, headers=headers)
    started = time.perf_counter()
    with httpx.stream("POST", url, json=payload, timeout=timeout,
                      headers=headers) as r:
        if (r.status_code >= 400 or "text/event-stream"
                not in r.headers.get("content-type", "")):
            r.read()
            return r
        body, timing = assemble(r.iter_lines(), started, timeout)
    return Streamed(body, timing)


class Streamed:
    """An SSE reply assembled into the body a non-streaming request returns."""
    status_code = 200
    text = ""

    def __init__(self, body: dict, timing: dict):
        self.body, self.timing = body, timing

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self.body


def assemble(lines, started: float, timeout: float = TIMEOUT_S):
    """(body, timing) from OpenAI-style SSE lines; offsets from `started`."""
    content, reasoning, tokens = [], [], []
    calls: dict[int, dict] = {}
    usage, timings, finish, model = {}, None, None, None
    first: dict = {"ttft_s": None, "first_reasoning_s": None}
    for line in lines:
        now = time.perf_counter() - started
        if now > timeout:
            # The non-streaming limit, kept as a total so a slow writer still times out.
            raise httpx.ReadTimeout(f"stream ran past {timeout}s")
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        if chunk.get("error"):
            raise CompletionError(
                f"server error mid-stream: {str(chunk['error'])[:400]}")
        usage = chunk.get("usage") or usage
        model = chunk.get("model") or model
        timings = chunk.get("timings") or timings
        for ch in chunk.get("choices") or []:
            delta = ch.get("delta") or {}
            if delta.get("content"):
                content.append(delta["content"])
                if first["ttft_s"] is None:
                    first["ttft_s"] = round(now, 4)
            think = delta.get("reasoning_content") or delta.get("reasoning")
            if think:
                reasoning.append(think)
                if first["first_reasoning_s"] is None:
                    first["first_reasoning_s"] = round(now, 4)
            for part in delta.get("tool_calls") or []:
                _merge_call(calls, part)
                first.setdefault("first_tool_s", round(now, 4))
            tokens.extend((ch.get("logprobs") or {}).get("content") or [])
            finish = ch.get("finish_reason") or finish
    message = {"role": "assistant",
               "content": "".join(content) if content else None,
               "reasoning_content": "".join(reasoning) if reasoning else None}
    if calls:
        message["tool_calls"] = [calls[i] for i in sorted(calls)]
    body = {"choices": [{"message": message, "finish_reason": finish,
                         "logprobs": {"content": tokens} if tokens else None}],
            "usage": usage, "timings": timings, "model": model}
    return body, first


def _merge_call(calls: dict, part: dict) -> None:
    """Fold one streamed tool_calls fragment into the call at its index."""
    at = part.get("index")
    at = len(calls) if not isinstance(at, int) else at
    got = calls.setdefault(at, {"id": "", "type": "function",
                                "function": {"name": "", "arguments": ""}})
    if part.get("id"):
        got["id"] = part["id"]
    fn = part.get("function") or {}
    if fn.get("name"):
        got["function"]["name"] += fn["name"]
    if isinstance(fn.get("arguments"), str):
        got["function"]["arguments"] += fn["arguments"]
    elif fn.get("arguments") is not None:
        got["function"]["arguments"] = json.dumps(fn["arguments"])


def chat(messages: list, model: str, gateway: str = DEFAULT_GATEWAY,
         tools: list | None = None, timeout: float = TIMEOUT_S,
         max_tokens: int = MAX_TOKENS, sampling: dict | None = None,
         stream: bool = True) -> Completion:
    """One turn of a multi-turn conversation with OpenAI tools. #474.

    `text` may be empty when the turn is tool calls; `message` is the whole
    assistant message, tool_calls included. timing["first_tool_s"] is the
    first tool-call fragment.
    """
    knobs = {"temperature": DEFAULT_TEMPERATURE, **(sampling or {})}
    payload = {"model": model, "max_tokens": max_tokens,
               "messages": list(messages), **knobs}
    if tools:
        payload["tools"] = list(tools)
        payload["tool_choice"] = "auto"
    if stream:
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
    try:
        r = _post(gateway, payload, timeout)
        if (stream and 400 <= getattr(r, "status_code", 200) < 500
                and "stream" in (getattr(r, "text", "") or "").lower()):
            payload.pop("stream")
            payload.pop("stream_options")
            r = _post(gateway, payload, timeout)
        r.raise_for_status()
        body = r.json()
        message = dict((body.get("choices") or [])[0]["message"])
        usage = body.get("usage") or {}
        timing = {"ttft_s": None, "first_reasoning_s": None,
                  "first_tool_s": None,
                  **(getattr(r, "timing", None) or {}),
                  "prefill_s": prefill_s(body.get("timings"))}
    except httpx.TimeoutException as exc:
        raise CompletionError(f"timed out after {timeout}s", reasons.TIMEOUT,
                              ("timeout_s", timeout)) from exc
    except httpx.HTTPStatusError as exc:
        raise CompletionError(
            f"gateway returned HTTP {exc.response.status_code}: "
            f"{exc.response.text[:400]}") from exc
    except httpx.HTTPError as exc:
        raise CompletionError(
            f"gateway unreachable at {gateway}: {exc} ({_START_HINT})",
            reasons.REFUSED_BY_GATEWAY) from exc
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise CompletionError(f"malformed response: {exc}", reasons.CRASHED) from exc
    return Completion(message.get("content") or "", usage, [], timing,
                      model=str(body.get("model") or ""), message=message)


def artifact(text: str, modality: str) -> str:
    """Recovered artifact for a known modality; the raw text otherwise.

    `code` maps to an empty tag tuple rather than being absent: extract() with
    no root tags still unwraps a markdown fence, which is exactly what code
    needs and what a bare .strip() would leave in.
    """
    if modality not in ROOT_TAGS:
        return text.strip()
    return extract(text, ROOT_TAGS[modality])
