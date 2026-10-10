"""A text subtask handed to a lane's adopted local model. #475.

The MCP server calls this in-process rather than shelling out to `soh`, because
the caller wants the first-token time and the model that answered, which the
CLI does not print. It stays the same product by sharing the route: the lane
commands resolve their model through route() here too.
"""
from __future__ import annotations

import json
import os
import statistics
import time
import urllib.request

from harness import completion, context, exclusive, gateway, reasons, serving

#: Lanes a delegated completion may name: the ones the gateway aliases as sohot-<lane>.
LANES = gateway.TEXT_LANES
MAX_PROMPT_CHARS = 100_000
#: The max_tokens cap when the served context is unknown.
MAX_TOKENS = 8192
#: A rough floor on characters per prompt token, so the estimate errs high.
CHARS_PER_TOKEN = 3
TIMEOUT_S = 300.0
#: Seconds to ask a server what it holds; an answer that late counts as not resident.
RESIDENCY_TIMEOUT_S = 2.0


class Refused(ValueError):
    """A request this module will not send, said before anything is loaded."""


def lane_model(lane: str, chosen: str | None = None) -> str:
    """-m when given, else the lane's adopted model here, else its typed constant. #297."""
    from harness import adopt, models, winners
    spec = chosen or adopt.default_for(lane, winners.typed().get(lane, ""))
    models.warn(spec)
    return spec


def route(lane: str, chosen: str | None = None, via: str = "") -> tuple[str, serving.Route]:
    """(spec, Route) for a text lane, as every lane command resolves it; a method routes as its base. #581."""
    from harness import methods
    spec = lane_model(lane, chosen)
    if methods.is_method(spec) and not methods.text_method(spec):
        raise Refused(f"the {lane} lane serves {spec}, which draws with an image engine "
                      f"rather than answering in text; run `soh svg` for it")
    return spec, serving.route(methods.base_of(spec) or spec, via)


def generate(spec: str, where: serving.Route, lane: str, prompt: str,
             call) -> tuple[str, dict, list]:
    """(text, cost, replies): `spec` served over `where`, each call made by
    call(prompt, model=, gateway=, modality=, sampling=) -> Completion. #581."""
    from harness import methods
    replies: list = []

    def ask(text: str, modality: str) -> str:
        got = call(text, model=where.model, gateway=where.base, modality=modality,
                   sampling=where.sampling or None)
        replies.append(got)
        return got.text
    text, cost = methods.run(spec, lane, prompt, ask)
    return text, cost, replies


def check_lane(lane: str) -> str:
    lane = (lane or "").strip().lower()
    if lane not in LANES:
        raise Refused(f"lane {lane!r} is not a text lane; delegation serves "
                      f"{', '.join(LANES)}. image and video go on the work queue.")
    return lane


def busy() -> str:
    """What holds the machine lock, or "" when nobody does. Never waits."""
    if os.environ.get(exclusive.HELD_ENV) == "1":
        return ""
    fd = os.open(exclusive.lock_path(), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        if exclusive._take(fd):
            exclusive._release(fd)
            return ""
    finally:
        os.close(fd)
    return exclusive.describe(exclusive.holder())


def _preflight(*texts: str, max_tokens: int = 1) -> None:
    size = sum(len(t or "") for t in texts)
    if size > MAX_PROMPT_CHARS:
        raise Refused(f"prompt is {size} characters, over the {MAX_PROMPT_CHARS} "
                      f"cap; pass less material or summarise it first")
    if int(max_tokens) < 1:
        raise Refused(f"max_tokens must be at least 1, not {max_tokens}")


def cap(ctx: int | None, *texts: str) -> int:
    """The largest max_tokens a request may ask for: the served context less the prompt. #590."""
    if ctx is None:
        return MAX_TOKENS
    chars = sum(len(t or "") for t in texts)
    return max(0, ctx - -(-chars // CHARS_PER_TOKEN))


def _budget(spec: str, max_tokens: int, *texts: str) -> None:
    ctx = context.served_ctx(spec)
    most = cap(ctx, *texts)
    if int(max_tokens) > most:
        where = (f"the {ctx}-token context {spec} is served at, less the prompt"
                 if ctx else f"the cap while {spec}'s served context is unknown")
        raise Refused(f"max_tokens must be 1..{most} ({where}), not {max_tokens}")


def _call_label(spec: str, n: int, modality: str) -> str:
    """Which call of a method this is, e.g. "call 2 of plan:<model> (the answer)"; "" for a plain model."""
    from harness import methods
    try:
        parsed = methods.parse(spec)
    except ValueError:
        parsed = None
    if parsed is None:
        return ""
    if parsed.method.name == "plan":
        what = "the answer" if modality else "the plan"
    else:
        what = f"sample {n}"
    return f"call {n} of {spec} ({what})"


def template(lane: str, thinking: bool | None) -> dict | None:
    """The lane's chat-template kwargs, with enable_thinking set when the caller says. #590."""
    kwargs = dict(completion.TEMPLATE.get(lane) or {})
    if thinking is not None:
        kwargs["enable_thinking"] = bool(thinking)
    return kwargs or None


def admit(holder: str, resident: bool, level: int | None) -> tuple[bool, str]:
    """(ok, why): may a delegated call go now? Only a model load needs the lock. #588."""
    from harness.pressure import NORMAL
    if not holder:
        return True, ""
    if not resident:
        return False, (f"this machine is busy with {holder}; answering would load the "
                       f"model under its lock, and delegation does not wait: retry "
                       f"when it finishes")
    if level is not None and level > NORMAL:
        return False, (f"this machine is busy with {holder} and memory pressure is "
                       f"level {level}; delegation does not wait: retry when it finishes")
    return True, f"the model is resident, so nothing loads beside {holder}"


def running_job(conn=None) -> str:
    """The queued job running here, with how long earlier runs of it took, or ""."""
    from harness import workqueue as wq
    try:
        job = wq.current(conn)
        took = wq.durations(job["title"], conn) if job else []
    except Exception:  # noqa: BLE001
        return ""
    if job is None:
        return ""
    started = wq._epoch(job.get("started"))
    ran = max(0.0, time.time() - started) / 60 if started else 0.0
    head = f"job {job['id']} ({job['title']}), running {ran:.0f} min"
    if not took:
        return f"{head}, no earlier run to estimate from"
    typical = statistics.median(took) / 60
    left = typical - ran
    if left <= 0:
        return f"{head}, past the {typical:.0f} min earlier runs of it took"
    return f"{head}, about {left:.0f} min left by earlier runs of it"


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=RESIDENCY_TIMEOUT_S) as r:
        return json.loads(r.read() or b"{}")


def upstream(where: serving.Route, config=None) -> tuple[str, str]:
    """(server base, model name) that answers `where`, a gateway alias resolved; ("", "") if unknown."""
    from harness.completion import DEFAULT_GATEWAY
    base = where.base.rstrip("/").removesuffix("/v1")
    if base != DEFAULT_GATEWAY:
        return base, where.model
    for entry in gateway.load(config).get("model_list") or []:
        if str(entry.get("model_name", "")).strip().lower() == where.model.lower():
            params = entry.get("litellm_params") or {}
            return (str(params.get("api_base", "")).rstrip("/").removesuffix("/v1"),
                    gateway.strip_provider(str(params.get("model", ""))))
    return "", ""


def loaded(where: serving.Route, get=None, config=None) -> bool:
    """True only when the server behind `where` holds its model now; unknown is False."""
    from harness import mlx_server, router
    get = get or _get_json
    base, model = upstream(where, config)
    if not model:
        return False
    try:
        if base in (serving.LLAMACPP_URL, router.url()):
            for m in get(base + "/models").get("data") or []:
                status = m.get("status")
                value = status.get("value") if isinstance(status, dict) else status
                if m.get("id") == model and value == "loaded":
                    return True
            return False
        if base == serving.MLX_URL:
            return get(base + mlx_server.LOADED_PATH).get("model") == model
        if base == serving.vllm_url():
            return any(m.get("id") == model
                       for m in get(base + "/v1/models").get("data") or [])
    except Exception:  # noqa: BLE001
        return False
    return False


def _gate(where: serving.Route) -> None:
    """Refuse a call that would load a model while another job holds the machine lock."""
    held = busy()
    if not held:
        return
    job = running_job()
    holder = f"{held} under {job}" if job else held
    resident = loaded(where)
    level = None
    if resident:
        from harness import pressure
        level = pressure.sample().level
    ok, why = admit(holder, resident, level)
    if not ok:
        raise Refused(why)


def _timing(got: completion.Completion, started: float, replies=()) -> dict:
    """Timing of `got`; token counts summed over every reply a method made."""
    replies = list(replies) or [got]

    def total(key):
        counts = [(r.usage or {}).get(key) for r in replies]
        return sum(counts) if all(c is not None for c in counts) else counts[-1]
    return {"ttft_s": (replies[0].timing or {}).get("ttft_s"),
            "seconds": round(time.perf_counter() - started, 4),
            "prompt_tokens": total("prompt_tokens"),
            "completion_tokens": total("completion_tokens")}


def complete(lane: str, prompt: str, system: str | None = None,
             max_tokens: int = 2048, temperature: float | None = None,
             timeout: float | None = None, thinking: bool | None = None) -> dict:
    """The lane's adopted model or method, timed; `method` is a method's cost. Raises Refused or CompletionError.

    `thinking` None serves the lane as it was measured; False sends enable_thinking false.
    """
    lane = check_lane(lane)
    if not (prompt or "").strip():
        raise Refused("empty prompt")
    _preflight(prompt, system or "", max_tokens=max_tokens)
    spec, where = route(lane)
    from harness import methods
    base = methods.base_of(spec) or spec
    _budget(base, max_tokens, prompt, system or completion.SYSTEM.get(lane, ""))
    _gate(where)
    started = time.perf_counter()
    made = []

    def call(text, modality="", **kw):
        made.append(modality)
        mine = system if modality else None
        label = _call_label(spec, len(made), modality)
        try:
            # Each call of a method gets the whole cap and the thinking switch. #581.
            if label:
                _budget(base, max_tokens, text, mine or completion.SYSTEM.get(modality, ""))
            if modality == "claims":
                from harness.checks import claims
                try:
                    kw = {**kw, "response_format": claims.response_format(claims.served_schema())}
                except claims.NoSchema as exc:
                    raise Refused(str(exc)) from exc
            return completion.complete_full(
                text, modality=modality, timeout=timeout or TIMEOUT_S,
                temperature=temperature, max_tokens=int(max_tokens), stream=True,
                system=mine, template=template(modality, thinking), **kw)
        except Refused as exc:
            raise Refused(f"{label}: {exc}") from exc
        except completion.CompletionError as exc:
            if exc.failure_class != reasons.TOKEN_BUDGET_EXHAUSTED:
                raise
            hint = "" if thinking is False else " Or pass thinking=false to answer without reasoning."
            raise completion.CompletionError(
                f"{label + ': ' if label else ''}{exc}{hint}",
                exc.failure_class, exc.limit) from exc
    text, cost, replies = generate(spec, where, lane, prompt, call)
    got = replies[-1]
    return {"text": text, "lane": lane, "spec": spec,
            "model": _answered_by(got.model or where.model), "method": cost or None,
            **_timing(got, started, replies)}


def _answered_by(model: str) -> str:
    """The real id behind a model name a server echoed; the name when nothing resolves it. #670."""
    from harness import models
    return models.resolve(model) or model


def check_schema(schema) -> dict:
    """The flat decide schema, validated as `soh decide` validates it."""
    from harness.checks import decide as decide_check
    if not isinstance(schema, dict) or not schema:
        raise ValueError("the schema must map field names to fields")
    for name, spec in schema.items():
        if not isinstance(spec, dict):
            raise ValueError(f"field {name!r} must be an object")
        decide_check.choices(spec)
        if not str(spec.get("description") or "").strip():
            raise ValueError(f"field {name!r} needs a description")
    return schema


def ask(question: str, schema: dict, where: serving.Route, context: str = "",
        timeout: float | None = None, stream: bool = False):
    """(answers/probabilities body, Completion): the core of `soh decide`. #423."""
    from harness.checks import decide as decide_check
    prompt = f"{question.rstrip()}\n\n{decide_check.render(schema)}"
    got = completion.complete_full(
        prompt, model=where.model, gateway=where.base, modality="decide",
        context=context, timeout=timeout or completion.TIMEOUT_S,
        sampling=where.sampling or None, top_logprobs=completion.TOP_LOGPROBS,
        stream=stream, response_format=decide_check.response_format(schema))
    parsed = decide_check.parse(
        decide_check.from_logprobs(got.text, got.tokens, schema), schema)
    missing = [n for n, f in parsed.items() if f["answer"] is None]
    if missing:
        raise ValueError(f"no usable answer for {', '.join(missing)}: "
                         f"{got.text.strip()[:200]}")
    body = {"answers": {n: f["answer"] for n, f in parsed.items()},
            "probabilities": {n: f["probs"] or {f["answer"]: 1.0}
                              for n, f in parsed.items()}}
    return body, got


def decide(question: str, schema: dict, context: str = "",
           timeout: float | None = None) -> dict:
    """`soh decide` on the decide lane's adopted model, timed."""
    if not (question or "").strip():
        raise Refused("empty question")
    check_schema(schema)
    _preflight(question, context, str(schema))
    spec, where = route("decide")
    _gate(where)
    started = time.perf_counter()
    # Whole, not streamed: the gateway drops logprobs from a stream, and they are the answer.
    body, got = ask(question, schema, where, context=context,
                    timeout=timeout or TIMEOUT_S)
    return {**body, "lane": "decide", "spec": spec,
            "model": got.model or where.model, **_timing(got, started)}
