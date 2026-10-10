"""The method registry: a named transform over a base candidate, `<name>:<args>:<base>`,
run over the base's own runner and route. #576."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

#: Where the loop's crossings are recorded: the sighting's source and the proposal's kind.
CROSSING_SOURCE = "method-crossing"
CROSSING_KIND = "method"

#: Text lanes whose cases are free text a second sample can change. decide samples
#: at temperature 0 and agent drives a tool loop, so neither is a best-of lane.
SAMPLED_TEXT_LANES = ("code", "web", "svg")
TEXT_LANES = ("code", "web", "svg", "extract")


@dataclass(frozen=True)
class Parsed:
    method: "Method"
    args: tuple
    base: str
    spec: str


@dataclass(frozen=True)
class Method:
    name: str
    grammar: str
    #: Lanes it applies to; the loop crosses it with these lanes' incumbents.
    lanes: tuple
    #: The Case.methods label it counts as; "" means whatever its base counts as.
    takes: str
    note: str
    #: (parsed, base_builder, outdir) -> runner. base_builder(spec) builds the base's runner.
    build: Callable
    #: Leading integer arguments before the base, with the defaults the loop crosses at.
    arity: int = 0
    defaults: tuple = ()
    minimum: int = 1
    #: The one modality its output is, or "" for its base's.
    modality: str = ""
    #: The lane whose incumbent is its base when the loop crosses it; "" is the lane itself.
    base_lane: str = ""
    #: Whether the loop crosses it with incumbents without anyone typing it.
    crosses: bool = True
    #: vector.TRACE_PRESETS key, for a trace method.
    preset: str = ""
    #: (parsed, lane, prompt, ask) -> (text, cost): how a lane command serves it; None for trace.
    serve: Callable | None = None
    extra: dict = field(default_factory=dict)

    def compose(self, base: str, *args) -> str:
        got = [str(a) for a in (args or self.defaults)]
        return ":".join([self.name, *got, base])


def parse(spec: str) -> Parsed | None:
    """The method, its arguments and its base; None when `spec` is not a method. Raises
    ValueError for a method spec that does not parse."""
    spec = (spec or "").strip()
    head, sep, rest = spec.partition(":")
    head = head.partition(",")[0].strip()
    method = REGISTRY.get(head)
    if method is None:
        return None
    args = []
    for _ in range(method.arity):
        arg, sep2, rest = rest.partition(":")
        try:
            n = int(arg)
        except ValueError:
            raise ValueError(f"{spec!r}: {method.grammar} needs a whole number, "
                             f"got {arg!r}") from None
        if n < method.minimum:
            raise ValueError(f"{spec!r}: {method.grammar} needs at least {method.minimum}")
        if not sep2:
            rest = ""
        args.append(n)
    if not sep or not rest.strip():
        raise ValueError(f"{spec!r} names no base; spelled {method.grammar}")
    return Parsed(method, tuple(args), rest.strip(), spec)


def is_method(spec: str) -> bool:
    try:
        return parse(spec) is not None
    except ValueError:
        return True


def base_of(spec: str) -> str:
    """The base a method spec composes over, or "" for anything else."""
    try:
        got = parse(spec)
    except ValueError:
        return ""
    return got.base if got else ""


def build(spec: str, base_builder, outdir=None):
    """The runner for a method spec. SystemExit on a bad spec, as every runner is built."""
    try:
        parsed = parse(spec)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    runner = parsed.method.build(parsed, base_builder, outdir)
    return runner


def applicable(lane: str) -> list[Method]:
    return [m for m in REGISTRY.values() if lane in m.lanes]


def crossings(lane: str, incumbent_of) -> list[str]:
    """Each crossing method applicable to `lane`, composed over the incumbent it takes."""
    out = []
    for m in applicable(lane):
        if not m.crosses:
            continue
        base = incumbent_of(m.base_lane or lane)
        if not base or is_method(base):
            continue
        out.append(m.compose(base))
    return out


def weights_of(spec: str) -> str:
    """The repo id a method's base downloads, or "" when its runner finds its own."""
    base = base_of(spec)
    if not base:
        return ""
    model = base.partition(",")[0].strip()
    from harness import models
    # A gateway entry is named by its repo id now (#670); the gateway serves it, as it did its nickname.
    if models.on_gateway(model):
        return ""
    return model if "/" in model and ":" not in model else ""


#: The first call of plan: a plan, and no answer yet.
PLAN_PROMPT = """Before answering, write a short numbered plan for the task below.
Do not write the answer itself.

TASK:
{task}"""

#: The second call of plan: the task again, with the plan to follow.
ANSWER_PROMPT = """{task}

Follow this plan:
{plan}"""


def run(spec: str, lane: str, prompt: str, ask) -> tuple[str, dict]:
    """(text, cost) for `prompt` served by `spec`, where ask(prompt, modality) -> text is one call
    on its base's route. A plain model is one ask and {}. ValueError for a method no text call serves. #581."""
    parsed = parse(spec)
    if parsed is None:
        return ask(prompt, lane), {}
    if parsed.method.serve is None:
        raise ValueError(f"{spec} draws with an image engine rather than answering in text; "
                         f"run `soh svg` for it")
    return parsed.method.serve(parsed, lane, prompt, ask)


def text_method(spec: str) -> bool:
    """Whether `spec` is a method a text call serves (best-of, plan)."""
    try:
        parsed = parse(spec)
    except ValueError:
        return False
    return parsed is not None and parsed.method.serve is not None


def _rank(lane: str, text: str) -> tuple:
    """The lane command's own check of a reply as a key: passing first, then fewer warnings."""
    from harness import completion
    from harness.checks import code as code_check, html as html_check, svg as svg_check
    body = completion.artifact(text, lane)
    if lane == "code":
        broken = code_check.syntax_error(body) if body.strip() else "empty"
        missing = [] if broken else code_check.unresolvable_imports(body)
        return (not broken, -len(missing))
    checker = {"svg": svg_check.check, "web": html_check.check}.get(lane)
    if checker is None:
        return (bool(body.strip()), 0)
    got = checker(text)
    return (bool(got.ok), -len(got.warnings))


def _serve_plan(parsed: Parsed, lane: str, prompt: str, ask) -> tuple[str, dict]:
    plan = ask(PLAN_PROMPT.format(task=prompt), "")
    answer = ask(ANSWER_PROMPT.format(task=prompt, plan=plan.strip()), lane)
    return answer, {"method": parsed.method.name, "base": parsed.base, "calls": 2}


def _serve_best_of(parsed: Parsed, lane: str, prompt: str, ask) -> tuple[str, dict]:
    from harness import completion
    n = parsed.args[0]
    best = best_key = failed = None
    chosen = 0
    for i in range(1, n + 1):
        try:
            text = ask(prompt, lane)
        except completion.CompletionError as exc:
            failed = exc
            continue
        key = _rank(lane, text)
        if best_key is None or key > best_key:
            best, best_key, chosen = text, key, i
    if best is None:
        raise failed or completion.CompletionError("no sample came back")
    return best, {"method": parsed.method.name, "base": parsed.base, "calls": n,
                  "samples": n, "chosen": chosen}


def cost(incumbent: dict, challenger: dict, spec: str) -> dict:
    """What a method paid for its result beside the incumbent: calls and median latency; {} for
    anything that is not a method."""
    try:
        parsed = parse(spec)
    except ValueError:
        return {}
    if parsed is None:
        return {}
    mine = float(challenger.get("median_s") or 0.0)
    theirs = float(incumbent.get("median_s") or 0.0)
    return {"method": parsed.method.name, "base": parsed.base,
            "calls": float(((challenger.get("metrics") or {}).get("method_calls")) or 1.0),
            "median_s": mine, "incumbent_median_s": theirs,
            "latency_ratio": round(mine / theirs, 2) if theirs > 0 else None}


def cost_note(incumbent: dict, challenger: dict, spec: str) -> str:
    """The cost as a sentence for the verdict a person reads."""
    got = cost(incumbent, challenger, spec)
    if not got:
        return ""
    ratio = (f", {got['latency_ratio']:.1f}x the incumbent's median latency"
             if got["latency_ratio"] is not None else "")
    return (f"cost: {got['method']} made {got['calls']:.1f} calls a case{ratio} "
            f"({got['median_s']:.2f}s against {got['incumbent_median_s']:.2f}s)")


# ---- the methods -----------------------------------------------------------------

def _trace(parsed: Parsed, base_builder, outdir):
    from evals.runners.trace import TraceRunner
    from harness.engines import resolve
    name, spec = parsed.method.name, parsed.base
    try:
        engine = resolve(spec)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if outdir is None:
        raise SystemExit(f"{parsed.spec} writes images; pass --out")
    return TraceRunner(engine, outdir, preset=parsed.method.preset)


def _best_of(parsed: Parsed, base_builder, outdir):
    from evals.runners.method import BestOfRunner
    return BestOfRunner(base_builder(parsed.base), parsed.args[0])


def _plan(parsed: Parsed, base_builder, outdir):
    from evals.runners.method import PlanRunner
    from evals.runners.text import CompletionRunner
    base = base_builder(parsed.base)
    if not isinstance(base, CompletionRunner):
        raise SystemExit(f"{parsed.spec}: plan needs a text base served through a text "
                         f"server, and {parsed.base} is not one")
    return PlanRunner(base)


REGISTRY: dict[str, Method] = {m.name: m for m in (
    Method("trace", "trace:<image engine>", ("svg",), "trace",
           "draw a raster with an image engine, then vectorize it", _trace,
           modality="svg", base_lane="image", preset="illustration"),
    Method("trace-icon", "trace-icon:<image engine>", ("svg",), "trace",
           "trace with the icon preset", _trace,
           modality="svg", base_lane="image", preset="icon", crosses=False),
    Method("best-of", "best-of:<n>:<base>", SAMPLED_TEXT_LANES, "",
           "sample n at the lane's temperature, keep what the lane's checks score highest",
           _best_of, arity=1, defaults=(3,), minimum=2, serve=_serve_best_of),
    Method("plan", "plan:<base>", TEXT_LANES, "",
           "ask for a plan, then answer with it: two calls", _plan,
           serve=_serve_plan),
)}


#: candidate prefix -> vector.TRACE_PRESETS key, read by evals.run.
TRACE_PRESETS = {m.name: m.preset for m in REGISTRY.values() if m.preset}
