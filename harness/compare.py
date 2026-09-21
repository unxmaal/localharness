"""Two models, one command, a verdict. Issue #259.

THE PRODUCT. Everything else in this package either feeds this or is a
discovery experiment that has never changed what a lane serves.

    lh compare code q3-4b local-large
    lh compare image mflux:z-image-turbo mflux:flux2-klein-4b

The eval suite underneath has produced every result this project has. What was
missing was a way to use it: a caller had to know the gateway URL, which of two
servers serves which alias, the candidate spelling for the lane, and to have
started the services. Four separate failures in one day came from getting one
of those wrong, and each one read as a model failing rather than as setup.

So this resolves the candidates, checks what they need is up BEFORE spending
anything, runs them paired in one run, and refuses to conclude when the control
did not run. The statistics are adopt.decide and paired.head_to_head unchanged.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from harness import lanes

#: A service a candidate needs, and the command that starts it. Keyed by the
#: port because that is what a check can actually test.
SERVICES = {
    4000: ("the gateway", "./scripts/services.sh start gateway"),
    8081: ("the text server", "./scripts/services.sh start llamacpp"),
    8890: ("the audio server", "./scripts/services.sh start audio"),
}

#: How long to wait for a service to answer before calling it absent. Short on
#: purpose: a wedged server that hangs for 180s is indistinguishable from a
#: slow model, and this exists so nobody spends an afternoon on that again.
PROBE_TIMEOUT = 5.0


@dataclass
class Plan:
    """What a comparison would do, before it does any of it."""
    lane: str
    incumbent: str
    challenger: str
    specs: tuple[str, str] = ("", "")
    gateway: str = ""
    needs: list[str] = field(default_factory=list)
    blocked: str = ""

    @property
    def ok(self) -> bool:
        return not self.blocked


@dataclass
class Outcome:
    """What a comparison found, or why it could not answer."""
    plan: Plan
    verdict: object = None
    summary: dict = field(default_factory=dict)
    receipt: str = ""
    blocked: str = ""

    @property
    def ok(self) -> bool:
        return not self.blocked


def answers(port: int, probe=None) -> bool:
    """Is a service on `port` able to serve a request, not merely bound?

    A BOUND PORT IS NOT A WORKING SERVICE. mlx_lm.server held :8081 for hours
    with an empty model list and every request hanging, and everything that
    asked only whether the port was open called it healthy. #255.
    """
    if probe is not None:
        return bool(probe(port))
    import urllib.error
    import urllib.request
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models",
                               timeout=PROBE_TIMEOUT)
        return True
    except (urllib.error.URLError, OSError):
        return False


def port_for(spec: str, gateway: str) -> int:
    """Which service this candidate's requests will reach.

    A process engine (`mflux:...`, `acestep:...`) runs a binary and needs no
    server at all, which is why this can answer 0.
    """
    from harness import engines

    head = spec.split(":", 1)[0].strip()
    if head in engines._BUILDERS:
        return 0
    if gateway:
        tail = gateway.rstrip("/").rsplit(":", 1)[-1]
        return int(tail) if tail.isdigit() else 4000
    return 4000


def plan(lane: str, incumbent: str, challenger: str, probe=None) -> Plan:
    """Resolve both candidates and check the way is clear. Spends nothing.

    SPELLING IS THIS FUNCTION'S JOB, not the caller's. `q3-4b` is a gateway
    alias, `mlx-community/Qwen3-4B-Instruct-2507-4bit` is a repo id the text
    server hot-swaps to, and `mflux:z-image-turbo` is a process engine. Asking
    a person to know which is which is how every comparison this week went
    wrong.
    """
    from harness import screen

    got = lanes.canonical(lane)
    out = Plan(got, incumbent, challenger)
    if not lanes.known(got):
        out.blocked = (f"no lane called {lane!r}. Known: "
                       f"{', '.join(lanes.ALL)}")
        return out
    if incumbent == challenger:
        out.blocked = "a candidate cannot be compared with itself"
        return out

    specs = []
    for name in (incumbent, challenger):
        spec = screen.candidate_for(got, name)
        if not spec:
            out.blocked = (f"the {got} lane has no way to run {name!r}; "
                           f"nothing here can measure it")
            return out
        specs.append(spec)
    out.specs = (specs[0], specs[1])

    # ONE GATEWAY SERVES THE WHOLE RUN. A repo id goes to the server that
    # hot-swaps to it; an alias goes to the gateway that knows the name. Mixing
    # them sent the alias to a server that had never heard of it, the control
    # scored 0/27, and the run had nothing to compare against (#223).
    #
    # TRANSLATE RATHER THAN REFUSE. An alias resolves to the upstream repo id
    # the gateway would have called anyway, so the pair meets at the same
    # server and the comparison still happens.
    routes = [screen.routed_gateway(n) for n in (incumbent, challenger)]
    direct = [r for r in routes if r]
    if direct and not all(routes):
        upstreams = []
        for name, route in zip((incumbent, challenger), routes):
            if route:
                upstreams.append(name)
                continue
            up = screen.upstream_of(name)
            if not up:
                out.blocked = (
                    f"{name!r} is a gateway alias and {direct[0]} serves repo "
                    f"ids, and the config does not say what {name!r} resolves "
                    f"to, so the two cannot meet at one server")
                return out
            upstreams.append(up)
        out.incumbent, out.challenger = upstreams
        out.specs = tuple(screen.candidate_for(got, n) or n for n in upstreams)
    out.gateway = direct[0] if direct else ""

    wanted = {port_for(s, out.gateway) for s in out.specs} - {0}
    for port in sorted(wanted):
        if answers(port, probe=probe):
            continue
        what, how = SERVICES.get(port, (f"the service on :{port}", ""))
        out.needs.append(f"{what} is not answering on :{port}"
                         + (f"  ->  {how}" if how else ""))
    if out.needs:
        out.blocked = "; ".join(out.needs)
    return out


def describe(p: Plan) -> str:
    """What this run will do, in the words a person would use."""
    if not p.ok:
        return f"cannot compare: {p.blocked}"
    where = p.gateway or "the gateway"
    return (f"{p.lane}: {p.challenger} against {p.incumbent}\n"
            f"  both on the {p.lane} cases, paired, served by {where}")


def run(lane: str, incumbent: str, challenger: str, *, repeat: int = 3,
        out: Path | None = None, runner=None, probe=None) -> Outcome:
    """Measure both, decide, and say why. The whole product in one call."""
    from harness import adopt

    p = plan(lane, incumbent, challenger, probe=probe)
    if not p.ok:
        return Outcome(p, blocked=p.blocked)

    out = Path(out) if out else _run_dir(p.lane)
    argv = ["uv", "run", "python", "-m", "evals.run",
            "--modality", p.lane, "--repeat", str(repeat),
            "--out", str(out),
            "--candidates", f"{p.specs[0]},{p.specs[1]}"]
    if p.gateway:
        argv += ["--gateway", p.gateway]

    proc = (runner or _run_eval)(argv)
    if proc != 0:
        return Outcome(p, blocked=f"the eval exited {proc}; see {out}")

    data = _receipt(out)
    if not data:
        return Outcome(p, blocked=f"the eval wrote no receipt to {out}")
    summary = data.get("summary") or {}

    # THE NAME LIVES IN THE KEY, NOT IN THE ROW. A receipt's summary row
    # carries case_ids, metrics and timings and no `candidate` field, while
    # adopt.decide and paired.head_to_head both identify a candidate by name.
    # Without this the verdict names nobody and pairs nothing, and reports
    # "they share no case in this run" for a run where both scored every case.
    key_i, rows_i = _row_for(summary, p.specs[0])
    key_c, rows_c = _row_for(summary, p.specs[1])
    if rows_i:
        rows_i = {**rows_i, "candidate": key_i}
    if rows_c:
        rows_c = {**rows_c, "candidate": key_c}
    # A CANDIDATE MEASURED BESIDE A CONTROL THAT DID NOT RUN SAYS NOTHING. The
    # incumbent scoring zero is the single most common way this project has
    # produced a confident wrong answer.
    if not rows_i:
        return Outcome(p, summary=summary, receipt=str(out), blocked=(
            f"{p.incumbent} contributed no rows, so there is no control and "
            f"nothing can be concluded. The receipt names "
            f"{sorted(summary) or 'nothing'}"))
    if not rows_c:
        return Outcome(p, summary=summary, receipt=str(out), blocked=(
            f"{p.challenger} contributed no rows, so it was never measured"))

    verdict = adopt.decide(p.lane, rows_i, rows_c, data.get("rows") or [])
    return Outcome(p, verdict=verdict, summary=summary, receipt=str(out))


def report(o: Outcome) -> str:
    """The answer, in plain words, with the numbers that produced it."""
    if not o.ok:
        return f"cannot answer: {o.blocked}"
    lines = []
    for name, row in sorted(o.summary.items()):
        metrics = " ".join(f"{k} {v:.3f}" for k, v in
                           sorted((row.get("metrics") or {}).items())
                           if isinstance(v, (int, float)))
        lines.append(f"  {name[:48]:48} {row.get('passed', 0)}/"
                     f"{row.get('total', 0)}"
                     + (f"  median {row['median_s']:.1f}s"
                        if row.get("median_s") else "")
                     + (f"  {metrics}" if metrics else ""))
    v = o.verdict
    head = (f"{v.challenger} REPLACES {v.incumbent}" if v.adopt
            else f"keeping {v.incumbent}")
    return "\n".join(lines + ["", f"  {head}: {v.why}", f"  {o.receipt}"])


def _run_dir(lane: str) -> Path:
    from harness import paths
    return (paths.home() / "runs"
            / f"{time.strftime('%Y%m%d-%H%M%S')}-compare-{lane}")


#: The checkout, so the eval runs from the right virtualenv however the
#: command was invoked. `uv run` walks UP to the nearest pyproject, and a
#: caller standing in a parent directory got that project's venv and a
#: ModuleNotFoundError for our own eval package.
REPO = Path(__file__).resolve().parent.parent


def _run_eval(argv: list[str]) -> int:
    return subprocess.run(argv, cwd=str(REPO)).returncode


def _receipt(out: Path) -> dict:
    path = Path(out) / "results.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return {}


def _row_for(summary: dict, spec: str) -> tuple[str, dict]:
    """The receipt row for this candidate.

    THE SPEC AND THE RECEIPT KEY ARE DIFFERENT NOTATIONS. `mflux:flux2-klein-4b`
    is written `mflux/flux2-klein-4b-q8` in a receipt: the separator differs and
    the quantisation is resolved. Matching them by equality reported that the
    incumbent had contributed no rows while its rows sat in the summary (#219).
    """
    from harness import winners

    if spec in summary:
        return spec, summary[spec]
    for key in summary:
        if winners.matches(spec, key, "engine") or winners.matches(
                spec, key, "alias"):
            return key, summary[key]
    tail = spec.rsplit(":", 1)[-1].rsplit("/", 1)[-1].lower()
    for key, row in summary.items():
        if tail and tail in key.lower():
            return key, row
    return "", {}
