"""localharness: generate media and talk to the machine, all locally.

    lh image "a red fox in snow" --width 768
    lh video "a fox running" --seconds 2
    lh svg   "a settings gear icon"
    lh web   "a landing page for a coffee roaster"
    lh code  "a python function that parses an ISO timestamp"
    lh extract --file build.log "how many tests failed?"
    lh say   "bonjour" --voice fr-male
    lh voices
    lh hear  --seconds 5

Every command is blocking, because on this hardware everything except video
finishes in around a second: images take about a minute, speech is sub-second.
Video is the exception and it streams its progress rather than going quiet for
forty minutes. There is deliberately no job dispatcher.

Exit status is 0 only when a usable artifact exists. "The command exited 0" is
not evidence: a broken diffusion pipeline emits a uniform grey square at the
right resolution with a clean exit status, so the output is checked before this
reports success.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from harness.checks import code as code_check
from harness import audio, completion, discover as discovery, env, exclusive, paths, proc, vector
from harness.checks import html as html_check
from harness.checks import image as image_check
from harness.checks import svg as svg_check
from harness.engines import resolve

DEFAULT_IMAGE_ENGINE = "mflux:flux2-klein-4b"
DEFAULT_VIDEO_ENGINE = "h3"

# Every evals/cases/{image,video}/*.yaml pins a resolution. The CLI did not, so
# it inherited whatever each engine defaults to -- 1024 for mflux -- and ran a
# different exam from the suite that chose its engine, which is how two correct
# measurements came to look like a regression (#157, #141, #142). Named rather
# than inlined so tests/test_cli_resolution.py can hold the two to each other.
DEFAULT_RESOLUTION = 512
# Per-lane defaults, set from the eval of 2026-09-06 rather than from a tier
# name. NO SINGLE MODEL WINS ALL FOUR LANES, so there is no one default to
# pick.
#
# EVERY NUMBER BELOW IS FROM ONE EXAM: an M2 Pro with 32 GB, mlx_lm.server
# behind the LiteLLM gateway, DEFAULT_TEMPERATURE 0.2, 2026-09-06. None of it
# has been re-run on a discrete card (#96), and a different temperature is a
# different exam (#90). Re-derive with:
#   uv run python -m evals.run --modality <lane> --repeat 3 \
#     --candidates local-mid,local-large,q3-4b,q3-8b,q3-14b
#
#   lane     winner       runner-up            why
#   svg      local-large  q3-14b               7/9 both; 4.7s vs 10.0s
#   web      q3-4b        local-large          5/5 vs 3/5 at 21.5s vs 19.1s
#
# svg and web had ONE default until the web lane was widened from two cases to
# five. At two cases local-large and q3-4b both scored 6/6, the lane
# discriminated nothing, and speed decided. At five they separate cleanly and
# they separate the OTHER WAY from svg -- so a single default could only ever
# have been wrong for one of the two lanes.
#
# q3-14b scores marginally better ink on both and is NOT used, because it is a
# hybrid THINKING model: it answers "reply with exactly: OK" in 152 completion
# tokens against 2, and on a real SVG it spends the entire budget reasoning and
# returns null content. `lh svg` timed out twice at 180s on it. q3-8b is worse
# still: 0/9 on svg, every run a timeout. The eval's pass rate and median hid
# this, because an aggregate does not show you HOW the failures fail.
#   code     q3-4b        q3-8b                6/9 both; 2.9s vs 130s
#   extract  local-large  q3-14b               9/10 both; 0.79s vs 13.4s
#
# Qwen2.5-7B (local-large) KEEPS the extract lane on merit: same accuracy as
# Qwen3-14B at seventeen times the speed. Being a generation behind did not
# make it wrong for a job that is one short answer from a log.
#
# Qwen2.5-1.5B (local-mid) was the default for svg and web and scored 2/9 and
# 3/6. That was the single worst consequence of never having compared anything.
DEFAULT_SVG_MODEL = "local-large"
DEFAULT_WEB_MODEL = "q3-4b"
DEFAULT_CODE_MODEL = "q3-4b"
DEFAULT_EXTRACT_MODEL = "local-large"

# Named per engine family because the fix differs, and because `uv tool install
# mflux` on its own silently picks Python 3.9, where every mflux entry point
# dies on `int | None`. The tell is 2 executables installed instead of 37.
INSTALL_HINT = {
    "mflux": " Install it with: uv tool install --python 3.12 mflux",
    "h3": " Build it: git clone https://github.com/antirez/h3.c && make -C h3.c",
}


#: Set by main() from --json. A module global because every verb reports
#: through err()/say(), and threading a flag through nine functions to reach
#: two print statements is worse than this.
_JSON = False
_VERB = ""


def err(msg: str) -> int:
    """Report a failure. Under --json it is DATA on stdout, not a stderr line.

    An agent that has to read stderr to discover something went wrong will not
    read stderr. The exit code stays 1 either way.
    """
    if _JSON:
        print(json.dumps({"ok": False, "verb": _VERB, "error": msg}))
    else:
        print(msg, file=sys.stderr)
    return 1


def say(*, path=None, body=None, seconds=None, peak_kb=None, size=None,
        human: str = "") -> int:
    """Report a success, in whichever shape the caller asked for."""
    if _JSON:
        out = {"ok": True, "verb": _VERB}
        if path is not None:
            out["path"] = str(path)
        if body is not None:
            out["body"] = body
        if seconds is not None:
            out["seconds"] = round(seconds, 3)
        if peak_kb:
            out["peak_gib"] = round(peak_kb / 1024 / 1024, 2)
        if size is not None:
            out["size"] = size
        print(json.dumps(out))
    else:
        print(human)
    return 0


def default_output(kind: str, suffix: str) -> Path:
    """Where an artifact goes when the caller did not say.

    Under $LOCALHARNESS_HOME/out, which is ABSOLUTE. It used to be a relative
    `out/`, and since `lh` installs onto PATH and runs from anywhere, that
    scattered artifacts into whatever directory the caller happened to be
    standing in.
    """
    return paths.artifact(kind, suffix)


def _generate(spec: str, prompt: str, out: Path, params: dict) -> int:
    """Shared body of `image` and `video`: build, run, check, report."""
    try:
        engine = resolve(spec)
    except ValueError as exc:
        return err(str(exc))

    out = out or default_output(engine.modality, engine.output_suffix)
    # Absolute, because an engine with its own working directory would otherwise
    # write a relative path inside that directory, silently, where nobody looks.
    out = Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    try:
        argv = engine.argv(prompt, out, params)
    except ValueError as exc:
        return err(str(exc))

    def waiting(message: str) -> None:
        print(message, file=sys.stderr)

    try:
        # Issue #137. Held across the RUN only: resolving a spec and building
        # an argv cost nothing and should not make anyone queue.
        with exclusive.held(engine.modality, announce=waiting):
            r = proc.run(argv, timeout=engine.timeout, stream=engine.stream,
                         cwd=engine.cwd)
    except FileNotFoundError as exc:
        return err(f"{exc} is not installed or not on PATH.{INSTALL_HINT.get(spec.split(':')[0], '')}")
    except subprocess.TimeoutExpired:
        return err(f"{engine.name} timed out after {engine.timeout}s")
    except OSError as exc:
        return err(f"could not launch {engine.name}: {exc}")

    if not r.ok:
        return err(f"{engine.name} exited {r.returncode}\n{r.stderr.strip()}")

    if engine.modality == "image":
        expect = None
        if params.get("width") and params.get("height"):
            expect = (params["width"], params["height"])
        checked = image_check.check(out, expect=expect)
        if not checked.ok:
            return err(f"{engine.name} produced an unusable image: {checked.reason}")
        for w in checked.warnings:
            print(f"warning: {w}", file=sys.stderr)
    elif not out.exists() or out.stat().st_size == 0:
        return err(f"{engine.name} exited 0 but left no output at {out}")

    # The resolution is printed because a wall time and a peak mean nothing
    # without it: 512 costs 11.4 GiB here and 1024 costs 23.9.
    size = (f"{params['width']}x{params['height']}"
            if params.get("width") and params.get("height") else None)
    return say(path=out, seconds=r.seconds, peak_kb=r.peak_kb, size=size,
               human=f"{out}  ({r.seconds:.1f}s, "
                     f"peak {r.peak_kb / 1024 / 1024:.1f} GiB"
                     + (f", {size}" if size else "") + ")")


def cmd_image(a) -> int:
    params = {k: getattr(a, k) for k in ("width", "height", "steps", "seed")}
    return _generate(a.model, a.prompt, a.output, params)


def cmd_prompt(a) -> int:
    """Write a prompt for whatever this machine actually runs. Issue #138.

    The caller says what they want a picture of. What engine serves that, and
    how to command it, is this command's problem.
    """
    from harness import authoring, completion

    try:
        engine = authoring.resolved_for(a.lane)
        guide = authoring.for_lane(a.lane)
    except authoring.NoGuide as exc:
        return err(str(exc))
    if not a.about:
        # The guide alone is useful: it is the only place this knowledge is
        # readable by a person as well as by a model.
        print(f"\n{a.lane} runs {engine.name}\n")
        print(authoring.instructions(guide))
        return 0
    ask = (f"{authoring.instructions(guide)}\n\n"
           f"Write ONE prompt for this engine. The caller asked for:\n"
           f"{a.about}\n\n"
           f"Reply with the prompt and nothing else.")
    try:
        got = completion.complete(ask, model=a.model, gateway=a.gateway,
                                  modality="extract")
    except Exception as exc:  # noqa: BLE001
        return err(f"{exc}")
    print(got.strip())
    if not a.quiet:
        # The provenance goes to stderr so the prompt itself can be piped.
        err(f"[{engine.name}, guide {guide['identity']}, written by {a.model}]")
    return 0


def cmd_video(a) -> int:
    params = {k: getattr(a, k) for k in
              ("width", "height", "frames", "seconds", "steps", "seed")}
    return _generate(a.model, a.prompt, a.output, params)


def _text(a, modality: str, suffix: str, checker) -> int:
    try:
        raw = completion.complete(a.prompt, model=a.model, gateway=a.gateway,
                                  modality=modality)
    except completion.CompletionError as exc:
        return err(str(exc))

    body = completion.artifact(raw, modality)
    out = Path(a.output or default_output(modality, suffix))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")

    checked = checker(raw)
    for w in checked.warnings:
        print(f"warning: {w}", file=sys.stderr)
    if not checked.ok:
        # Written anyway: you cannot debug what was deleted.
        return err(f"wrote {out}, but it does not check out: {checked.reason}")
    return say(path=out, body=body, human=str(out))


#: The prompt an image model needs to produce something a tracer can use. A
#: photographic fox vectorizes into thousands of paths; flat shapes on white
#: vectorize into an icon.
TRACE_STYLE = ("flat vector illustration, simple clean shapes, bold outlines, "
               "solid colours, white background, no gradients, no texture")


def cmd_svg(a) -> int:
    method = getattr(a, "method", "llm")
    if method in ("trace", "icon"):
        return _svg_by_tracing(a, preset="illustration" if method == "trace"
                                          else "icon")
    return _text(a, "svg", ".svg", svg_check.check)


def _svg_by_tracing(a, preset: str = "illustration") -> int:
    """Draw it, then vectorize it.

    The measured answer for this lane. Five language models were compared on
    it and all five produce valid markup that is not the picture, because an
    LLM writes bezier coordinates it cannot see. Diffusion draws in pixel
    space, where "frog" is a shape it has seen.
    """
    out = Path(a.output or default_output("svg", ".svg")).resolve()
    png = out.with_suffix(".png")
    # The raster is KEPT. When the SVG is wrong the first question is always
    # whether the raster was wrong too, and deleting it throws away the only
    # way to answer.
    rc = _generate(a.engine, f"{a.prompt}, {TRACE_STYLE}", png,
                   {"width": a.width, "height": a.height,
                    "steps": None, "seed": a.seed})
    if rc != 0:
        return rc
    try:
        svg = vector.trace(png, preset=preset)
    except vector.VectorError as exc:
        return err(f"{png} was generated but could not be vectorized: {exc}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg, encoding="utf-8")
    checked = svg_check.check(svg)
    for w in checked.warnings:
        print(f"warning: {w}", file=sys.stderr)
    if not checked.ok:
        return err(f"wrote {out}, but it does not check out: {checked.reason}")
    return say(path=out, body=svg, human=str(out))


def cmd_web(a) -> int:
    return _text(a, "web", ".html", html_check.check)


def _answer(a, modality: str, context: str = "") -> int:
    """Ask for text and print it. The lanes that answer rather than draw.

    Stdout by default, because both of these produce something you pipe, read
    or paste. An SVG is an artifact you open in a viewer; a two-token answer to
    "how many tests failed" is not, and writing it to out/extract-<stamp>.txt
    would be a worse place to leave it than the terminal.
    """
    try:
        raw = completion.complete(a.prompt, model=a.model, gateway=a.gateway,
                                  modality=modality, context=context)
    except completion.CompletionError as exc:
        return err(str(exc))

    body = completion.artifact(raw, modality)

    # #143. `svg` and `web` are checked before the caller sees them and `code`
    # was not, which is backwards: it is the one lane whose output is meant to
    # be executed. Neither check here RUNS anything -- harness/checks/code.py
    # does that, and it needs a case's assertions, which a one-off prompt has
    # no equivalent of. These are warnings rather than a verdict because the
    # caller's target environment is not necessarily this machine.
    if modality == "code":
        broken = code_check.syntax_error(body)
        if broken:
            print(f"warning: does not parse, {broken}", file=sys.stderr)
        missing = code_check.unresolvable_imports(body)
        if missing:
            print(f"warning: imports not installed here: {', '.join(missing)}",
                  file=sys.stderr)

    if getattr(a, "output", None):
        out = Path(a.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(body if body.endswith("\n") else body + "\n", encoding="utf-8")
        return say(path=out, body=body, human=str(out))
    return say(body=body, human=body)


def cmd_code(a) -> int:
    return _answer(a, "code")


def cmd_extract(a) -> int:
    if a.file:
        source = Path(a.file)
        if not source.exists():
            return err(f"no such file: {source}")
        context = source.read_text(encoding="utf-8")
    else:
        # A terminal with nobody piping into it reads as an empty string, which
        # is the case below rather than a hang.
        context = sys.stdin.read() if not sys.stdin.isatty() else ""
    if not context.strip():
        return err("no material to read: pass --file, or pipe it in. "
                   "Answering a question about a log nobody supplied would "
                   "invent one.")
    return _answer(a, "extract", context=context)


def cmd_say(a) -> int:
    text = sys.stdin.read() if a.text == "-" else a.text
    if not text.strip():
        return err("nothing to say")
    out = Path(a.output or default_output("speech", ".wav"))
    try:
        # The voice name carries its model, reference clip and language code.
        # A cloned voice is three coupled settings, and getting one wrong fails
        # by naming another.
        audio.speak_as(a.voice, text, out=out, speed=a.speed,
                       base_url=a.base_url)
    except ValueError as exc:
        return err(str(exc))
    except audio.AudioError as exc:
        return err(str(exc))
    if a.play:
        # THE FILE IS ALREADY WRITTEN. A machine with no player is a missing
        # convenience, not a failed synthesis, so this warns and still reports
        # where the audio is.
        try:
            proc.run(audio.play_argv(out))
        except audio.AudioError as exc:
            print(f"warning: {exc}", file=sys.stderr)
    return say(path=out, human=str(out))


def cmd_discover(a) -> int:
    """What can this machine do, and what has never been measured?

    Built as a command rather than done by hand because the answer changes
    every time anything is installed or any eval is run. A number in a document
    is wrong by the next commit.
    """
    if getattr(a, "inspect", False):
        return _report_inspect(a)
    if getattr(a, "judge", False) and getattr(a, "from_store", False):
        return _report_judge_store(a)
    if getattr(a, "queue", False):
        return _report_queue(a)
    if getattr(a, "screen", False):
        return _report_screen(a)
    if getattr(a, "winners", False):
        return _report_winners(a)
    if getattr(a, "coverage", False):
        return _report_coverage(a)
    if getattr(a, "neighbors", False):
        return _report_neighbors(a)
    if getattr(a, "control", False):
        return _report_control(a)
    if getattr(a, "recurrence", False):
        return _report_recurrence(a)
    if getattr(a, "sources", False):
        return _report_sources(a)
    if getattr(a, "feeds", False):
        return _report_feeds(a)
    if a.external:
        if not a.lane:
            return err("--external needs a --lane: the registries are asked "
                       "different questions per modality")
        try:
            found = discovery.external(a.lane)
        except ValueError as exc:
            return err(str(exc))
        if a.json:
            print(json.dumps({"candidates": [vars(c) for c in found]}, indent=2))
            return 0
        if not found:
            print("nothing new found. Either this machine has measured what "
                  "the registry knows about, or there is no network.")
            return 0
        print(f"\n{a.lane}: candidates the registry has that nothing here has "
              f"measured")
        print("(proposals, not conclusions -- the eval decides)")
        for c in found:
            print(f"\n  {c.name}")
            print(f"    {c.source}")
            print(f"    {c.note}")
            print(f"    -> uv run python -m evals.run {c.how}")
        return 0

    caps = discovery.annotate(discovery.capabilities())
    if a.lane:
        caps = [c for c in caps if c.lane == a.lane]
    if a.gap:
        # A broken row is not a gap: running the command only refuses.
        caps = [c for c in caps
                if not c.measured and c.present and not c.blocked]

    if a.json:
        print(json.dumps({"capabilities": [vars(c) for c in caps]}, indent=2))
        return 0

    if not caps:
        print("nothing found" if not a.gap else "no gaps: everything here has been measured")
        return 0

    by_lane: dict[str, list] = {}
    for c in caps:
        by_lane.setdefault(c.lane, []).append(c)
    for lane in sorted(by_lane):
        print(f"\n{lane}")
        for c in sorted(by_lane[lane], key=lambda c: (c.measured, c.name)):
            # A recorded decision is not a defect, so it gets its own mark.
            if c.kind == "decision":
                mark = "declined"
            # BROKEN outranks measured: a thing can be measured and broken.
            elif c.blocked:
                mark = "BROKEN"
            elif not c.present:
                mark = "MISSING"
            elif c.measured:
                mark = "measured"
            else:
                mark = "NEVER RUN"
            print(f"  {mark:9} {c.kind:7} {c.name}")
            if c.blocked:
                print(f"            !! {c.blocked}")
            elif not c.present and c.note:
                print(f"            !! {c.note}")
            elif not c.measured:
                print(f"            -> {c.how}")
    # A declined tool is not an unmeasured gap, so it stays out of the ratio.
    countable = [c for c in caps if c.kind != "decision"]
    total, done = len(countable), sum(1 for c in countable if c.measured)
    print(f"\n{done}/{total} measured. The rest have never been run here.")
    _warn_stale_sources()
    return 0


def _warn_stale_sources() -> None:
    """Discovery nobody remembers to run is discovery that does not happen."""
    from harness import feeds

    try:
        stale = [r for r in feeds.staleness() if r["stale"]]
    except Exception:  # noqa: BLE001
        return
    if not stale:
        return
    names = ", ".join(r["name"] for r in stale[:4])
    print(f"\n{len(stale)} discovery source(s) not read in "
          f"{feeds.interval_days()} days: {names}")
    print("  lh discover --feeds     read them now")
    print("  lh discover --sources   when each was last read")


def _report_control(a) -> int:
    """A judge is a metric, and a metric without a control is noise.

    REPEATED, and that is the whole point of this command. The judge samples
    and nothing pins a seed, so one run gave gap +4 and the next +2 on
    identical inputs. control_repeated() was built for that, was tested, and
    nothing called it -- this command went to the single-shot form, so the
    sentence authorising every score in the project came from one draw.
    Issue #172.
    """
    from harness import judge
    runs = max(1, getattr(a, "runs", 3))
    try:
        got = judge.control_repeated(runs=runs, shape=getattr(a, "shape", "")
                                     or "described",
                                     gateway=getattr(a, "gateway", "") or "")
    except Exception as exc:  # noqa: BLE001
        return err(f"control failed: {exc}")
    if a.json:
        print(json.dumps(got, indent=2))
        return 0
    print(f"\nrubric {got['rubric']}, judge {got['model']}, "
          f"shape {got['shape']}, {got['runs']} run(s)")
    for r in got["rows"]:
        print(f"  {r['outcome']:5} {r['score']:2}  {r['name']:20} {r['why'][:52]}")
    gaps = ", ".join(f"{g:+d}" for g in got["gaps"])
    print(f"\n  gap per run {gaps}   spread {got['spread']}")
    if got["separates"]:
        print(f"  SEPARATES in all {got['runs']}. Scores from this rubric may "
              f"be used to rank.")
        return 0
    print(f"  DOES NOT SEPARATE: {got['separated_in']} of {got['runs']} runs. "
          f"No ranking may be drawn from this rubric.")
    return 1


def _report_recurrence(a) -> int:
    from harness import memory_store as ms
    conn = ms.connect()
    try:
        rows = ms.recurrence(conn, minimum=2)
        stats = ms.precision(conn)
        per_source = ms.by_source(conn)
        extract = ms.extraction(conn)
    finally:
        conn.close()
    if a.json:
        print(json.dumps({"recurrence": rows, "totals": stats,
                          "by_source": per_source,
                          "extraction": extract}, indent=2))
        return 0
    if not rows:
        print("nothing seen more than once yet. Run `lh discover --feeds`.")
    else:
        print("\nseen more than once (recurrence beats a single mention):")
        for r in rows:
            span = (r["last_seen"] - r["first_seen"]) / 86400.0
            print(f"  {r['times']}x over {span:5.1f}d  {r['name'][:44]:46}"
                  f" {r['sources']} source(s)")
    print(f"\n{stats['proposals']} proposals, {stats['resolved']} resolved, "
          f"{stats['verdict_measured']} measured, "
          f"{stats['verdict_declined']} declined")
    if per_source:
        print("\nper source (issue #49: precision is a query, not a count):")
        print(f"  {'source':24} {'proposed':>8} {'resolved':>8} {'settled':>8}")
        for r in per_source:
            print(f"  {r['source']:24} {r['proposals']:8d} "
                  f"{r['resolved']:8d} {r['settled']:8d}")
    if extract:
        print("\nextraction precision: of the names pulled out of prose, how "
              "many were real")
        print(f"  {'source':24} {'kept':>6} {'dropped':>8} {'precision':>10}")
        for r in extract:
            reasons = ", ".join(f"{k} {v}" for k, v in
                                sorted(r["reasons"].items(), key=lambda kv: -kv[1]))
            print(f"  {r['source']:24} {r['kept']:6d} {r['dropped']:8d} "
                  f"{r['precision']:10.2f}   {reasons}")
    return 0


def shard(names: list, spec: str) -> list:
    """The slice of the work this worker owns, as `i/n`.

    WITHOUT THIS A FAN-OUT IS FICTION. `parallelism: 4` on a Job whose pods all
    receive the same arguments is four workers doing identical work: four times
    the API budget, four times the clones, one result, and four writers racing
    on the same rows.

    A STRIDE, not a contiguous block. The candidate list arrives ranked, so
    splitting it into blocks would give worker 0 every strong candidate and the
    last worker the tail -- the slowest repos to clone are not evenly spread
    either, and blocks turn that into one straggler. `names[i::n]` interleaves,
    which is both simpler and better balanced.

    Empty spec means the whole list, so the CLI on a laptop is unchanged.
    """
    if not spec:
        return names
    try:
        i, n = (int(part) for part in spec.split("/", 1))
    except ValueError:
        raise SystemExit(f"--shard wants i/n, got {spec!r}")
    if n < 1 or not 0 <= i < n:
        raise SystemExit(f"--shard {spec} is not a slice of {n} workers")
    return names[i::n]


def resolve_registry(name: str, client, model=None) -> tuple[str, dict | None]:
    """Which registry answers for a name nothing wrote a registry down for.

    THE MIGRATION CANNOT ANSWER THIS. A name lifted from prose was resolved
    against HuggingFace by the sweep and the store kept the reddit permalink,
    so 223 of 235 rows carry no evidence either way (#167). Guessing from the
    shape of the string would be free and wrong: an `org/name` is a valid id in
    both namespaces, and a definite 404 is now terminal, so a wrong guess
    settles a real model as missing for good.

    HuggingFace first, on the same argument feeds.candidates() already makes in
    its two passes: where both answer, the model is the thing the eval can run.

    Returns the registry and whatever the answering registry already handed
    over, so the caller does not ask a second time: both of these rate-limit,
    and a repeated question is a request spent on nothing. Raises Gone only
    when BOTH say no; an empty registry means nothing could be told, which
    settles nothing.
    """
    from harness import github, inspect as ins
    from harness import memory_store as ms

    model = model or ins.hf_model
    reachable = False
    try:
        return ms.HUGGINGFACE, model(name)
    except ins.Gone:
        reachable = True
    except ins.InspectError:
        pass
    try:
        return ms.GITHUB, client.repo(name)
    except github.NotFound:
        if reachable:
            raise ins.Gone(f"{name}: neither registry has anything by that name")
    except github.GitHubError:
        pass
    return "", None


def _report_inspect(a) -> int:
    """Read a candidate's source before anyone downloads its weights. #61."""
    from harness import github, inspect as ins
    from harness import memory_store as ms

    client = github.Client(budget=getattr(a, "budget", 900))
    work = paths.home() / "cache" / "clones"
    work.mkdir(parents=True, exist_ok=True)
    store = ms.connect()
    try:
        if a.repos:
            # The flag says repos, so they are read as repos.
            work_items = [(n, ms.GITHUB) for n in a.repos]
        elif getattr(a, "from_store", False):
            # The rung the ladder was missing: what the sweep found, rather
            # than the crowd. Without this the two tiers read different
            # sources and nothing consumes a swept proposal.
            #
            # ASKED OF EACH REGISTRY SEPARATELY. One list handed to one API is
            # how 227 of 235 swept candidates 404ed: every name the sweep
            # writes is a HuggingFace id and this tier only knew how to clone
            # from GitHub. Issue #167.
            limit = getattr(a, "top", 10) * 5
            work_items = [(n, r) for r in ms.REGISTRIES
                          for n in ms.pending(store, limit=limit, registry=r)]
            # And the ones the store cannot route, which it resolves rather
            # than guesses at. See resolve_registry().
            work_items += [(n, "") for n in
                           ms.pending(store, limit=limit, registry="")]
        else:
            work_items = [(n.repo, ms.GITHUB) for n in
                          __import__("harness.neighbors", fromlist=["x"])
                          .neighbors(client=client, top=getattr(a, "top", 10))]
        work_items = shard(work_items, getattr(a, "shard", ""))
        out = []
        for repo, registry in work_items:
            card = None
            if not registry:
                try:
                    registry, card = resolve_registry(repo, client)
                except ins.Gone as exc:
                    err(f"{repo}: {exc}")
                    ms.decide(store, repo, "broken", tier=ms.INSPECT,
                              detail=str(exc)[:200])
                    continue
                if not registry:
                    err(f"{repo}: neither registry could be reached, so it "
                        f"stays unanswered")
                    continue
                ms.set_registry(store, repo, registry)
            if registry == ms.HUGGINGFACE:
                try:
                    fit = ins.inspect_model(repo, data=card)
                except ins.Gone as exc:
                    # A definite 404 is an answer about the candidate, so it is
                    # recorded as one. An unreachable registry is not.
                    err(f"{repo}: {exc}")
                    try:
                        ms.decide(store, repo, "broken", tier=ms.INSPECT,
                                  detail=str(exc)[:200])
                    except KeyError:
                        pass   # named on the command line, never proposed
                    continue
                except ins.InspectError as exc:
                    err(f"{repo}: {exc}")
                    continue
            else:
                try:
                    meta = client.repo(repo) if card is None else card
                except github.GitHubError as exc:
                    err(f"{repo}: {exc}")
                    continue
                try:
                    fit = ins.inspect(repo, work, meta=meta)
                except ins.InspectError as exc:
                    err(f"{repo}: {exc}")
                    continue
            out.append(fit)
            ms.record(store, ms.Seen(
                name=repo, source="inspect", registry=registry,
                kind="weights" if registry == ms.HUGGINGFACE else "repo",
                url=(f"https://huggingface.co/{repo}"
                     if registry == ms.HUGGINGFACE
                     else f"https://github.com/{repo}"),
                resolved=repo, lane=fit.lanes.get(repo, ""), why=fit.why,
                # WHAT IT IS, beside what the verdict said about it. The judge
                # reads this; with only a name it cannot rank at all (#175).
                description=fit.description))
            # A thing that cannot run here is ANSWERED, so it is terminal and
            # never proposed again. "unknown" settles nothing, deliberately.
            outcome = {"fits": "queued", "unknown": ""}.get(fit.verdict, "declined")
            if outcome:
                ms.decide(store, repo, outcome, tier=ms.INSPECT,
                          detail=f"{fit.verdict}: {fit.why}"[:200])
            # The WEIGHTS are what a download queue can act on. The repo is
            # something to install and screen, and the two are not the same
            # queue: queueing the repo sent GitHub names to snapshot_download,
            # which wants a HuggingFace id, and every one of them 401'd.
            if fit.verdict != "fits":
                continue
            if registry == ms.HUGGINGFACE:
                # A model IS the weight, so there is no second queue to fill
                # and nothing to retire: what `queued` means here is already
                # recorded above.
                continue
            # In HEADLINE order, not smallest-first: the smallest named
            # weight is almost always a tokenizer or a helper, and the first
            # queue built that way filled with them. Issue #68.
            ranked = [m for m in fit.headline if m in fit.weights][:3]
            # Re-inspecting CORRECTS the queue rather than only extending it:
            # weights this repo queued under an older ranking, and no longer
            # ranks, are retired. Issue #73.
            ms.retire_unlisted(
                store, repo, keep=ranked, reason=(
                    "no longer among this repo's top-ranked weights"))
            for model_id in ranked:
                size = fit.weights[model_id]
                if size > ins.MEMORY_CEILING:
                    continue
                ms.record(store, ms.Seen(
                    name=model_id, source="inspect", kind="weights",
                    registry=ms.HUGGINGFACE,
                    url=f"https://huggingface.co/{model_id}",
                    resolved=model_id, lane=fit.lanes.get(model_id, ""),
                    why=f"named by {repo}"))
                ms.link(store, repo, model_id, "needs")
                ms.decide(store, model_id, "queued", tier=ms.INSPECT,
                          detail=f"bytes={size} lane={fit.lanes.get(model_id) or '-'} "
                                 f"named by {repo}")
    finally:
        store.close()
    if getattr(a, "judge", False):
        _judge_fits(out, store_path=None)
    if a.json:
        print(json.dumps({"inspected": [vars(f) for f in out]}, indent=2))
        return 0
    print("\nread from source, with nothing downloaded and nothing run")
    for f in out:
        print(f"\n  {f.verdict.upper():14} {f.repo}")
        print(f"    {f.why}")
        bits = []
        if f.mlx:
            bits.append("MLX-native")
        if f.mps and not f.mlx:
            bits.append("torch/MPS")
        if f.cuda_mentioned:
            bits.append(f"mentions {', '.join(f.cuda_mentioned[:2])}")
        if f.unsized:
            bits.append(f"{len(f.unsized)} weight(s) unsized")
        if bits:
            print(f"    {'; '.join(bits)}")
    return 0


def cmd_fetch(a) -> int:
    """Download what the inspect tier queued. Issue #62."""
    from harness import fetching, inspect as ins
    from harness import memory_store as ms

    store = ms.connect()
    try:
        rows = fetching.queued(store)
        if not rows:
            print("nothing queued. `lh discover --inspect` fills the queue.")
            return 0
        if not a.run:
            testable = [r for r in rows if r.get("lane")]
            orphans = [r for r in rows if not r.get("lane")]
            print(f"\n{len(testable)} queued, "
                  f"{fetching.free_bytes() / fetching.GIB:.0f} GiB free. "
                  f"--run to start, one at a time. The score is the judged "
                  f"score of the repo that named the weight.")
            for r in testable[:20]:
                size = fetching.size_of(r)
                gib = f"{size / fetching.GIB:5.1f} GiB" if size else "  no size"
                print(f"  {r['score'] or 0:>4.0f}  {gib}  {r['lane']:6s} "
                      f"{r['resolved'] or r['name']}")
            if orphans:
                print(f"\n{len(orphans)} named but NOT queued: nothing here can "
                      f"measure them. Not a verdict on the model -- the eval "
                      f"suite has no case, runner or metric for this kind of "
                      f"thing, and building one is sometimes the work.")
                for r in orphans[:10]:
                    print(f"        {r['resolved'] or r['name']}")
            return 0
        sizes = {(r["resolved"] or r["name"]): fetching.size_of(r)
                 for r in rows[:a.limit]}
        for got in fetching.run(store, sizes, limit=a.limit):
            print(f"  {'OK  ' if got['ok'] else 'skip'} {got['repo']}: {got['why']}")
    finally:
        store.close()
    return 0


def _judge_fits(fits, store_path=None) -> int:
    """Score inspected candidates with the facts the clone produced. #69.

    Ordering matters: INSPECT runs before JUDGE now. Cheapest-first was never
    the real justification for the old order -- inspect costs seconds and the
    judge costs about a second -- and the judge scoring a generic description
    3/10 while the inspect tier had already proved the thing MLX-native was the
    tier with more evidence losing to the tier with less.
    """
    from harness import judge
    from harness import memory_store as ms
    try:
        rubric = judge.load()
    except judge.JudgeError as exc:
        return err(str(exc))
    store = ms.connect(store_path)
    try:
        print("\n  judged, with the source read first:")
        for f in fits:
            weights = (f"{f.smallest / (1024**3):.1f} to "
                       f"{f.largest / (1024**3):.1f} GiB" if f.largest else "")
            item = judge.describe(
                f.repo, why=f.description, source="github-crowd",
                inspected=f"{f.verdict}: {f.why}",
                platform=("MLX-native" if f.mlx else
                          "torch/MPS" if f.mps else ""),
                weights=weights)
            try:
                score, why = judge.score(item, rubric)
            except Exception as exc:  # noqa: BLE001
                err(f"{f.repo}: {exc}")
                continue
            print(f"    {score:2d}/10  {f.repo:36.36s} {why[:56]}")
            try:
                ms.decide(store, f.repo, "queued", tier=ms.JUDGE, score=score,
                          rubric=rubric.stamp, judge=rubric.model,
                          detail=why[:200])
            except KeyError:
                pass
    finally:
        store.close()
    return 0


def _report_queue(a) -> int:
    """What a screen would teach us, best first. Issue #175.

    NOT A PREDICTION OF WHO WINS. The judge tier tried that and could not: two
    controls shaped like this data both failed to separate six models with known
    opposite outcomes, because the fact that separated them was produced by
    RUNNING them and a registry card has never held it.

    Arithmetic over what the store already knows, so it is deterministic and
    needs no gateway -- which is worth as much as the ordering, given the
    instrument it stands in for had to be run three times to be believed.
    """
    from harness import rank
    from harness import memory_store as ms

    store = ms.connect()
    try:
        rows = ms.judgeable(store, limit=getattr(a, "top", 25) * 4)
        waiting = ms.judgeable_total(store)
    finally:
        store.close()
    if not rows:
        print("nothing queued that a screen has not answered")
        return 0
    ranked = rank.rank(rows, serving=rank.serving(),
                       measured_lanes=rank.lanes_with_receipts(),
                       ceiling_gib=22.0)[:getattr(a, "top", 25)]
    if a.json:
        print(json.dumps({"queue": ranked, "waiting": waiting}, indent=2))
        return 0
    print(f"\n{len(ranked)} of {waiting} waiting, by what a screen would teach:")
    for r in ranked:
        print(f"\n  {r['value']:+6.1f}  {r['name']}")
        print(f"          {r['value_why'] or 'nothing known about it'}")
    return 0


def _report_coverage(a) -> int:
    """What discovery never saw. Issue #99.

    Extraction precision measures the quality of what is CAUGHT. This measures
    reach: of the things this project actually adopted, which did a configured
    source ever surface. A source list is not a measurement of coverage.
    """
    from harness import coverage
    from harness import memory_store as ms

    store = ms.connect()
    try:
        got = coverage.report(store)
    finally:
        store.close()
    if a.json:
        print(json.dumps(got, indent=2))
        return 0
    print(f"\n{got['adopted']} things this machine runs or measured:")
    print(f"  {len(got['found']):3d} surfaced by a discovery source first")
    print(f"  {len(got['late']):3d} surfaced only after they were already run")
    print(f"  {len(got['holes']):3d} never surfaced by any source")
    if got["by_source"]:
        print("\n  credited with a find:")
        for source, n in got["by_source"].items():
            print(f"    {source:24} {n}")
    if got["proposed"]:
        # A source producing plenty that nobody has run is a different problem
        # from one producing nothing, and they want opposite fixes.
        print("\n  proposals produced (adopted or not):")
        for source, n in got["proposed"].items():
            print(f"    {source:24} {n}")
    holes = [h for h in got["holes"] if "why" not in h]
    ours = [h for h in got["holes"] if "why" in h]
    if ours:
        print(f"\n  {len(ours)} in the store only because one of our own tiers "
              f"put it there:")
        for h in ours[:12]:
            print(f"    {h['name']}")
    if holes:
        print(f"\n  {len(holes)} no configured source ever produced:")
        for h in holes[:20]:
            print(f"    {h['name']:44} {'/'.join(h['how'])}")
    return 0


def _report_winners(a) -> int:
    """What the receipts say won each lane, against what this file has typed in.

    The four DEFAULT_*_MODEL constants above are a hand copy of a measurement
    that lives in the run receipts. Both are worth having -- a default that
    moved because somebody ran an eval last night is a CLI two machines
    disagree about -- but a hand copy with nothing watching it is this
    project's most-bitten failure class.
    """
    from harness import winners

    best = winners.beaten_in()
    rows = winners.disagreements()
    if a.json:
        print(json.dumps({"typed": winners.typed(), "measured": best,
                          "disagreements": rows}, indent=2))
        return 0
    #: exact agreement needs no mark; the other two each say which they are.
    MARK = {"exact": " ", "quantised": "~", "": "*"}
    print(f"\n  {'lane':9} {'typed':34} {'measured here':30} run")
    for lane, name in sorted(winners.typed().items()):
        got = best.get(lane)
        if not got:
            print(f"  {lane:9} {name:34} {'-- not in any receipt':30}")
        else:
            mark = MARK.get(got["match"], "*")
            print(f" {mark}{lane:9} {name:34} "
                  f"{got['candidate'] + ' ' + str(got['pass_rate']):30} "
                  f"{got['run']}")
    print("\n  * beaten in a run it was in   ~ only a quantisation of it ran")
    beaten = [r for r in rows if r["state"] == "beaten"]
    quantised = [r for r in rows if r["state"] == "under-specified"]
    unmeasured = [r for r in rows if r["state"] == "unmeasured"]
    if beaten:
        print(f"\n  {len(beaten)} default(s) lost a comparison they were in:")
        for r in beaten:
            print(f"    {r['modality']}: {r['measured']} beat {r['typed']} "
                  f"in {r['run']}")
    if quantised:
        # NOT a disagreement about which is better. The constant names an
        # artifact no run here has produced, because the engine quantises and
        # the candidate name does not say so.
        print(f"\n  {len(quantised)} default(s) name something only a "
              f"quantisation of which has run here:")
        for r in quantised:
            print(f"    {r['modality']}: {r['typed']} -> {r['measured']} "
                  f"in {r['run']}")
    if unmeasured:
        # NOT a disagreement. A default that appears in no receipt was never in
        # the room, and reporting silence as conflict is how an inventory
        # becomes noise nobody reads.
        print(f"\n  {len(unmeasured)} default(s) appear in no receipt here, so "
              f"nothing on this machine can check them:")
        for r in unmeasured:
            print(f"    {r['modality']}: {r['typed']}")
    return 0


def _report_screen(a) -> int:
    """Run the cheapest real thing, and record whether it ran at all. #53.

    SAYS WHAT IT WOULD DO AND STOPS, unless told otherwise. `--run` is the same
    convention `lh fetch` uses, and for the same reason: this is the first tier
    that spends real time and real memory, and one that starts doing so because
    something ranked well is how a laptop ends up unusable overnight.
    """
    import subprocess

    from harness import rank, screen
    from harness import memory_store as ms

    store = ms.connect()
    try:
        rows = ms.judgeable(store, limit=getattr(a, "top", 25) * 4)
    finally:
        store.close()
    ranked = rank.rank(rows, serving=rank.serving(),
                       measured_lanes=rank.lanes_with_receipts())
    planned = screen.plan(ranked)[:getattr(a, "top", 5)]
    if not planned:
        print("nothing queued to screen")
        return 0
    ready = [r for r in planned if r["state"] == screen.READY]
    if not getattr(a, "run", False):
        print(f"\n{len(planned)} candidate(s), {len(ready)} ready to screen:")
        for r in planned:
            print(f"\n  {r['state']:16} {r['name']}")
            print(f"    {r['why_not']}")
            if r["state"] == screen.READY:
                print(f"    {' '.join(screen.argv(r))}")
        print("\n  --run to screen the ready ones; nothing is downloaded either "
              "way")
        return 0
    if not ready:
        return err("nothing is ready to screen: fetch weights first, and note "
                   "that a screen never downloads")

    store = ms.connect()
    try:
        for r in ready[:getattr(a, "limit", 1)]:
            print(f"\n── {r['name']}", flush=True)
            proc = subprocess.run(screen.argv(r), capture_output=True,
                                  text=True)
            # The RUN's own summary, read from what it wrote rather than parsed
            # out of its chatter: a tier that infers an outcome from stdout is
            # a tier that reports success when the format changes.
            summary = _latest_summary(r["modality"])
            got, why = screen.outcome(proc.returncode, summary)
            print(f"   {got.upper()}: {why}")
            if proc.returncode != 0:
                err(proc.stderr.strip()[-400:] or "no stderr")
            ms.decide(store, r["name"], got, tier=ms.SCREEN, detail=why[:200])
    finally:
        store.close()
    return 0


def _latest_summary(modality: str) -> dict | None:
    """The summary of the newest run receipt for this modality."""
    newest, when = None, 0.0
    try:
        receipts = sorted(paths.runs().rglob("results.json"))
    except OSError:
        return None
    for f in receipts:
        try:
            stat = f.stat()
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (data.get("receipt") or {}).get("modality") != modality:
            continue
        if stat.st_mtime > when:
            newest, when = data.get("summary"), stat.st_mtime
    return newest


def _report_judge_store(a) -> int:
    """Score what the inspect tier queued and no judge has read. #148 phase 3.

    THE RUNG ABOVE --inspect --from-store. _judge_fits() can only see the Fit
    objects produced in its own process, so a judge has never been able to
    reach the store: after #167 that left real candidates with real verdicts
    and nothing ranking them.

    THE CONTROL RUNS FIRST AND THE TIER REFUSES WITHOUT IT. A judge is a
    metric, this one samples, and a tier that scores unattended on a schedule
    has nobody present to doubt it. `--no-control` exists for a person watching
    the output, never for a Job.
    """
    from harness import judge
    from harness import memory_store as ms

    gateway = getattr(a, "gateway", "") or ""
    try:
        rubric = judge.load()
    except judge.JudgeError as exc:
        return err(str(exc))

    # THE WORK FIRST, THEN THE GATE, and in that order for two reasons. The
    # control costs a model call per control item, so a Job whose queue is
    # empty would otherwise pay eighteen of them to prove a rubric separates
    # and then score nothing. And a store that cannot be reached should be
    # found in the first second rather than after the judging budget is spent:
    # the first run of this tier in a cluster did exactly that, spending its
    # control on a gateway and then dying on a Postgres in recovery.
    store = ms.connect()
    scored = 0
    try:
        rows = ms.judgeable(store, limit=getattr(a, "top", 25))
        rows = shard(rows, getattr(a, "shard", ""))
        waiting = ms.judgeable_total(store)
        if not rows:
            print("nothing queued by inspect that a judge has not already read")
            return 0

        if not getattr(a, "no_control", False):
            runs = max(1, getattr(a, "runs", 3))
            # THE SHAPE OF ITS OWN INPUT. This tier feeds the judge store
            # rows, so a control made of hand-written project descriptions
            # answers a question about different data -- it separated at +7
            # while every real candidate came back 3/10 (#175).
            shape = getattr(a, "shape", "") or "carded"
            try:
                got = judge.control_repeated(runs=runs, gateway=gateway,
                                             shape=shape)
            except Exception as exc:  # noqa: BLE001
                return err(f"control failed, so nothing was scored: {exc}")
            gaps = ", ".join(f"{g:+d}" for g in got["gaps"])
            print(f"control: rubric {got['rubric']}, judge {got['model']}, "
                  f"shape {got['shape']}, gap per run {gaps}, "
                  f"spread {got['spread']}")
            if not got["separates"]:
                return err(
                    f"the rubric separates in only {got['separated_in']} of "
                    f"{got['runs']} control runs, so nothing was scored. A "
                    f"score from a rubric that does not discriminate is a "
                    f"number, not a ranking")

        print(f"\njudging {len(rows)} candidate(s) the sweep found and the "
              f"source tier answered:")
        for row in rows:
            item = judge.describe(
                # The card first: a sighting's `why` is what one source said
                # on one day, and for a prose-swept id it is usually empty.
                row["name"], why=row.get("description") or row.get("why") or "",
                source=row.get("source") or "", times_seen=row.get("times") or 0,
                relevance=row.get("relevance") or 0,
                inspected=row.get("inspected") or "")
            try:
                score, why = judge.score(item, rubric, gateway=gateway)
            except Exception as exc:  # noqa: BLE001 - one bad reply is not a run
                err(f"{row['name']}: {exc}")
                continue
            print(f"  {score:2d}/10  {row['name']:40.40s} {why[:48]}")
            ms.decide(store, row["name"], "queued", tier=ms.JUDGE, score=score,
                      rubric=rubric.stamp, judge=rubric.model,
                      detail=why[:200])
            scored += 1
    finally:
        store.close()
    left = max(0, waiting - scored)
    print(f"\n{scored} scored, under rubric {rubric.identity} judged by "
          f"{rubric.model}"
          + (f"; {left} still waiting" if left else ""))
    return 0


def _report_neighbors(a) -> int:
    """What the people who build what we run are looking at. Issue #58."""
    from harness import feeds, github, memory_store as ms, neighbors as nb

    client = github.Client(budget=getattr(a, "budget", 900))
    try:
        people = nb.cohort(client=client, limit=getattr(a, "crowd", 250))
    except github.GitHubError as exc:
        return err(f"{exc}. `gh auth status` to check the token.")
    if getattr(a, "control", False):
        got = nb.control(list(nb.DEFAULT_SEEDS), people, client)
        if a.json:
            print(json.dumps(got, indent=2))
            return 0
        print(f"\ncrowd of {got['crowd']}, ranking {len(got['top'])}")
        print(f"  expected and found : {', '.join(got['expected_found']) or 'NONE'}")
        print(f"  expected but absent: {', '.join(got['missing']) or 'none'}")
        print(f"  decoys in the top  : {', '.join(got['decoys_in_top']) or 'none'}")
        print(f"\n  SEPARATES: {got['separates']}")
        if not got["separates"]:
            print("  Do not quote a score from this run.")
        print("\n  by shared count alone, which is what we do NOT ship:")
        for r in got["raw_top"][:5]:
            print(f"    {r}")
        return 0

    found = nb.neighbors(people, client, top=getattr(a, "top", 25),
                         exclude=nb.DEFAULT_SEEDS)
    if a.json:
        print(json.dumps({"crowd": len(people),
                          "neighbors": [vars(n) for n in found]}, indent=2))
        return 0
    print(f"\nrepos concentrated in the crowd that builds what this machine "
          f"runs\n({len(people)} people, {client.spent} requests; a popularity "
          f"signal, not a measurement)")
    if client.stale:
        print(f"  {len(client.stale)} answer(s) served from a stale cache")
    for n in found:
        flag = "  ARCHIVED" if n.archived else ""
        print(f"\n  {n.score:.5f}  {n.repo}{flag}")
        print(f"    {n.shared} of {n.crowd} starred it; {n.stars} stars, "
              f"pushed {n.pushed}")
        if n.description:
            print(f"    {n.description[:100]}")
    # Into the store, so a sighting counts towards recurrence and the graph
    # finally has edges to walk. Seeds are recorded too, because link() needs
    # both ends to exist.
    store = ms.connect()
    try:
        feeds.record_fetch("github-crowd")
        for seed in nb.DEFAULT_SEEDS:
            ms.record(store, ms.Seen(name=seed, source="installed", kind="repo",
                                     registry=ms.GITHUB,
                                     url=f"https://github.com/{seed}",
                                     resolved=seed))
        for n in found:
            ms.record(store, ms.Seen(name=n.repo, source="github-crowd",
                                     registry=ms.GITHUB,
                                     url=f"https://github.com/{n.repo}",
                                     why=n.description, kind="repo",
                                     relevance=feeds.relevance(
                                         f"{n.repo} {n.description} "
                                         f"{' '.join(n.topics)}"),
                                     resolved=n.repo))
            for seed in nb.DEFAULT_SEEDS:
                ms.link(store, seed, n.repo, "crowd",
                        note=f"{n.shared}/{n.crowd} at {n.score:.5f}")
        if getattr(a, "judge", False):
            _judge_neighbors(found, store)
    finally:
        store.close()
    return 0


def _judge_neighbors(found, store):
    """Score crowd proposals with the rubric. Same cheapest tier as the feeds.

    A repo card says more than a recap blurb, so the judge is shown the
    description, topics and how much of the crowd starred it.
    """
    from harness import judge
    from harness import memory_store as ms
    try:
        rubric = judge.load()
    except judge.JudgeError as exc:
        return err(str(exc))
    print("\n  judged:")
    for n in found:
        why = f"{n.description} [{n.language}; {', '.join(n.topics[:6])}]"
        row = store.execute(
            "SELECT v.detail FROM verdicts v JOIN proposals p "
            "ON p.id = v.proposal_id WHERE p.name = ? AND v.tier = 'inspect' "
            "ORDER BY v.id DESC LIMIT 1", (n.repo,)).fetchone()
        try:
            score, reason = judge.score(
                judge.describe(n.repo, why=why, source="github-crowd",
                               times_seen=n.shared,
                               inspected=(row["detail"] if row else "")),
                rubric)
        except Exception as exc:  # noqa: BLE001
            err(f"{n.repo}: {exc}")
            continue
        print(f"    {score:2d}/10  {n.repo:38.38s} {reason[:60]}")
        try:
            ms.decide(store, n.repo, "queued", tier=ms.JUDGE, score=score,
                      rubric=rubric.stamp, judge=rubric.model,
                      detail=reason[:200])
        except KeyError:
            pass
    return 0


def _report_sources(a) -> int:
    from harness import feeds

    rows = feeds.staleness()
    proposed = discovery.feed_sources()
    if a.json:
        print(json.dumps({"sources": rows,
                          "proposed": [vars(c) for c in proposed]}, indent=2))
        return 0
    print(f"\ndiscovery sources (interval {feeds.interval_days()} days, "
          f"${feeds.INTERVAL_ENV} to change)")
    for r in rows:
        age = ("never read" if r["age_days"] is None
               else f"{r['age_days']:.1f} days ago")
        mark = "STALE" if r["stale"] else "ok   "
        print(f"  {mark} {r['name']:24} {age}")
        print(f"        {r['url']}")
    if proposed:
        print("\nsources these feeds point at that we do not read:")
        for c in proposed:
            # Probing is the difference between a shortlist and a guess: half
            # of these hosts serve no feed at all. Issue #50.
            ok, why = feeds.probe(c.how or c.url)
            mark = "FEED " if ok else "none "
            print(f"  {mark} {c.name:20} {c.note}")
            print(f"        {why}")
        print(f"\n  Add one to {feeds.config_path()} to start reading it. "
              f"Deliberately manual: a source URL out of untrusted prose "
              f"should need a human nod.")
    return 0


def _report_feeds(a) -> int:
    from harness import memory_store as ms
    store = ms.connect()
    try:
        found = discovery.from_feeds(
            verify=not a.no_verify,
            min_relevance=1 if a.platform else None, store=store,
            comments=getattr(a, "comments", 0),
            mentions=(_mention_extractor()
                      if getattr(a, "comments", 0) and getattr(a, "judge", False)
                      else None))
        if getattr(a, "judge", False):
            found = _judge_proposals(found, store)
    finally:
        store.close()
    if a.json:
        print(json.dumps({"candidates": [vars(c) for c in found]}, indent=2))
        return 0
    broken = [c for c in found if c.blocked]
    for c in broken:
        err(f"{c.name}: {c.blocked}")
    found = [c for c in found if not c.blocked]
    if not found:
        print("nothing new in the feeds, or no network.")
        return 0
    updates = [c for c in found if c.kind == "update"]
    found = [c for c in found if c.kind != "update"]
    if updates:
        print("\nBEHIND on something already installed:")
        for c in updates:
            print(f"  {c.name:12} {c.note}")
            print(f"    -> {c.how}")
    if not found:
        return 0
    print("\ncandidates the community is talking about that nothing here "
          "has measured")
    print("(a popularity signal, not a measurement -- the eval decides)")
    for c in found:
        print(f"\n  {c.name}")
        print(f"    {c.source}")
        print(f"    {c.note}")
        print(f"    -> uv run python -m evals.run {c.how}")
    return 0


def _mention_extractor():
    """The judge, used to read names out of freeform comment prose. #79."""
    from harness import judge
    return lambda text: judge.mentions(text)


def _judge_proposals(found, store):
    """Score, record and re-sort. The cheapest tier: text only, no GPU."""
    from harness import judge
    from harness import memory_store as ms
    try:
        rubric = judge.load()
    except judge.JudgeError as exc:
        err(str(exc))
        return found
    for c in found:
        if c.kind != "proposal":
            continue
        try:
            score, why = judge.score(
                judge.describe(c.name, why=c.note, relevance=c.relevance),
                rubric)
        except Exception as exc:  # noqa: BLE001
            err(f"{c.name}: {exc}")
            continue
        c.relevance = score
        c.note = f"[{score}/10] {why[:90]} | {c.note}"
        try:
            ms.decide(store, c.name, "queued", tier="judge", score=score,
                      rubric=rubric.stamp, judge=rubric.model, detail=why[:200])
        except KeyError:
            pass
    found.sort(key=lambda c: -getattr(c, "relevance", 0))
    return found


def cmd_voices(a) -> int:
    """Three coupled settings behind one name is only usable if the names are
    discoverable."""
    def star(name):
        return " (default)" if name == audio.DEFAULT_VOICE else ""

    print("cloned (a reference clip, so any language, any accent):")
    for name in sorted(audio.VOICE_PRESETS):
        v = audio.resolve_voice(name)
        print(f"  {name}{star(name)}  {v.model.split('/')[-1]}"
              f"  speaks {v.lang_code}  from {Path(v.ref_audio).name}")
    print("\nkokoro (a fixed table, English unless noted; sub-second, "
          "where a cloned voice takes seconds):")
    for name in audio.KNOWN_VOICES:
        note = "  French, female" if name == "ff_siwis" else ""
        print(f"  {name}{star(name)}{note}")
    return 0


def cmd_sensitivity(a) -> int:
    """Vary a constant and say whether anything downstream moved."""
    from harness import probes, sensitivity

    if getattr(a, "list", False):
        if a.json:
            print(json.dumps({"probes": sorted(probes.PROBES),
                              "uncovered": probes.UNCOVERED}, indent=2))
            return 0
        print("\nprobes:")
        for name in sorted(probes.PROBES):
            print(f"  {name}")
        print("\nnot covered, and why:")
        for name, why in sorted(probes.UNCOVERED.items()):
            print(f"  {name}\n      {why}")
        return 0

    unknown = [n for n in a.names if n not in probes.PROBES]
    if unknown:
        return err(f"unknown probe(s) {', '.join(unknown)}; "
                   f"known: {', '.join(sorted(probes.PROBES))}")
    try:
        found = probes.run(a.names or None)
    except Exception as exc:  # noqa: BLE001
        return err(f"{exc}")
    if a.json:
        print(json.dumps({"coverage": probes.coverage(),
                          "findings": [
                              {"name": f.name, "default": f.default,
                               "verdict": f.verdict, "band": list(f.band),
                               "readings": [vars(r) for r in f.readings]}
                              for f in found]}, indent=2, default=str))
        return 0
    cover = probes.coverage()
    print(f"\nagainst cached data only: a crowd of {cover['crowd']} and "
          f"{cover['clones']} clones on disk\n")
    print(sensitivity.report(found))
    return 0


def cmd_hear(a) -> int:
    clip = Path(a.file) if a.file else Path(a.output or
                                            default_output("clip", ".wav"))
    if not a.file:
        clip.parent.mkdir(parents=True, exist_ok=True)
        print(f"listening for {a.seconds}s...", file=sys.stderr)
        try:
            r = proc.run(audio.record_argv(clip, a.seconds),
                         timeout=a.seconds + 30)
        except FileNotFoundError:
            return err("`rec` not found. Install it with: brew install sox")
        if not r.ok:
            return err(f"recording failed (exit {r.returncode})\n{r.stderr.strip()}")

    try:
        text = audio.transcribe(clip, base_url=a.base_url)
    except audio.AudioError as exc:
        return err(str(exc))
    return say(path=clip, body=text, human=text)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="lh", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command")

    def media(name, help_, engine_default, func):
        p = sub.add_parser(name, help=help_)
        p.add_argument("prompt")
        p.add_argument("-o", "--output")
        p.add_argument("-m", "--model", default=engine_default,
                       help="engine spec, e.g. mflux:z-image-turbo,quantize=4")
        p.add_argument("--width", type=int, default=DEFAULT_RESOLUTION)
        p.add_argument("--height", type=int, default=DEFAULT_RESOLUTION)
        p.add_argument("--steps", type=int)
        p.add_argument("--seed", type=int)
        p.set_defaults(func=func)
        return p

    media("image", "generate an image", DEFAULT_IMAGE_ENGINE, cmd_image)
    v = media("video", "generate a video", DEFAULT_VIDEO_ENGINE, cmd_video)
    v.add_argument("--frames", type=int)
    v.add_argument("--seconds", type=int, help="duration at 24fps")

    for name, help_, func in (("svg", "generate an SVG", cmd_svg),
                              ("web", "generate a web page", cmd_web)):
        p = sub.add_parser(name, help=help_)
        p.add_argument("prompt")
        p.add_argument("-o", "--output")
        p.add_argument("-m", "--model",
                       default=DEFAULT_SVG_MODEL if name == "svg"
                       else DEFAULT_WEB_MODEL,
                       help="gateway alias")
        p.add_argument("--gateway", default=completion.DEFAULT_GATEWAY)
        p.set_defaults(func=func)
        if name == "svg":
            # `llm` is still the default because it is seconds against a
            # minute, and for a two-shape icon it is sometimes enough. `trace`
            # is the one that draws the picture.
            p.add_argument("--method", choices=("llm", "trace", "icon"),
                           default="llm",
                           help="llm: a language model writes the paths. "
                                "trace: generate an image and vectorize it. "
                                "icon: the same, tuned for a small file")
            p.add_argument("--engine", default=DEFAULT_IMAGE_ENGINE,
                           help="image engine used by --method trace")
            p.add_argument("--width", type=int, default=512)
            p.add_argument("--height", type=int, default=512)
            p.add_argument("--seed", type=int)

    pr = sub.add_parser("prompt",
                        help="write a prompt for whatever engine this machine "
                             "runs in a lane")
    pr.add_argument("lane", choices=["image", "video"],
                    help="which generating lane the prompt is for")
    pr.add_argument("about", nargs="?", default="",
                    help="what the caller wants. Without it, print the guide")
    pr.add_argument("-m", "--model", default=DEFAULT_EXTRACT_MODEL,
                    help="gateway alias that writes the prompt")
    pr.add_argument("--gateway", default=completion.DEFAULT_GATEWAY)
    pr.add_argument("--quiet", action="store_true",
                    help="omit the engine and guide line from stderr")
    pr.set_defaults(func=cmd_prompt)

    c = sub.add_parser("code", help="generate code")
    c.add_argument("prompt")
    c.add_argument("-o", "--output", help="write to a file instead of stdout")
    c.add_argument("-m", "--model", default=DEFAULT_CODE_MODEL,
                   help="gateway alias")
    c.add_argument("--gateway", default=completion.DEFAULT_GATEWAY)
    c.set_defaults(func=cmd_code)

    x = sub.add_parser("extract",
                       help="answer a question about a file or piped input")
    x.add_argument("prompt", help="the question")
    x.add_argument("-f", "--file", help="the material; omit to read stdin")
    x.add_argument("-o", "--output", help="write to a file instead of stdout")
    x.add_argument("-m", "--model", default=DEFAULT_EXTRACT_MODEL,
                   help="gateway alias")
    x.add_argument("--gateway", default=completion.DEFAULT_GATEWAY)
    x.set_defaults(func=cmd_extract)

    s = sub.add_parser("say", help="speak text aloud")
    s.add_argument("text", help="the text, or - to read stdin")
    s.add_argument("-o", "--output")
    s.add_argument("--voice", default=audio.DEFAULT_VOICE,
                   help="a cloned preset or a kokoro voice; see `lh voices`")
    s.add_argument("--speed", type=float, default=1.0)
    s.add_argument("--base-url", default=audio.DEFAULT_BASE_URL)
    s.add_argument("--no-play", dest="play", action="store_false", default=True)
    s.set_defaults(func=cmd_say)

    sub.add_parser("voices", help="list the voices that can be spoken"
                   ).set_defaults(func=cmd_voices)

    d = sub.add_parser("discover",
                       help="what this machine can do, and what has never "
                            "been measured")
    d.add_argument("--lane", help="only this modality")
    d.add_argument("--gap", action="store_true",
                   help="only what has never been run")
    d.add_argument("--external", action="store_true",
                   help="ask the registries what exists that this machine has "
                        "never measured (needs --lane)")
    d.add_argument("--feeds", action="store_true",
                   help="read the community aggregation feeds for candidates")
    d.add_argument("--sources", action="store_true",
                   help="list discovery sources, when each was last read, and "
                        "any new sources the feeds point at")
    d.add_argument("--judge", action="store_true",
                   help="with --feeds, score each proposal 1-10 with the "
                        "rubric before anything is run")
    d.add_argument("--control", action="store_true",
                   help="score items whose outcome is already known, and report "
                        "whether the rubric separates them. Run this before "
                        "trusting any score")
    d.add_argument("--shape", default="", choices=["", "described", "bare",
                                                   "carded"],
                   help="which known set the control scores. A tier must gate "
                        "on the shape of its own input: `described` is "
                        "hand-written project prose, `carded` is what a "
                        "registry says, `bare` is a name and nothing else")
    d.add_argument("--runs", type=int, default=3,
                   help="how many times to run the control. The judge samples "
                        "and nothing pins a seed, so one run is one draw")
    d.add_argument("--no-control", action="store_true",
                   help="with --judge --from-store, score without running the "
                        "control first. For a person watching the output, "
                        "never for a Job")
    d.add_argument("--gateway", default=completion.DEFAULT_GATEWAY,
                   help="where the judge model is served. A pod reaches the "
                        "host's gateway, not its own localhost")
    d.add_argument("--neighbors", action="store_true",
                   help="repos concentrated in the crowd that builds what "
                        "this machine runs; add --control to check the metric "
                        "before trusting it")
    d.add_argument("--crowd", type=int, default=250,
                   help="with --neighbors, how many people to ask")
    d.add_argument("--budget", type=int, default=900,
                   help="with --neighbors, cap on GitHub API requests")
    d.add_argument("--top", type=int, default=25,
                   help="with --neighbors, how many to show")
    d.add_argument("--inspect", action="store_true",
                   help="clone a candidate's source and say whether it can run "
                        "here, before anything is downloaded")
    d.add_argument("--repos", nargs="*", default=[],
                   help="with --inspect, specific repos instead of the crowd")
    d.add_argument("--from-store", action="store_true",
                   help="with --inspect, take candidates the sweep already "
                        "found and nothing has answered, most-corroborated "
                        "first, instead of rebuilding the crowd. With --judge "
                        "and without --inspect, score what the source tier "
                        "queued and no judge has read")
    d.add_argument("--shard", default="", metavar="I/N",
                   help="with --inspect, take only this worker's slice of the "
                        "candidates. Kubernetes passes the index of an Indexed "
                        "Job; without it every worker does the same work")
    d.add_argument("--coverage", action="store_true",
                   help="of the things this machine runs, which a configured "
                        "source ever surfaced. Precision measures what is "
                        "caught; this measures reach")
    d.add_argument("--winners", action="store_true",
                   help="what the run receipts say won each lane, against the "
                        "defaults this CLI has typed in")
    d.add_argument("--screen", action="store_true",
                   help="run the cheapest real thing on the top of the queue "
                        "and record whether it ran at all. Says what it would "
                        "do unless given --run, and NEVER downloads")
    d.add_argument("--run", action="store_true",
                   help="with --screen, actually run it")
    d.add_argument("--limit", type=int, default=1,
                   help="with --screen --run, how many to screen. One at a "
                        "time: a screen holds a model in memory")
    d.add_argument("--queue", action="store_true",
                   help="what a screen would teach us, best first, from what "
                        "the store already knows. Arithmetic, not a judge")
    d.add_argument("--recurrence", action="store_true",
                   help="what keeps coming back, from the discovery store")
    d.add_argument("--comments", type=int, default=0, metavar="N",
                   help="with --feeds, also read the replies on the N newest "
                        "posts per source. The comparative judgements live "
                        "there, not in the post")
    d.add_argument("--platform", action="store_true",
                   help="with --feeds, only what looks like it runs on "
                        "THIS machine")
    d.add_argument("--no-verify", action="store_true",
                   help="with --feeds, skip resolving prose names against the "
                        "registry. Faster, and QUIETER: every unresolved name "
                        "is dropped rather than offered")
    d.set_defaults(func=cmd_discover)

    f = sub.add_parser("fetch",
                       help="download weights the inspect tier queued")
    f.add_argument("--run", action="store_true",
                   help="actually download; without it, only says what would")
    f.add_argument("--limit", type=int, default=1,
                   help="how many to fetch. One at a time by default: this "
                        "machine holds one working set")
    f.set_defaults(func=cmd_fetch)

    sens = sub.add_parser(
        "sensitivity",
        help="does a constant change anything? Issue #98")
    sens.add_argument("names", nargs="*",
                      help="probes to run (default: all). "
                           "`--list` names them")
    sens.add_argument("--list", action="store_true",
                      help="name the probes and the constants nothing covers")
    sens.set_defaults(func=cmd_sensitivity)

    h = sub.add_parser("hear", help="transcribe a clip, or record and transcribe")
    h.add_argument("file", nargs="?", help="an existing audio file")
    h.add_argument("-o", "--output", help="where to save a new recording")
    h.add_argument("--seconds", type=float, default=5.0)
    h.add_argument("--base-url", default=audio.DEFAULT_BASE_URL)
    h.set_defaults(func=cmd_hear)

    # On EVERY verb. A flag that only some subcommands accept is worse than no
    # flag: the caller cannot rely on it without first knowing which.
    for parser in sub.choices.values():
        parser.add_argument("--json", action="store_true",
                            help="machine-readable result on stdout, including "
                                 "on failure")
    return ap


def main(argv: list[str] | None = None) -> int:
    global _JSON, _VERB
    ap = build_parser()
    a = ap.parse_args(argv)
    _JSON = bool(getattr(a, "json", False))
    _VERB = getattr(a, "command", "") or ""
    # Before anything spawns mflux or h3. Installed on PATH this runs with
    # nothing sourced, and an unset HF_HOME sends huggingface_hub to
    # ~/.cache/huggingface to re-download weights that are already on the
    # volume. Silently, and onto the disk this machine has least of.
    if env.apply() is None:
        print(f"warning: no weights location with {env.HF_MIN_FREE_GB}GB free "
              f"(tried {env.configured()}); set HF_ROOT or HF_HOME",
              file=sys.stderr)
    if not getattr(a, "func", None):
        ap.print_help(sys.stderr)
        return 2
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
