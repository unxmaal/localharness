"""SoHoT: generate media and talk to the machine, all locally.

    soh image "a red fox in snow" --width 768
    soh video "a fox running" --seconds 2
    soh svg   "a settings gear icon"
    soh web   "a landing page for a coffee roaster"
    soh code  "a python function that parses an ISO timestamp"
    soh extract --file build.log "how many tests failed?"
    soh say   "bonjour" --voice fr-male
    soh voices
    soh hear  --seconds 5

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
import sys

from harness import adopt, audio, completion, env, power, proc, vector  # noqa: F401
from harness.commands import adopt as adopt_cmd
from harness.commands import benchmarks as benchmarks_cmd
from harness.commands import cases as cases_cmd
from harness.commands import chaos as chaos_cmd
from harness.commands import common
from harness.commands import discover as discover_cmd
from harness.commands import gateway as gateway_cmd
from harness.commands import gauntlet as gauntlet_cmd
from harness.commands import jobs as jobs_cmd
from harness.commands import judge as judge_cmd
from harness.commands import lanes as lanes_cmd
from harness.commands import measure as measure_cmd
from harness.commands import report as report_cmd
from harness.commands import reverify as reverify_cmd
from harness.commands import screen as screen_cmd
from harness.commands import store as store_cmd
from harness.commands import usage as usage_cmd
from harness.commands import volume as volume_cmd

# What scripts, tests and other modules take from harness.cli, from where it now lives.
from harness.commands.common import say  # noqa: F401
from harness.commands.discover import (SOURCE_TIERS, _report_control, _report_feeds,
    _report_inspect, _report_neighbors, cmd_discover, resolve_registry)  # noqa: F401
from harness.commands.judge import _report_judge_store  # noqa: F401
from harness.commands.lanes import (_generate, _text_route, cmd_svg, lane_model)  # noqa: F401
from harness.commands.loop import (_loop_spend, _queueable, _reopen_retests,
    _spend_and_settle)  # noqa: F401
from harness.commands.measure import (NO_ROWS_FOR_CANDIDATE, _all_refused, _measure,
    _measure_and_adopt, _receipt_at, _summary_row, cmd_throughput, measurable)  # noqa: F401
from harness.commands.report import (_report_queue, retest_line)  # noqa: F401
from harness.commands.screen import (_judge_fits, _report_screen, _screen_plan,
    cmd_fetch, screenable_backlog)  # noqa: F401
from harness.commands.store import cmd_memory  # noqa: F401


DEFAULT_IMAGE_ENGINE = "mflux:flux2-klein-4b"
DEFAULT_VIDEO_ENGINE = "h3"
#: The music lane's incumbent. Turbo at 8 steps rather than the base model:
#: it is what was measured on this machine (issue #236) and the base model's
#: 32-100 steps have never been run here.
DEFAULT_MUSIC_ENGINE = "acestep:acestep-v15-turbo"

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
# different exam (#90). The names are the gateway nicknames of the time, which
# gateway/config.yaml maps to model ids (#670). Re-derive with:
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
DEFAULT_SVG_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"
DEFAULT_WEB_MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
DEFAULT_CODE_MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
DEFAULT_EXTRACT_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"
# Best measured on the decide lane (#311), on llama-server, which enforces the lane's schema. #572.
DEFAULT_DECIDE_MODEL = "imajev-4b-Q8_0"
#: infovore claim extraction: Qwen2.5-7B on llama-server, which enforces the reply schema. #654.
DEFAULT_CLAIMS_MODEL = "Qwen2.5-7B-Instruct-Q4_K_M"
#: The agent lane starts on the code lane's typed default until it adopts. #474.
DEFAULT_AGENT_MODEL = DEFAULT_CODE_MODEL
#: The ocr lane starts on the reader the image lane's text check already trusts. #562.
DEFAULT_OCR_ENGINE = "osocr:auto"
#: The retrieval lane starts on BM25: no weights, and the bar a model must clear. #563.
DEFAULT_RETRIEVAL_ENGINE = "bm25"
#: The pii lane starts on patterns: no weights, and blind to every name. #564.
DEFAULT_PII_ENGINE = "pii-regex"

TEXT_MODEL_HELP = ("gateway alias, mlx repo id or llamacpp:<stem> "
                   "(default: the lane's adopted model, else {})")
GATEWAY_HELP = ("send the request here verbatim; by default the model's own "
                "server is chosen (gateway, mlx_lm.server or llama-server)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="soh", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command")

    def media(name, help_, engine_default, func):
        p = sub.add_parser(name, help=help_)
        p.add_argument("prompt")
        p.add_argument("-o", "--output")
        p.add_argument("-m", "--model", default=None,
                       help="engine spec, e.g. mflux:z-image-turbo,quantize=4 "
                            f"(default: the lane's adopted model, else {engine_default})")
        p.add_argument("--width", type=int, default=DEFAULT_RESOLUTION)
        p.add_argument("--height", type=int, default=DEFAULT_RESOLUTION)
        p.add_argument("--steps", type=int)
        p.add_argument("--seed", type=int)
        p.set_defaults(func=func)
        return p

    media("image", "generate an image", DEFAULT_IMAGE_ENGINE, lanes_cmd.cmd_image)
    v = media("video", "generate a video", DEFAULT_VIDEO_ENGINE, lanes_cmd.cmd_video)
    v.add_argument("--frames", type=int)
    v.add_argument("--seconds", type=int, help="duration at 24fps")

    for name, help_, func in (("svg", "generate an SVG", lanes_cmd.cmd_svg),
                              ("web", "generate a web page", lanes_cmd.cmd_web)):
        p = sub.add_parser(name, help=help_)
        p.add_argument("prompt")
        p.add_argument("-o", "--output")
        p.add_argument("-m", "--model", default=None,
                       help=TEXT_MODEL_HELP.format(DEFAULT_SVG_MODEL if name == "svg"
                                                   else DEFAULT_WEB_MODEL))
        p.add_argument("--gateway", default=None, help=GATEWAY_HELP)
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
            p.add_argument("--engine", default=None,
                           help="image engine used by --method trace (default: "
                                "the image lane's adopted model, else "
                                f"{DEFAULT_IMAGE_ENGINE})")
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
    pr.add_argument("-m", "--model", default=None,
                    help="the model that writes the prompt (default: the "
                         "extract lane's adopted model, else "
                         f"{DEFAULT_EXTRACT_MODEL})")
    pr.add_argument("--gateway", default=None, help=GATEWAY_HELP)
    pr.add_argument("--quiet", action="store_true",
                    help="omit the engine and guide line from stderr")
    pr.set_defaults(func=lanes_cmd.cmd_prompt)

    c = sub.add_parser("code", help="generate code")
    c.add_argument("prompt")
    c.add_argument("-o", "--output", help="write to a file instead of stdout")
    c.add_argument("-m", "--model", default=None,
                   help=TEXT_MODEL_HELP.format(DEFAULT_CODE_MODEL))
    c.add_argument("--gateway", default=None, help=GATEWAY_HELP)
    c.set_defaults(func=lanes_cmd.cmd_code)

    x = sub.add_parser("extract",
                       help="answer a question about a file or piped input")
    x.add_argument("prompt", help="the question")
    x.add_argument("-f", "--file", help="the material; omit to read stdin")
    x.add_argument("-o", "--output", help="write to a file instead of stdout")
    x.add_argument("-m", "--model", default=None,
                   help=TEXT_MODEL_HELP.format(DEFAULT_EXTRACT_MODEL))
    x.add_argument("--gateway", default=None, help=GATEWAY_HELP)
    x.set_defaults(func=lanes_cmd.cmd_extract)

    dc = sub.add_parser("decide", help="answer a schema's fields, with a "
                        "probability per choice")
    dc.add_argument("prompt", help="the question")
    dc.add_argument("--schema", required=True,
                    help="a JSON file or inline JSON: {field: {type: enum|boolean, "
                         "description, choices}}")
    dc.add_argument("-f", "--file", help="the material; omit to read stdin")
    dc.add_argument("-m", "--model", default=None,
                    help=TEXT_MODEL_HELP.format(DEFAULT_DECIDE_MODEL))
    dc.add_argument("--gateway", default=None, help=GATEWAY_HELP)
    dc.set_defaults(func=lanes_cmd.cmd_decide)

    s = sub.add_parser("say", help="speak text aloud")
    s.add_argument("text", help="the text, or - to read stdin")
    s.add_argument("-o", "--output")
    s.add_argument("--voice", default=audio.DEFAULT_VOICE,
                   help="a cloned preset or a kokoro voice; see `soh voices`")
    s.add_argument("--speed", type=float, default=1.0)
    s.add_argument("-m", "--model", default=None,
                   help="tts model, overriding the voice's (a kokoro voice uses "
                        "the lane's adopted model, else "
                        f"{audio.DEFAULT_TTS_MODEL})")
    s.add_argument("--base-url", default=audio.DEFAULT_BASE_URL)
    s.add_argument("--no-play", dest="play", action="store_false", default=True)
    s.set_defaults(func=lanes_cmd.cmd_say)

    sub.add_parser("voices", help="list the voices that can be spoken"
                   ).set_defaults(func=lanes_cmd.cmd_voices)

    d = sub.add_parser("discover",
                       help="what this machine can do, and what has never "
                            "been measured")
    d.add_argument("--lane", help="only this modality")
    d.add_argument("--gap", action="store_true",
                   help="only what has never been run")
    d.add_argument("--external", action="store_true",
                   help="ask the registries what exists that this machine has "
                        "never measured (needs --lane)")
    d.add_argument("--budget-gib", type=float, default=20.0, dest="budget_gib",
                   help="with --loop --run, the ceiling on what this "
                        "invocation will download")
    d.add_argument("--repeat", type=int, default=None,
                   help="with --loop --run, repetitions per case when "
                        "measuring a challenger against the incumbent; "
                        "default: what the power calculation needs (#479)")
    d.add_argument("--effect", type=float, default=None,
                   help="the smallest per-cell pass-rate gain worth adopting for")
    d.add_argument("--power-budget-min", type=float, default=power.BUDGET_MIN,
                   help="minutes one paired measure may take; a powered repeat "
                        "over it runs the fallback repeat and says the cost (#591)")
    d.add_argument("--loop", action="store_true",
                   help="every step from a sweep to an adopted winner. Says "
                        "what it would do; --run spends the disk and minutes")
    d.add_argument("--sweep", action="store_true",
                   help="read every source family, then report. What running "
                        "a discovery means: --feeds alone leaves the star "
                        "graph unread")
    d.add_argument("--benchmarks", action="store_true",
                   help="sweep the registries for benchmark sources per lane (#491)")
    d.add_argument("--papers", action="store_true",
                   help="read HuggingFace daily papers; each is a technique for a lane (#576)")
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
                   help="what the stored runs say won each lane, against the "
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
    d.add_argument("--evidence", action="store_true",
                   help="verdicts whose run receipt is no longer on disk, so "
                        "nothing can re-judge them")
    d.add_argument("--requeue", action="store_true",
                   help="with --revisit, retract each one back to inspect")
    d.add_argument("--revisit", action="store_true",
                   help="candidates another machine refused whose reason no "
                        "longer applies here. A verdict is a fact about the "
                        "machine that made it")
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
    d.set_defaults(func=discover_cmd.cmd_discover)

    f = sub.add_parser("fetch",
                       help="download weights the inspect tier queued")
    f.add_argument("--run", action="store_true",
                   help="actually download; without it, only says what would")
    f.add_argument("--limit", type=int, default=1,
                   help="how many to fetch. One at a time by default: this "
                        "machine holds one working set")
    f.set_defaults(func=screen_cmd.cmd_fetch)

    rep = sub.add_parser(
        "report",
        help="one self-contained HTML page showing where the harness stands")
    rep.add_argument("--out", default="",
                     help="where to write it (default: $LOCALHARNESS_HOME/report.html)")
    rep.add_argument("--export", action="store_true",
                     help="write this machine's report as privacy-checked JSON "
                          "(default: $LOCALHARNESS_HOME/reports/<machine>.json)")
    rep.add_argument("--publish", action="store_true",
                     help="publish that JSON to the reports branch and rebuild "
                          "the GitHub Pages site")
    rep.add_argument("--auto-publish", choices=["on", "off"], default="",
                     help="publish at the end of every discovery loop on this "
                          "machine; off until turned on")
    rep.set_defaults(func=report_cmd.cmd_report)

    rub = sub.add_parser(
        "rubric",
        help="label an eval set by hand and score local models on it. #286")
    rub.add_argument("action", choices=["label", "status", "run"])
    rub.add_argument("set", help="eval set directory, or a name under "
                                 "$LOCALHARNESS_HOME/evalsets")
    rub.add_argument("--candidates", default="",
                     help="with run, comma-separated gateway aliases")
    rub.add_argument("--gateway", default=completion.DEFAULT_GATEWAY)
    rub.add_argument("--port", type=int, default=8766)
    rub.add_argument("--repeat-rate", type=float, default=0.2)
    rub.add_argument("--no-browser", action="store_true")
    rub.set_defaults(func=judge_cmd.cmd_rubric)

    thr = sub.add_parser("throughput", help="requests per hour at several "
                         "in-flight levels against one text spec")
    thr.add_argument("--model", required=True)
    thr.add_argument("--texts",
                     help="JSONL file; each line's --field is one request")
    thr.add_argument("--claims", action="append",
                     help="claims export JSONL; each case is the request the claims lane sends (#665)")
    thr.add_argument("--field", default="text")
    thr.add_argument("--n", type=int, default=16)
    thr.add_argument("--levels", default="1,2,4")
    thr.add_argument("--max-tokens", type=int, default=300)
    thr.add_argument("--gateway", default="http://127.0.0.1:4000")
    thr.add_argument("--serve", choices=("vllm-mlx", "vllm-metal"),
                     help="start this engine for a vllm: --model, sweep, then stop it")
    thr.add_argument("--server-pid", type=int,
                     help="sample this server's peak memory (with its children)")
    thr.set_defaults(func=measure_cmd.cmd_throughput)
    gw = sub.add_parser("gateway", help="the gateway's key (#482)")
    gw.add_argument("action", choices=("key",))
    gw.add_argument("--rotate", action="store_true",
                    help="replace the key; the gateway must restart")
    gw.set_defaults(func=gateway_cmd.cmd_gateway)
    use = sub.add_parser("usage", help="real use through the gateway: requests, "
                         "tokens, TTFT, errors and bad tool calls per alias")
    use.add_argument("--lane", default="", help="only sohot-<lane>")
    use.add_argument("--since", default="7d", help="window: 7d, 12h, 30m, 90s")
    use.add_argument("--text", choices=("on", "off"), default="",
                     help="keep bounded prompt and completion text samples (off by default)")
    use.set_defaults(func=usage_cmd.cmd_usage)
    dsk = sub.add_parser("disk", help="what the weights cache holds, what "
                         "uses it, and what is safe to delete")
    dsk.add_argument("--delete", choices=("rejected", "unknown"), default="",
                     help="remove this group, only what is safe to delete")
    dsk.add_argument("--yes", action="store_true",
                     help="do not ask; required with --json")
    dsk.add_argument("--record", action="store_true",
                     help="give every path with no download row a row, and "
                          "stamp rows whose path is gone as removed")
    dsk.add_argument("action", nargs="?", default="", choices=("", "speed"),
                     help="speed: uncached sequential read of a volume, stored (#612)")
    dsk.add_argument("path", nargs="?", default="",
                     help="with speed, a directory on the volume (default: the models volume)")
    dsk.add_argument("--mib", type=int, default=2560, help="with speed, how much to write and read")
    dsk.set_defaults(func=store_cmd.cmd_disk)
    vol = sub.add_parser("volume", help="stop and start everything that reads the "
                         "models volume, to swap its disk safely (#612)")
    vol.add_argument("action", choices=("stop", "start", "status"))
    vol.add_argument("--path", default="", help="the volume, or a path on it "
                     "(default: the volume holding HF_HOME)")
    vol.set_defaults(func=volume_cmd.cmd_volume)
    mem = sub.add_parser("memory", help="measure how much memory a run can "
                         "take before macOS starts pushing back")
    mem.add_argument("action", choices=("ramp", "show"))
    mem.add_argument("--step-gb", type=float, default=1.0)
    mem.add_argument("--settle", type=float, default=3.0,
                     help="seconds to wait after each step before sampling")
    mem.add_argument("--floor-pct", type=int, default=10,
                     help="stop if kern.memorystatus_level falls to this")
    mem.add_argument("--cap-gb", type=float, default=None,
                     help="never allocate more than this; default RAM - 4 GB")
    mem.set_defaults(func=store_cmd.cmd_memory)
    sto = sub.add_parser("store", help="dry-run: migrate a copy of a store "
                         "with this checkout and check it, before a deploy")
    sto.add_argument("action", choices=("dry-run",))
    sto.add_argument("path", nargs="?", default="",
                     help="the store to copy (default: this home's discovery.db)")
    sto.add_argument("--keep", action="store_true",
                     help="keep the migrated copy and say where")
    sto.set_defaults(func=store_cmd.cmd_store)
    bch = sub.add_parser("benchmarks", help="benchmark sources found per lane, their "
                         "contamination status, and what was imported (#491)")
    bch.add_argument("--lane", default="", help="only this lane")
    bch.add_argument("--sweep", action="store_true", help="read the registries now")
    bch.add_argument("--force", action="store_true", help="with --sweep, ignore the interval")
    bch.add_argument("--training", nargs="+", metavar="REPO[=ALIAS]",
                     help="read these candidates' cutoffs and declared datasets from their cards")
    bch.add_argument("--import", dest="import_source", default="", metavar="SOURCE",
                     help="import cases from a source with a converter, e.g. hf:CodeEval-Pro/mbpp-pro")
    bch.add_argument("--n", type=int, default=0, help="with --import, how many cases")
    bch.add_argument("--probe", nargs="+", metavar="MODEL",
                     help="prefix-completion contamination probe of the lane's cases (default decide)")
    bch.add_argument("--limit", type=int, default=0, help="with --probe, at most this many cases")
    bch.add_argument("--gateway", default=completion.DEFAULT_GATEWAY)
    bch.set_defaults(func=benchmarks_cmd.cmd_benchmarks)
    cas = sub.add_parser("cases", help="re-run a case importer, or its negative control (#603)")
    cas.add_argument("action", choices=("import", "control"))
    cas.add_argument("source", help="an importer in evals/importers, e.g. leetcode")
    cas.add_argument("--out", default="", help="cases root (default: evals/cases)")
    cas.set_defaults(func=cases_cmd.cmd_cases)
    aud = sub.add_parser("audit", help="check the live store and what the gateway serves, "
                         "read-only; run after every deploy and nightly. #492")
    aud.add_argument("path", nargs="?", default="",
                     help="the store to audit (default: this home's discovery.db)")
    aud.set_defaults(func=store_cmd.cmd_audit)
    jobs = sub.add_parser("jobs", help="the work queue: runs in order while "
                          "memory pressure is normal")
    jobs.add_argument("action", choices=("add", "list", "pause", "resume",
                                         "cancel", "priority"))
    jobs.add_argument("rest", nargs=argparse.REMAINDER,
                      help="add: -- <command...>; cancel: <id>")
    jobs.add_argument("--title", default="")
    jobs.add_argument("--priority", type=int, default=0,
                      help="higher runs first; ties run in the order added")
    jobs.add_argument("--cwd", default="",
                      help="add: run here instead of the deploy checkout. #645")
    jobs.set_defaults(func=jobs_cmd.cmd_jobs)
    jud = sub.add_parser(
        "judge",
        help="decide a human-judged lane by looking and listening. Issue #273")
    # THE RUN DIRECTORY IS REQUIRED, never a newest receipt. That helper
    # says so itself: "NOT FOR DECIDING ANYTHING. Reporting only", because
    # newest means sorts-highest-by-name and one badly named directory wins
    # forever (#222). A human verdict decides something.
    jud.add_argument("run", help="the run directory whose artifacts to compare")
    jud.add_argument("--lane", default="",
                     help="the lane being judged (default: read from the "
                          "receipt)")
    jud.add_argument("--port", type=int, default=8765)
    jud.add_argument("--no-browser", action="store_true")
    jud.add_argument("--all-machines", action="store_true",
                     help="serve the winner on every machine it fits, not only "
                          "this one (#485)")
    jud.add_argument("--force", action="store_true",
                     help="adopt on fewer than --min-votes answers, recorded "
                          "as forced")
    jud.add_argument("--min-votes", type=int, default=adopt.MIN_VOTES,
                     help="answers a by-hand adoption needs without --force")
    jud.set_defaults(func=judge_cmd.cmd_judge)
    ado = sub.add_parser(
        "adopt",
        help="measure a challenger against a lane's incumbent and adopt it "
             "on a significant win. Issue #464")
    ado.add_argument("--lane", default="")
    ado.add_argument("--challenger", default="",
                     help="a spec or gateway model id, e.g. mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit")
    ado.add_argument("--power", action="store_true",
                     help="print, per lane, the repeat and holdout size needed to "
                          "reach power 0.8, and the projected wall time (#591)")
    ado.add_argument("--receipt", action="append", default=[],
                     help="with --power, read the incumbent's draws from this "
                          "results.json instead of the store (repeatable)")
    ado.add_argument("--power-budget-min", type=float, default=power.BUDGET_MIN,
                     help="minutes one paired measure may take (#591)")
    ado.add_argument("--repeat", type=int, default=None,
                     help="default: the smallest repeat with enough power to "
                          "detect --effect on the holdout, capped (#479)")
    ado.add_argument("--effect", type=float, default=None,
                     help="the smallest per-cell pass-rate gain worth adopting for")
    ado.set_defaults(func=adopt_cmd.cmd_adopt)

    ver = sub.add_parser(
        "verify",
        help="run each lane's OWN default, to find out whether the lane works")
    ver.add_argument("--lane", default="", help="only this lane")
    ver.add_argument("--run", action="store_true",
                     help="actually run; without it, only says what it would")
    ver.add_argument("--all", action="store_true",
                     help="every lane, not only the unverified and stale ones")
    ver.set_defaults(func=measure_cmd.cmd_verify)
    rev = sub.add_parser(
        "reverify",
        help="queue a re-run of each wanted lane's served model when its runtime "
             "moved, it aged, or real use regressed; flag what failed")
    rev.add_argument("--lane", default="", help="only this lane")
    rev.add_argument("--dry-run", action="store_true",
                     help="say what would be queued and settled; write nothing")
    rev.add_argument("--days", type=float, default=None,
                     help="with --lane: days without a passing run before it is due "
                          "(default 7)")
    rev.set_defaults(func=reverify_cmd.cmd_reverify)

    sens = sub.add_parser(
        "sensitivity",
        help="does a constant change anything? Issue #98")
    sens.add_argument("names", nargs="*",
                      help="probes to run (default: all). "
                           "`--list` names them")
    sens.add_argument("--list", action="store_true",
                      help="name the probes and the constants nothing covers")
    sens.add_argument("--sweeps", action="store_true",
                      help="binding knobs, the sweeps that would be queued, and the decisions so far; "
                           "writes nothing (the loop queues them) (#636)")
    sens.add_argument("--inventory", action="store_true",
                      help="gating constants the knob registry covers and the ones it does not (#636)")
    sens.set_defaults(func=lanes_cmd.cmd_sensitivity)

    gnt = sub.add_parser("gauntlet", help="closed defects with no class, and the "
                         "review questions a diff raises. Issue #492")
    gnt.add_argument("action", choices=("audit", "review"))
    gnt.add_argument("range", nargs="?", default="",
                     help="review: the git range to read (default: origin/main...HEAD)")
    gnt.add_argument("--ref", default="",
                     help="audit: the history whose commits are read (default: origin/main)")
    gnt.add_argument("--offline", action="store_true",
                     help="audit: the committed defect snapshot, no gh")
    gnt.set_defaults(func=gauntlet_cmd.cmd_gauntlet)
    cha = sub.add_parser("chaos", help="break things on purpose on a scratch home and "
                         "check the harness recovers; never scheduled")
    cha.add_argument("--yes-break-things", action="store_true",
                     help="required: kills a scratch server, fills a scratch disk, "
                          "drops the HF client's network")
    cha.add_argument("--only", action="append", default=[],
                     help="run only this scenario (repeatable)")
    cha.set_defaults(func=chaos_cmd.cmd_chaos)

    h = sub.add_parser("hear", help="transcribe a clip, or record and transcribe")
    h.add_argument("file", nargs="?", help="an existing audio file")
    h.add_argument("-o", "--output", help="where to save a new recording")
    h.add_argument("--seconds", type=float, default=5.0)
    h.add_argument("-m", "--model", default=None,
                   help="stt model (default: the lane's adopted model, else "
                        f"{audio.DEFAULT_STT_MODEL})")
    h.add_argument("--base-url", default=audio.DEFAULT_BASE_URL)
    h.add_argument("--worst", choices=("stt", "tts"), default="",
                   help="list the stored clips with the highest WER, to listen to (#94)")
    h.add_argument("-n", type=int, default=10, help="with --worst, how many clips")
    h.add_argument("--rates", action="store_true",
                   help="seconds per word over stored tts clips, against the runaway ceiling (#91)")
    h.add_argument("--cases", default="", help="cases root (default: evals/cases)")
    h.add_argument("--store", default="", help="the store (default: this home's discovery.db)")
    h.set_defaults(func=lanes_cmd.cmd_hear)

    # On EVERY verb. A flag that only some subcommands accept is worse than no
    # flag: the caller cannot rely on it without first knowing which.
    for parser in sub.choices.values():
        parser.add_argument("--json", action="store_true",
                            help="machine-readable result on stdout, including "
                                 "on failure")
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    a = ap.parse_args(argv)
    common._JSON = bool(getattr(a, "json", False))
    common._VERB = getattr(a, "command", "") or ""
    # Before anything spawns mflux or h3. Installed on PATH this runs with
    # nothing sourced, and an unset HF_HOME sends huggingface_hub to
    # ~/.cache/huggingface to re-download weights that are already on the
    # volume. Silently, and onto the disk this machine has least of.
    env.guard()
    if not getattr(a, "func", None):
        ap.print_help(sys.stderr)
        return 2
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
