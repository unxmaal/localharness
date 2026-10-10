"""Run the eval suite and print a comparison.

    uv run python -m evals.run --modality svg --candidates mlx-community/Qwen2.5-1.5B-Instruct-4bit,mlx-community/Qwen2.5-0.5B-Instruct-4bit
    uv run python -m evals.run --modality image \
        --candidates mflux:flux2-klein-4b,mflux:z-image-turbo

Artifacts and results.json land in $LOCALHARNESS_HOME/runs/<stamp>-<modality>/
unless --out says otherwise.

A candidate is either a gateway alias (text modalities) or an engine spec
(anything that runs as a process). Which one it is decides the runner, and a
candidate is only handed the cases it can actually run: giving mflux an SVG
case produces a failure row that says nothing about mflux.

Sequenced by candidate, never interleaved: mlx_lm.server hot-swaps models per
request, so alternating between two aliases pays a model load on every case and
measures disk speed instead of the model.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

from harness import audio, completion, context, delegate, ds4, env, knobs, paths, reasons, router
from harness.engines import Engine, names as engine_names, parse_options, resolve

from dataclasses import replace

from evals.core import (AGENT_MODALITIES, MODALITIES, TEXT_MODALITIES,
                        TEXT_SUFFIX, Case,
                        Receipt, cases_digest, comparable, direction_of,
                        summarize)
from evals import core, environment, private
from evals.runners.base import RunnerError
from evals.runners.process import ProcessRunner
from evals.runners.chain import ChainRunner
from evals.runners.repair import RepairRunner
from evals.runners.speech import SpeechRunner
from evals.runners.omnisvg import DEFAULT_CANDIDATES, OmniSVGRunner
from evals.runners.text import CompletionRunner
from evals.runners.transcription import TranscriptionRunner

ROOT = Path(__file__).resolve().parent
ALL_MODALITIES = sorted(MODALITIES)


#: READ FROM THE BUILDER TABLE, not restated. This was a hand-written tuple of
#: three and `diffusers` was not in it, so kind_of("diffusers:org/m") answered
#: `gateway` and an image spec was routed to the TEXT gateway. The engine has
#: been registered in harness/engines.py since the CUDA machine was brought up
#: and had never once reached a ProcessRunner, on any machine. Fourth copy of
#: "which engines exist" in this repo; RULE #237 says the fix is a read, not
#: more care.
PROCESS_ENGINES = tuple(sorted(engine_names()))
#: The svg lane's second METHOD: draw a raster, then vectorize it. Written as
#: `trace:<engine spec>` so the engine underneath stays the ordinary spec.
TRACE_PREFIX = "trace"
#: Every method, trace among them, lives in harness/methods.py. #576.
from harness import methods, models  # noqa: E402
# candidate prefix -> vector.TRACE_PRESETS key
TRACE_PREFIXES = methods.TRACE_PRESETS
#: Generate, check, repair. A WORKFLOW rather than a model: `<model>` and
#: `repair:<model>` are different products. See evals/runners/repair.py.
REPAIR_PREFIX = "repair"
REPAIR_OPTIONS = {"attempts"}
#: The svg lane's THIRD method: a model that emits draw commands as tokens.
#: `omnisvg:4B` -- a size, not an engine spec, because the weights it needs
#: are two fixed repos rather than anything the caller chooses.
OMNISVG_PREFIX = "omnisvg"
#: A frontier model through headless Claude Code (#359): `claude-code:claude-opus-5-5`.
CLAUDE_CODE_PREFIX = "claude-code"
OMNISVG_OPTIONS = {"candidates"}
#: Two-stage image workflows, named for their second stage. See
#: evals/runners/chain.py -- this is the ComfyUI vocabulary mflux already ships.
from evals.runners.chain import STAGES as CHAIN_STAGES  # noqa: E402
TTS_OPTIONS = {"voice", "ref_audio", "lang_code", "ear"}
STT_OPTIONS = {"backend", "language"}


#: A GGUF file by stem, sent straight to llama-server. #295.
from harness.serving import DS4_PREFIX, LLAMACPP_PREFIX, VLLM_PREFIX  # noqa: E402

LLAMACPP_KIND = LLAMACPP_PREFIX.rstrip(":")
#: Cactus-Compute/needle3 through its own CLI; decide and agent lanes. #312.
from harness.needle import LANES as NEEDLE_LANES, PREFIX as NEEDLE_KIND  # noqa: E402
#: A model served by vLLM on its own port. #310.
VLLM_KIND = VLLM_PREFIX.rstrip(":")
#: One of antirez/ds4's own GGUFs through ds4-server. #611.
DS4_KIND = DS4_PREFIX.rstrip(":")
#: Kinds a text server answers through serving.route.
TEXT_KINDS = ("gateway", LLAMACPP_KIND, VLLM_KIND, DS4_KIND)


def kind_of(candidate: str) -> str:
    """process, tts, or gateway. The prefix decides, so a typo in the rest of
    the spec is reported as a bad spec rather than silently becoming a model
    name the gateway has never heard of."""
    head = candidate.split(":", 1)[0].split(",", 1)[0].strip()
    if head in PROCESS_ENGINES:
        return "process"
    if head in ("tts", "stt"):
        return head
    if head in methods.REGISTRY:
        return head
    if head == OMNISVG_PREFIX:
        return OMNISVG_PREFIX
    if head == REPAIR_PREFIX:
        return REPAIR_PREFIX
    if head == CLAUDE_CODE_PREFIX:
        return CLAUDE_CODE_PREFIX
    if head in CHAIN_STAGES:
        return "chain"
    if head in (LLAMACPP_KIND, VLLM_KIND, DS4_KIND, NEEDLE_KIND):
        return head
    return "gateway"


def modality_of(candidate: str) -> str | None:
    """The one modality this candidate can run, or None for text candidates,
    which can run all of them."""
    kind = kind_of(candidate)
    if kind in ("tts", "stt"):
        return kind
    if kind in methods.REGISTRY:
        # A trace answers svg cases while its engine makes images; a text method
        # answers whatever its base answers.
        return methods.REGISTRY[kind].modality or modality_of(methods.base_of(candidate))
    if kind == OMNISVG_PREFIX:
        return "svg"
    if kind in (REPAIR_PREFIX, CLAUDE_CODE_PREFIX):
        # A text candidate wearing a loop: it runs every text lane, same as
        # the model it wraps.
        return None
    if kind == "chain":
        return "image"
    if kind == NEEDLE_KIND:
        return "decide"
    if kind == "process":
        engine = engine_for(candidate)
        return engine.modality if engine else None
    return None


def language_of(candidate: str) -> str | None:
    """The language a speech candidate can be measured in, or None for the
    lanes that have no language at all.

    For tts it is the EAR's language, not the model's: a French sentence can
    only be scored by a transcriber that speaks French, so the two are the same
    constraint rather than two settings that happen to agree.
    """
    kind = kind_of(candidate)
    if kind not in ("tts", "stt"):
        return None
    options = _options_of(candidate)
    if kind == "stt":
        return options.get("language") or "en"
    _, _, language = options.get("ear", "server").partition(":")
    return language.strip() or "en"


def _options_of(candidate: str) -> dict:
    """The key=value tail of a candidate spec, or {} if it does not parse.

    Selection must not raise: a bad spec is reported by build_runner, with the
    full message, rather than here as a filtering accident.
    """
    _, _, rest = candidate.partition(":")
    _, _, optstr = rest.partition(",")
    try:
        return parse_options(optstr, candidate)
    except ValueError:
        return {}


def engine_for(candidate: str) -> Engine | None:
    try:
        return resolve(candidate)
    except ValueError:
        return None


def _speech_runner(candidate: str, outdir: Path | None) -> SpeechRunner:
    _, _, rest = candidate.partition(":")
    model, _, optstr = rest.partition(",")
    try:
        options = parse_options(optstr, candidate)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    unknown = set(options) - TTS_OPTIONS
    if unknown:
        raise SystemExit(f"unknown tts option(s) {', '.join(sorted(unknown))}; "
                         f"allowed: {', '.join(sorted(TTS_OPTIONS))}")
    if not model:
        raise SystemExit("a tts candidate needs a model, e.g. "
                         "tts:mlx-community/Kokoro-82M-bf16,voice=am_adam")
    if outdir is None:
        raise SystemExit(f"{candidate} writes audio; pass --out")
    ref = options.get("ref_audio", "")
    if ref and not Path(ref).exists():
        # Cloning is the slow lane. Finding the typo forty cases in is the
        # expensive way to find it.
        raise SystemExit(f"{candidate}: no reference audio at {ref}")
    try:
        # Only Kokoro has a voice table; a candidate may legitimately name none.
        # ONLY KOKORO HAS A VOICE TABLE, so "" is right for everything else --
        # bm_george sent to Qwen3-TTS or Marvis is a Kokoro name on a model that
        # has never heard of it. But "" for KOKORO means the server falls back
        # to af_heart, which this machine does not have cached, and the stream
        # dies mid-body. So the default applies to Kokoro and nothing else.
        # Issue #195.
        voice = options.get("voice")
        if voice is None:
            voice = audio.DEFAULT_KOKORO_VOICE if is_kokoro(model) else ""
        return SpeechRunner(model=model, outdir=outdir,
                            voice=voice,
                            ref_audio=ref or None,
                            lang_code=options.get("lang_code", ""),
                            ear=options.get("ear", "server"))
    except ValueError as exc:
        # A bad ear is a bad candidate string, not a traceback.
        raise SystemExit(f"{candidate}: {exc}") from exc


def _transcription_runner(candidate: str) -> TranscriptionRunner:
    _, _, rest = candidate.partition(":")
    model, _, optstr = rest.partition(",")
    model = model.strip()
    try:
        options = parse_options(optstr, candidate)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    unknown = set(options) - STT_OPTIONS
    if unknown:
        raise SystemExit(f"unknown stt option(s) {', '.join(sorted(unknown))}; "
                         f"allowed: {', '.join(sorted(STT_OPTIONS))}")
    if not model:
        raise SystemExit("an stt candidate needs a model, e.g. "
                         "stt:mlx-community/parakeet-tdt-0.6b-v2")
    try:
        return TranscriptionRunner(
            model=model,
            backend=options.get("backend", "server"),
            language=options.get("language", ""))
    except ValueError as exc:
        # A bad backend is a bad candidate string, and gets the same treatment
        # as a bad engine spec: named before the corpus runs, not a traceback.
        raise SystemExit(f"{candidate}: {exc}") from exc


def is_kokoro(model: str) -> bool:
    """Does this model use Kokoro's voice table?

    The voice names are Kokoro's own, so they are meaningful for Kokoro and
    meaningless anywhere else. Issue #195.
    """
    return "kokoro" in model.lower()


def text_candidate(candidate: str) -> bool:
    """A text server answers it: a text spec, or a method over one. #576."""
    if kind_of(candidate) in TEXT_KINDS:
        return True
    base = methods.base_of(candidate)
    return bool(base) and kind_of(base) in TEXT_KINDS


def text_spec(candidate: str) -> str:
    """The text spec under a candidate, for the server it reaches."""
    return methods.base_of(candidate) or candidate


def method_receipts(specs: dict) -> dict:
    """receipt key -> the method and the base it composed over, for each method spec. #576."""
    out = {}
    for key, spec in specs.items():
        try:
            got = methods.parse(spec)
        except ValueError:
            got = None
        if got is not None:
            out[key] = {"spec": spec, "method": got.method.name,
                        "args": list(got.args), "base": got.base}
    return out


def resolved_models(specs: dict, config=None) -> dict:
    """receipt key -> the model id each text candidate resolved to now, for the receipt. #670."""
    return {key: models.resolve(text_spec(spec), config) for key, spec in specs.items()
            if text_candidate(spec)}


def greedy(candidates: list[str]) -> list[str]:
    """Pin text candidates to temperature 0, so a screen is one fixed draw. #308."""
    out = []
    for c in candidates:
        if (text_candidate(c)
                and "temperature" not in parse_options(c.partition(",")[2], c)):
            c = f"{c},temperature=0"
        out.append(c)
    return out


def effective_sampling(modality: str, candidates: list[str]) -> dict:
    """What this run actually asked for, per modality.

    The receipt used to record completion.SAMPLING verbatim, which is the
    SHIPPED table rather than what the run used. A candidate carrying
    `temperature=0.7` produced a receipt identical to one at the default, so
    comparable() answered "same exam" and two temperatures would rank in one
    table. Issue #90.

    The default falls through per modality exactly as complete_with_usage
    resolves it, so a run with no overrides records what it always recorded.
    """
    out = {m: dict(v) for m, v in sorted(completion.SAMPLING.items())}
    seen: dict[str, dict] = {}
    for m in sorted({modality} | set(out)):
        if m and m != "all":
            out.setdefault(m, {})
            out[m].setdefault("temperature", completion.DEFAULT_TEMPERATURE)
    for candidate in candidates:
        if not text_candidate(candidate):
            continue
        _, _, optstr = candidate.partition(",")
        if not optstr:
            continue
        for key, value in parse_options(optstr, candidate).items():
            if kind_of(candidate) == DS4_KIND and key in ds4.OPTIONS:
                continue  # the server's launch, on the receipt as `launch`. #611.
            try:
                value = float(value)
            except ValueError:
                pass
            seen.setdefault(key, {}).setdefault(value, []).append(candidate)
            out.setdefault(modality, {})[key] = value
    # ONE RECEIPT DESCRIBES ONE RUN. Candidates disagreeing on a sampling knob
    # would leave the receipt recording whichever was parsed last, and
    # comparable() treats sampling as an axis precisely because two values are
    # two exams. Refuse rather than record a number that was true of a third of
    # the run. Issue #90.
    for key, values in sorted(seen.items()):
        if len(values) > 1:
            spread = "; ".join(f"{v} ({', '.join(c)})"
                               for v, c in sorted(values.items(), key=str))
            raise SystemExit(
                f"candidates disagree on {key}: {spread}. Two values are two "
                f"exams, so run them separately and compare the results.")
    return out


def run_budget(modality: str, asked: int | None) -> int:
    """The reply budget a run records: the one it was given, else its lane's; 0 where no text is generated. #628."""
    if asked is not None and int(asked) < 1:
        raise SystemExit(f"--max-tokens must be at least 1, not {asked}")
    lane = completion.budget(modality)
    if modality != "all" and not lane:
        return 0
    return int(asked) if asked is not None else lane


def run_ladder(modality: str, asked: str | None) -> tuple[int, ...]:
    """The budget ladder a run climbs: none unless asked, the lane's when asked bare. #668."""
    if asked is None or not completion.ladder(modality):
        return ()
    if asked == "default":
        return completion.ladder(modality)
    try:
        rungs = tuple(int(part.strip()) for part in asked.split(","))
    except ValueError:
        raise SystemExit(f"--budget-ladder takes comma-separated token counts, not {asked!r}") from None
    if len(rungs) < 2 or rungs[0] < 1 or any(b <= a for a, b in zip(rungs, rungs[1:])):
        raise SystemExit(f"--budget-ladder needs two or more budgets of at least 1, smallest "
                         f"first, not {asked!r}")
    return rungs


def rungs_for(ladder: tuple[int, ...], ctx: int | None, *texts: str) -> tuple[int, ...]:
    """The ladder a case climbs: rungs past the served context less the prompt become that room. #668."""
    if ctx is None:
        return tuple(ladder)
    room = delegate.cap(ctx, *texts)
    if room < 1:
        return tuple(ladder[:1])
    kept = [rung for rung in ladder if rung < room]
    if len(kept) < len(ladder):
        kept.append(room)
    return tuple(kept)


def set_budget(runner, max_tokens: int) -> None:
    """Point a runner, and any base a method wraps, at one reply budget. #668."""
    runner.max_tokens = int(max_tokens)
    base = getattr(runner, "base", None)
    if base is not None:
        set_budget(base, max_tokens)


def climb(runner, case: Case, rungs: tuple[int, ...]) -> core.Result:
    """Run `case` at each rung while its reply is cut off at the budget; the last row, with every attempt. #668."""
    attempts = []
    for rung in rungs:
        set_budget(runner, rung)
        row = runner.run(case)
        exhausted = row.failure_class == reasons.TOKEN_BUDGET_EXHAUSTED
        # A reply with no content and no usage spent the whole budget.
        tokens = (row.metrics or {}).get("completion_tokens") or (rung if exhausted else None)
        attempts.append({"max_tokens": rung, "tokens": tokens, "seconds": row.seconds,
                         "passed": bool(row.passed), "failure_class": row.failure_class})
        if row.passed or not exhausted:
            break
    row.attempts = attempts
    row.seconds = round(sum(a["seconds"] for a in attempts), 3)
    return row


def build_runner(candidate: str, gateway: str, outdir: Path | None,
                 adherence: str | None = None, modality: str = "",
                 max_tokens: int = 0):
    if modality in AGENT_MODALITIES:
        runner = _agent_runner(candidate, gateway, max_tokens)
    else:
        runner = _build_runner(candidate, gateway, outdir, adherence, max_tokens)
    runner.spec = candidate
    return runner


def _agent_runner(candidate: str, gateway: str, max_tokens: int = 0):
    """A tool loop for a text candidate, or claude -p on the same sandbox. #474."""
    kind = kind_of(candidate)
    if kind == CLAUDE_CODE_PREFIX:
        from evals.runners.claude_agent import ClaudeAgentRunner
        model = candidate.partition(":")[2].strip()
        if not model:
            raise SystemExit("claude-code needs a model, e.g. claude-code:claude-opus-5-5")
        return ClaudeAgentRunner(model)
    if kind == NEEDLE_KIND:
        from evals.runners.needle import needle_agent_runner
        try:
            return needle_agent_runner(candidate)
        except (ValueError, RunnerError) as exc:
            raise SystemExit(f"{candidate}: {exc}") from None
    if kind not in TEXT_KINDS:
        raise SystemExit(f"{candidate} cannot drive a tool loop; the agent lane "
                         f"takes a text candidate or claude-code:<model>")
    from evals.runners.agent import AgentRunner
    from harness import serving
    try:
        where = serving.route(candidate, gateway)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    from harness import context
    return AgentRunner(where.base, candidate.partition(",")[0].strip(),
                       model=where.model, sampling=where.sampling or None,
                       served_ctx=context.served_ctx(candidate), max_tokens=max_tokens)


def _build_runner(candidate: str, gateway: str, outdir: Path | None,
                  adherence: str | None = None, max_tokens: int = 0):
    kind = kind_of(candidate)
    if kind in TEXT_KINDS:
        # A gateway alias may carry sampling overrides, so a sweep is a command
        # rather than an edit to a constant. `temperature` is the one that had
        # never been varied: completion.SAMPLING pins svg and web at 0.4 and
        # everything else falls through to DEFAULT_TEMPERATURE. Issue #90.
        from harness import serving
        try:
            where = serving.route(candidate, gateway)
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
        return CompletionRunner(where.base, candidate.partition(",")[0].strip(),
                                sampling=where.sampling or None, model=where.model,
                                max_tokens=max_tokens)
    if kind == NEEDLE_KIND:
        from evals.runners.needle import NeedleDecideRunner
        if outdir is None:
            raise SystemExit(f"{candidate} writes decide artifacts; pass --out")
        try:
            return NeedleDecideRunner(candidate, outdir)
        except ValueError as exc:
            raise SystemExit(f"{candidate}: {exc}") from None
    if kind == "tts":
        return _speech_runner(candidate, outdir)
    if kind == "stt":
        return _transcription_runner(candidate)
    if kind == CLAUDE_CODE_PREFIX:
        from evals.runners.claude_code import ClaudeCodeRunner
        model = candidate.partition(":")[2].strip()
        if not model:
            raise SystemExit("claude-code needs a model, e.g. claude-code:claude-opus-5-5")
        return ClaudeCodeRunner(model)
    if kind == REPAIR_PREFIX:
        _, _, rest = candidate.partition(":")
        model, _, optstr = rest.partition(",")
        try:
            options = parse_options(optstr, candidate)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        unknown = set(options) - REPAIR_OPTIONS
        if unknown:
            raise SystemExit(
                f"unknown repair option(s) {', '.join(sorted(unknown))}; "
                f"allowed: {', '.join(sorted(REPAIR_OPTIONS))}")
        if not model.strip():
            raise SystemExit("a repair candidate needs a model, e.g. "
                             "repair:mlx-community/Qwen2.5-7B-Instruct-4bit")
        return RepairRunner(gateway, model.strip(),
                            attempts=int(options.get("attempts", 3)))
    if kind == "chain":
        stage, _, spec = candidate.partition(":")
        if not spec.strip():
            raise SystemExit(f"{candidate} needs a base engine, e.g. "
                             f"{stage}:mflux:flux2-klein-4b")
        try:
            engine = resolve(spec.strip())
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if outdir is None:
            raise SystemExit(f"{candidate} writes images; pass --out")
        return ChainRunner(engine, stage, outdir)
    if kind == OMNISVG_PREFIX:
        _, _, rest = candidate.partition(":")
        size, _, optstr = rest.partition(",")
        try:
            options = parse_options(optstr, candidate)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        unknown = set(options) - OMNISVG_OPTIONS
        if unknown:
            raise SystemExit(
                f"unknown omnisvg option(s) {', '.join(sorted(unknown))}; "
                f"allowed: {', '.join(sorted(OMNISVG_OPTIONS))}")
        try:
            return OmniSVGRunner(
                size.strip() or "4B",
                candidates=int(options.get("candidates", DEFAULT_CANDIDATES)))
        except ValueError as exc:
            raise SystemExit(f"{candidate}: {exc}") from exc
    if kind in methods.REGISTRY:
        def base(spec):
            runner = _build_runner(spec, gateway, outdir, adherence, max_tokens)
            runner.spec = spec
            return runner
        return methods.build(candidate, base, outdir)
    try:
        engine = resolve(candidate)
    except ValueError as exc:
        # Before anything runs: a typo must not cost a forty-minute generation.
        raise SystemExit(str(exc)) from exc
    if outdir is None:
        raise SystemExit(
            f"{candidate} writes files; pass --out to say where they go")
    return ProcessRunner(engine, outdir, adherence=adherence)


def receipt_key(candidate: str) -> str:
    """The summary key a run of `candidate` writes, or "" if it cannot run. #407."""
    import tempfile
    try:
        with tempfile.TemporaryDirectory() as scratch:
            return build_runner(candidate, "", Path(scratch)).candidate
    except (SystemExit, Exception):  # noqa: BLE001
        return ""


def method_of(candidate: str) -> str:
    """Which METHOD this candidate is, for cases that declare what they can
    fairly test.

    Not a taxonomy of models. The only distinction any case has needed is
    whether a method can put a <text> element in the document at all, and two
    of the three here cannot: `trace` turns glyphs into outlines, and OmniSVG's
    tokenizer emits move/line/curve/arc/close and nothing else. Calling
    everything that is not `trace` an LLM handed chart-bars to OmniSVG, which
    then failed it for missing <text> -- measuring the method, which is the
    exact thing Case.methods exists to prevent.
    """
    kind = kind_of(candidate)
    if kind in methods.REGISTRY:
        return methods.REGISTRY[kind].takes or method_of(methods.base_of(candidate))
    if kind == OMNISVG_PREFIX:
        return OMNISVG_PREFIX
    return "llm"


def unrunnable(candidate: str, modality: str | None) -> str:
    """Why a candidate was skipped, naming the typed spelling a bare repo needed. #645."""
    why = "no cases of a modality it can run, skipped"
    if modality in ("tts", "stt") and kind_of(candidate) != modality:
        why += f"; a {modality} candidate is spelled {modality}:{candidate}"
    return why


def cases_for(candidate: str, cases: list[Case]) -> list[Case]:
    """The cases this candidate can actually run."""
    modality = modality_of(candidate)
    if kind_of(candidate) == NEEDLE_KIND:
        return [c for c in cases if c.modality in NEEDLE_LANES]
    if modality is None:
        text = [c for c in cases if c.modality in TEXT_MODALITIES | AGENT_MODALITIES]
        return _fair(candidate, text) if kind_of(candidate) in methods.REGISTRY else text
    picked = [c for c in cases if c.modality == modality]
    return _fair(candidate, picked)


def _fair(candidate: str, picked: list[Case]) -> list[Case]:
    """The cases whose declared methods take this candidate, in its method's own lanes."""
    method = methods.REGISTRY.get(kind_of(candidate))
    if method is not None and method.lanes:
        picked = [c for c in picked if c.modality in method.lanes]
    # A case may declare which methods it can fairly test. See Case.methods.
    picked = [c for c in picked
              if not c.methods or method_of(candidate) in c.methods]
    language = language_of(candidate)
    if language is not None:
        picked = [c for c in picked if c.language == language]
    return picked


# Modalities whose output varies run to run. Diffusion varies enormously with
# the seed and a language model at temperature 0.2 is not deterministic either;
# Kokoro at a fixed voice and speed produces the same bytes every time, so
# repeating it burns time averaging three identical numbers. `extract` is
# excluded for a different reason: the answer is one token and the whole point
# of the lane is that it is cheap, so three samples of "137" buys nothing.
#
# `music` is here on MEASUREMENT rather than on principle: four runs of the
# same case at a pinned seed scored wer 0.1515, 0.0000, 0.3333 and 0.1053,
# because ACE-Step's 5Hz chain-of-thought samples at its own temperature and
# `seed` only reaches the diffusion below it (RULE #280). Omitting it made
# --repeat a silent no-op for the one lane that needs it most.
STOCHASTIC_MODALITIES = {"image", "video", "svg", "web", "code", "music", "agent"}


#: Screen settings: the smallest thing that still proves the pipeline ran.
SCREEN_PARAMS = {"width": 256, "height": 256, "steps": 2, "seconds": None,
                 "frames": None}


def screen_pool(cases: list[Case]) -> list[Case]:
    """The dev cases, plus any (modality, language) only holdout covers. #479."""
    from harness import holdout
    dev = holdout.only(cases, "dev")
    have = {(c.modality, c.language) for c in dev}
    return dev + [c for c in cases if c not in dev
                  and (c.modality, c.language) not in have]


def _first_each(cases: list[Case]) -> list[Case]:
    """The first case by id for each (modality, language)."""
    picked: dict[tuple[str, str], Case] = {}
    for c in sorted(cases, key=lambda c: c.id):
        picked.setdefault((c.modality, c.language), c)
    return list(picked.values())


def screen_cases(cases: list[Case], candidates=()) -> list[Case]:
    """One case per modality AND LANGUAGE, shrunk. Cheap enough to be wrong about.

    Keyed on modality alone this took the alphabetically first id, so the tts
    lane always picked `fr-liaison` and cases_for -- which filters by the
    candidate's language -- then matched nothing for every English candidate.
    The tier returned "no cases of a modality it can run" for the lane's own
    adopted default. Issue #194.

    Per candidate the cost is unchanged: cases_for still narrows to the one
    case in that candidate's language.

    Given `candidates`, each one's pick comes from the cases its method can
    take, so omnisvg is not handed chart-bars and left with nothing. #555.
    """
    import dataclasses
    groups = [cases_for(c, cases) for c in candidates] or [cases]
    picked: dict[str, Case] = {}
    for group in groups:
        for c in _first_each(group):
            picked.setdefault(c.id, c)
    out = []
    for c in picked.values():
        params = dict(c.params)
        for k, v in SCREEN_PARAMS.items():
            if k in params and v is not None:
                params[k] = min(params[k], v) if isinstance(params[k], int) else v
        out.append(dataclasses.replace(c, params=params))
    return out


def expand_cases(cases: list[Case], repeat: int) -> list[Case]:
    """Turn each stochastic case into `repeat` cases with different seeds.

    One sample per prompt ranks noise. A single generation decides a comparison
    on luck, and the whole point of the suite is that the comparison means
    something.
    """
    if repeat < 1:
        raise SystemExit(f"--repeat must be at least 1, got {repeat}")
    if repeat == 1:
        return cases

    out: list[Case] = []
    for case in cases:
        if case.modality not in STOCHASTIC_MODALITIES:
            out.append(case)
            continue
        base = case.params.get("seed", 0)
        for i in range(repeat):
            out.append(replace(case, id=f"{case.id}#{i + 1}",
                               # A copy, never the same dict: Case is frozen but
                               # its params are not, and mutating them would
                               # rewrite the case the caller still holds.
                               params={**case.params, "seed": base + i}))
    return out


def unrun_summary(cases: list[Case], candidates: list[str]) -> str:
    """Which cases no candidate in this run can execute.

    A case that silently never runs is invisible, and the summary below it
    looks complete. That is the same class of problem as a silent pass: an
    entire lane goes unmeasured and nothing says so.
    """
    covered = {c.id for candidate in candidates
               for c in cases_for(candidate, cases)}
    missed = [c for c in cases if c.id not in covered]
    if not missed:
        return ""
    by_modality: dict[str, list[str]] = {}
    for case in missed:
        by_modality.setdefault(case.modality, []).append(case.id)
    return "; ".join(f"{modality}: {', '.join(ids)}"
                     for modality, ids in sorted(by_modality.items()))


def select_cases(cases: list[Case], modality: str) -> list[Case]:
    if modality != "all" and modality not in MODALITIES:
        raise SystemExit(f"unknown modality '{modality}'; known: "
                         f"{', '.join(ALL_MODALITIES)}")
    # The agent lane drives a tool loop for minutes per case, so `all` leaves it out. #474.
    chosen = ([c for c in cases if c.modality not in AGENT_MODALITIES]
              if modality == "all" else [c for c in cases if c.modality == modality])
    if not chosen:
        raise SystemExit(f"no cases for modality '{modality}'")
    return chosen


def load_selected(args) -> list[Case]:
    """The run's cases: --cases plus, when --cases is the shipped tree, the local-only ones. #654."""
    shipped = Path(args.cases).resolve() == core.SHIPPED.resolve()
    local = Path(args.local_cases) if getattr(args, "local_cases", None) else core.local_root()
    loaded = core.load_suite(args.cases, local) if shipped else core.load_cases(args.cases)
    return select_cases(loaded, args.modality)


def parse_args(argv: list[str] | None = None):
    return _parser().parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return _main(args)


def _parser():
    ap = argparse.ArgumentParser(prog="evals.run")
    ap.add_argument("--compare", nargs="+", metavar="RESULTS.JSON",
                    help="put finished runs in one table, or refuse if their "
                         "receipts say they are not comparable")
    ap.add_argument("--across", default="",
                    help="with --compare, the ONE receipt axis these runs vary "
                         "on. Reports what changed cell by cell instead of "
                         "ranking them, and still refuses if a second axis "
                         "also differs")
    ap.add_argument("--modality", required=False,
                    help=f"one of {', '.join(ALL_MODALITIES)}, or 'all'")
    ap.add_argument("--candidates", required=False,
                    help="comma-separated gateway aliases and/or engine specs")
    ap.add_argument("--from-winners", action="store_true",
                    help="take the candidate from what the stored runs say "
                         "won this lane, rather than from a constant. A "
                         "deployment built around 'the current winner' must "
                         "read it rather than be told it")
    ap.add_argument("--gateway", default="http://127.0.0.1:4000")
    ap.add_argument("--cases", default=str(ROOT / "cases"))
    ap.add_argument("--local-cases", default=None,
                    help="local-only cases added to the shipped ones "
                         "(default: $LOCALHARNESS_HOME/cases)")
    ap.add_argument("--out", default=None,
                    help="write artifacts and results.json here "
                         "(default: a new directory under "
                         "$LOCALHARNESS_HOME/runs)")
    ap.add_argument("--adherence", choices=("pickscore", "hpsv2"), default=None,
                    help="score how well each image matches its prompt. Loads a "
                         "multi-GB preference model, so it is opt-in and needs "
                         "`uv sync --group metrics`. Scores from the two "
                         "backends are NOT comparable to each other")
    ap.add_argument("--screen", action="store_true",
                   help="cheapest tier: one case per candidate at minimal "
                        "settings, answering only whether it runs. Never a "
                        "ranking")
    ap.add_argument("--repeat", type=int, default=1,
                    help="samples per case, each with a different seed. One "
                         "sample per prompt ranks noise; 3 is the usual "
                         "minimum for an image comparison you would act on")
    ap.add_argument("--split", choices=("all", "dev", "holdout"), default="all",
                    help="which side of the per-lane holdout split to run; the "
                         "screen always runs dev (#479)")
    ap.add_argument("--knob", action="append", default=[], metavar="NAME=VALUE",
                    help="run at this setting of a registered exam knob, recorded on the "
                         "receipt; repeatable (harness/knobs.py, #636)")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="reply budget for every text request, recorded on the "
                         "receipt (default: the lane's, completion.BUDGET) (#628)")
    ap.add_argument("--budget-ladder", nargs="?", const="default", default=None,
                    metavar="N,N,...",
                    help="retry a reply cut off at its budget at the next larger one, e.g. "
                         "4000,16000,32000,65536; bare, the lane's (completion.LADDER). Capped by "
                         "each candidate's served context, recorded on the receipt (#668)")
    return ap


def _main(args) -> int:
    # SAME GUARD AS `lh`, and this is the entry point that actually downloads:
    # `lh discover --screen` prints this very command for a user to copy, so
    # without it the screen tier hands out the unguarded path. Issue #191.
    env.guard()
    if args.compare:
        return compare_runs(args.compare, getattr(args, "across", ""))
    # Required for a RUN, not for a comparison. Left off `required=True` so
    # `--compare` can stand alone; enforced here so a normal run still fails
    # loudly rather than halfway through.
    missing = [f"--{n}" for n in ("modality", "candidates")
               if not getattr(args, n)]
    if missing:
        raise SystemExit(f"{' and '.join(missing)} required (or use --compare)")
    from harness import exclusive
    # One model-loading run on the machine at a time, across projects. #314.
    with exclusive.held("eval", announce=lambda m: print(
            f"evals.run: {m}", file=sys.stderr, flush=True)):
        return _execute(args)


def knob_overrides(pairs) -> dict:
    """{knob: value} from NAME=VALUE pairs; a name that is not an exam knob is refused. #636."""
    out = {}
    for pair in pairs or ():
        name, sep, value = str(pair).partition("=")
        if not sep:
            raise SystemExit(f"--knob takes NAME=VALUE, not {pair!r}")
        try:
            out[name.strip()] = float(value)
        except ValueError:
            raise SystemExit(f"--knob {name}: {value!r} is not a number") from None
    try:
        knobs.settings("", out)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    return out


def _execute(args) -> int:
    overrides = knob_overrides(getattr(args, "knob", None))
    with knobs.rebind(overrides):
        return _execute_at(args, overrides)


def _execute_at(args, overrides: dict) -> int:
    if args.screen:
        # A screen is allowed to be statistically worthless. Its job is to
        # reject what does not run at all, which is how most things here have
        # failed: seedvr2 crashed 0/3, local-small never closed a tag 0/9.
        args.repeat = 1
        args.adherence = ""
    from harness import holdout
    side = "dev" if args.screen else (getattr(args, "split", "") or "all")
    chosen = load_selected(args)
    cases = expand_cases(screen_pool(chosen) if args.screen
                         else holdout.only(chosen, side), args.repeat)
    # An engine spec contains commas, which are also the candidate separator.
    # Split on commas that start a new candidate, i.e. those followed by a
    # known engine prefix or by something with no '=' in it.
    if getattr(args, "from_winners", False):
        if args.candidates:
            raise SystemExit("--from-winners and --candidates say two "
                             "different things about what to run; pass one")
        args.candidates = winner_for(args.modality)
        print(f"── winner for {args.modality}: {args.candidates}", flush=True)
    candidates = split_candidates(args.candidates)
    ladder = run_ladder(args.modality, getattr(args, "budget_ladder", None))
    if ladder and getattr(args, "max_tokens", None) is not None:
        raise SystemExit("--budget-ladder and --max-tokens say two things about the reply "
                         "budget; pass one")
    budget = ladder[-1] if ladder else run_budget(args.modality, getattr(args, "max_tokens", None))
    if args.screen:
        candidates = greedy(candidates)
        cases = screen_cases(cases, candidates)
    outdir = resolve_outdir(args.out, args.modality)
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    pressed = warn_if_pressed()
    skipped = unrun_summary(cases, candidates)
    if skipped:
        print(f"\nnot run, no candidate for them -- {skipped}", file=sys.stderr)

    results, specs, planned, launches, evicted = [], {}, [], {}, {}
    for candidate in candidates:
        mine = cases_for(candidate, cases)
        if args.screen:
            mine = _first_each(mine)
        if not mine:
            print(f"\n── {candidate}: {unrunnable(candidate, args.modality)}",
                  file=sys.stderr)
            continue
        runner = build_runner(candidate, args.gateway, outdir,
                              adherence=args.adherence, modality=args.modality,
                              max_tokens=budget)
        knobs.apply(runner, overrides)
        if runner.candidate in specs:
            # One key, one row set and one artifact name: the second would overwrite the first. #429.
            raise SystemExit(f"{specs[runner.candidate]} and {candidate} both "
                             f"run as {runner.candidate}; run them separately")
        specs[runner.candidate] = candidate
        _servable(candidate)
        planned.append((candidate, runner, mine))
        models.warn(candidate)
    resolved = resolved_models(specs)
    for candidate, runner, mine in planned:
        with _served(candidate) as launched:
            if launched:
                launches[runner.candidate] = ds4.launch_text(launched["launch"])
            _run_candidate(args, candidate, runner, mine, results, outdir, evicted, ladder)

    if not results:
        raise SystemExit("nothing ran: no candidate matched any case")

    report(summarize(results), resolved)
    if outdir:
        # The RECEIPT: what this run was, so a later run can be told apart
        # from it before anyone ranks the two together. See core.comparable().
        receipt = Receipt(
            modality=args.modality,
            case_ids=tuple(sorted({c.id for c in cases})),
            repeat=args.repeat,
            sampling=effective_sampling(args.modality, candidates),
            gateway=args.gateway,
            adherence=getattr(args, "adherence", "") or "",
            tier="screen" if getattr(args, "screen", False) else "measure",
            accelerator=accelerator_id(),
            instruments=instruments(candidates),
            engines=engines(candidates),
            where=where_id(),
            swap_used_mb=swap_used_mb(),
            pressure=pressed.as_dict(),
            cases_digest=cases_digest(cases),
            split=side, split_version=holdout.VERSION,
            methods=method_receipts(specs),
            devices=devices(results),
            launch=launches,
            max_tokens=budget,
            knobs=knobs.settings(args.modality, overrides),
            router_swaps=evicted,
            budget_ladder=ladder,
            resolved=resolved)
        now = time.time()
        ids = candidate_ids(specs, args.modality)
        for r in results:
            r.candidate_id = ids.get(r.candidate)
        payload = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S",
                                              time.localtime(now)),
                   "environment": environment.capture(),
                   "receipt": receipt.as_dict(),
                   # A name is a label and can drop the repo org; the spec
                   # runs. #337.
                   "specs": specs,
                   "summary": summarize(results),
                   "rows": [export_row(r) for r in results]}
        (outdir / "results.json").write_text(json.dumps(payload, indent=2),
                                             encoding="utf-8")
        run_id = store_run(outdir, payload, now)
        print(f"\nartifacts + results.json in {outdir}; stored as run {run_id}")
    return 0


def _servable(candidate: str) -> None:
    """Refuse before anything runs when ds4-server cannot serve this candidate. #611."""
    if kind_of(candidate) != DS4_KIND:
        return
    try:
        ds4.preflight(candidate)
    except (ds4.Unservable, ValueError) as exc:
        raise SystemExit(f"{candidate}: {exc}") from None


@contextlib.contextmanager
def _served(candidate: str):
    """The ds4-server launch a ds4 candidate runs against, started for it if none is up."""
    if kind_of(candidate) != DS4_KIND:
        yield None
        return
    try:
        with ds4.serving(candidate) as state:
            yield state
    except ds4.Unservable as exc:
        raise SystemExit(f"{candidate}: {exc}") from None


def _run_candidate(args, candidate, runner, mine, results, outdir, evicted=None,
                   ladder=()) -> None:
    runner.screening = bool(args.screen)
    model = router.model_for(candidate, args.gateway)
    watch = router.Watch(model) if model else None
    print(f"\n── {runner.candidate}", flush=True)
    # A cold load is not the case's to pay for; a failed one is every case's. #406.
    cold = None
    if args.screen:
        try:
            runner.warm()
        except RunnerError as exc:
            cold = exc
    ctx = _served_ctx(candidate) if ladder else None
    for case in mine:
        if watch:
            watch.look(case.id, "start")
        if cold:
            r = runner.failed(case, cold)
        elif ladder:
            r = climb(runner, case, rungs_for(ladder, ctx, case.prompt, case.context))
        else:
            r = runner.run(case)
        if watch:
            watch.look(case.id, "end")
        private.withhold_detail(r, case)
        results.append(r)
        mark = "pass" if r.passed else "FAIL"
        note = "" if r.passed else f"  {r.detail}"
        warn = f"  ({len(r.warnings)} warn)" if r.warnings else ""
        first = ("" if r.ttft_s is None else
                 f"  ttft {r.ttft_s:.2f}s{' cold' if r.cold else ''}")
        print(f"  {mark}  {r.seconds:6.2f}s  {case.id}{warn}{first}{note}",
              flush=True)
        if outdir and r.output:
            f = outdir / runner.artifact(
                case, TEXT_SUFFIX.get(case.modality, ".txt"))
            f.write_text(r.output, encoding="utf-8")
            r.artifact_path = str(f.resolve())
        if case.private:
            # The reply stays in the run dir on this machine; the row carries no text. #654.
            r.output = None
    if watch and watch.swaps and evicted is not None:
        evicted[runner.candidate] = watch.swaps
        print(f"  router evicted {model} {len(watch.swaps)} time(s) during this run; "
              f"it will not be ranked", file=sys.stderr)


def _served_ctx(candidate: str) -> int | None:
    """The per-slot context a candidate is served at, which caps its ladder; None when unknown. #668."""
    try:
        return context.served_ctx(candidate)
    except Exception:  # noqa: BLE001
        return None


def export_row(r) -> dict:
    """A results.json row; `artifact` is the deprecated pre-#463 field."""
    return {**vars(r), "artifact": r.output if r.output is not None
            else r.artifact_path}


def candidate_ids(specs: dict, lane: str) -> dict:
    """receipt key -> the candidates row this run's spec for it is. #429."""
    from harness import candidates as C
    from harness import memory_store as ms
    conn = ms.connect()
    try:
        return {key: C.ensure(conn, spec, key=key, lane=lane)
                for key, spec in specs.items()}
    finally:
        conn.close()


def store_run(outdir, payload: dict, at: float) -> int | None:
    """The run and its rows into the store; results.json is the export. #410."""
    from harness import memory_store as ms
    from harness import runs, workqueue
    conn = ms.connect()
    try:
        run_id = runs.record(conn, outdir, payload, at=at)
        workqueue.link_run(conn, run_id)
        return run_id
    finally:
        conn.close()


def devices(results) -> dict:
    """receipt key -> the "device:attention" pairs its rows answered on, where a runner said. #604."""
    seen: dict = {}
    for r in results:
        rt = getattr(r, "runtime", None) or {}
        if rt.get("device"):
            seen.setdefault(r.candidate, set()).add(f"{rt['device']}:{rt.get('attn') or 'default'}")
    return {k: ",".join(sorted(v)) for k, v in seen.items()}


def engines(candidates) -> dict:
    """Which server answered each text candidate. #295."""
    from harness import serving
    return {c: serving.engine_for(text_spec(c)) for c in candidates
            if text_candidate(c)}


def instruments(candidates=()) -> dict:
    """What measured this run, for the receipt. See core.Receipt.instruments.

    Best effort by design: a receipt must not fail a finished run, and an
    instrument nobody could name is recorded as absent rather than as a
    mismatch -- comparable() compares only the keys both runs carry.
    """
    found = {}
    try:
        from harness import proc
        found["peak"] = proc.PEAK_METHOD
    except Exception:  # noqa: BLE001
        pass
    try:
        from harness.checks import ocr
        found["ocr"] = ocr.available_backend() or ""
    except Exception:  # noqa: BLE001
        pass
    try:
        # WHICH SERVER PRODUCED THE TOKENS. Without it two runs across
        # different engines look like the same exam to comparable(). Issue #190.
        from harness import serving
        found["serving"] = ("+".join(sorted(set(engines(candidates).values())))
                            or serving.text_engine())
    except Exception:  # noqa: BLE001
        pass
    return {k: v for k, v in found.items() if v}


#: Below this, wall-clock is not comparable with a quiet machine. #283.
FREE_PCT_WARN = 25


def swap_used_mb() -> int:
    """Swap in use, or 0 if it cannot be told. Descriptive only. #283."""
    try:
        return int(environment.capture().get("swap_used_mb") or 0)
    except Exception:  # noqa: BLE001 - a receipt must not fail a finished run
        return 0


def winner_for(modality: str, conn=None) -> str:
    """The candidate the stored runs say won this lane. #148 phase 5.

    RAISES WHEN NOTHING MEASURED IT: a lane built around the current winner
    that falls back to a typed constant is the constant again.
    """
    from harness import memory_store as ms
    from harness import winners
    close = conn is None
    conn = conn or ms.connect()
    try:
        best = winners.beaten_in(conn) or winners.from_receipts(conn)
    finally:
        if close:
            conn.close()
    got = best.get((modality or "").strip().lower())
    if not got:
        raise SystemExit(
            f"--from-winners: no stored run names a winner for {modality!r} "
            f"on this machine, so there is nothing to build a run around. "
            f"Measure the lane first, or pass --candidates explicitly.")
    return got["candidate"]


def warn_if_pressed(got=None, out=None):
    """Warn before the numbers appear if the machine is not quiet. #283."""
    import sys
    from harness import pressure
    got = pressure.sample() if got is None else got
    why = ""
    if got.alarming:
        why = f"macOS reports memory pressure level {got.level}"
    elif got.free_pct is not None and got.free_pct <= FREE_PCT_WARN:
        why = f"{got.free_pct}% of memory free"
    if why:
        print(f"\nWARNING: {why}. Wall-clock numbers from this run are not "
              f"comparable with ones taken on a quiet machine; peak memory is "
              f"unaffected.", file=out or sys.stderr, flush=True)
    return got


def where_id() -> str:
    """The receipt's `where` field, empty if it cannot be told. Same rule as
    accelerator_id: a receipt must not fail a finished run, and an unknown
    place is read as unknown rather than as a mismatch."""
    try:
        from harness import machine
        return machine.where()
    except Exception:  # noqa: BLE001
        return ""


def accelerator_id() -> str:
    """The receipt's accelerator field. `<kind>:<name>`, or empty if the
    machine cannot be asked -- an empty field is treated as unknown by
    comparable() rather than as a mismatch."""
    try:
        from harness import machine
        acc = machine.detect().accelerator
    except Exception:  # noqa: BLE001 - a receipt must not fail a finished run
        return ""
    return f"{acc.kind}:{acc.name}" if acc.kind else ""


def compare_across(axis: str, loaded: list) -> int:
    """A sweep over ONE axis, compared cell by cell rather than ranked.

    comparable() refuses these runs, correctly: two sampling settings are two
    exams and must not share a leaderboard. Asking what the axis DOES is the
    opposite question, so this is a named exception and not a bypass -- a run
    differing on a SECOND axis is still refused, because a sweep whose
    configurations differ in two ways cannot say which one moved the result.
    """
    from harness import paired

    if not paired.is_axis(axis):
        print(f"unknown axis {axis!r}; one of {', '.join(paired.AXES)}, or knobs.<name>")
        return 1
    base_file, base, _, base_rows = loaded[0]
    for f, receipt, _, _ in loaded:
        why = core.contaminated(receipt)
        if why:
            print(f"REFUSED: {f} cannot be compared -- {why}.")
            return 1
    for f, receipt, _, rows in loaded[1:]:
        differs = paired.differences(base, receipt)
        if axis not in differs:
            print(f"REFUSED: {base_file} and {f} agree on {axis}, so there is "
                  f"nothing to sweep. Nothing varied.")
            return 1
        extra = [d for d in differs if d != axis]
        if extra:
            print(f"REFUSED: {base_file} and {f} differ on {', '.join(extra)} "
                  f"as well as {axis}. A sweep that varies two things cannot "
                  f"say which one moved the result.")
            return 1
        print(f"\n{axis}: {paired.value(base, axis)!r} -> {paired.value(receipt, axis)!r}")
        print(f"  {'candidate':34} {'lost':>5} {'gained':>7} {'same':>6} "
              f"{'p':>7}  verdict")
        for cell in paired.cells(base_rows, rows):
            print(f"  {cell.candidate:34} {cell.lost:5d} {cell.gained:7d} "
                  f"{cell.unchanged:6d} {cell.p:7.2f}  {cell.verdict}")
    print("\n  Paired by case and repeat index. p is the exact McNemar test on "
          "discordant\n  cells; concordant cells carry no information about a "
          "change.")
    return 0


def compare_runs(files: list[str], across: str = "") -> int:
    """Put two or more finished runs in one table -- or refuse to.

    `comparable()` has existed and been tested since the receipts were added,
    and nothing called it. A guard that nothing invokes is a function.

    Refusing is the feature. Rows have been ranked here across runs with
    different sampling and different candidate sets, and the reader had no way
    to know.
    """
    import json

    loaded = []
    for f in files:
        data = stored_receipt(f)
        try:
            data = data or json.loads(Path(f).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"cannot read {f}: {exc}")
            return 1
        raw = data.get("receipt")
        if not raw:
            print(f"{f} carries no receipt, so it cannot be compared with "
                  f"anything. Runs written before receipts existed are in this "
                  f"state; re-run to get one.")
            return 1
        loaded.append((f, Receipt.from_dict(raw),
                       data.get("summary") or {},
                       data.get("rows") or []))

    first_file, first, _, _ = loaded[0]
    if across:
        return compare_across(across, loaded)
    for f, receipt, _, _ in loaded[1:]:
        ok, why = comparable(first, receipt)
        if not ok:
            print(f"REFUSED: {first_file} and {f} are not comparable -- {why}.")
            print("Ranking them in one table would compare two different exams.")
            return 1

    merged: dict = {}
    for f, _, summary, _ in loaded:
        for name, row in summary.items():
            # Same candidate in two comparable runs: keep them apart by file,
            # since two samples of one candidate is a repeat, not a duplicate.
            key = name if name not in merged else f"{name} ({Path(f).parent.name})"
            merged[key] = row
    report(merged)
    return 0


def stored_receipt(f) -> dict | None:
    """The stored run a path names (its dir or its results.json), or None."""
    from harness import memory_store as ms
    from harness import runs
    p = Path(f)
    where = p.parent if p.name == "results.json" else p
    conn = ms.connect()
    try:
        return runs.receipt_at(conn, where)
    finally:
        conn.close()


def resolve_outdir(out: str | None, modality: str) -> Path:
    """Where this run's artifacts and results.json go.

    Always somewhere, and always the same shape. --out used to be REQUIRED for
    any lane that writes a file, so every invocation in this repo's history
    picked a directory by hand -- .logs/img, .logs/voices, .logs/ev-svg-fair --
    and none of them agreed.
    """
    if out:
        return Path(out)
    return paths.new_run(modality)


def split_candidates(raw: str) -> list[str]:
    """Split a candidate list on commas, keeping engine options attached.

    `mflux:z-image-turbo,quantize=4,local-mid` is two candidates, not three:
    a chunk containing '=' belongs to the candidate before it.
    """
    out: list[str] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" in chunk and out:
            out[-1] += f",{chunk}"
        else:
            out.append(chunk)
    return out


def _note_ranking_disagreements(summary: dict, metric_names: list) -> None:
    """Say so when the pass rate and a metric order the candidates differently.

    Seen live on the code lane: one model passed 50% of cases to another's 33%
    while writing 52.8% correct code to the other's 75.6%. A case with six
    strict checks fails outright on a single miss, so the binary view punishes
    the better model. Both numbers are true and neither is the answer on its
    own, so the reader has to be told they point different ways rather than
    reading the top line as a verdict.
    """
    if len(summary) < 2:
        return
    by_rate = [n for n, _ in sorted(summary.items(),
                                    key=lambda kv: -kv[1]["pass_rate"])]
    for metric in metric_names:
        # A TIE on pass rate cannot disagree with anything. Three parakeets all
        # scored 40/40 and the note still announced that one "passes more
        # cases", which is not a disagreement, it is the note describing sort
        # order noise.
        rates = {s.get("pass_rate", 0) for s in summary.values()}
        if len(rates) < 2:
            continue

        # A metric computed over a DIFFERENT NUMBER OF CASES per candidate is
        # not a ranking, and this note was making exactly that comparison: on
        # its first live run it announced that local-mid scored better on ink,
        # from the one case it passed, against a rival scored over two.
        partial = {name for name, s in summary.items()
                   if s.get("metric_n", {}).get(metric, s["total"]) < s["total"]}
        if partial:
            continue

        # Only candidates that actually reported this metric. A model that
        # failed every case has no opinion about it, and reading its absence as
        # a score is how a broken candidate wins.
        scored = {n: s for n, s in summary.items() if metric in s["metrics"]}
        if len(scored) < 2:
            continue
        by_metric = [n for n, _ in sorted(
            scored.items(), key=lambda kv: kv[1]["metrics"][metric],
            reverse=direction_of(metric) == "higher")]
        by_rate_scored = [n for n in by_rate if n in scored]
        if by_metric[0] != by_rate_scored[0]:
            print(f"\n  Note: pass rate and {metric} DISAGREE. {by_rate_scored[0]} "
                  f"passes more cases; {by_metric[0]} scores better on "
                  f"{metric}. A case fails outright on one missed check, so "
                  f"the pass rate punishes a strong model that slips once.")


def report(summary: dict, resolved: dict | None = None) -> None:
    """Print the comparison, ordered by what actually distinguishes candidates.

    Pass rate first, then the quality metrics, then latency. Sorting on latency
    before quality is how a faster-but-worse candidate ends up on the top line.
    """
    metric_names = sorted({name for s in summary.values()
                           for name in s.get("metrics", {})})

    def rank(item):
        _, s = item
        # Negated for a higher-is-better metric so one ascending sort handles
        # both. Ranking every metric as an error rate put the candidate that
        # drew LEAST on the top line.
        scores = []
        for name in metric_names:
            # A NEUTRAL metric is reported and never ranked on. completion
            # tokens is the case: more is not better, and as higher-is-better
            # it would have put the most verbose candidate first.
            if direction_of(name) == "neutral":
                continue
            value = (s.get("metrics") or {}).get(name)
            if value is None:
                # A candidate that reported nothing must sort LAST, not as a
                # perfect score. Defaulting a lower-is-better metric to 0.0 put
                # a model that failed all 40 cases at the top of the ranking.
                scores.append(float("inf"))
            else:
                scores.append(value if direction_of(name) == "lower" else -value)
        return (-s.get("pass_rate", 0), scores, s.get("median_s", 0))

    arrows = {n: {"lower": "v", "higher": "^"}.get(direction_of(n), "-")
              for n in metric_names}
    header = (f"{'candidate':30} {'pass':>7} {'rate':>6} {'median':>8} "
              f"{'ttft':>8} {'first':>8} {'peak':>9}")
    for name in metric_names:
        header += f" {name + ' ' + arrows[name]:>9} {name + '.worst':>13}"
    # A wrong answer and a reply cut off at the budget are two findings. #628.
    header += f" {'wrong':>6} {'budget':>6}"
    print("\n" + "=" * len(header))
    print(header)

    for name, s in sorted(summary.items(), key=rank):
        # .get throughout: report() now also renders summaries READ FROM DISK
        # for --compare, and a run written by an older version will not have
        # every key this one expects. Degrading is right; a KeyError on a
        # historical result is not.
        peak_kb = s.get("peak_kb") or 0
        peak = f"{peak_kb / 1024 / 1024:.1f}GiB" if peak_kb else "-"
        ttft = s.get("ttft_median_s")
        ttft = f"{ttft:>7.2f}s" if ttft is not None else f"{'-':>8}"
        line = (f"{name:30} {s.get('passed', 0):>3}/{s.get('total', 0):<3} "
                f"{s.get('pass_rate', 0):>6.0%} "
                f"{s.get('median_s', 0):>7.2f}s {ttft} "
                f"{s.get('first_s', 0):>7.2f}s {peak:>9}")
        for metric in metric_names:
            value = (s.get("metrics") or {}).get(metric)
            worst = s.get("metrics_worst", {}).get(metric)
            line += (f" {value:>9.3f}" if value is not None else f" {'-':>9}")
            line += (f" {worst:>13.3f}" if worst is not None else f" {'-':>13}")
        line += f" {s.get('wrong', 0):>6} {s.get('budget', 0):>6}"
        print(line)

    named = {n: models.display(n, resolved) for n in summary if models.is_alias(n)}
    if named:
        print("\nmodels behind the aliases above:")
        for n, shown in named.items():
            print(f"  {n} = {shown}")

    agents = {n: (s2.get("agent") or {}).get("ctx") for n, s2 in summary.items()
              if "agent" in s2}
    if agents:
        print("\nserved context (tokens per slot; a long case past it fails on step 1):")
        for n, ctx in agents.items():
            print(f"  {n}: {ctx or 'unknown'}")

    rungs = {n: s2["by_rung"] for n, s2 in summary.items() if s2.get("by_rung")}
    if rungs:
        print("\npass by rung (passed / reached each budget, and the seconds spent there):")
        for n, got in rungs.items():
            print(f"  {n}: " + ", ".join(f"{m} {g['passed']}/{g['tried']} in {g['seconds']:.0f}s"
                                         for m, g in got.items()))

    cold = [n for n, s2 in summary.items() if s2.get("first_is_cold")]
    if cold:
        print(f"\n`first` is the FIRST case, not a warm number. For {cold[0]} it "
              f"is a cold start;\nfor the others the runtime was already up. A "
              f"one-shot caller meets `first`.")

    if metric_names:
        low = [n for n in metric_names if direction_of(n) == "lower"]
        high = [n for n in metric_names if direction_of(n) == "higher"]
        flat = [n for n in metric_names if direction_of(n) == "neutral"]
        legend = []
        if low:
            legend.append(f"{', '.join(low)}: lower is better (v)")
        if high:
            legend.append(f"{', '.join(high)}: higher is better (^)")
        if flat:
            legend.append(f"{', '.join(flat)}: reported, not ranked on (-)")
        print("  " + "; ".join(legend))

    # HOW they failed, not just how often. Qwen3-14B scored 7/9 on svg and
    # looked like a winner; both failures were null content from a thinking
    # model that spent the budget reasoning. The pass rate could not say so.
    for name, s in sorted(summary.items(), key=rank):
        kinds = [(k, s.get(k, 0)) for k in ("wrong", "empty", "errored", "budget")]
        shown = [f"{n} {k}" for k, n in kinds if n]
        if len(shown) > 0 and s.get("passed", 0) < s.get("total", 0):
            print(f"  {name}: {', '.join(shown)}")

    # Two candidates in ONE run can sit different exams: a case may be unfair to
    # a method (chart-bars asserts <text>, which tracing cannot emit) or to a
    # language. Their pass rates are then not comparable, and 4/4 against 2/6
    # reads exactly as though they were.
    sat = {name: set(s.get("case_ids", [])) for name, s in summary.items()}
    if len({frozenset(v) for v in sat.values()}) > 1:
        shared = set.intersection(*sat.values()) if sat else set()
        print("  Note: candidates sat DIFFERENT cases, so the pass rates are "
              "not directly comparable.")
        for name in sorted(sat):
            extra = sorted(sat[name] - shared)
            if extra:
                print(f"    {name}: also ran {', '.join(extra)}")
        print(f"    shared by all: {', '.join(sorted(shared)) or 'none'}")

    # A mean over two of nine cases is not comparable with a mean over nine.
    # local-mid's svg ink of 0.564 was exactly that, printed beside 0.229.
    for name, s in sorted(summary.items(), key=rank):
        thin = {m: n for m, n in (s.get("metric_n") or {}).items()
                if m in metric_names and n < s.get("total", 0)}
        if thin:
            parts = ", ".join(f"{m} over {n}/{s.get('total', 0)}"
                              for m, n in sorted(thin.items()))
            print(f"  {name}: PARTIAL -- {parts}. Not comparable with a row "
                  f"scored over all {s.get('total', 0)}.")

    for name, s in summary.items():
        for f in s.get("failures") or []:
            print(f"  {name}: {f}")

    _note_ranking_disagreements(summary, metric_names)

    # Only worth saying when nothing separates the candidates. With a quality
    # metric in hand the suite CAN rank them, and repeating the disclaimer
    # would train the reader to skip it.
    all_pass = [s for s in summary.values() if s["pass_rate"] == 1.0]
    distinct = len({tuple(sorted(s["metrics"].items()))
                    for s in summary.values()}) > 1
    if len(all_pass) > 1 and not distinct:
        print("\nNote: every candidate passed everything and no metric "
              "separates them. These checks are a competence gate, not a "
              "quality ranking.")


if __name__ == "__main__":
    raise SystemExit(main())
