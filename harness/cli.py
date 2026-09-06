"""localharness: generate media and talk to the machine, all locally.

    lh image "a red fox in snow" --width 768
    lh video "a fox running" --seconds 2
    lh svg   "a settings gear icon"
    lh web   "a landing page for a coffee roaster"
    lh say   "bonjour"
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
import subprocess
import sys
import time
from pathlib import Path

from harness import audio, completion, proc
from harness.checks import html as html_check
from harness.checks import image as image_check
from harness.checks import svg as svg_check
from harness.engines import resolve

DEFAULT_IMAGE_ENGINE = "mflux:flux2-klein-4b"
DEFAULT_VIDEO_ENGINE = "h3"
DEFAULT_TEXT_MODEL = "local-mid"
OUTDIR = Path("out")

# Named per engine family because the fix differs, and because `uv tool install
# mflux` on its own silently picks Python 3.9, where every mflux entry point
# dies on `int | None`. The tell is 2 executables installed instead of 37.
INSTALL_HINT = {
    "mflux": " Install it with: uv tool install --python 3.12 mflux",
    "h3": " Build it: git clone https://github.com/antirez/h3.c && make -C h3.c",
}


def err(msg: str) -> int:
    print(msg, file=sys.stderr)
    return 1


def default_output(kind: str, suffix: str) -> Path:
    """out/image-20260905-142233-041.png — sortable, and unique per run.

    Milliseconds are in there because two `lh image` calls a second apart must
    not overwrite each other; losing a generation to a name collision is the
    kind of thing you only notice much later.
    """
    stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
    return OUTDIR / f"{kind}-{stamp}{suffix}"


def _generate(spec: str, prompt: str, out: Path, params: dict) -> int:
    """Shared body of `image` and `video`: build, run, check, report."""
    try:
        engine = resolve(spec)
    except ValueError as exc:
        return err(str(exc))

    out = out or default_output(engine.modality, engine.output_suffix)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    try:
        argv = engine.argv(prompt, out, params)
    except ValueError as exc:
        return err(str(exc))

    try:
        r = proc.run(argv, timeout=engine.timeout, stream=engine.stream)
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

    print(f"{out}  ({r.seconds:.1f}s, peak {r.peak_kb / 1024 / 1024:.1f} GiB)")
    return 0


def cmd_image(a) -> int:
    params = {k: getattr(a, k) for k in ("width", "height", "steps", "seed")}
    return _generate(a.model, a.prompt, a.output, params)


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
    out.write_text(body)

    checked = checker(raw)
    for w in checked.warnings:
        print(f"warning: {w}", file=sys.stderr)
    if not checked.ok:
        # Written anyway: you cannot debug what was deleted.
        return err(f"wrote {out}, but it does not check out: {checked.reason}")
    print(f"{out}")
    return 0


def cmd_svg(a) -> int:
    return _text(a, "svg", ".svg", svg_check.check)


def cmd_web(a) -> int:
    return _text(a, "web", ".html", html_check.check)


def cmd_say(a) -> int:
    text = sys.stdin.read() if a.text == "-" else a.text
    if not text.strip():
        return err("nothing to say")
    out = Path(a.output or default_output("speech", ".wav"))
    try:
        audio.speak(text, out=out, voice=a.voice, speed=a.speed,
                    base_url=a.base_url)
    except audio.AudioError as exc:
        return err(str(exc))
    if a.play:
        proc.run(audio.play_argv(out))
    print(f"{out}")
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
        print(audio.transcribe(clip, base_url=a.base_url))
    except audio.AudioError as exc:
        return err(str(exc))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="lh", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command")

    def media(name, help_, engine_default, func):
        p = sub.add_parser(name, help=help_)
        p.add_argument("prompt")
        p.add_argument("-o", "--output")
        p.add_argument("-m", "--model", default=engine_default,
                       help="engine spec, e.g. mflux:z-image-turbo,quantize=4")
        p.add_argument("--width", type=int)
        p.add_argument("--height", type=int)
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
        p.add_argument("-m", "--model", default=DEFAULT_TEXT_MODEL,
                       help="gateway alias")
        p.add_argument("--gateway", default=completion.DEFAULT_GATEWAY)
        p.set_defaults(func=func)

    s = sub.add_parser("say", help="speak text aloud")
    s.add_argument("text", help="the text, or - to read stdin")
    s.add_argument("-o", "--output")
    s.add_argument("--voice", default=audio.DEFAULT_VOICE,
                   help=f"one of {', '.join(audio.KNOWN_VOICES)}")
    s.add_argument("--speed", type=float, default=1.0)
    s.add_argument("--base-url", default=audio.DEFAULT_BASE_URL)
    s.add_argument("--no-play", dest="play", action="store_false", default=True)
    s.set_defaults(func=cmd_say)

    h = sub.add_parser("hear", help="transcribe a clip, or record and transcribe")
    h.add_argument("file", nargs="?", help="an existing audio file")
    h.add_argument("-o", "--output", help="where to save a new recording")
    h.add_argument("--seconds", type=float, default=5.0)
    h.add_argument("--base-url", default=audio.DEFAULT_BASE_URL)
    h.set_defaults(func=cmd_hear)

    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    a = ap.parse_args(argv)
    if not getattr(a, "func", None):
        ap.print_help(sys.stderr)
        return 2
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
