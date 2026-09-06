"""Engines: a candidate string in, a command line out.

An engine is data. Adding one is a spec string rather than a code change, which
is the same bet the gateway makes for text models, and it is what lets the CLI
and the eval suite issue byte-identical commands instead of two copies that
drift.

    mflux:z-image-turbo                 image, 8-bit quantized (the default)
    mflux:flux2-klein-4b,quantize=none  image, bf16
    mflux:z-image-turbo,steps=4         image, with a pinned default
    h3                                  video

Options after the model are `key=value`, comma separated. They are the engine's
DEFAULTS; a case or a CLI flag overrides them at call time. Unknown options are
rejected here rather than by the generator forty minutes in.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

Argv = Callable[[str, Path, dict], list[str]]

GRAMMAR = "engine:model[,key=value,...]  (engines: mflux, h3)"


def spec_error(spec: str) -> str:
    return f"bad engine spec {spec!r}; expected {GRAMMAR}"


@dataclass(frozen=True)
class Engine:
    """How to invoke one generator.

    `argv(prompt, output_path, params)` takes a prompt and not a Case: this
    package is below the eval suite, and an engine that knows about test cases
    cannot be used by the product.
    """
    name: str
    spec: str
    argv: Argv
    modality: str = "image"
    output_suffix: str = ".png"
    #: Wall-clock ceiling, seconds. Video is minutes per frame on this machine.
    timeout: float = 900.0
    #: Whether the child's output should reach the terminal live.
    stream: bool = False


def resolve(spec: str) -> Engine:
    """Parse a candidate spec into an Engine."""
    head, _, optstr = spec.partition(",")
    engine, sep, model = head.partition(":")
    engine = engine.strip()
    model = model.strip()
    if engine not in _BUILDERS:
        raise ValueError(
            f"unknown engine {engine!r}; known engines: "
            f"{', '.join(sorted(_BUILDERS))}. Expected {GRAMMAR}")

    options = parse_options(optstr, spec)
    return _BUILDERS[engine](spec, model, options)


def parse_options(optstr: str, spec: str) -> dict:
    out: dict[str, str] = {}
    for chunk in optstr.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        key, eq, value = chunk.partition("=")
        if not eq:
            raise ValueError(f"{spec_error(spec)}: option {chunk!r} is not key=value")
        out[key.strip()] = value.strip()
    return out


def _check_options(options: dict, allowed: set[str], spec: str) -> None:
    unknown = set(options) - allowed
    if unknown:
        raise ValueError(
            f"{spec_error(spec)}: unknown option(s) "
            f"{', '.join(sorted(unknown))}; allowed: {', '.join(sorted(allowed))}")


def _flag(cmd: list[str], name: str, value) -> None:
    """Append `--name value`, but only when a value was actually given.

    An unset param must be omitted, not passed as the string "None": mflux
    accepts --steps None and fails deep inside the scheduler.
    """
    if value is not None:
        cmd += [name, str(value)]


# ---- mflux ----------------------------------------------------------------

_MFLUX_OPTIONS = {"quantize", "steps", "width", "height", "guidance"}

# Model family -> entry point. Longest prefix wins; the empty prefix is the
# FLUX.1 family, which owns the bare `mflux-generate`.
#
# This table is NOT optional and cannot be replaced by `mflux-generate --model
# X`. Every mflux binary shares one argument parser, so `mflux-generate --help`
# advertises every built-in model under --model regardless of which binary can
# actually run it, and then rejects the wrong ones at RUNTIME:
#
#     mflux-generate: error: FLUX.2 Klein is not supported by mflux-generate.
#                            Use mflux-generate-flux2 instead.
#
# Read off mflux's entry_points.txt and its models/<family>/cli/ tree, not off
# the help output. test_every_entry_point_in_the_table_actually_exists is what
# catches this going stale.
ENTRY_POINTS = {
    "flux2-klein": "mflux-generate-flux2",
    "z-image-turbo": "mflux-generate-z-image-turbo",
    "z-image": "mflux-generate-z-image",
    "krea-2": "mflux-generate-krea2",
    "krea2": "mflux-generate-krea2",
    "qwen-image": "mflux-generate-qwen",
    "fibo": "mflux-generate-fibo",
    "ernie-image-turbo": "mflux-generate-ernie-image-turbo",
    "ernie-image": "mflux-generate-ernie-image",
    "lens-turbo": "mflux-generate-lens",
    "boogu": "mflux-generate-boogu",
    "ideogram-4": "mflux-generate-ideogram4",
}

# The FLUX.1 family, which `mflux-generate` itself serves. Listed rather than
# inferred so an unknown model is a spec error instead of a binary that rejects
# it several seconds into loading.
FLUX1_MODELS = {
    "dev", "schnell", "krea-dev", "dev-kontext", "dev-fill", "dev-redux",
    "dev-depth", "dev-controlnet-canny", "schnell-controlnet-canny",
    "dev-controlnet-upscaler", "dev-fill-catvton",
}


def mflux_binary(model: str) -> str:
    """The mflux entry point that can actually run `model`."""
    if model in FLUX1_MODELS:
        return "mflux-generate"
    for prefix in sorted(ENTRY_POINTS, key=len, reverse=True):
        if model.startswith(prefix):
            return ENTRY_POINTS[prefix]
    raise ValueError(
        f"unknown mflux model {model!r}. Known families: "
        f"{', '.join(sorted(ENTRY_POINTS))}; "
        f"FLUX.1: {', '.join(sorted(FLUX1_MODELS))}")


def _mflux(spec: str, model: str, options: dict) -> Engine:
    if not model:
        raise ValueError(
            f"{spec_error(spec)}: mflux needs a model, e.g. mflux:z-image-turbo")
    _check_options(options, _MFLUX_OPTIONS, spec)
    binary = mflux_binary(model)

    quantize = options.get("quantize", "8")
    # bf16 and q4 of the same model are different candidates with different
    # speed and memory. Sharing a row makes the comparison meaningless.
    label = "bf16" if quantize == "none" else f"q{quantize}"

    defaults = {k: v for k, v in options.items() if k != "quantize"}

    def argv(prompt: str, out: Path, params: dict) -> list[str]:
        p = {**defaults, **{k: v for k, v in params.items() if v is not None}}
        cmd = [binary, "--model", model,
               "--prompt", prompt, "--output", str(out),
               # Writes the full generation config beside the image, which is
               # the only way a result from three weeks ago is reproducible.
               "--metadata"]
        for name in ("width", "height", "steps", "seed", "guidance"):
            _flag(cmd, f"--{name}", p.get(name))
        if quantize != "none":
            cmd += ["--quantize", str(quantize)]
        return cmd

    return Engine(name=f"mflux/{model}-{label}", spec=spec, argv=argv,
                  modality="image", output_suffix=".png", timeout=900.0)


# ---- h3 (antirez/h3.c, MiniMax-H3 video) ----------------------------------

_H3_OPTIONS = {"steps", "layers", "reuse", "width", "height", "ssd_streaming"}
H3_DEFAULT_BIN = str(Path.home() / "projects/github/antirez/h3.c/h3")
H3_DEFAULT_MODEL_DIR = "/Volumes/Models/MiniMax-H3"


def _h3(spec: str, model: str, options: dict) -> Engine:
    _check_options(options, _H3_OPTIONS, spec)
    defaults = dict(options)

    def argv(prompt: str, out: Path, params: dict) -> list[str]:
        p = {**defaults, **{k: v for k, v in params.items() if v is not None}}
        if p.get("frames") is not None and p.get("seconds") is not None:
            raise ValueError("h3: pass --frames or --seconds, not both")

        # Read at call time, not import time, so a test or a different volume
        # can point this somewhere else without reloading the module.
        cmd = [os.environ.get("H3_BIN", H3_DEFAULT_BIN),
               "-d", os.environ.get("H3_MODEL_DIR", H3_DEFAULT_MODEL_DIR),
               "-p", prompt, "-o", str(out)]
        for name in ("width", "height", "frames", "seconds", "steps",
                     "layers", "reuse", "seed"):
            _flag(cmd, f"--{name}", p.get(name))
        # 134GiB of weights against 32GB of RAM. Without streaming the DiT off
        # the SSD it cannot start at all; the Studio can turn it off.
        if str(p.get("ssd_streaming", True)).lower() not in ("false", "0", "no"):
            cmd.append("--ssd-streaming")
        return cmd

    return Engine(name="h3/minimax-h3", spec=spec, argv=argv, modality="video",
                  output_suffix=".mp4",
                  # 512x512x22 frames measured at 40.5 minutes on the M2 Pro.
                  timeout=6 * 3600.0, stream=True)


_BUILDERS: dict[str, Callable[[str, str, dict], Engine]] = {
    "mflux": _mflux,
    "h3": _h3,
}
