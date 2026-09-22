#!/usr/bin/env python3
"""Generate one track with ACE-Step, for the music lane's engine to invoke.

THE ONLY PYTHON IN scripts/, and deliberately. ACE-Step lives in its own
virtualenv with torch, mlx and a pinned transformers, so it cannot be imported
from this repo's environment. This file is run BY that environment, using that
checkout's own interpreter directly:

    $ACESTEP_ROOT/.venv/bin/python scripts/acestep_generate.py ...

and NOT through `uv run`, which forks rather than execs and so hides this
process from the peak-memory instrument. See engines.acestep_python().

It imports nothing from `harness`: it is a program the harness shells into,
the same relationship `$LLAMACPP_BIN` and `$FLUIDAUDIO_CLI` have.

`generate_music` names its own output with a uuid under `save_dir`, so this
generates into a temporary directory and moves the single file it finds to
`--out`. An engine that cannot say where its artifact went is an engine whose
artifact the runner cannot check.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

#: ACE-Step init reads proxies out of the environment and then cannot reach a
#: local checkpoint through them.
for _var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
             "ALL_PROXY"):
    os.environ.pop(_var, None)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True, help="where to write the wav")
    p.add_argument("--caption", default="", help="the style prompt")
    p.add_argument("--lyrics", default="", help="lyrics, or [Instrumental]")
    p.add_argument("--lyrics-file", default="",
                   help="read lyrics from a file instead, for anything with "
                        "newlines in it")
    p.add_argument("--instrumental", action="store_true")
    # STYLE TRANSFER. `cover` is in TASK_TYPES_TURBO, so the config this
    # project runs supports it; it was simply never passed. Read off
    # GenerationParams rather than guessed: task_type, src_audio and
    # audio_cover_strength are the three fields that matter. Issue #275.
    p.add_argument("--task", default="text2music",
                   help="text2music, or cover for style transfer from --ref")
    p.add_argument("--ref", default="",
                   help="reference audio to take the style from (task=cover)")
    p.add_argument("--cover-strength", type=float, default=0.2,
                   help="how much of the reference to keep, 0..1. ACE-Step's "
                        "own docs say to set this SMALL for style transfer; "
                        "the field defaults to 1.0, which reproduces the "
                        "reference rather than restyling a new piece")
    p.add_argument("--bpm", type=float, default=None)
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("--keyscale", default="")
    p.add_argument("--timesignature", default="")
    p.add_argument("--language", default="en")
    p.add_argument("--seed", type=int, default=-1)
    p.add_argument("--steps", type=int, default=8,
                   help="8 for turbo, 32-100 for the base model")
    p.add_argument("--guidance", type=float, default=1.0)
    # ACE-Step's OWN DEFAULTS, and they are here as knobs rather than as a
    # correction. The 5Hz chain-of-thought samples at 0.85/0.9 regardless of
    # `seed`, so identical runs differ: lyrics-plain scored 0.1515 then
    # 0.0000, lyrics-dense 0.1974 then 0.1053.
    #
    # Forcing greedy (0.0 / 1.0) was tried as the fix and MADE IT WORSE in
    # both directions: lyrics-plain went to wer 1.000 on both of two runs --
    # nothing intelligible at all -- and lyrics-dense still varied, 0.0789
    # then 0.1579. So the sampling is not the only source of variance and the
    # CoT needs its temperature to produce a singable line.
    #
    # The consequence is a property of the lane rather than a bug to fix here:
    # one music run is ONE DRAW, and a music number must come from --repeat.
    p.add_argument("--lm-temperature", type=float, default=0.85)
    p.add_argument("--lm-top-p", type=float, default=0.9)
    p.add_argument("--config", default="acestep-v15-turbo",
                   help="checkpoint directory name or repo id, passed through "
                        "untouched: a discovered candidate is a repo id and a "
                        "table of known names would refuse it")
    p.add_argument("--lm", default="acestep-5Hz-lm-0.6B",
                   help="the 5Hz LM. Unlike the DiT it does NOT self-download")
    p.add_argument("--root", default=os.environ.get("ACESTEP_ROOT", ""),
                   help="the ACE-Step checkout; defaults to $ACESTEP_ROOT")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    # ARGUMENTS BEFORE ENVIRONMENT. A missing --ref is wrong however the
    # machine is configured, and reporting the checkout first sends whoever
    # ran it to fix the wrong thing.
    if args.task == "cover" and not args.ref:
        print("cover needs --ref: it restyles a reference, and without one "
              "there is nothing to take the style from", file=sys.stderr)
        return 2
    if args.ref and not Path(args.ref).expanduser().is_file():
        print(f"no reference audio at {args.ref}", file=sys.stderr)
        return 2
    root = Path(args.root or ".").expanduser().resolve()
    if not (root / "acestep").is_dir():
        print(f"no ACE-Step checkout at {root}; set $ACESTEP_ROOT",
              file=sys.stderr)
        return 2
    sys.path.insert(0, str(root))

    lyrics = args.lyrics
    if args.lyrics_file:
        lyrics = Path(args.lyrics_file).read_text(encoding="utf-8")
    if args.instrumental and not lyrics.strip():
        lyrics = "[Instrumental]"

    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler
    from acestep.inference import (GenerationConfig, GenerationParams,
                                   generate_music)

    dit = AceStepHandler()
    msg, ok = dit.initialize_service(project_root=str(root),
                                     config_path=args.config,
                                     device="auto", offload_to_cpu=False)
    if not ok:
        print(f"DiT init failed: {msg}", file=sys.stderr)
        return 3

    lm = LLMHandler()
    msg, ok = lm.initialize(checkpoint_dir=str(root / "checkpoints"),
                            lm_model_path=args.lm, backend="mlx",
                            device="auto", offload_to_cpu=False, dtype=None)
    if not ok:
        # The DiT self-downloads and the LM does not, so this is the ordinary
        # first-run failure and the message has to say what to fetch.
        print(f"LM init failed: {msg}\n"
              f"the 5Hz LM does not self-download; fetch ACE-Step/{args.lm} "
              f"into {root / 'checkpoints'}", file=sys.stderr)
        return 4

    params = GenerationParams(
        task_type=args.task, caption=args.caption, lyrics=lyrics,
        src_audio=(str(Path(args.ref).expanduser().resolve())
                   if args.ref else None),
        audio_cover_strength=args.cover_strength,
        instrumental=args.instrumental, bpm=args.bpm, duration=args.duration,
        keyscale=args.keyscale, timesignature=args.timesignature,
        vocal_language=args.language, seed=args.seed,
        inference_steps=args.steps, guidance_scale=args.guidance,
        lm_temperature=args.lm_temperature, lm_top_p=args.lm_top_p,
        thinking=True)

    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        result = generate_music(dit, lm, params=params,
                                config=GenerationConfig(batch_size=1,
                                                        audio_format="wav"),
                                save_dir=tmp)
        if not result.success:
            print(f"generation failed: {result.status_message}",
                  file=sys.stderr)
            return 5
        made = sorted(Path(tmp).glob("*.wav"))
        if not made:
            print(f"generation reported success and wrote no wav to {tmp}",
                  file=sys.stderr)
            return 6
        shutil.move(str(made[0]), out)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
