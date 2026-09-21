"""The image lane's generator through diffusers, on whatever accelerator is here.

It is a command line rather than a library call for one reason:
harness/engines.py treats an engine as data, a spec string that becomes an
argv, and the CLI and the eval suite then issue byte-identical commands.
mflux-generate has that shape and so does this.

IT WAS CUDA-ONLY, and was named image_cuda for it. That was not a fact about
diffusers, it was the CUDA machine being the only place anyone had tried: this
file selected `cuda` or fell back to refusing, so on Apple Silicon the lane had
exactly one engine and mflux had never faced a challenger. torch runs on Metal
through mps, and the svg lane's audit recorded that path as viable here two
weeks before anything used it.

IT STILL REFUSES THE CPU unless told to. SDXL on a CPU is minutes per image
where an accelerator is seconds, and a lane that silently took that path would
report the model as slow rather than the machine as misconfigured. The same
reasoning as the transcription server's device message.

A .json sidecar goes beside every PNG, holding the model, the seed and the
step count. mflux writes one under --metadata for the same reason: a result
from three weeks ago is only reproducible if it says what produced it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

#: Steps a turbo model wants. SDXL-Turbo is trained for 1 to 4 and produces
#: mush at 30, which is the opposite of the usual more-is-better assumption.
DEFAULT_STEPS = 4
DEFAULT_SIZE = 512
#: Turbo models are distilled to run without classifier-free guidance, and a
#: guidance above 1.0 makes them worse rather than more faithful.
DEFAULT_GUIDANCE = 0.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="image-diffusers", description=__doc__)
    p.add_argument("--model", required=True,
                   help="a diffusers repo id, e.g. stabilityai/sdxl-turbo")
    p.add_argument("--prompt", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p.add_argument("--width", type=int, default=DEFAULT_SIZE)
    p.add_argument("--height", type=int, default=DEFAULT_SIZE)
    p.add_argument("--guidance", type=float, default=DEFAULT_GUIDANCE)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--allow-cpu", action="store_true",
                   help="generate without a card, which takes minutes")
    return p.parse_args(argv)


#: The accelerators this can use, in the order they are preferred. `cpu` is
#: absent on purpose: it is reachable only through --allow-cpu.
def accelerator(torch) -> str:
    """The device to generate on, or "" when there is no accelerator.

    mps IS AN ACCELERATOR HERE. Asking only about cuda made "no card" and "no
    Apple GPU" the same answer, which is how a Mac ended up with one image
    engine. Split out so a test can pin it: torch.cuda.is_available() reads the
    machine, and a verdict test that reads the machine tests the machine.
    """
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return ""


def _pipeline(model: str, device: str):
    import torch
    from diffusers import AutoPipelineForText2Image

    # HALF PRECISION ON BOTH ACCELERATORS. Metal supports float16 and the
    # weights are published in it; loading fp32 on a 32 GB unified machine
    # doubles the working set for no gain. `variant="fp16"` is a request for a
    # separate set of files that many repos do not publish, so it stays cuda-
    # only rather than becoming a download that 404s.
    half = device in ("cuda", "mps")
    pipe = AutoPipelineForText2Image.from_pretrained(
        model,
        torch_dtype=torch.float16 if half else torch.float32,
        variant="fp16" if device == "cuda" else None,
        safety_checker=None)
    return pipe.to(device)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    import torch
    device = accelerator(torch)
    if not device:
        if not args.allow_cpu:
            print("image-diffusers: no cuda device and no mps device. "
                  "Generating on the CPU takes minutes per image and would "
                  "read as a slow model rather than a machine without an "
                  "accelerator; pass --allow-cpu to do it anyway.",
                  file=sys.stderr)
            return 2
        device = "cpu"

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    pipe = _pipeline(args.model, device)
    image = pipe(prompt=args.prompt, num_inference_steps=args.steps,
                 guidance_scale=args.guidance, width=args.width,
                 height=args.height, generator=generator).images[0]
    image.save(out)

    out.with_suffix(".json").write_text(json.dumps({
        "model": args.model,
        "prompt": args.prompt,
        "steps": args.steps,
        "guidance": args.guidance,
        "width": args.width,
        "height": args.height,
        "seed": args.seed,
        "device": device,
        "torch": torch.__version__,
    }, indent=2), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
