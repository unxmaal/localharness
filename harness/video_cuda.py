"""The video lane's generator on a machine with an NVIDIA card.

h3 is Metal shaders over a 134 GiB checkpoint that streams off an SSD. There is
no build of it that runs here and nothing that size fits a 12 GB card, so this
lane changes model as well as tool. WHICH model is left open on purpose: this
takes any diffusers video repo id, so the answer comes out of an eval table
rather than out of this file.

IT OFFLOADS BY DEFAULT. A video pipeline held resident does not fit 12 GB, and
diffusers can move each stage to host memory between steps. That costs time and
buys the lane existing at all, which is the same trade h3 makes by streaming
its DiT off the SSD.

FRAMES ARE ENCODED BY FFMPEG rather than by an imageio helper. The video check
already needs ffmpeg and ffprobe to score the result, so the encoder is
something the machine has by the time this runs, and the alternative is another
dependency that wraps the same binary.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_FRAMES = 25
DEFAULT_FPS = 8
DEFAULT_STEPS = 30
DEFAULT_SIZE = 512
DEFAULT_GUIDANCE = 3.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="video-cuda", description=__doc__)
    p.add_argument("--model", required=True,
                   help="a diffusers video repo id, e.g. Lightricks/LTX-Video")
    p.add_argument("--prompt", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    p.add_argument("--fps", type=int, default=DEFAULT_FPS)
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p.add_argument("--width", type=int, default=DEFAULT_SIZE)
    p.add_argument("--height", type=int, default=DEFAULT_SIZE)
    p.add_argument("--guidance", type=float, default=DEFAULT_GUIDANCE)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-offload", action="store_true",
                   help="keep the pipeline resident; needs a larger card")
    p.add_argument("--allow-cpu", action="store_true")
    return p.parse_args(argv)


def _as_image(frame):
    """A PIL image, whichever way the pipeline hands frames back.

    Video pipelines disagree: LTX returns PIL images and TextToVideoSD returns
    a float array in 0..1. Assuming either one crashes on the other after the
    generation has already been paid for, which is the worst place to find out.
    """
    if hasattr(frame, "save"):
        return frame
    import numpy as np
    from PIL import Image

    array = np.asarray(frame)
    if array.dtype != np.uint8:
        array = (array.clip(0, 1) * 255).round().astype(np.uint8)
    return Image.fromarray(array)


def encode(frames, out: Path, fps: int) -> None:
    """PNG frames through ffmpeg into an mp4 the video check can read."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit(
            "video-cuda: no ffmpeg on PATH. The video check needs it to score "
            "the result as well, so the lane cannot run without one.")
    with tempfile.TemporaryDirectory() as d:
        for index, frame in enumerate(frames):
            _as_image(frame).save(Path(d) / f"{index:05d}.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-framerate", str(fps),
             "-i", str(Path(d) / "%05d.png"),
             # yuv420p and an even frame size, or the file plays nowhere: a
             # 513-pixel width encodes fine and then fails to open.
             "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(out)],
            capture_output=True, text=True)
        if proc.returncode != 0 or not out.exists():
            raise SystemExit(f"video-cuda: ffmpeg failed: "
                             f"{proc.stderr.strip()[:300]}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    import torch
    if torch.cuda.is_available():
        device = "cuda"
    elif args.allow_cpu:
        device = "cpu"
    else:
        print("video-cuda: no CUDA device. A video on the CPU is hours rather "
              "than minutes and would read as a slow model rather than a "
              "machine without a card; pass --allow-cpu to do it anyway.",
              file=sys.stderr)
        return 2

    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32)
    if device == "cuda" and not args.no_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(args.seed)

    result = pipe(prompt=args.prompt, num_frames=args.frames,
                  num_inference_steps=args.steps, guidance_scale=args.guidance,
                  width=args.width, height=args.height, generator=generator)
    frames = result.frames[0]

    out = Path(args.output)
    encode(frames, out, args.fps)

    out.with_suffix(".json").write_text(json.dumps({
        "model": args.model,
        "prompt": args.prompt,
        "frames": args.frames,
        "fps": args.fps,
        "steps": args.steps,
        "guidance": args.guidance,
        "width": args.width,
        "height": args.height,
        "seed": args.seed,
        "device": device,
        "offloaded": device == "cuda" and not args.no_offload,
        "torch": torch.__version__,
    }, indent=2), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
