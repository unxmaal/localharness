"""Is this a usable generated video?

The image checker's argument, one dimension further. "The command exited 0" is
not evidence, and neither is "ffprobe demuxed it": a broken sampler emits a
technically valid MP4 of exactly the right length in which nothing moves, and a
stalled one emits N copies of the first frame. Both play. Both are failures,
and both cost forty minutes to produce.

So the checks are: does it demux, is it the size and length that was asked for,
does each frame contain an image rather than one colour, and does anything
actually change from frame to frame.

`motion` is published as a metric rather than only a threshold, because "more
of a video than the other one" is a real difference between two clips that both
pass the gate.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# Below this, consecutive frames are the same picture. A real generation lands
# far above it; a stalled sampler sits at exactly 0.
MIN_MOTION = 0.001
# Enough to catch a stall without decoding a whole clip.
SAMPLE_FRAMES = 6
# Encoders drop or pad a frame at the boundary. Failing a forty-minute
# generation over one frame would be measuring the muxer.
FRAME_TOLERANCE = 1


class VideoError(RuntimeError):
    """The file could not be inspected as a video."""


@dataclass
class VideoInfo:
    width: int
    height: int
    frames: int
    duration: float
    codec: str = ""


@dataclass
class VideoResult:
    ok: bool
    reason: str = ""
    warnings: list[str] = field(default_factory=list)
    info: VideoInfo | None = None
    motion: float = 0.0
    stddev: float = 0.0

    @property
    def metrics(self) -> dict:
        return {"motion": round(self.motion, 4)}


def probe(path: str | Path) -> VideoInfo:
    """Container and stream facts, via ffprobe."""
    path = Path(path)
    if not path.exists():
        raise VideoError(f"no video file at {path}")
    if shutil.which("ffprobe") is None:
        raise VideoError("ffprobe is not installed (brew install ffmpeg)")

    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-count_frames", "-show_entries",
         "stream=width,height,nb_read_frames,codec_name,duration",
         "-show_entries", "format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        raise VideoError(f"not a decodable video: {proc.stderr.strip()[:200]}")

    try:
        data = json.loads(proc.stdout)
        stream = (data.get("streams") or [{}])[0]
        if not stream.get("width"):
            raise VideoError(f"{path.name} has no video stream")
        # nb_read_frames is a count of frames actually decoded, not the
        # container's claim, which generators frequently get wrong.
        frames = int(stream.get("nb_read_frames") or 0)
        duration = float(stream.get("duration")
                         or data.get("format", {}).get("duration") or 0.0)
    except (ValueError, KeyError, IndexError) as exc:
        raise VideoError(f"could not read {path.name}: {exc}") from exc

    return VideoInfo(int(stream["width"]), int(stream["height"]), frames,
                     duration, stream.get("codec_name", ""))


def _sample(path: Path, count: int = SAMPLE_FRAMES) -> list:
    """Decode `count` evenly spaced frames as greyscale images."""
    from PIL import Image

    with tempfile.TemporaryDirectory() as d:
        proc = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path),
             "-vf", f"select='not(mod(n\\,{max(1, count // 2)}))'",
             # -fps_mode, not -vsync: ffmpeg 8 removed the old spelling and
             # fails the whole command with "Unrecognized option 'vsync'".
             "-fps_mode", "vfr", "-frames:v", str(count),
             str(Path(d) / "f%03d.png")],
            capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            raise VideoError(f"could not decode frames: {proc.stderr.strip()[:200]}")
        out = []
        for f in sorted(Path(d).glob("f*.png")):
            with Image.open(f) as im:
                out.append(im.convert("L").copy())
        return out


def motion(path: str | Path) -> float:
    """Mean absolute difference between consecutive sampled frames, 0-1.

    Zero means every sampled frame is identical: the sampler stalled, or the
    prompt produced a still. Computed on greyscale, because a colour shift with
    no movement is still movement and this is not trying to be clever.
    """
    frames = _sample(Path(path))
    if len(frames) < 2:
        return 0.0

    from PIL import ImageChops
    total = 0.0
    for a, b in zip(frames, frames[1:]):
        diff = ImageChops.difference(a, b)
        histogram = diff.histogram()
        pixels = sum(histogram)
        total += sum(i * n for i, n in enumerate(histogram)) / (pixels * 255)
    return round(total / (len(frames) - 1), 4)


def _stddev(image) -> float:
    histogram = image.histogram()
    total = sum(histogram)
    if not total:
        return 0.0
    mean = sum(i * n for i, n in enumerate(histogram)) / total
    var = sum(n * (i - mean) ** 2 for i, n in enumerate(histogram)) / total
    return var ** 0.5


def check(path: str | Path, expect: tuple[int, int] | None = None,
          frames: int | None = None,
          min_motion: float = MIN_MOTION) -> VideoResult:
    path = Path(path)
    try:
        info = probe(path)
    except VideoError as exc:
        return VideoResult(False, str(exc))

    if expect and (info.width, info.height) != tuple(expect):
        return VideoResult(
            False,
            f"expected {expect[0]}x{expect[1]}, got {info.width}x{info.height}",
            info=info)

    if frames is not None and abs(info.frames - frames) > FRAME_TOLERANCE:
        return VideoResult(
            False, f"expected {frames} frames, got {info.frames}", info=info)

    try:
        sampled = _sample(path)
    except VideoError as exc:
        return VideoResult(False, str(exc), info=info)
    if not sampled:
        return VideoResult(False, "no frames could be decoded", info=info)

    # A uniform frame is the video form of the grey square, and it is a
    # different diagnosis from "nothing moved" even though both look still.
    worst = max(_stddev(f) for f in sampled)
    if worst < 2.0:
        return VideoResult(
            False,
            f"every sampled frame is a uniform canvas (stddev {worst:.2f})",
            info=info, stddev=worst)

    moved = motion(path)
    if moved < min_motion:
        return VideoResult(
            False,
            f"static: nothing changes between frames (motion {moved:.4f}). "
            f"The container is valid and it plays; the sampler did not move.",
            info=info, motion=moved, stddev=worst)

    return VideoResult(True, "", info=info, motion=moved, stddev=worst)
