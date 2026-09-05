"""Is this a usable generated image?

"The command exited 0" is not evidence. A broken diffusion pipeline emits a
uniform grey square, at the right resolution, with a clean exit status. So the
checks are: does it decode, is it the size that was asked for, and does it
actually contain an image rather than one colour.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, UnidentifiedImageError

# Below this, the pixels carry no structure worth calling an image. A real
# generation lands orders of magnitude above it; a solid fill sits at 0.
MIN_STDDEV = 2.0
SMALL_EDGE = 64


@dataclass
class ImageResult:
    ok: bool
    reason: str = ""
    warnings: list[str] = field(default_factory=list)
    width: int = 0
    height: int = 0
    stddev: float = 0.0
    bytes: int = 0


def check(path: str | Path, expect: tuple[int, int] | None = None) -> ImageResult:
    path = Path(path)
    if not path.exists():
        return ImageResult(False, f"no output file at {path}")
    size = path.stat().st_size
    if size == 0:
        return ImageResult(False, "output file is empty")

    try:
        with Image.open(path) as im:
            im.load()
            width, height = im.size
            rgb = im.convert("RGB")
            # Per-channel stddev; a uniform fill is 0 on every channel.
            stat_bands = rgb.split()
            stddev = max(_stddev(b) for b in stat_bands)
    except UnidentifiedImageError:
        return ImageResult(False, f"not a decodable image: {path.name}",
                           bytes=size)
    except OSError as exc:
        return ImageResult(False, f"decode failed: {exc}", bytes=size)

    warnings: list[str] = []
    if min(width, height) < SMALL_EDGE:
        warnings.append(f"small image: {width}x{height}")

    if expect and (width, height) != tuple(expect):
        return ImageResult(False,
                           f"expected {expect[0]}x{expect[1]}, got {width}x{height}",
                           warnings, width, height, stddev, size)

    if stddev < MIN_STDDEV:
        return ImageResult(False,
                           f"uniform canvas (stddev {stddev:.2f}): the pipeline "
                           f"produced a blank image",
                           warnings, width, height, stddev, size)

    return ImageResult(True, "", warnings, width, height, stddev, size)


def _stddev(band: Image.Image) -> float:
    hist = band.histogram()
    total = sum(hist)
    if not total:
        return 0.0
    mean = sum(i * n for i, n in enumerate(hist)) / total
    var = sum(n * (i - mean) ** 2 for i, n in enumerate(hist)) / total
    return var ** 0.5
