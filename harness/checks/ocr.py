"""Reading text back out of a generated image.

Text rendering is the one image ability the pixel checks are completely blind
to. A sign reading OPEM decodes cleanly, is exactly the size that was asked
for, and has plenty of variance: every check the suite had says it is a good
image.

What comes back is a character error rate rather than a verdict, so two models
that both render something legible can still be ordered. That is the whole
reason the suite needed a quality axis.

ONE LANE, TWO IMPLEMENTATIONS. Apple's Vision on macOS and Windows.Media.Ocr
on Windows. Both ship with the operating system -- no model download, no
server, a tenth of a second -- which is what makes either usable as a metric
rather than a second thing to install. Which one runs is decided here, by what
the machine has, and not by the caller.

A machine with NEITHER is a third case, and it is not a failure: an image whose
text could not be read is an unmeasured lane, not a bad render. Scoring it as a
loss would blame the generator for a missing dependency.
"""
from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
import re
from pathlib import Path

import jiwer

# Punctuation and whitespace at the EDGES of a recognized string. A shop sign
# reading "OPEN." with a period is a perfectly good sign, and quotes around a
# word are a design choice; scoring either against "OPEN" is one character in
# four, which showed up in a real comparison as the only quality difference
# between two image models. Only the edges: OP-EN is not the word that was
# asked for, and stripping the middle would hide that.
_EDGE = re.compile(r"^[^\w]+|[^\w]+$")

# Vision is not perfect on synthetic renders either. A threshold of exactly
# zero would measure the OCR rather than the generator under test.
DEFAULT_MAX_CER = 0.25


class OcrUnavailable(RuntimeError):
    """Nothing on this machine can read text out of an image."""


@dataclass
class OcrResult:
    ok: bool
    reason: str = ""
    warnings: list[str] = field(default_factory=list)
    cer: float = 1.0
    text: list[str] = field(default_factory=list)
    #: False when no backend ran. A cer of 0 would rank as a perfect render and
    #: a cer of 1 as a total failure; neither happened, so the table gets
    #: neither number.
    measured: bool = True

    @property
    def metrics(self) -> dict:
        return {"cer": round(self.cer, 4)} if self.measured else {}


def _read_vision(path: Path) -> list[str]:
    """Every text region Vision finds, most confident candidate per region."""
    import Vision
    from Foundation import NSURL

    url = NSURL.fileURLWithPath_(str(path.resolve()))
    handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(url, None)
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    ok, err = handler.performRequests_error_([request], None)
    if not ok:
        raise RuntimeError(f"Vision failed on {path}: {err}")

    out = []
    for observation in request.results() or []:
        candidates = observation.topCandidates_(1)
        if candidates:
            out.append(str(candidates[0].string()))
    return out


def _read_windows(path: Path) -> list[str]:
    """Every line Windows.Media.Ocr finds.

    The WinRT surface is asynchronous throughout, so the whole read is one
    coroutine driven by asyncio.run rather than four awaits stitched together.
    """
    import asyncio

    from winsdk.windows.graphics.imaging import BitmapDecoder
    from winsdk.windows.media.ocr import OcrEngine
    from winsdk.windows.storage import FileAccessMode, StorageFile

    async def recognize() -> list[str]:
        handle = await StorageFile.get_file_from_path_async(str(path.resolve()))
        stream = await handle.open_async(FileAccessMode.READ)
        decoder = await BitmapDecoder.create_async(stream)
        bitmap = await decoder.get_software_bitmap_async()
        engine = OcrEngine.try_create_from_user_profile_languages()
        if engine is None:
            # Windows ships the engine; the language packs are per-install.
            raise OcrUnavailable(
                "Windows.Media.Ocr has no language pack for this user profile")
        result = await engine.recognize_async(bitmap)
        return [line.text for line in result.lines]

    return asyncio.run(recognize())


#: backend -> (the platform it belongs to, the module that must import, the
#: reader). Ordered, so "auto" takes the first that answers.
BACKENDS = {
    "vision": ("darwin", "Vision", _read_vision),
    "windows": ("win32", "winsdk.windows.media.ocr", _read_windows),
}


def available_backend():
    """The backend this machine can actually run, or None.

    Both halves are asked. The platform alone is not enough -- Windows OCR is
    an optional component and pyobjc is an optional install -- and an import
    alone is not enough either.
    """
    for name, (platform_name, module, _) in BACKENDS.items():
        if sys.platform != platform_name:
            continue
        try:
            importlib.import_module(module)
        except ImportError:
            continue
        return name
    return None


def read(path: str | Path, backend: str = "auto") -> list[str]:
    """Every text region in the image, via whichever backend this machine has."""
    path = Path(path)
    if not path.exists():
        # An OCR engine answers "no text" for a missing file, which would read
        # as a generator that drew nothing rather than as a broken path.
        raise FileNotFoundError(path)

    name = available_backend() if backend == "auto" else backend
    if name is None:
        raise OcrUnavailable(
            f"no OCR backend on {sys.platform}; tried {', '.join(BACKENDS)}")
    if name not in BACKENDS:
        raise ValueError(f"unknown ocr backend {name!r}; "
                         f"known: {', '.join(BACKENDS)}")
    return BACKENDS[name][2](path)


def normalize(text: str) -> str:
    """Lowercase, and drop punctuation at the edges but never in the middle."""
    return _EDGE.sub("", text.strip().lower())


def cer(expected: str, got: str) -> float:
    """Character error rate, case-insensitive, edge punctuation ignored.

    0.0 is a perfect render.
    """
    expected = normalize(expected)
    if not expected:
        raise ValueError("cannot score against empty expected text")
    got = normalize(got)
    if not got:
        return 1.0
    return min(1.0, jiwer.cer(expected, got))


def check(path: str | Path, expect: str, max_cer: float = DEFAULT_MAX_CER) -> OcrResult:
    """Is `expect` legibly rendered in the image at `path`?"""
    try:
        regions = read(path)
    except FileNotFoundError:
        return OcrResult(False, f"no image at {path}")
    except OcrUnavailable as exc:
        # Not a failed render. See the module docstring.
        return OcrResult(True, "", [f"text not measured: {exc}"],
                         measured=False)

    if not regions:
        return OcrResult(False, f"no text found in the image; expected {expect!r}",
                         cer=1.0)

    # Models add flourishes and OCR splits the image into regions, so score
    # the best-matching region: the question is whether the word was rendered,
    # not whether it was the only thing on the sign.
    best = min(regions, key=lambda region: cer(expect, region))
    rate = cer(expect, best)
    if rate > max_cer:
        return OcrResult(
            False,
            f"expected {expect!r}, read {best!r} "
            f"(character error rate {rate:.2f} over {max_cer})",
            cer=rate, text=regions)
    return OcrResult(True, "", cer=rate, text=regions)
