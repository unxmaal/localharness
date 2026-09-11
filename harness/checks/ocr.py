"""Reading text back out of a generated image.

Text rendering is the one image ability the pixel checks are completely blind
to. A sign reading OPEM decodes cleanly, is exactly the size that was asked
for, and has plenty of variance: every check the suite had says it is a good
image.

What comes back is a character error rate rather than a verdict, so two models
that both render something legible can still be ordered. That is the whole
reason the suite needed a quality axis.

ONE LANE, THREE IMPLEMENTATIONS. Apple's Vision on macOS and Windows.Media.Ocr
on Windows both ship with the operating system -- no model download, no server,
a tenth of a second -- which is what makes either usable as a metric rather
than a second thing to install. Linux ships nothing of the kind, so there the
lane is RapidOCR, an ONNX detector plus recogniser that installs as a wheel and
runs on the CPU.

NOT TESSERACT, and this was measured rather than assumed. On the 18 text-render
images this project has actually generated, asking whether the sign's word
appears anywhere in the output at all:

    Vision                  18 / 18
    RapidOCR                18 / 18, exact, no surrounding noise
    tesseract                6 / 18 across four page-segmentation modes
    tesseract + greyscale, autocontrast, 2x    0 / 18

Tesseract is built for scanned documents and this lane is scene text in a
photograph. Shipping it would have reported twelve good renders as total
failures, which is the exact failure the DEFAULT_MAX_CER note below warns
about: measuring the OCR instead of the generator.

WHICH ENGINE RAN IS PART OF THE RESULT. It goes in the eval receipt and
comparable() refuses to rank across two of them, the same way PickScore is not
HPSv2. RapidOCR is the one that runs on all three machines, so pinning it
everywhere is the way to make a CER comparable ACROSS them; the OS engines are
preferred here only because they are free and already installed.

A machine with NEITHER is a third case, and it is not a failure: an image whose
text could not be read is an unmeasured lane, not a bad render. Scoring it as a
loss would blame the generator for a missing dependency.
"""
from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
from functools import lru_cache
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


@lru_cache(maxsize=1)
def _rapidocr():
    """The engine, built once. Constructing it loads two ONNX models, and this
    is called per image in a sweep."""
    from rapidocr_onnxruntime import RapidOCR
    return RapidOCR()


def _read_rapidocr(path: Path) -> list[str]:
    """Every text region RapidOCR finds, in the order it found them."""
    result, _ = _rapidocr()(str(path))
    return [str(line[1]).strip() for line in (result or []) if str(line[1]).strip()]


def _module_probe(name: str):
    """Whether an optional binding imports. pyobjc and winsdk are both
    installs that can be absent on the platform they belong to."""
    def probe() -> bool:
        try:
            importlib.import_module(name)
        except ImportError:
            return False
        return True
    return probe


#: backend -> (the platform it belongs to or "" for any, a probe saying whether
#: it is actually there, the reader). ORDERED, so "auto" takes the first that
#: answers: the OS's own engine where there is one, and RapidOCR where there
#: is not. RapidOCR is not pinned to Linux, so a Mac without pyobjc measures the
#: lane instead of skipping it -- and the receipt records which one ran.
BACKENDS = {
    "vision": ("darwin", _module_probe("Vision"), _read_vision),
    "windows": ("win32", _module_probe("winsdk.windows.media.ocr"),
                _read_windows),
    "rapidocr": ("", _module_probe("rapidocr_onnxruntime"), _read_rapidocr),
}


def available_backend():
    """The backend this machine can actually run, or None.

    Both halves are asked. The platform alone is not enough -- Windows OCR is
    an optional component and pyobjc is an optional install -- and a platform
    match alone is not enough either.
    """
    for name, (platform_name, probe, _) in BACKENDS.items():
        if platform_name and sys.platform != platform_name:
            continue
        if not probe():
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
