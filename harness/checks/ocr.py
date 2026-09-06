"""Reading text back out of a generated image, with Apple's Vision framework.

Text rendering is the one image ability the pixel checks are completely blind
to. A sign reading OPEM decodes cleanly, is exactly the size that was asked
for, and has plenty of variance: every check the suite had says it is a good
image.

What comes back is a character error rate rather than a verdict, so two models
that both render something legible can still be ordered. That is the whole
reason the suite needed a quality axis.

Vision ships with macOS: no model download, no server, and it runs in about a
tenth of a second.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import jiwer

# Vision is not perfect on synthetic renders either. A threshold of exactly
# zero would measure the OCR rather than the generator under test.
DEFAULT_MAX_CER = 0.25


@dataclass
class OcrResult:
    ok: bool
    reason: str = ""
    warnings: list[str] = field(default_factory=list)
    cer: float = 1.0
    text: list[str] = field(default_factory=list)

    @property
    def metrics(self) -> dict:
        return {"cer": round(self.cer, 4)}


def read(path: str | Path) -> list[str]:
    """Every text region Vision finds, most confident candidate per region."""
    import Vision
    from Foundation import NSURL

    path = Path(path)
    if not path.exists():
        # Vision answers "no text" for a missing file, which would read as a
        # generator that drew nothing rather than as a broken path.
        raise FileNotFoundError(path)

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


def cer(expected: str, got: str) -> float:
    """Character error rate, case-insensitive. 0.0 is a perfect render."""
    expected = expected.strip().lower()
    if not expected:
        raise ValueError("cannot score against empty expected text")
    got = got.strip().lower()
    if not got:
        return 1.0
    return min(1.0, jiwer.cer(expected, got))


def check(path: str | Path, expect: str, max_cer: float = DEFAULT_MAX_CER) -> OcrResult:
    """Is `expect` legibly rendered in the image at `path`?"""
    try:
        regions = read(path)
    except FileNotFoundError:
        return OcrResult(False, f"no image at {path}")

    if not regions:
        return OcrResult(False, f"no text found in the image; expected {expect!r}",
                         cer=1.0)

    # Models add flourishes and Vision splits the image into regions, so score
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
