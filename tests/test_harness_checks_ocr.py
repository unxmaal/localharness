"""Reading text back out of a generated image.

Text rendering is the one image ability the pixel checks are completely blind
to: a sign reading OPEM decodes, is the right size, and has plenty of variance.
This is what tells OPEN from OPEM, and the character error rate it reports is a
graded score rather than a verdict, so two models that both render something
legible can still be ordered.

Apple's Vision framework, which ships with macOS. No model download, no server.
"""
import sys

import pytest

from harness.checks import ocr


def render(path, text, size=(480, 200)):
    """A high-contrast image of `text`, which any OCR should read."""
    from PIL import Image, ImageDraw, ImageFont
    im = Image.new("RGB", size, (255, 255, 255))
    d = ImageDraw.Draw(im)
    # A real typeface on whichever machine this is. load_default() is a small
    # bitmap font that OCR reads badly, so falling straight to it would measure
    # the fixture rather than the check.
    font = None
    for candidate in ("/System/Library/Fonts/Supplemental/Arial.ttf",
                      "C:/Windows/Fonts/arial.ttf",
                      "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                      "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                      "/usr/share/fonts/truetype/liberation/"
                      "LiberationSans-Regular.ttf"):
        try:
            font = ImageFont.truetype(candidate, 96)
            break
        except OSError:
            continue
    if font is None:  # pragma: no cover - fallback for a stripped system
        font = ImageFont.load_default()
    d.text((20, 40), text, fill=(0, 0, 0), font=font)
    im.save(path)
    return path


def test_reads_text_out_of_an_image(tmp_path):
    got = ocr.read(render(tmp_path / "a.png", "OPEN"))
    assert any("OPEN" in s.upper() for s in got), got


def test_an_image_with_no_text_reads_as_nothing(tmp_path):
    from PIL import Image
    import random
    p = tmp_path / "n.png"
    im = Image.new("RGB", (256, 256))
    im.putdata([(random.randint(0, 255),) * 3 for _ in range(256 * 256)])
    im.save(p)
    assert ocr.read(p) == []


def test_a_missing_file_raises_rather_than_reading_as_empty(tmp_path):
    with pytest.raises(FileNotFoundError):
        ocr.read(tmp_path / "nope.png")


# ---- the check ------------------------------------------------------------

def test_the_expected_word_rendered_correctly_scores_zero_error(tmp_path):
    r = ocr.check(render(tmp_path / "a.png", "OPEN"), expect="OPEN")
    assert r.ok
    assert r.cer == 0.0


def test_a_near_miss_is_a_graded_score_not_just_a_failure(tmp_path):
    """OPEM for OPEN is one character in four. A pass/fail check throws that
    away; the point of the number is that a model which almost renders text is
    distinguishable from one that renders noise."""
    r = ocr.check(render(tmp_path / "a.png", "OPEM"), expect="OPEN",
                  max_cer=0.0)
    assert not r.ok
    assert 0 < r.cer <= 0.5
    assert "OPEM" in r.reason.upper()


def test_text_that_is_not_there_at_all_is_total_error(tmp_path):
    from PIL import Image
    p = tmp_path / "n.png"
    Image.new("RGB", (256, 256), (255, 255, 255)).save(p)
    r = ocr.check(p, expect="OPEN")
    assert not r.ok
    assert r.cer == 1.0
    assert "no text" in r.reason.lower()


def test_case_and_surrounding_words_do_not_count_against_it(tmp_path):
    """Vision returns each text region separately and models add flourishes.
    The check is whether the requested word is rendered, not whether it is the
    only thing on the sign."""
    r = ocr.check(render(tmp_path / "a.png", "open"), expect="OPEN")
    assert r.ok and r.cer == 0.0


def test_the_best_matching_region_is_the_one_scored(tmp_path):
    r = ocr.check(render(tmp_path / "a.png", "CAFE\nOPEN", size=(480, 320)),
                  expect="OPEN")
    assert r.ok, r.reason


def test_the_metric_is_published_for_ranking(tmp_path):
    r = ocr.check(render(tmp_path / "a.png", "OPEN"), expect="OPEN")
    assert r.metrics == {"cer": 0.0}


def test_the_default_tolerance_allows_a_little_ocr_noise(tmp_path):
    """Vision is not perfect on synthetic renders either, so a threshold of
    exactly zero would measure the OCR rather than the generator."""
    assert 0 < ocr.DEFAULT_MAX_CER < 0.5


# ---- normalization ---------------------------------------------------------
#
# Found by a real comparison: Z-Image Turbo rendered a shop sign reading
# "OPEN." with a period, which is a perfectly good sign. Scored against "OPEN"
# that is one character in four, and it showed up in a summary as the only
# quality difference between two image models. It was punctuation.

def test_trailing_punctuation_is_not_a_rendering_error(tmp_path):
    r = ocr.check(render(tmp_path / "a.png", "OPEN."), expect="OPEN")
    assert r.ok
    assert r.cer == 0.0


def test_surrounding_quotes_are_not_a_rendering_error(tmp_path):
    r = ocr.check(render(tmp_path / "a.png", '"OPEN"'), expect="OPEN")
    assert r.ok and r.cer == 0.0


def test_internal_punctuation_still_counts(tmp_path):
    """Stripping the edges must not turn into ignoring punctuation. A sign
    reading OP-EN is not the word that was asked for."""
    assert ocr.cer("open", "op-en") > 0


def test_a_genuinely_wrong_word_is_still_wrong(tmp_path):
    r = ocr.check(render(tmp_path / "a.png", "OPEM"), expect="OPEN", max_cer=0.0)
    assert not r.ok


# ---- per-OS alternates ----------------------------------------------------


def test_the_backend_is_chosen_by_what_the_machine_has():
    """One lane, two implementations. Apple's Vision on macOS and
    Windows.Media.Ocr on Windows -- both ship with the OS, neither needs a
    download or a server, and which one runs is not a decision any caller
    should be making."""
    name = ocr.available_backend()
    assert name in ocr.BACKENDS
    if sys.platform == "darwin":
        assert name == "vision"
    if sys.platform == "win32":
        assert name == "windows"


def test_an_unmeasurable_lane_warns_rather_than_failing(tmp_path, monkeypatch):
    """A check that cannot RUN is not a render that failed.

    Reporting it as a failure blames the generator for a missing dependency,
    and the score sheet then carries a defeat that never happened. This is the
    distinction `adherence` already draws, and the one the repo draws when it
    refuses to fetch a weight nothing can measure.
    """
    monkeypatch.setattr(ocr, "available_backend", lambda: None)
    r = ocr.check(render(tmp_path / "a.png", "OPEN"), expect="OPEN")
    assert r.ok, r.reason
    assert any("not measured" in w for w in r.warnings), r.warnings


def test_an_unmeasured_lane_publishes_no_metric(tmp_path, monkeypatch):
    """A cer of 0 would rank as a perfect render and a cer of 1 as a total
    failure. Neither happened, so neither number belongs in the table."""
    monkeypatch.setattr(ocr, "available_backend", lambda: None)
    r = ocr.check(render(tmp_path / "a.png", "OPEN"), expect="OPEN")
    assert "cer" not in r.metrics


def test_a_machine_with_no_os_engine_still_measures_the_lane():
    """Linux ships no OCR engine. RapidOCR is the backend there, and it is not
    pinned to Linux -- a Mac without pyobjc measures the lane instead of
    skipping it, and the receipt records which engine ran."""
    platform_name, probe, reader = ocr.BACKENDS["rapidocr"]
    assert platform_name == "", "rapidocr must not be pinned to one platform"
    assert callable(probe) and callable(reader)


def test_every_backend_is_gated_on_being_installed_not_just_on_the_platform():
    """Windows OCR is an optional component and pyobjc is an optional install.
    A platform match alone said yes to both and then failed on the import."""
    for name, (_, probe, _) in ocr.BACKENDS.items():
        assert callable(probe), name
        assert probe() in (True, False), name


def test_tesseract_is_not_a_backend():
    """Measured on this project's own 18 text-render images: Vision 18/18,
    RapidOCR 18/18, tesseract 6/18, tesseract with preprocessing 0/18. It would
    have reported twelve good renders as total failures."""
    assert "tesseract" not in ocr.BACKENDS
