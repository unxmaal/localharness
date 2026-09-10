"""The harness itself: loading cases, running candidates, scoring comparably.

The whole point is answering "which candidate should I use" in twenty minutes
without editing code, so the contract under test is: a case file plus a
candidate name produces a comparable row.
"""
import json

import pytest
import yaml

from harness.checks import render
from evals.core import Case, Result, load_cases, score, summarize


def write_case(d, name, **over):
    body = {"id": name, "modality": "svg",
            "prompt": "Draw a red circle.",
            "assert": {"min_shapes": 1}}
    body.update(over)
    p = d / f"{name}.yaml"
    p.write_text(yaml.safe_dump(body), encoding="utf-8")
    return p


# ---- case loading ----------------------------------------------------------

def test_loads_cases_from_a_directory(tmp_path):
    write_case(tmp_path, "circle")
    write_case(tmp_path, "square", prompt="Draw a square.")
    cases = load_cases(tmp_path)
    assert {c.id for c in cases} == {"circle", "square"}
    assert all(isinstance(c, Case) for c in cases)


def test_cases_are_ordered_deterministically(tmp_path):
    for n in ("c", "a", "b"):
        write_case(tmp_path, n)
    assert [c.id for c in load_cases(tmp_path)] == ["a", "b", "c"]


def test_malformed_case_names_the_file(tmp_path):
    """A broken case must not fail anonymously in a 50-case run."""
    (tmp_path / "bad.yaml").write_text("id: bad\nmodality: svg\n", encoding="utf-8")  # no prompt
    with pytest.raises(ValueError, match="bad.yaml"):
        load_cases(tmp_path)


def test_unknown_modality_is_rejected_at_load(tmp_path):
    """Fail before spending model time, not after."""
    write_case(tmp_path, "x", modality="telepathy")
    with pytest.raises(ValueError, match="telepathy"):
        load_cases(tmp_path)


# ---- scoring ---------------------------------------------------------------

GOOD_SVG = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 2 2">'
            '<circle cx="1" cy="1" r="1"/></svg>')


def test_score_passes_a_good_artifact(tmp_path):
    case = load_cases_one(tmp_path)
    r = score(case, GOOD_SVG)
    assert r.passed
    assert r.detail == ""


def test_score_fails_an_unparseable_artifact(tmp_path):
    case = load_cases_one(tmp_path)
    r = score(case, "<svg><circle</svg>")
    assert not r.passed
    assert "parse" in r.detail.lower()


def test_min_shapes_assertion_is_enforced(tmp_path):
    case = load_cases_one(tmp_path, assert_={"min_shapes": 3})
    r = score(case, GOOD_SVG)
    assert not r.passed
    assert "3" in r.detail


def test_must_contain_assertion(tmp_path):
    case = load_cases_one(tmp_path, assert_={"must_contain": ["circle"]})
    assert score(case, GOOD_SVG).passed
    case2 = load_cases_one(tmp_path, assert_={"must_contain": ["polygon"]})
    assert not score(case2, GOOD_SVG).passed


def load_cases_one(tmp_path, **over):
    a = over.pop("assert_", {"min_shapes": 1})
    write_case(tmp_path, "one", **{"assert": a, **over})
    return load_cases(tmp_path)[0]


# ---- summarizing across candidates ----------------------------------------

def rows():
    return [
        Result("circle", "fast", True, 1.0, 900, ""),
        Result("square", "fast", False, 1.2, 900, "draws nothing"),
        Result("circle", "big", True, 8.0, 4200, ""),
        Result("square", "big", True, 9.0, 4200, ""),
    ]


def test_summary_groups_by_candidate():
    s = summarize(rows())
    assert s["fast"]["passed"] == 1 and s["fast"]["total"] == 2
    assert s["big"]["passed"] == 2 and s["big"]["total"] == 2


def test_summary_reports_median_not_mean_latency():
    """One cold model load must not decide which candidate looks fastest."""
    s = summarize([
        Result("a", "x", True, 1.0, 0, ""),
        Result("b", "x", True, 1.0, 0, ""),
        Result("c", "x", True, 60.0, 0, ""),   # cold start
    ])
    assert s["x"]["median_s"] == 1.0


def test_summary_is_json_serializable():
    json.dumps(summarize(rows()))


def test_summary_of_nothing_is_empty_not_a_crash():
    assert summarize([]) == {}


# ---- params vs assertions --------------------------------------------------
#
# `params:` are generation knobs handed to the engine. `assert:` are checks.
# Width lived under `assert:` and was doing both jobs, which meant the engine
# read the assertion block and the checker read generation settings.

def test_params_are_loaded_separately_from_assertions(tmp_path):
    (tmp_path / "i.yaml").write_text(yaml.safe_dump({
        "id": "i", "modality": "image", "prompt": "a fox",
        "params": {"width": 512, "height": 512, "seed": 42}}), encoding="utf-8")
    c = load_cases(tmp_path)[0]
    assert c.params == {"width": 512, "height": 512, "seed": 42}
    assert c.assertions == {}


def test_a_case_with_neither_block_is_fine(tmp_path):
    (tmp_path / "i.yaml").write_text(yaml.safe_dump({
        "id": "i", "modality": "image", "prompt": "a fox"}), encoding="utf-8")
    c = load_cases(tmp_path)[0]
    assert c.params == {} and c.assertions == {}


def test_a_misspelled_param_is_caught_at_load_not_after_the_run(tmp_path):
    """`widht: 512` used to be accepted silently and generate at the default
    size, and the case would pass."""
    (tmp_path / "i.yaml").write_text(yaml.safe_dump({
        "id": "i", "modality": "image", "prompt": "a fox",
        "params": {"widht": 512}}), encoding="utf-8")
    with pytest.raises(ValueError, match="widht"):
        load_cases(tmp_path)


def test_a_misspelled_assertion_is_caught_at_load(tmp_path):
    write_case(tmp_path, "x", **{"assert": {"must_contian": ["circle"]}})
    with pytest.raises(ValueError, match="must_contian"):
        load_cases(tmp_path)


def test_a_text_assertion_on_an_image_case_is_rejected_rather_than_ignored(tmp_path):
    """There is no text to search, so this can only ever pass vacuously. Until
    the suite can OCR, saying so is better than a silent pass."""
    (tmp_path / "i.yaml").write_text(yaml.safe_dump({
        "id": "i", "modality": "image", "prompt": "a sign reading OPEN",
        "assert": {"must_contain": ["OPEN"]}}), encoding="utf-8")
    with pytest.raises(ValueError, match="must_contain"):
        load_cases(tmp_path)


# ---- unified scoring -------------------------------------------------------

def test_score_judges_images_through_the_same_entry_point(tmp_path):
    """score() was bypassed for images, so the two modalities were judged by
    two rulers and image cases silently lost every shared assertion."""
    import random
    from PIL import Image
    png = tmp_path / "a.png"
    im = Image.new("RGB", (64, 64))
    im.putdata([(random.randint(0, 255),) * 3 for _ in range(64 * 64)])
    im.save(png)
    case = Case(id="i", modality="image", prompt="a fox",
                params={"width": 64, "height": 64})
    assert score(case, png).passed


def test_score_fails_an_image_at_the_wrong_size(tmp_path):
    from PIL import Image
    png = tmp_path / "a.png"
    Image.new("RGB", (64, 64), (1, 2, 3)).save(png)
    case = Case(id="i", modality="image", prompt="a fox",
                params={"width": 512, "height": 512})
    r = score(case, png)
    assert not r.passed and "512" in r.detail


def test_score_fails_a_uniform_image(tmp_path):
    from PIL import Image
    png = tmp_path / "a.png"
    Image.new("RGB", (64, 64), (128, 128, 128)).save(png)
    case = Case(id="i", modality="image", prompt="a fox", params={})
    assert not score(case, png).passed


def test_score_of_a_modality_with_no_checker_is_not_a_silent_pass():
    """Every shipped modality has a checker now, so this constructs one that
    does not. The behaviour still matters: a lane with nothing measuring it
    would otherwise report a 100% pass rate, which is the most misleading
    number the suite could print."""
    case = Case(id="t", modality="telepathy", prompt="a fox")
    r = score(case, "anything")
    assert not r.passed
    assert "no checker" in r.detail.lower()


def test_the_cases_that_ship_with_the_suite_actually_load():
    """Load-time validation is only worth having if the shipped cases pass it.
    Nothing else in the tests reads the real cases directory."""
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent / "evals" / "cases"
    try:
        cases = load_cases(root)
    except ValueError as exc:
        # The generated stt cases reference audio on an external volume. When
        # that drive did not come back after a reboot, this raised and every
        # other assertion below went untested. Skipping is right ONLY for that
        # cause: a genuinely malformed committed case must still fail here.
        if "audio_file" in str(exc) and "/Volumes/" in str(exc):
            import pytest as _pytest
            _pytest.skip(f"stt corpus volume is not mounted: {exc}")
        raise
    assert len(cases) >= 8
    committed = {"svg", "web", "image", "tts", "video", "code", "extract"}
    present = {c.modality for c in cases}
    assert committed <= present, f"missing lanes: {committed - present}"
    # stt cases are generated per machine and gitignored (the audio lives on a
    # volume), so they may or may not be here. Anything else is a typo.
    assert present <= committed | {"stt"}, f"unexpected: {present - committed}"
    assert all(c.prompt.strip() for c in cases)


# ---- metrics: the quality axis --------------------------------------------
#
# A pass rate separates working from broken. It cannot order two candidates
# that both work, which is exactly the situation the suite kept landing in.
# Metrics are numbers a checker produces alongside the verdict.

def test_a_result_carries_metrics_from_its_checker(tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"x" * 9000)
    case = Case(id="s", modality="tts", prompt="the quick brown fox",
                assertions={"max_wer": 0.5})
    import harness.checks.speech as speech
    r = score(case, wav, transcriber=lambda p: "the quick brown box")
    assert r.passed
    assert r.metrics["wer"] == 0.25
    assert speech  # the checker used


def test_metrics_survive_a_failing_row(tmp_path):
    """A candidate that fails the threshold still has a number worth seeing."""
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"x" * 9000)
    case = Case(id="s", modality="tts", prompt="the quick brown fox",
                assertions={"max_wer": 0.1})
    r = score(case, wav, transcriber=lambda p: "a slow green ox")
    assert not r.passed
    assert r.metrics["wer"] >= 1.0


def test_a_quality_metric_is_averaged_not_medianed():
    """Latency takes a median so one cold model load cannot decide the winner.
    A quality metric is the opposite: most cases score a clean 0.0 and the
    whole signal is in the few that do not, so a median reports 0.0 for a
    candidate that failed a case outright."""
    clean = [Result(str(i), "a", True, 1.0, 0, "", metrics={"wer": 0.0})
             for i in range(4)]
    rows = clean + [Result("x", "a", True, 1.0, 0, "", metrics={"wer": 0.5})]
    assert summarize(rows)["a"]["metrics"]["wer"] == 0.1


def test_the_worst_case_of_each_metric_is_kept_so_an_outlier_is_visible():
    rows = [Result("a", "k", True, 1.0, 0, "", metrics={"wer": 0.0}),
            Result("b", "k", True, 1.0, 0, "", metrics={"wer": 0.4})]
    assert summarize(rows)["k"]["metrics_worst"]["wer"] == 0.4


def test_a_candidate_with_no_metrics_reports_none_rather_than_zero():
    """Zero would rank an unmeasured candidate first."""
    s = summarize([Result("a", "x", True, 1.0, 0, "")])
    assert s["x"]["metrics"] == {}


def test_metrics_are_json_serializable():
    json.dumps(summarize([Result("a", "x", True, 1.0, 0, "",
                                 metrics={"wer": 0.1})]))


def test_max_wer_is_a_valid_assertion_for_tts(tmp_path):
    (tmp_path / "t.yaml").write_text(yaml.safe_dump({
        "id": "t", "modality": "tts", "prompt": "hello there",
        "assert": {"max_wer": 0.2}, "params": {"speed": 1.0}}), encoding="utf-8")
    c = load_cases(tmp_path)[0]
    assert c.assertions["max_wer"] == 0.2


def test_min_shapes_is_not_a_valid_assertion_for_tts(tmp_path):
    (tmp_path / "t.yaml").write_text(yaml.safe_dump({
        "id": "t", "modality": "tts", "prompt": "hello",
        "assert": {"min_shapes": 2}}), encoding="utf-8")
    with pytest.raises(ValueError, match="min_shapes"):
        load_cases(tmp_path)


# ---- image text assertions -------------------------------------------------

def render_text(path, text):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.new("RGB", (512, 512), (250, 250, 250))
    d = ImageDraw.Draw(im)
    # A real typeface on whichever machine this is. load_default() is a small
    # bitmap font that OCR reads as nothing at all, which fails the test for a
    # reason that has nothing to do with what it is checking.
    font = None
    for candidate in ("/System/Library/Fonts/Supplemental/Arial.ttf",
                      "C:/Windows/Fonts/arial.ttf"):
        try:
            font = ImageFont.truetype(candidate, 96)
            break
        except OSError:
            continue
    if font is None:  # pragma: no cover - fallback for a stripped system
        font = ImageFont.load_default()
    d.text((40, 200), text, fill=(10, 10, 10), font=font)
    im.save(path)
    return path


def test_an_image_case_can_assert_the_text_it_should_render(tmp_path):
    """Until now the suite could not tell OPEN from OPEM: both decode, both are
    the right size, both have variance."""
    case = Case(id="sign", modality="image", prompt="a sign reading OPEN",
                params={"width": 512, "height": 512},
                assertions={"text": "OPEN"})
    assert score(case, render_text(tmp_path / "a.png", "OPEN")).passed


def test_a_misrendered_word_fails_and_reports_what_was_read(tmp_path):
    case = Case(id="sign", modality="image", prompt="a sign reading OPEN",
                params={"width": 512, "height": 512},
                assertions={"text": "OPEN", "max_cer": 0.0})
    r = score(case, render_text(tmp_path / "a.png", "OPEM"))
    assert not r.passed
    assert "OPEM" in r.detail.upper()
    assert r.metrics["cer"] > 0


def test_the_character_error_rate_reaches_the_row_for_ranking(tmp_path):
    case = Case(id="sign", modality="image", prompt="a sign reading OPEN",
                params={"width": 512, "height": 512},
                assertions={"text": "OPEN"})
    assert score(case, render_text(tmp_path / "a.png", "OPEN")).metrics["cer"] == 0.0


def test_the_pixel_check_runs_first_so_a_blank_image_is_not_an_ocr_failure(tmp_path):
    """"no text found" for a uniform grey square is a true statement and a
    useless diagnosis."""
    from PIL import Image
    p = tmp_path / "a.png"
    Image.new("RGB", (512, 512), (128, 128, 128)).save(p)
    case = Case(id="sign", modality="image", prompt="x",
                params={"width": 512, "height": 512},
                assertions={"text": "OPEN"})
    r = score(case, p)
    assert not r.passed
    assert "uniform" in r.detail.lower()


def test_an_image_case_with_no_text_assertion_never_runs_ocr(tmp_path):
    import random
    from PIL import Image
    p = tmp_path / "a.png"
    im = Image.new("RGB", (64, 64))
    im.putdata([(random.randint(0, 255),) * 3 for _ in range(64 * 64)])
    im.save(p)
    case = Case(id="i", modality="image", prompt="a fox", params={})
    r = score(case, p)
    assert r.passed
    assert "cer" not in r.metrics


def test_text_is_a_valid_assertion_for_image_cases(tmp_path):
    (tmp_path / "i.yaml").write_text(yaml.safe_dump({
        "id": "i", "modality": "image", "prompt": "a sign reading OPEN",
        "assert": {"text": "OPEN", "max_cer": 0.3}}), encoding="utf-8")
    c = load_cases(tmp_path)[0]
    assert c.assertions["text"] == "OPEN"


# ---- svg is rasterized as well as parsed -----------------------------------

BLANK_SVG = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
             '<circle cx="12" cy="12" r="10" fill="white"/>'
             '<rect x="0" y="0" width="4" height="4" fill="white"/></svg>')


def test_an_svg_that_parses_but_draws_nothing_visible_fails():
    """Two shapes, valid markup, an xmlns, a viewBox -- and it renders as an
    empty rectangle. Every structural check the suite had says it is fine."""
    import shutil
    if render.rasterizer_path() is None:
        pytest.skip("needs rsvg-convert")
    case = Case(id="s", modality="svg", prompt="a gear",
                assertions={"min_shapes": 2})
    r = score(case, BLANK_SVG)
    assert not r.passed
    assert "blank" in r.detail.lower()


def test_a_drawn_svg_reports_its_ink_coverage():
    import shutil
    if render.rasterizer_path() is None:
        pytest.skip("needs rsvg-convert")
    case = Case(id="s", modality="svg", prompt="a gear",
                assertions={"min_shapes": 1})
    r = score(case, GOOD_SVG)
    assert r.passed
    assert r.metrics["ink"] > 0


def test_the_structural_check_runs_first(tmp_path):
    """Unparseable markup should be reported as unparseable, not as something
    that failed to render."""
    case = Case(id="s", modality="svg", prompt="a gear")
    r = score(case, "<svg><circle</svg>")
    assert not r.passed
    assert "parse" in r.detail.lower()


# ---- video -----------------------------------------------------------------

def make_clip(path, source="testsrc", frames=8, size="128x128"):
    import subprocess
    sep = ":" if "=" in source else "="
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", f"{source}{sep}size={size}:rate=8", "-frames:v", str(frames),
         "-pix_fmt", "yuv420p", str(path)], check=True, capture_output=True)
    return path


def needs_ffmpeg():
    import shutil
    if shutil.which("ffmpeg") is None:
        pytest.skip("needs ffmpeg")


def test_video_is_scored_through_the_same_entry_point(tmp_path):
    """It used to have no checker at all, so a video candidate FAILED by
    construction with 'no checker for modality video'."""
    needs_ffmpeg()
    case = Case(id="v", modality="video", prompt="a fox running",
                params={"width": 128, "height": 128, "frames": 8})
    assert score(case, make_clip(tmp_path / "v.mp4")).passed


def test_a_video_where_nothing_moves_fails(tmp_path):
    needs_ffmpeg()
    case = Case(id="v", modality="video", prompt="a fox running",
                params={"width": 128, "height": 128, "frames": 8})
    r = score(case, make_clip(tmp_path / "v.mp4", source="color=c=gray"))
    assert not r.passed


def test_the_requested_frame_count_is_read_from_params(tmp_path):
    needs_ffmpeg()
    case = Case(id="v", modality="video", prompt="x",
                params={"width": 128, "height": 128, "frames": 22})
    r = score(case, make_clip(tmp_path / "v.mp4", frames=8))
    assert not r.passed
    assert "22" in r.detail


def test_seconds_are_converted_to_frames_at_24fps(tmp_path):
    """A case may ask for a duration instead; h3 takes either and the checker
    has to know they mean the same thing."""
    needs_ffmpeg()
    case = Case(id="v", modality="video", prompt="x", params={"seconds": 1})
    r = score(case, make_clip(tmp_path / "v.mp4", frames=8))
    assert not r.passed
    assert "24" in r.detail


def test_motion_reaches_the_row_for_ranking(tmp_path):
    needs_ffmpeg()
    case = Case(id="v", modality="video", prompt="x", params={})
    assert score(case, make_clip(tmp_path / "v.mp4")).metrics["motion"] > 0


# ---- metric direction ------------------------------------------------------
#
# Every metric so far was an error rate, so "lower is better" was baked into
# both the ranking and the worst-case column. `ink` and `motion` broke that
# silently the moment they were added -- more ink is better -- and prompt
# adherence breaks it again. A metric that does not declare its direction
# cannot be ranked on.

def test_every_published_metric_declares_a_direction():
    from evals.core import METRIC_DIRECTION
    for name in ("wer", "cer", "ink", "motion"):
        assert name in METRIC_DIRECTION, f"{name} has no declared direction"
        assert METRIC_DIRECTION[name] in ("lower", "higher", "neutral")


def test_error_rates_are_lower_is_better():
    from evals.core import METRIC_DIRECTION
    assert METRIC_DIRECTION["wer"] == "lower"
    assert METRIC_DIRECTION["cer"] == "lower"


def test_coverage_metrics_are_higher_is_better():
    from evals.core import METRIC_DIRECTION
    assert METRIC_DIRECTION["motion"] == "higher"


def test_ink_is_a_floor_and_is_not_ranked_on():
    """It catches SVG that renders as an empty rectangle, and the checker
    already fails those outright, so above zero the direction says nothing.
    As "higher" it crowned the worst candidate in the svg lane: OmniSVG drew
    one filled blob where three shapes were asked for, and marking 85% of the
    canvas is what that looks like to this metric."""
    from evals.core import METRIC_DIRECTION
    assert METRIC_DIRECTION["ink"] == "neutral"


def test_the_worst_case_of_a_higher_is_better_metric_is_its_minimum():
    """metrics_worst took a max unconditionally, so the 'worst' motion was the
    liveliest case in the run."""
    rows = [Result("a", "k", True, 1.0, 0, "", metrics={"motion": 0.02}),
            Result("b", "k", True, 1.0, 0, "", metrics={"motion": 0.40})]
    assert summarize(rows)["k"]["metrics_worst"]["motion"] == 0.02


def test_the_worst_case_of_a_lower_is_better_metric_is_still_its_maximum():
    rows = [Result("a", "k", True, 1.0, 0, "", metrics={"wer": 0.0}),
            Result("b", "k", True, 1.0, 0, "", metrics={"wer": 0.4})]
    assert summarize(rows)["k"]["metrics_worst"]["wer"] == 0.4


def test_an_undeclared_metric_is_treated_as_lower_is_better_and_says_so():
    """A new checker that forgets to declare a direction should not silently
    invert the ranking."""
    from evals.core import direction_of
    with pytest.warns(UserWarning, match="direction"):
        assert direction_of("some_new_metric") == "lower"


def test_adherence_is_measured_only_when_a_backend_is_asked_for(tmp_path):
    """It loads a 4GB model, so it is opt-in rather than a tax on every image
    run. Without the flag, an image case reports no adherence."""
    import random
    from PIL import Image
    p = tmp_path / "a.png"
    im = Image.new("RGB", (64, 64))
    im.putdata([(random.randint(0, 255),) * 3 for _ in range(64 * 64)])
    im.save(p)
    case = Case(id="i", modality="image", prompt="a fox", params={})
    assert "adherence" not in score(case, p).metrics


def test_an_unknown_adherence_backend_is_rejected(tmp_path):
    from PIL import Image
    import random
    p = tmp_path / "a.png"
    im = Image.new("RGB", (64, 64))
    im.putdata([(random.randint(0, 255),) * 3 for _ in range(64 * 64)])
    im.save(p)
    case = Case(id="i", modality="image", prompt="a fox", params={})
    with pytest.raises(ValueError, match="pickscore"):
        score(case, p, adherence="vibes")


def test_a_page_that_parses_but_renders_blank_fails():
    """The same failure SVG had: valid markup, real content, white on white."""
    from harness.checks import render
    if render.chrome_path() is None:
        pytest.skip("needs Chrome")
    blank = ("<!doctype html><html><head><title>t</title><style>"
             "body{background:#fff;color:#fff}</style></head>"
             "<body><h1>Invisible</h1></body></html>")
    case = Case(id="p", modality="web", prompt="a page")
    r = score(case, blank)
    assert not r.passed
    assert "blank" in r.detail.lower()


def test_a_real_page_reports_its_ink():
    from harness.checks import render
    if render.chrome_path() is None:
        pytest.skip("needs Chrome")
    page = ("<!doctype html><html><head><title>t</title></head>"
            "<body><h1>Coffee Roaster</h1><p>Beans since 1994.</p></body></html>")
    r = score(Case(id="p", modality="web", prompt="a page"), page)
    assert r.passed, r.detail
    assert r.metrics["ink"] > 0


def test_the_structural_html_check_runs_before_rendering():
    case = Case(id="p", modality="web", prompt="a page")
    r = score(case, "I would rather not write a web page.")
    assert not r.passed
    assert "blank" not in r.detail.lower()


# ---- context: input the model works ON, not just a prompt -------------------
#
# svg and image cases are instructions with no input. A log-triage case is the
# opposite: the instruction is one line and the material is forty. Keeping them
# separate means the prompt stays readable and the material can be swapped.

def test_a_case_can_carry_context(tmp_path):
    (tmp_path / "c.yaml").write_text(yaml.safe_dump({
        "id": "c", "modality": "extract", "prompt": "Which line is the error?",
        "context": "INFO ok\nERROR disk full\nINFO done",
        "assert": {"must_contain": ["disk full"]}}), encoding="utf-8")
    c = load_cases(tmp_path)[0]
    assert "disk full" in c.context
    assert c.prompt == "Which line is the error?"


def test_context_defaults_to_empty(tmp_path):
    write_case(tmp_path, "x")
    assert load_cases(tmp_path)[0].context == ""


def test_context_can_come_from_a_file_beside_the_case(tmp_path):
    """A realistic log is hundreds of lines and does not belong inline."""
    (tmp_path / "build.log").write_text("INFO ok\nFATAL out of memory\n", encoding="utf-8")
    (tmp_path / "c.yaml").write_text(yaml.safe_dump({
        "id": "c", "modality": "extract", "prompt": "What failed?",
        "context_file": "build.log"}), encoding="utf-8")
    assert "out of memory" in load_cases(tmp_path)[0].context


def test_a_missing_context_file_names_the_case(tmp_path):
    (tmp_path / "c.yaml").write_text(yaml.safe_dump({
        "id": "c", "modality": "extract", "prompt": "x",
        "context_file": "nope.log"}), encoding="utf-8")
    with pytest.raises(ValueError, match="c.yaml"):
        load_cases(tmp_path)


def test_context_and_context_file_together_are_rejected(tmp_path):
    (tmp_path / "c.yaml").write_text(yaml.safe_dump({
        "id": "c", "modality": "extract", "prompt": "x",
        "context": "inline", "context_file": "f.log"}), encoding="utf-8")
    with pytest.raises(ValueError, match="context"):
        load_cases(tmp_path)


# ---- extract: the small, fast lane -----------------------------------------

def test_extract_is_scored_on_what_it_answered(tmp_path):
    case = Case(id="e", modality="extract", prompt="Which line failed?",
                context="INFO ok\nFATAL out of memory",
                assertions={"must_contain": ["out of memory"]})
    assert score(case, "The build failed: out of memory.").passed


def test_extract_fails_when_the_answer_is_missing_the_fact(tmp_path):
    case = Case(id="e", modality="extract", prompt="Which line failed?",
                context="INFO ok\nFATAL out of memory",
                assertions={"must_contain": ["out of memory"]})
    r = score(case, "Something went wrong somewhere.")
    assert not r.passed


def test_extract_supports_an_exact_answer(tmp_path):
    """The narrowest and most useful shape: one token out, nothing else."""
    case = Case(id="e", modality="extract", prompt="What is the exit code?",
                context="process exited with 137", assertions={"equals": "137"})
    assert score(case, "137").passed
    assert score(case, " 137\n").passed, "surrounding whitespace is not an error"
    assert not score(case, "The exit code was 137.").passed


def test_an_exact_answer_is_case_insensitive():
    case = Case(id="e", modality="extract", prompt="x", context="y",
                assertions={"equals": "FATAL"})
    assert score(case, "fatal").passed


def test_extract_publishes_no_metric_of_its_own():
    """Its quality axis is latency, which every row already carries. Inventing
    a score here would be decoration."""
    case = Case(id="e", modality="extract", prompt="x", context="y",
                assertions={"equals": "z"})
    assert score(case, "z").metrics == {}


# ---- code ------------------------------------------------------------------

SLUGIFY = ('def slugify(text):\n    import re\n'
           '    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")\n')


def code_case(**over):
    base = dict(id="slugify", modality="code",
                prompt="Write slugify(text).",
                assertions={"checks": ["slugify('A B') == 'a-b'"]})
    base.update(over)
    return Case(**base)


def test_code_is_scored_by_running_it():
    assert score(code_case(), SLUGIFY).passed


def test_code_that_gets_it_wrong_fails_with_the_check_that_broke():
    r = score(code_case(), 'def slugify(text):\n    return text\n')
    assert not r.passed
    assert "a-b" in r.detail


def test_the_pass_fraction_reaches_the_row_for_ranking():
    case = code_case(assertions={"checks": [
        "slugify('A B') == 'a-b'", "slugify('--x--') == 'x'"]})
    r = score(case, 'def slugify(text):\n    return text.lower().replace(" ", "-")\n')
    assert r.metrics["code_pass"] == 0.5


def test_code_pass_is_higher_is_better():
    from evals.core import METRIC_DIRECTION
    assert METRIC_DIRECTION["code_pass"] == "higher"


def test_a_code_case_must_declare_checks(tmp_path):
    """Without them the case passes every model."""
    (tmp_path / "c.yaml").write_text(yaml.safe_dump({
        "id": "c", "modality": "code", "prompt": "Write slugify."}), encoding="utf-8")
    with pytest.raises(ValueError, match="checks"):
        load_cases(tmp_path)


def test_checks_is_a_valid_assertion_only_for_code(tmp_path):
    (tmp_path / "c.yaml").write_text(yaml.safe_dump({
        "id": "c", "modality": "svg", "prompt": "x",
        "assert": {"checks": ["1 == 1"]}}), encoding="utf-8")
    with pytest.raises(ValueError, match="checks"):
        load_cases(tmp_path)


# ---- stt -------------------------------------------------------------------
#
# The mirror of tts. There the prompt is the text to speak and the artifact is
# audio; here the audio is the input and the prompt is the reference transcript
# a human wrote. That is what makes this measurement of the STT model ALONE,
# unlike the tts round trip, which is joint with whatever reads it back.

def test_an_stt_case_carries_the_audio_it_transcribes(tmp_path):
    clip = tmp_path / "a.wav"
    clip.write_bytes(b"x" * 9000)
    (tmp_path / "c.yaml").write_text(yaml.safe_dump({
        "id": "c", "modality": "stt", "prompt": "the quick brown fox",
        "audio_file": "a.wav", "assert": {"max_wer": 0.2}}), encoding="utf-8")
    c = load_cases(tmp_path)[0]
    assert c.audio == clip


def test_a_missing_audio_file_names_the_case(tmp_path):
    (tmp_path / "c.yaml").write_text(yaml.safe_dump({
        "id": "c", "modality": "stt", "prompt": "x", "audio_file": "nope.wav"}), encoding="utf-8")
    with pytest.raises(ValueError, match="c.yaml"):
        load_cases(tmp_path)


def test_an_stt_case_needs_audio(tmp_path):
    (tmp_path / "c.yaml").write_text(yaml.safe_dump({
        "id": "c", "modality": "stt", "prompt": "x"}), encoding="utf-8")
    with pytest.raises(ValueError, match="audio_file"):
        load_cases(tmp_path)


def test_an_absolute_audio_path_is_used_as_given(tmp_path):
    """Corpus cases are generated against a volume, not shipped with the repo."""
    clip = tmp_path / "a.wav"
    clip.write_bytes(b"x" * 9000)
    (tmp_path / "c.yaml").write_text(yaml.safe_dump({
        "id": "c", "modality": "stt", "prompt": "x",
        "audio_file": str(clip)}), encoding="utf-8")
    assert load_cases(tmp_path)[0].audio == clip


def test_stt_is_scored_against_the_human_transcript():
    case = Case(id="s", modality="stt", prompt="the quick brown fox",
                assertions={"max_wer": 0.3})
    r = score(case, "The quick brown fox.")
    assert r.passed
    assert r.metrics["wer"] == 0.0


def test_stt_fails_a_bad_transcript():
    case = Case(id="s", modality="stt", prompt="the quick brown fox",
                assertions={"max_wer": 0.1})
    r = score(case, "a slow green ox")
    assert not r.passed
    assert r.metrics["wer"] > 0.1


def test_stt_measures_without_a_threshold():
    case = Case(id="s", modality="stt", prompt="the quick brown fox")
    r = score(case, "the quick brown box")
    assert r.passed
    assert r.metrics["wer"] == 0.25


def test_max_wer_is_the_assertion_for_stt(tmp_path):
    clip = tmp_path / "a.wav"
    clip.write_bytes(b"x" * 9000)
    (tmp_path / "c.yaml").write_text(yaml.safe_dump({
        "id": "c", "modality": "stt", "prompt": "x", "audio_file": "a.wav",
        "assert": {"min_shapes": 2}}), encoding="utf-8")
    with pytest.raises(ValueError, match="min_shapes"):
        load_cases(tmp_path)


def test_a_ratio_metric_aggregates_as_a_corpus_rate_not_a_mean_of_rates():
    """A two-word utterance must not weigh the same as a forty-word one.
    Measured on LibriSpeech: one error in "Ay me" scores 0.500 and was the
    worst row in a run whose mean was 0.023, purely because the clip is short.
    Corpus wer -- total errors over total words -- is what ASR reports."""
    rows = [Result("short", "m", True, 0.1, 0, "",
                   metrics={"wer": 0.5, "wer_errors": 1, "wer_words": 2}),
            Result("long", "m", True, 0.1, 0, "",
                   metrics={"wer": 0.0, "wer_errors": 0, "wer_words": 38})]
    got = summarize(rows)["m"]["metrics"]
    assert got["wer"] == round(1 / 40, 4)   # corpus rate, not (0.5 + 0.0) / 2


def test_the_count_companions_are_not_shown_as_metrics_of_their_own():
    """wer_errors is bookkeeping. A column of it in the table is noise."""
    rows = [Result("a", "m", True, 0.1, 0, "",
                   metrics={"wer": 0.5, "wer_errors": 1, "wer_words": 2})]
    assert set(summarize(rows)["m"]["metrics"]) == {"wer"}


def test_a_metric_with_no_counts_still_averages():
    rows = [Result("a", "m", True, 0.1, 0, "", metrics={"adherence": 20.0}),
            Result("b", "m", True, 0.1, 0, "", metrics={"adherence": 24.0})]
    assert summarize(rows)["m"]["metrics"]["adherence"] == 22.0


def test_the_worst_row_is_still_the_worst_row():
    """The corpus rate is the headline; the outlier must stay visible."""
    rows = [Result("short", "m", True, 0.1, 0, "",
                   metrics={"wer": 0.5, "wer_errors": 1, "wer_words": 2}),
            Result("long", "m", True, 0.1, 0, "",
                   metrics={"wer": 0.0, "wer_errors": 0, "wer_words": 38})]
    assert summarize(rows)["m"]["metrics_worst"]["wer"] == 0.5


# ---- case language ---------------------------------------------------------

def test_a_case_declares_its_language_and_defaults_to_english(tmp_path):
    (tmp_path / "a.yaml").write_text(
        "id: en\nmodality: tts\nprompt: hello there\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text(
        "id: fr\nmodality: tts\nprompt: bonjour\nlanguage: fr\n", encoding="utf-8")
    by_id = {c.id: c for c in load_cases(tmp_path)}
    assert by_id["en"].language == "en"
    assert by_id["fr"].language == "fr"


def test_a_french_case_is_scored_with_french_number_words(tmp_path):
    """The case knows its language; the checker has to be told."""
    from evals.core import score
    r = score(Case(id="n", modality="stt", prompt="92 jetons", language="fr"),
              "quatre-vingt-douze jetons")
    assert r.metrics["wer"] == 0.0


# ---- how a candidate failed, not just how often ---------------------------
# A pass rate hid Qwen3-14B returning NULL CONTENT behind a 7/9 score: both its
# failures were "no answer", not "wrong answer", and the row read like a
# winner. It hid whisper the other way round too -- 0/40, no metric at all, and
# a missing lower-is-better number sorted as a perfect 0.0.

def test_a_result_records_why_it_failed():
    from evals.core import failure_kind
    assert failure_kind("") == ""
    assert failure_kind("word error rate 0.47 over the limit") == "wrong"
    assert failure_kind("2/5 checks passed") == "wrong"


def test_producing_nothing_is_not_the_same_as_producing_something_wrong():
    """The distinction the report exists to make."""
    from evals.core import failure_kind
    for detail in ("empty completion",
                   "q3-14b returned no answer: it spent the whole 4000-token "
                   "budget on reasoning",
                   "mflux exited 0 but left no output at /tmp/x.png"):
        assert failure_kind(detail) == "empty", detail


def test_a_broken_instrument_is_its_own_kind():
    """An outage is not the candidate scoring badly. Recording it as a wrong
    answer is how a dead server becomes a model ranking."""
    from evals.core import failure_kind
    for detail in ("timed out after 180.0s",
                   "gateway unreachable at http://127.0.0.1:4000",
                   "could not transcribe: stt unreachable"):
        assert failure_kind(detail) == "error", detail


def test_summarize_counts_each_kind_separately():
    from evals.core import summarize
    s = summarize([
        Result("a", "m", True, 1.0, 0, ""),
        Result("b", "m", False, 1.0, 0, "2/5 checks passed"),
        Result("c", "m", False, 1.0, 0, "empty completion"),
        Result("d", "m", False, 1.0, 0, "timed out after 180.0s"),
    ])["m"]
    assert s["wrong"] == 1 and s["empty"] == 1 and s["errored"] == 1


# ---- comparability --------------------------------------------------------
# From EnviousWispr's model_registry.comparable(). Rows have been ranked here
# across runs with different sampling and different warm/cold conditions, and
# local-mid's svg ink of 0.564 -- computed over the TWO cases it passed -- was
# printed beside numbers computed over nine.

def test_a_metric_carries_the_sample_it_was_computed_over():
    from evals.core import summarize
    s = summarize([
        Result("a", "m", True, 1.0, 0, "", metrics={"ink": 0.5}),
        Result("b", "m", False, 1.0, 0, "no ink here"),
        Result("c", "m", True, 1.0, 0, "", metrics={"ink": 0.3}),
    ])["m"]
    assert s["metric_n"]["ink"] == 2, "a mean over 2 of 3 must say so"


def test_two_runs_of_the_same_shape_are_comparable():
    from evals.core import Receipt, comparable
    a = Receipt(modality="svg", case_ids=("x", "y"), repeat=3,
                sampling={"temperature": 0.4}, gateway="http://gw")
    ok, why = comparable(a, Receipt(**{**a.__dict__}))
    assert ok, why


def test_a_different_case_set_is_not_comparable():
    from evals.core import Receipt, comparable
    a = Receipt(modality="svg", case_ids=("x", "y"), repeat=3,
                sampling={}, gateway="http://gw")
    b = Receipt(modality="svg", case_ids=("x", "z"), repeat=3,
                sampling={}, gateway="http://gw")
    ok, why = comparable(a, b)
    assert not ok and "case" in why.lower()


def test_different_sampling_is_not_comparable():
    """Adding a repetition penalty changed what the svg lane produces. Ranking
    a run from before it against one from after is comparing two exams."""
    from evals.core import Receipt, comparable
    a = Receipt(modality="svg", case_ids=("x",), repeat=1,
                sampling={"temperature": 0.2}, gateway="http://gw")
    b = Receipt(modality="svg", case_ids=("x",), repeat=1,
                sampling={"temperature": 0.4, "repetition_penalty": 1.1},
                gateway="http://gw")
    ok, why = comparable(a, b)
    assert not ok and "sampling" in why.lower()


def test_a_gpu_timeout_is_an_instrument_failure_not_a_bad_drawing():
    """Caught live on the first trace run. Metal's wording is 'GPU Timeout
    Error', not 'timed out', so the classifier called a driver failure a wrong
    answer and it counted against the model's score."""
    from evals.core import failure_kind
    detail = ("mflux exited 1: RuntimeError: [METAL] Command buffer execution "
              "failed: Caused GPU Timeout Error "
              "(00000002:kIOGPUCommandBufferCallbackErrorTimeout).")
    assert failure_kind(detail) == "error"


def test_an_out_of_memory_crash_is_also_the_instrument():
    from evals.core import failure_kind
    assert failure_kind("mflux exited 3: out of memory") == "error"


# ---- a case can be unfair to a method -------------------------------------
# chart-bars asserts must_contain: ["text"]. A vectorizer CANNOT satisfy it:
# tracing turns glyphs into outlines, so a traced SVG has no <text> element by
# construction. The case was written when an LLM was the only method, and it
# encodes that assumption. Comparing methods on it measures the assumption.

def test_a_case_can_declare_which_methods_it_is_fair_to(tmp_path):
    (tmp_path / "a.yaml").write_text(
        "id: bars\nmodality: svg\nprompt: draw bars\nmethods: [llm]\n"
        "assert:\n  must_contain: ['text']\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("id: gear\nmodality: svg\nprompt: a gear\n", encoding="utf-8")
    by = {c.id: c for c in load_cases(tmp_path)}
    assert by["bars"].methods == ("llm",)
    assert by["gear"].methods == (), "no declaration means fair to every method"


# ---- must_contain was a naive substring ------------------------------------
# From the 2026-09-05 hostile review, never closed: `must_contain` lowercases
# both sides and asks `needle in artifact`, so "text" is satisfied by the word
# "context" anywhere in the document -- including inside a comment or a CSS
# class name. chart-bars asserted exactly that needle.

def test_a_bare_word_needle_needs_a_word_boundary():
    from evals.core import contains
    assert not contains("<svg><desc>a context diagram</desc></svg>", "text")
    assert contains("<svg><text>9</text></svg>", "text")


def test_a_needle_with_punctuation_is_still_a_raw_substring():
    """Authors already write "<table" and 'type="password"'. Those must keep
    working exactly as they read."""
    from evals.core import contains
    assert contains("<table class=x>", "<table")
    assert contains('<input type="password">', 'type="password"')
    assert not contains("<tablet>", "<table ")


def test_matching_stays_case_insensitive():
    from evals.core import contains
    assert contains("<SVG><TEXT>9</TEXT></SVG>", "text")


# ---- degenerate shapes counted as drawing ----------------------------------

def test_zero_sized_shapes_do_not_count_towards_min_shapes():
    """Also from the hostile review: eight `<rect width="0" height="0"/>`
    satisfied min_shapes: 6 and the case passed. A shape with no extent is not
    a shape."""
    from harness.checks import svg
    doc = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 9 9">'
           + '<rect width="0" height="0"/>' * 8 + "</svg>")
    r = svg.check(doc)
    assert r.shape_count == 0, f"counted {r.shape_count} invisible rects"
    assert not r.ok, "a document of zero-size rects draws nothing"


def test_real_shapes_still_count():
    from harness.checks import svg
    doc = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 9 9">'
           + '<rect width="3" height="3"/>' * 8 + "</svg>")
    r = svg.check(doc)
    assert r.ok and r.shape_count == 8


def test_a_metric_can_be_reported_without_being_ranked():
    """completion_tokens is context, not quality: more tokens is not better,
    and it went in as higher-is-better which would rank the most verbose
    candidate first. It explains a latency rather than scoring anything."""
    from evals.core import direction_of
    assert direction_of("completion_tokens") == "neutral"
    assert direction_of("tokens_per_s") == "higher"


def test_a_neutral_metric_is_not_used_to_rank(capsys):
    from evals.core import summarize
    from evals.run import report
    s = summarize([
        Result("a", "terse", True, 1.0, 0, "",
               metrics={"completion_tokens": 10, "ink": 0.5}),
        Result("a", "windy", True, 1.0, 0, "",
               metrics={"completion_tokens": 900, "ink": 0.1}),
    ])
    report(s)
    out = capsys.readouterr().out
    # Ranked on ink, so the terse one leads despite writing far less.
    assert out.index("terse") < out.index("windy")
    assert "completion_tokens" in out, "still reported, just not ranked on"


def test_a_scaling_workflow_can_declare_the_size_it_meant_to_produce():
    """The image check asserts the output matches the case's width/height. An
    upscaler deliberately produces a different size, so it failed every case
    for doing its job -- the same shape as chart-bars asserting <text>, which
    a vectorizer can never emit. The case says what to GENERATE; a workflow
    that scales says what it INTENDED."""
    from pathlib import Path
    import tempfile
    from PIL import Image, ImageDraw
    from evals.core import score

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "big.png"
        im = Image.new("RGB", (128, 128), "white")
        ImageDraw.Draw(im).ellipse((8, 8, 120, 120), fill="green")
        im.save(p)
        c = Case(id="fox", modality="image", prompt="a fox",
                 params={"width": 64, "height": 64})
        # Without the override the case's 64x64 is the expectation and this
        # fails for being the wrong size.
        assert not score(c, p).passed
        assert score(c, p, expect_size=(128, 128)).passed


def test_a_tool_exiting_non_zero_is_an_instrument_failure():
    """Third time this classifier has been too narrow. A subprocess that exits
    non-zero CRASHED; it did not produce a bad artifact. Counting it as a wrong
    answer scores a broken tool against the model's quality."""
    from evals.core import failure_kind
    assert failure_kind("upscale-seedvr2 (stage 2) exited 1: are supported: "
                        "repeat(array, repeats: int, axis)") == "error"
    assert failure_kind("mflux/x exited 3: something") == "error"


def test_exit_zero_with_no_output_is_still_empty_not_error():
    """The tool ran fine and produced nothing, which is a different thing and
    the ordering has to keep them apart."""
    from evals.core import failure_kind
    assert failure_kind("mflux exited 0 but left no output at /tmp/x.png") == "empty"


def test_a_screen_may_not_be_ranked_against_a_measurement():
    """Issue #53. A screen asks whether it ran; a measurement asks whether it is
    better. Ranking them together is the confound comparable() exists for."""
    from evals.core import Receipt, comparable
    base = dict(modality="image", case_ids=("a",), repeat=1, sampling={},
                gateway="g")
    ok, why = comparable(Receipt(**base, tier="screen"),
                         Receipt(**base, tier="measure"))
    assert not ok
    assert "tier" in why


def test_two_screens_are_comparable_with_each_other():
    from evals.core import Receipt, comparable
    base = dict(modality="image", case_ids=("a",), repeat=1, sampling={},
                gateway="g", tier="screen")
    assert comparable(Receipt(**base), Receipt(**base))[0]


def test_a_receipt_defaults_to_the_measure_tier():
    """Every run written before tiers existed was a measurement."""
    from evals.core import Receipt
    r = Receipt(modality="svg", case_ids=("a",), repeat=1, sampling={},
                gateway="g")
    assert r.tier == "measure"
    assert r.as_dict()["tier"] == "measure"


# ---- the machine is part of the exam ---------------------------------------

def test_two_accelerators_are_not_one_table():
    """Every lane has an implementation per machine and a different tool behind
    each, so two results.json files from one lane may have been produced by
    different programs on different silicon."""
    from evals.core import Receipt, comparable
    mini = Receipt("image", ("fox",), 3, {}, "", accelerator="unified:arm64")
    card = Receipt("image", ("fox",), 3, {}, "", accelerator="discrete:RTX 4070")
    ok, why = comparable(mini, card)
    assert not ok
    assert "accelerator" in why


def test_a_run_from_before_this_existed_is_still_comparable():
    """Refusing an empty field would invalidate every measurement recorded
    before the field did -- the STT corpus and the svg three-way included."""
    from evals.core import Receipt, comparable
    mini = Receipt("image", ("fox",), 3, {}, "", accelerator="unified:arm64")
    old = Receipt("image", ("fox",), 3, {}, "")
    assert comparable(mini, old)[0]
    assert comparable(old, old)[0]


def test_the_same_accelerator_is_one_table():
    from evals.core import Receipt, comparable
    a = Receipt("image", ("fox",), 3, {}, "", accelerator="unified:arm64")
    assert comparable(a, a)[0]
