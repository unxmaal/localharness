"""The harness itself: loading cases, running candidates, scoring comparably.

The whole point is answering "which candidate should I use" in twenty minutes
without editing code, so the contract under test is: a case file plus a
candidate name produces a comparable row.
"""
import json

import pytest
import yaml

from evals.core import Case, Result, load_cases, score, summarize


def write_case(d, name, **over):
    body = {"id": name, "modality": "svg",
            "prompt": "Draw a red circle.",
            "assert": {"min_shapes": 1}}
    body.update(over)
    p = d / f"{name}.yaml"
    p.write_text(yaml.safe_dump(body))
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
    (tmp_path / "bad.yaml").write_text("id: bad\nmodality: svg\n")  # no prompt
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
        "params": {"width": 512, "height": 512, "seed": 42}}))
    c = load_cases(tmp_path)[0]
    assert c.params == {"width": 512, "height": 512, "seed": 42}
    assert c.assertions == {}


def test_a_case_with_neither_block_is_fine(tmp_path):
    (tmp_path / "i.yaml").write_text(yaml.safe_dump({
        "id": "i", "modality": "image", "prompt": "a fox"}))
    c = load_cases(tmp_path)[0]
    assert c.params == {} and c.assertions == {}


def test_a_misspelled_param_is_caught_at_load_not_after_the_run(tmp_path):
    """`widht: 512` used to be accepted silently and generate at the default
    size, and the case would pass."""
    (tmp_path / "i.yaml").write_text(yaml.safe_dump({
        "id": "i", "modality": "image", "prompt": "a fox",
        "params": {"widht": 512}}))
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
        "assert": {"must_contain": ["OPEN"]}}))
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
    """video/tts/stt have no checker yet. Reporting them as passed would put a
    100% pass rate next to nothing measured."""
    case = Case(id="v", modality="video", prompt="a fox")
    r = score(case, "/tmp/nope.mp4")
    assert not r.passed
    assert "no checker" in r.detail.lower()


def test_the_cases_that_ship_with_the_suite_actually_load():
    """Load-time validation is only worth having if the shipped cases pass it.
    Nothing else in the tests reads the real cases directory."""
    from pathlib import Path
    cases = load_cases(Path(__file__).resolve().parent.parent / "evals" / "cases")
    assert len(cases) >= 8
    assert {c.modality for c in cases} == {"svg", "web", "image", "tts"}
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
        "assert": {"max_wer": 0.2}, "params": {"speed": 1.0}}))
    c = load_cases(tmp_path)[0]
    assert c.assertions["max_wer"] == 0.2


def test_min_shapes_is_not_a_valid_assertion_for_tts(tmp_path):
    (tmp_path / "t.yaml").write_text(yaml.safe_dump({
        "id": "t", "modality": "tts", "prompt": "hello",
        "assert": {"min_shapes": 2}}))
    with pytest.raises(ValueError, match="min_shapes"):
        load_cases(tmp_path)


# ---- image text assertions -------------------------------------------------

def render_text(path, text):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.new("RGB", (512, 512), (250, 250, 250))
    d = ImageDraw.Draw(im)
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 96)
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
        "assert": {"text": "OPEN", "max_cer": 0.3}}))
    c = load_cases(tmp_path)[0]
    assert c.assertions["text"] == "OPEN"
