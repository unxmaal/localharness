"""One lane vocabulary, asserted across the four modules that each had one.

Issue #207. Four modules decided the set of lanes independently and no two
agreed, so `image` was the only lane that could travel the whole ladder:
`code` could be measured and screened and not searched for, `web` could be
measured and nothing else, `svg` could be searched for and not screened.

These tests are a GATE, not an inventory: every gap they name was closed in
the same change, so they start green and must be held there.
"""
import subprocess
from pathlib import Path

import pytest

from harness import discover, inspect as ins, lanes, rank, screen

CASES = Path(__file__).resolve().parents[1] / "evals" / "cases"


#: Lanes whose cases are GENERATED rather than committed, with the step that
#: creates each. An allowlist that outlives its reason is how a check rots, so
#: test_a_generated_lane_is_still_absent_from_the_tree asserts each one is
#: genuinely untracked.
GENERATED = {
    "stt": "uv run python -m evals.corpora  (LibriSpeech test-clean, "
           ".gitignore:10)",
}


def published() -> set[str]:
    """Case directories a fresh clone gets, asked of git rather than the disk.

    THE DESK IS NOT THE RUNNER. This read `CASES.iterdir()` and went red on
    three runners at once while `make check` was green here, because
    `evals/cases/stt/` holds 300 generated LibriSpeech clips and is gitignored.
    The directory exists on the machine that generated it and in no clone.
    Resolve against what a push would PUBLISH.
    """
    out = subprocess.run(["git", "ls-files", "evals/cases"],
                         cwd=CASES.parents[1], capture_output=True, text=True,
                         check=True).stdout
    return {line.split("/")[2] for line in out.splitlines()
            if line.startswith("evals/cases/") and line.count("/") > 2}


def test_every_measurable_modality_is_a_named_lane():
    """A directory of cases with no lane name cannot be discovered for."""
    on_disk = published()
    assert on_disk <= set(lanes.ALL), (
        f"evals/cases has {sorted(on_disk - set(lanes.ALL))}, "
        f"which lanes.ALL does not name")


def test_every_named_lane_has_cases_to_measure_it():
    """A lane nothing can measure is a lane that cannot finish the ladder."""
    missing = set(lanes.ALL) - published() - set(GENERATED)
    assert not missing, (
        f"lanes.ALL names {sorted(missing)} with no cases under evals/cases "
        f"and no entry in GENERATED saying how they are made")


def test_a_generated_lane_is_still_absent_from_the_tree():
    """The exception must keep earning its place. If someone commits the stt
    cases, this fails and the allowlist entry gets deleted rather than quietly
    excusing a lane that no longer needs excusing."""
    for lane, how in GENERATED.items():
        assert lane not in published(), (
            f"{lane} cases are tracked now; drop it from GENERATED ({how})")


def test_every_named_lane_can_be_searched_for():
    """lane_queries("code") and ("web") both raised ValueError before #207."""
    for lane in lanes.ALL:
        assert discover.lane_queries(lane, machine=_mac()), (
            f"nothing to ask a registry for lane {lane!r}")


#: The one lane that can be measured and searched for and NOT screened, with
#: the reason, so the exception can be challenged rather than discovered. An
#: allowlist that outlives its reason is how a check rots.
NO_RUNNER = {"video": "no per-model video runner: discover._HOW says so too"}


def test_every_named_lane_can_be_screened_except_the_argued_ones():
    for lane in lanes.ALL:
        spec = screen.candidate_for(lane, "org/model")
        if lane in NO_RUNNER:
            assert not spec, f"{lane} can be screened now; drop its exception"
        else:
            assert spec, f"lane {lane!r} has no candidate spelling"


def test_the_wanted_order_is_the_one_that_was_asked_for():
    """image, code, web, svg, video. tts and stt are deliberately absent."""
    assert rank.LANE_PRIORITY == ("image", "code", "web", "svg", "video")
    assert not {"tts", "stt"} & set(rank.LANE_PRIORITY)


def test_priority_falls_monotonically_down_the_wanted_list():
    scores = [rank.priority_of(lane) for lane in lanes.WANTED]
    assert scores == sorted(scores, reverse=True)
    assert rank.priority_of("tts") == 0.0


def test_text_is_code_everywhere_it_is_read():
    """Two names for one lane put half of it in the queue and half nowhere."""
    assert lanes.canonical("text") == "code"
    assert rank.lane_of({"lane": "text"}) == "code"
    assert screen.candidate_for("text", "org/m") == "org/m"
    assert discover.lane_queries("text") == discover.lane_queries("code")


def test_a_sources_coverage_claim_is_not_a_candidates_lane():
    assert lanes.canonical("all") == ""
    assert rank.lane_of({"lane": "all"}) == ""


def test_the_text_served_lanes_share_one_spelling():
    """web, svg, code and extract are one prompt to mlx_lm.server."""
    for lane in lanes.TEXT_SERVED:
        assert screen.candidate_for(lane, "org/m") == "org/m"


# --- the prose classifier, and its two controls ---------------------------

#: Real store rows. The lane is what a person reading the line would say.
NAMES_A_LANE = [
    ("awni/voxmlx", "Realtime Transcription with Voxtral in MLX", "stt"),
    ("Blaizzy/mlx-video", "inference and finetuning of video models", "video"),
    ("x/y", "Speeds up SDXL workflows on Mac using native MLX", "image"),
    ("x/y", "Ethical language model from Danish Foundation Models.", "code"),
    ("x/y", "Generates clean SVG icons from a description", "svg"),
    ("x/y", "Builds a responsive web page from a brief", "web"),
]

#: Also real store rows, and the more important half: a classifier that routed
#: these would be worse than the empty lane it replaced. 287 of 394 laneless
#: sightings look like this.
NAMES_NOTHING = [
    ("x/y", "Packs 82GB power for Apple Silicon."),
    ("x/y", "Runs large AI models efficiently on Mac computers."),
    ("stockeh/mlx-optimizers", "A collection of optimizers for MLX"),
    ("ml-explore/mlx-data", "Efficient framework-agnostic data loading"),
    ("apple/container", "A tool for creating and running Linux containers"),
    ("x/y", "MoE quantized version for smoother GPU runs."),
    ("x/y", ""),
]


@pytest.mark.parametrize("name,why,want", NAMES_A_LANE)
def test_prose_names_the_lane(name, why, want):
    assert lanes.from_prose(f"{name} {why}") == want


@pytest.mark.parametrize("name,why", NAMES_NOTHING)
def test_prose_that_names_no_lane_stays_empty(name, why):
    assert lanes.from_prose(f"{name} {why}") == ""


def test_prose_naming_two_lanes_refuses_to_pick_one():
    """A wrong lane costs a download and a screen that answers nothing."""
    assert lanes.from_prose(
        "Juggles a million tokens of text, images, and video.") == ""


def test_the_registry_outranks_the_prose():
    """pipeline_tag is the publisher's answer; prose is a guess at it."""
    assert ins.lane_for({"pipeline_tag": "text-to-speech"},
                        "writes html and css") == "tts"
    assert ins.lane_for({}, "writes html and css") == "web"


def _mac():
    """An Apple Silicon machine, so the queries do not vary with the runner."""
    from harness import machine
    from harness.memory import Accelerator
    return machine.Machine(frozenset({"mlx", "cpu"}),
                           Accelerator("unified", 32.0, 25.0, "Mac14,12"))


# --- what the queue must not spend its first page on -----------------------

def _row(name, lane="video", description="", times=2):
    return {"name": name, "lane": lane, "description": description,
            "times": times, "registry": "huggingface"}


def test_an_adapter_never_reaches_the_queue():
    """screen.is_attachment() refused these only AFTER they took the top of
    the ranking: the first six rows were video LoRAs. Issue #207."""
    rows = [_row("LiconStudio/Ltx2.3-VBVR-lora-I2V"),
            _row("Robert1212star/TaoMate-H3-3Step-ComfyUI", lane="image"),
            _row("org/real-model", description="a text-to-video model")]
    got = rank.rank(rows, serving=(), measured_lanes=())
    assert [r["name"] for r in got] == ["org/real-model"]


def test_an_adapter_is_caught_by_its_card_as_well_as_its_name():
    rows = [_row("org/innocuous-name", description="a style LoRA for FLUX")]
    assert rank.rank(rows, serving=(), measured_lanes=()) == []


def test_a_real_model_whose_card_says_lora_ready_is_kept():
    """The one enumerated exception, so the refusal cannot creep."""
    rows = [_row("org/base", description="lora-ready base checkpoint")]
    assert [r["name"] for r in rank.rank(rows, serving=(), measured_lanes=())
            ] == ["org/base"]


def test_value_and_rank_agree_about_what_a_lane_is():
    """value() kept its own copy of the all/text rule and disagreed with
    lane_of() about `text`, so the score said no-lane while the filter said
    code. One question, two answers. Issue #207."""
    scored, why = rank.value(_row("org/x", lane="text"), measured_lanes=set())
    assert "no lane can measure it" not in why
    assert "code is a wanted lane" in why


# --- one text model serves four lanes --------------------------------------

def test_a_text_candidate_can_be_tested_in_every_text_served_lane():
    """`lh svg` and `lh web` have always resolved to the code lane's model.
    Filing a candidate under one of them hid it from the other three: web and
    svg read as having ZERO candidates while holding 103. Issue #207."""
    assert set(lanes.testable_in("code")) == set(lanes.TEXT_SERVED)
    for target in lanes.TEXT_SERVED:
        assert lanes.serves("code", target)
        assert lanes.serves("text", target), "the alias travels too"


def test_an_image_candidate_is_only_an_image_candidate():
    """The negative control. mflux does not write markup."""
    assert lanes.testable_in("image") == ("image",)
    for target in lanes.TEXT_SERVED + ("video", "stt", "tts"):
        assert not lanes.serves("image", target)


def test_a_laneless_candidate_serves_nothing():
    assert lanes.testable_in("") == ()
    assert lanes.testable_in("all") == ()
    assert not lanes.serves("", "code")


def test_a_speech_candidate_does_not_leak_into_the_wanted_lanes():
    for lane in ("stt", "tts"):
        assert lanes.testable_in(lane) == (lane,)
        assert not any(lanes.serves(lane, w) for w in lanes.WANTED)
