"""Issue #54: the rubric judge, and the control that decides if it means anything."""
import pytest

from harness import judge
from harness.judge import JudgeError, describe, load, parse_score


def fake(replies):
    """A completion stub returning canned text per call."""
    seq = list(replies)

    def complete(prompt, **kw):
        return seq.pop(0) if seq else seq_default
    seq_default = "5\nno opinion"
    return complete


def test_the_shipped_rubric_loads_and_is_versioned():
    r = load()
    assert r.version >= 1
    assert r.identity == f"{r.name}@{r.version}"
    assert (r.low, r.high) == (1, 10)


def test_the_rubric_prompt_carries_both_directions():
    p = load().prompt
    assert "SCORES HIGH" in p and "SCORES LOW" in p
    assert "composition" in p.lower()
    assert "cuda" in p.lower()


def test_an_unknown_rubric_names_the_ones_that_exist():
    with pytest.raises(JudgeError) as exc:
        load("nonexistent")
    assert "novelty" in str(exc.value)


# ---- score parsing ---------------------------------------------------------

@pytest.mark.parametrize("text,want", [
    ("9\nA composition of installed tools.", 9),
    ("10", 10),
    ("1 - marketing with no measurement", 1),
    ("Score: 7. Runs natively on MLX.", 7),
])
def test_a_score_is_read_off_the_first_line(text, want):
    assert parse_score(text)[0] == want


def test_the_reasoning_survives():
    s, why = parse_score("8\nComposes two things already installed.")
    assert s == 8
    assert "Composes" in why


def test_a_reply_with_no_score_is_an_error_not_a_guess():
    """Silently defaulting to a middling score would put junk in the ranking."""
    with pytest.raises(JudgeError):
        parse_score("I am not able to score this item.")


def test_a_score_outside_the_scale_is_not_accepted():
    with pytest.raises(JudgeError):
        parse_score("42\nvery good", low=1, high=10)


# ---- what the model is shown ----------------------------------------------

def test_provenance_is_shown_alongside_the_prose():
    """A judge given only a description is partly scoring copywriting, and the
    recap entries this reads are one-line marketing blurbs."""
    d = describe("SDMLX", why="SDXL through MLX", source="recap",
                 times_seen=3, relevance=4)
    assert "TIMES SEEN: 3" in d
    assert "APPLE SILICON RELEVANCE: +4" in d
    assert "SEEN IN: recap" in d


def test_a_bare_item_still_describes_cleanly():
    assert describe("x") == "NAME: x"


# ---- the control -----------------------------------------------------------

def test_a_separating_judge_reports_that_it_separates():
    replies = {"trace": "9\ncomposition", "repair": "8\nfree",
               "upscale-seedvr2": "3\nanother upscaler",
               "local-small": "2\ntiny", "q3-14b": "4\ncrowded family"}

    def complete(prompt, **kw):
        for name, r in replies.items():
            if f"NAME: {name}" in prompt:
                return r
        raise AssertionError(prompt)

    got = judge.control(complete=complete)
    assert got["separates"] is True
    assert got["won_min"] == 8 and got["lost_max"] == 4
    assert got["gap"] == 4


def test_an_overlapping_judge_is_reported_as_not_separating():
    """The failure that matters: individually plausible scores that do not rank.
    This is the shape that got the speaker-similarity numbers retracted."""
    replies = {"trace": "6\n", "repair": "5\n", "upscale-seedvr2": "7\n",
               "local-small": "2\n", "q3-14b": "4\n"}

    def complete(prompt, **kw):
        for name, r in replies.items():
            if f"NAME: {name}" in prompt:
                return r
        raise AssertionError(prompt)

    got = judge.control(complete=complete)
    assert got["separates"] is False
    assert got["gap"] < 0


def test_the_control_carries_rubric_and_model_identity():
    """Two runs under different rubrics or judges are different exams."""
    got = judge.control(complete=lambda p, **kw: "5\n")
    assert "@" in got["rubric"]
    assert got["model"]


def test_the_control_set_has_both_classes():
    outcomes = {o for _, _, o in judge.CONTROL}
    assert outcomes == {"won", "lost"}
    assert len(judge.CONTROL) >= 4


def test_the_control_never_tells_the_model_the_answer():
    """It has to reach the right order from the description alone."""
    for name, why, outcome in judge.CONTROL:
        assert outcome not in why.lower()
        assert "won" not in why.lower() and "beat" not in why.lower()


# ---- issue #69: the judge was working from less than the inspect tier ------

def test_the_inspect_result_reaches_the_judge():
    """apple/coreai-models scored 3/10 off a generic description while the
    inspect tier had already established it was MLX-native with weights that
    fit. The tier holding more evidence lost to the tier holding less."""
    d = describe("apple/coreai-models", why="Model export recipes",
                 inspected="fits: MLX-native", platform="MLX-native",
                 weights="1.4 to 15.3 GiB")
    assert "READ FROM ITS SOURCE: fits: MLX-native" in d
    assert "RUNTIME: MLX-native" in d
    assert "WEIGHTS IT NAMES: 1.4 to 15.3 GiB" in d


def test_a_candidate_never_inspected_still_describes_cleanly():
    assert describe("x", why="y") == "NAME: x\nDESCRIPTION: y"


def test_the_rubric_scores_an_out_of_domain_tool_low():
    """CellSeg3D, a napari plugin for 3D cell segmentation, scored 7/10.
    Genuinely good software, and nothing here can measure it."""
    p = load().prompt.lower()
    assert "domain" in p
    assert "segmentation" in p or "genomics" in p or "robotics" in p


def test_changing_what_the_judge_sees_bumps_the_rubric_version():
    """Two runs under different rubrics are different exams, so a scored run
    from before this change must not be ranked against one from after."""
    assert load().version >= 2


def test_every_control_item_carries_the_same_inspect_verdict():
    """A control has to be shaped like the data the judge actually meets. When
    the inspect result was first shown, six unrelated candidates all scored
    10/10 on "MLX-native, weights fit", and the control could not see it
    because control items carried no inspect fields at all. A fact shared by
    every item cannot be what separates them."""
    prompts = []
    judge.control(complete=lambda p, **kw: prompts.append(p) or "5\n")
    assert len(prompts) == len(judge.CONTROL)
    assert all(judge.CONTROL_INSPECTED in p for p in prompts)


def test_the_rubric_says_running_here_is_not_merit():
    p = load().prompt.lower()
    assert "price of entry" in p or "not merit" in p
    assert "5 at most" in p


# ---- issue #79: reading names out of freeform prose ------------------------

def test_a_named_thing_and_its_claim_are_both_kept():
    """The value of a comment is the CLAIM, not just the name: "king for image
    editing" is the part a registry cannot tell you."""
    reply = "Qwen Image Edit | king for image editing\nBreeze TTS | best for cloning"
    got = judge.mentions("...", complete=lambda p, **kw: reply)
    assert got == [("Qwen Image Edit", "king for image editing"),
                   ("Breeze TTS", "best for cloning")]


def test_a_comment_naming_nothing_yields_nothing():
    assert judge.mentions("thanks!", complete=lambda p, **kw: "NONE") == []


def test_empty_text_never_reaches_the_model():
    def boom(p, **kw):
        raise AssertionError("must not be called")
    assert judge.mentions("   ", complete=boom) == []


def test_a_model_that_errors_does_not_empty_the_sweep():
    """One unparseable comment is not a failure of the sweep."""
    def boom(p, **kw):
        raise RuntimeError("model is down")
    assert judge.mentions("something", complete=boom) == []


def test_a_sentence_is_not_a_name():
    reply = ("A | ok\n" + "x" * 80 + " | too long\n| no name\nNAME | header")
    got = judge.mentions("...", complete=lambda p, **kw: reply)
    assert [n for n, _ in got] == []


def test_a_narrating_model_is_cut_off():
    reply = "\n".join(f"thing{i} | claim" for i in range(40))
    assert len(judge.mentions("...", complete=lambda p, **kw: reply)) == \
        judge.MAX_MENTIONS


def test_list_markers_are_stripped():
    got = judge.mentions("...", complete=lambda p, **kw: "- ComfyUI | a UI")
    assert got == [("ComfyUI", "a UI")]


# ---- issue #85: one control run is one draw --------------------------------

def test_the_control_reports_a_spread_not_a_single_gap():
    """Same rubric, same model, gap +4 then +2 on identical inputs. The judge
    samples and nothing pins a seed, so near the boundary the verdict is a coin
    flip -- and it is the sentence authorising every score."""
    seq = iter(["10\n", "9\n", "3\n", "3\n", "3\n",     # run 1: gap +6
                "10\n", "5\n", "4\n", "3\n", "3\n"])    # run 2: gap +1
    got = judge.control_repeated(runs=2, complete=lambda p, **kw: next(seq))
    assert got["gaps"] == [6, 1] and got["spread"] == 5


def test_separating_once_out_of_three_is_not_separating():
    seq = iter(["10\n", "9\n", "3\n", "3\n", "3\n",     # separates
                "3\n", "3\n", "10\n", "3\n", "3\n"])    # does not
    got = judge.control_repeated(runs=2, complete=lambda p, **kw: next(seq))
    assert got["separated_in"] == 1
    assert got["separates"] is False


def test_the_composition_criterion_excludes_a_new_dependency():
    """A second judge scored upscale-seedvr2 10/10 reading "composition of
    existing tools" off the rubric's own top criterion. It is a new dependency
    wearing the word."""
    p = load().prompt.lower()
    assert "already installed here" in p
    assert "new dependency" in p


def test_changing_the_criterion_bumped_the_rubric():
    assert load().version >= 4
