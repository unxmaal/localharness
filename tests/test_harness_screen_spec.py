"""A spec must survive being spelled twice. Issue #214.

The challenger arrives from the store as a bare repo id; the incumbent arrives
from winners.typed() already spelled `mflux:flux2-klein-4b`. candidate_for
wrapped both, so the lane's own control became `mflux:mflux:flux2-klein-4b`,
which resolve() rejects. The control contributed no cases, the challenger ran
alone, and the loop printed the pairing it intended above a one-sided receipt.
"""
import pytest

from harness import screen


@pytest.mark.parametrize("lane,model,want", [
    ("image", "flux2-klein-4b", "mflux:flux2-klein-4b"),
    ("image", "mflux:flux2-klein-4b", "mflux:flux2-klein-4b"),
    ("image", "filipstrand/Z-Image-Turbo-mflux-4bit",
     "mflux:filipstrand/Z-Image-Turbo-mflux-4bit"),
    ("tts", "mlx-community/Kokoro-82M-bf16", "tts:mlx-community/Kokoro-82M-bf16"),
    ("tts", "tts:mlx-community/Kokoro-82M-bf16", "tts:mlx-community/Kokoro-82M-bf16"),
    ("stt", "stt:org/m", "stt:org/m"),
])
def test_spelling_a_spec_twice_is_spelling_it_once(lane, model, want):
    assert screen.candidate_for(lane, model) == want


def test_it_is_idempotent_for_every_lane_that_has_a_spelling():
    for lane in screen.LANE_CANDIDATES:
        once = screen.candidate_for(lane, "org/model")
        assert screen.candidate_for(lane, once) == once, lane


def test_a_text_lane_has_no_prefix_to_double():
    """The code lane passes the model through, so there is nothing to strip
    and a repo id beginning with a lane name must not be mangled."""
    assert screen.candidate_for("code", "mflux/some-text-model") == \
        "mflux/some-text-model"


def test_an_attachment_is_still_refused_whichever_spelling_it_arrives_in():
    assert screen.candidate_for("image", "mflux:org/x", "a style LoRA") == ""


# --- the receipt key's shape differs by family, and so did the matcher ------

ENGINE_SUMMARY = {
    "mflux/flux2-klein-4b-q8": {"passed": 9, "total": 9},
    "mflux/filipstrand/Z-Image-Turbo-mflux-4bit-q8": {"passed": 9, "total": 9},
}
SPEECH_SUMMARY = {
    "Kokoro-82M-bf16/bm_george": {"passed": 1, "total": 1},
    "marvis-tts-250m-v0.2-MLX-8bit/none": {"passed": 0, "total": 1},
}


def test_an_engine_receipt_names_the_engine_first():
    """`mflux:flux2-klein-4b` -> `mflux/flux2-klein-4b-q8`. The old matcher
    compared `mflux` against `mflux:flux2-klein-4b` and reported the image
    lane's own control absent from a receipt it is named in. Issue #219."""
    from harness import cli

    got = cli._summary_row(ENGINE_SUMMARY, "mflux:flux2-klein-4b", "image")
    assert got and got["candidate"] == "mflux/flux2-klein-4b-q8"


def test_a_discovered_engine_candidate_matches_through_its_spec():
    from harness import cli

    got = cli._summary_row(
        ENGINE_SUMMARY, "mflux:filipstrand/Z-Image-Turbo-mflux-4bit", "image")
    assert got
    assert got["candidate"] == "mflux/filipstrand/Z-Image-Turbo-mflux-4bit-q8"


def test_a_speech_receipt_names_the_model_first_and_still_matches():
    """The shape the old matcher WAS written for must keep working."""
    from harness import cli

    got = cli._summary_row(
        SPEECH_SUMMARY, "tts:mlx-community/Kokoro-82M-bf16", "tts")
    assert got and got["candidate"] == "Kokoro-82M-bf16/bm_george"


def test_the_two_candidates_do_not_match_each_other():
    """The negative control. A matcher loose enough to pair the incumbent with
    the challenger would adopt a model against itself."""
    from harness import cli

    inc = cli._summary_row(ENGINE_SUMMARY, "mflux:flux2-klein-4b", "image")
    ch = cli._summary_row(
        ENGINE_SUMMARY, "mflux:filipstrand/Z-Image-Turbo-mflux-4bit", "image")
    assert inc["candidate"] != ch["candidate"]


def test_a_candidate_genuinely_absent_is_still_absent():
    from harness import cli

    assert cli._summary_row(ENGINE_SUMMARY, "mflux:qwen-image", "image") is None


# --- a screen that never started says nothing about the candidate (#201) ---

@pytest.mark.parametrize("detail", [
    "~/.venv/bin/python: Error while finding module specification for "
    "'evals.run' (ModuleNotFoundError: No module named 'evals')",
    "bash: uv: command not found",
    "No such file or directory: 'mflux-generate'",
])
def test_a_screen_that_could_not_start_is_requeued_not_broken(detail):
    """SEVENTH OCCURRENCE of the class. A loop run screened four freshly
    fetched candidates BROKEN because the subprocess resolved a PARENT
    directory's virtualenv and could not import our own eval package.

    `broken` is terminal. Four real models were declined forever for
    something they never did.
    """
    from harness import screen
    got, why = screen.outcome(1, None, detail=detail)
    assert got == "queued", why
    assert "says nothing about the candidate" in why


def test_a_candidate_that_genuinely_failed_is_still_broken():
    """THE NEGATIVE CONTROL. A guard that requeues every failure means no
    candidate is ever answered and the queue never shrinks."""
    from harness import screen
    got, _ = screen.outcome(
        1, None, detail="generated 0 of 3 cases; the model returned empty output")
    assert got == "broken"
