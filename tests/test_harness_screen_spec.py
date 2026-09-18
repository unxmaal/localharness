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
    for lane in screen.LANE_CANDIDATE:
        once = screen.candidate_for(lane, "org/model")
        assert screen.candidate_for(lane, once) == once, lane


def test_a_text_lane_has_no_prefix_to_double():
    """The code lane passes the model through, so there is nothing to strip
    and a repo id beginning with a lane name must not be mangled."""
    assert screen.candidate_for("code", "mflux/some-text-model") == \
        "mflux/some-text-model"


def test_an_attachment_is_still_refused_whichever_spelling_it_arrives_in():
    assert screen.candidate_for("image", "mflux:org/x", "a style LoRA") == ""
