"""A discovered candidate is always a repo id, and the image lane refused one.

`engines.mflux_binary` picked the entry point by `model.startswith(prefix)`
against a fixed family list, so the owner prefix of
`filipstrand/Z-Image-Turbo-mflux-4bit` hid `z-image-turbo`. engine_for
swallowed the ValueError, modality_of returned None, and cases_for fell
through to the text modalities, of which image is not one. The image lane
could screen its own typed defaults and nothing the loop found. Issue #213.
"""
import pytest

from harness import engines, screen


@pytest.mark.parametrize("model,binary", [
    ("filipstrand/Z-Image-Turbo-mflux-4bit", "mflux-generate-z-image-turbo"),
    ("Tongyi-MAI/Z-Image", "mflux-generate-z-image"),
    ("mlx-community/flux2-klein-9b-4bit", "mflux-generate-flux2"),
])
def test_a_repo_id_resolves_to_its_family(model, binary):
    assert engines.mflux_binary(model) == binary


@pytest.mark.parametrize("model,binary", [
    ("flux2-klein-4b", "mflux-generate-flux2"),
    ("z-image-turbo", "mflux-generate-z-image-turbo"),
    ("dev", "mflux-generate"),
    ("schnell", "mflux-generate"),
])
def test_a_bare_family_name_still_resolves(model, binary):
    """The typed defaults, which are how the lane runs today."""
    assert engines.mflux_binary(model) == binary


def test_a_repo_id_naming_no_family_is_still_refused():
    """The negative control. Guessing an entry point would hand mflux a model
    the binary rejects several seconds into loading, and the error would read
    as the candidate's fault."""
    with pytest.raises(ValueError):
        engines.mflux_binary("someone/A-Totally-New-Architecture")


def test_the_longest_family_wins():
    """z-image-turbo and z-image are both prefixes of the same name."""
    assert engines.family_of("filipstrand/Z-Image-Turbo-mflux-4bit").startswith(
        "z-image-turbo")
    assert engines.mflux_binary("x/Z-Image-Turbo-4bit").endswith("z-image-turbo")


def test_the_candidate_now_reaches_the_image_cases():
    """The whole point: a spec the screen builds must select image cases."""
    from pathlib import Path

    from evals import core
    from evals import run as R

    cases = [c for c in core.load_cases(Path("evals/cases")) if c.modality == "image"]
    spec = screen.candidate_for("image", "filipstrand/Z-Image-Turbo-mflux-4bit")
    assert R.modality_of(spec) == "image"
    assert [c.id for c in R.cases_for(spec, cases)] == [c.id for c in cases]


# --- a third harness-side refusal that was recorded as terminal -------------

def test_the_runner_having_no_spec_is_our_gap_not_the_candidates():
    for why in ("mflux:org/x: no cases of a modality it can run, skipped",
                "nothing ran: no candidate matched any case"):
        assert screen.refused_by_harness(why)


def test_a_real_failure_is_still_the_candidates():
    """The negative control, and the one that matters: without it the ladder
    settles nothing and re-offers a broken model every sweep."""
    for why in ("mflux exited 1: out of memory",
                "the image was a blank canvas",
                "0 of 3 case(s) passed"):
        assert not screen.refused_by_harness(why)
