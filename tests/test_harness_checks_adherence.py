"""Does the picture match the words?

The axis the suite was missing. Every other image check is about competence --
it decodes, it is the right size, it is not one colour, the text is legible --
and all of them scored FLUX.2 klein and Z-Image Turbo identically. None of them
looks at whether the image is of what was asked for.

Two backends, because they disagree in the literature and there was no reason
to guess which suits this machine:

  pickscore  a CLIP-H fine-tune on 500k human preference pairs
  hpsv2      a CLIP-H fine-tune on the HPD v2 preference dataset

Both are torch, both run on MPS, and both live behind the optional `metrics`
dependency group. Absent, the check warns and passes -- a missing optional tool
that failed every candidate at once would look exactly like a model regression.
"""
import shutil

import pytest

from harness.checks import adherence

torch = pytest.importorskip("torch", reason="needs the `metrics` dependency group")

pytestmark = pytest.mark.slow


def picture_of(tmp_path, kind):
    """Two obviously different images, so a working scorer must separate them
    and a broken one cannot fake it."""
    from PIL import Image, ImageDraw
    p = tmp_path / f"{kind}.png"
    im = Image.new("RGB", (256, 256), (250, 250, 250))
    d = ImageDraw.Draw(im)
    if kind == "circle":
        d.ellipse((40, 40, 216, 216), fill=(200, 30, 30))
    else:
        d.polygon([(128, 30), (226, 226), (30, 226)], fill=(30, 60, 200))
    im.save(p)
    return p


@pytest.mark.parametrize("backend", ["pickscore", "hpsv2"])
def test_a_matching_prompt_scores_higher_than_a_mismatched_one(tmp_path, backend):
    """The only property that makes the number worth reporting."""
    circle = picture_of(tmp_path, "circle")
    right = adherence.score(circle, "a large red circle", backend=backend)
    wrong = adherence.score(circle, "a blue triangle", backend=backend)
    assert right > wrong, f"{backend}: {right} vs {wrong}"


@pytest.mark.parametrize("backend", ["pickscore", "hpsv2"])
def test_the_score_is_a_finite_number(tmp_path, backend):
    s = adherence.score(picture_of(tmp_path, "circle"), "a red circle",
                        backend=backend)
    assert isinstance(s, float)
    assert s == s and abs(s) < 1e6  # not NaN, not absurd


def test_the_model_is_loaded_once_not_per_image(tmp_path):
    """A 4GB load per case would dominate a 19-second generation."""
    adherence.reset_cache()
    img = picture_of(tmp_path, "circle")
    adherence.score(img, "a red circle", backend="pickscore")
    loaded = adherence.cache_info()
    adherence.score(img, "a red circle", backend="pickscore")
    assert adherence.cache_info() == loaded


def test_an_unknown_backend_names_the_ones_that_exist(tmp_path):
    with pytest.raises(ValueError) as e:
        adherence.score(picture_of(tmp_path, "circle"), "x", backend="vibes")
    assert "pickscore" in str(e.value) and "hpsv2" in str(e.value)


def test_a_missing_image_raises_rather_than_scoring_zero(tmp_path):
    with pytest.raises(FileNotFoundError):
        adherence.score(tmp_path / "nope.png", "x")


# ---- the check -------------------------------------------------------------

def test_check_publishes_adherence_as_a_metric(tmp_path):
    r = adherence.check(picture_of(tmp_path, "circle"), "a large red circle")
    assert r.ok
    assert "adherence" in r.metrics


def test_check_fails_below_a_declared_minimum(tmp_path):
    r = adherence.check(picture_of(tmp_path, "circle"), "a blue triangle",
                        min_adherence=99.0)
    assert not r.ok
    assert "99" in r.reason


def test_check_with_no_minimum_measures_without_judging(tmp_path):
    """Ranking needs the number even where there is no threshold to fail."""
    r = adherence.check(picture_of(tmp_path, "circle"), "a blue triangle")
    assert r.ok
    assert r.metrics["adherence"] is not None


def test_without_torch_the_check_warns_and_passes(tmp_path, monkeypatch):
    """A missing optional dependency that failed every candidate at once would
    look exactly like a model regression."""
    monkeypatch.setattr(adherence, "_backend", _raise_missing)
    r = adherence.check(picture_of(tmp_path, "circle"), "a red circle")
    assert r.ok
    assert r.warnings and "metrics" in r.warnings[0]
    assert r.metrics == {}


def _raise_missing(*_a, **_k):
    raise adherence.MetricsUnavailable(
        "torch is not installed; uv sync --group metrics")


def test_the_embedding_helper_accepts_both_transformers_api_shapes():
    """transformers 4 returned a bare tensor; 5 returns a ModelOutput whose
    pooler_output is the projected embedding. A version bump should not become
    an AttributeError deep inside a scoring run."""
    import torch
    from transformers.modeling_outputs import BaseModelOutputWithPooling
    from harness.checks.adherence import _embedding

    bare = torch.zeros(1, 8)
    assert _embedding(bare) is bare
    wrapped = BaseModelOutputWithPooling(last_hidden_state=torch.zeros(1, 2, 8),
                                         pooler_output=bare)
    assert _embedding(wrapped) is bare
