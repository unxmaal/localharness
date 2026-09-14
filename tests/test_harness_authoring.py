"""Issue #138: a prompt-authoring lane that knows which engine it writes for."""
import pytest

from harness import authoring


# --- the engine is resolved, never carried a second time ------------------

def test_the_lane_asks_the_same_resolver_the_generating_command_asks(
        monkeypatch):
    """THE SILENT FAILURE THIS PREVENTS: change the image default, and a lane
    with its own copy keeps writing fluent prompts aimed at the model you
    stopped using. No error, just worse pictures."""
    from harness import cli
    monkeypatch.setattr(cli, "DEFAULT_IMAGE_ENGINE", "h3")
    assert authoring.engine_for("image") == "h3"


def test_the_resolver_gives_the_whole_identity_not_the_bare_constant():
    """`mflux:flux2-klein-4b` resolves to `mflux/flux2-klein-4b-q8`: the
    quantisation the constant does not carry is part of what actually runs."""
    got = authoring.resolved_for("image").name
    assert got.startswith("mflux/")
    assert got != "mflux"


def test_a_lane_that_generates_nothing_is_refused_by_name():
    with pytest.raises(authoring.NoGuide) as caught:
        authoring.engine_for("svg")
    assert "image" in str(caught.value) and "video" in str(caught.value)


# --- the knowledge lives in the tree, versioned ---------------------------

def test_every_shipped_guide_carries_an_identity_and_a_source():
    """Same shape as the rubric the judge reads: an asset whose identity can go
    in a record beside what it produced."""
    import yaml
    for path in sorted(authoring.GUIDES.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert data.get("engine") == path.stem, path.name
        assert isinstance(data.get("version"), int), path.name
        assert data.get("lane"), path.name


def test_the_h3_grammar_is_in_the_tree_rather_than_in_a_knowledge_base():
    """It was expensive to reverse-engineer and verified against the vendor's
    own guide, and no running code could reach it."""
    guide = authoring.load("h3")
    text = authoring.instructions(guide)
    assert "subject_definitions" in text
    assert "multiples of 32" in text
    assert "mutually exclusive" in text
    # The two marker vocabularies are disjoint, and conflating them is the
    # error the first version of this made.
    assert "fully_preserved" in text


def test_a_guide_says_what_it_does_not_know_rather_than_inventing_it():
    """A prompt guide that invents its own confident tone is worse than none:
    a caller cannot tell which half was measured."""
    text = authoring.instructions(authoring.load("mflux"))
    assert "NOT ESTABLISHED" in text
    assert "Not established here" in text


def test_an_engine_with_no_guide_is_refused_rather_than_guessed_at():
    with pytest.raises(authoring.NoGuide) as caught:
        authoring.load("nothing-like-this")
    assert "guessing at it in prose" in str(caught.value)


def test_an_empty_field_is_omitted_rather_than_rendered_as_a_blank_heading():
    text = authoring.instructions(
        {"engine": "x", "identity": "x@1", "responds_to": "", "ignores": " "})
    assert "WHAT IT RESPONDS TO" not in text
    assert "WHAT IT IGNORES" not in text


def test_the_instructions_name_the_guide_so_a_result_can_cite_it():
    assert "h3@" in authoring.instructions(authoring.load("h3"))
