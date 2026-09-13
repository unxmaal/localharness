"""Gauntlet, "external state with no assertion on it".

Every gateway alias points at somebody else's published artifact. Each was
obtainable when written, and nothing here notices when one stops being. #147 is
the proof: four GGUFs in config.cuda.yaml cannot be fetched as written, found
by hand on a bare Linux box, by trying.

Red-proofed the way harness/privacy.py is: the negative cases matter more,
because a check that flags every alias gets switched off within a week.
"""
from pathlib import Path

import pytest
import yaml

from harness import obtainable as ob

REPO = Path(__file__).resolve().parent.parent


def config(tmp_path, name, entries):
    (tmp_path / "gateway").mkdir(exist_ok=True)
    p = tmp_path / "gateway" / name
    p.write_text(yaml.safe_dump({"model_list": entries}), encoding="utf-8")
    return f"gateway/{name}"


# ---- reading what is named ------------------------------------------------

def test_the_provider_prefix_is_not_part_of_the_identifier():
    """`openai/` means "speak the OpenAI protocol", not an OpenAI model.
    Leaving it on turns every repo id into a three-part path that resolves to
    nothing, which would make this check fire on all sixteen and be deleted."""
    assert ob.strip_provider("openai/mlx-community/Qwen3-8B-4bit") == \
        "mlx-community/Qwen3-8B-4bit"
    assert ob.strip_provider("mlx-community/Qwen3-8B-4bit") == \
        "mlx-community/Qwen3-8B-4bit"


def test_a_repo_id_and_a_stem_are_told_apart():
    assert ob.Named("c", "a", "mlx-community/Qwen3-8B-4bit").is_repo_id
    assert not ob.Named("c", "a", "qwen3-8b-q4_k_m").is_repo_id


def test_every_alias_in_the_real_configs_is_read():
    got = ob.named(REPO)
    assert len(got) > 10, "the configs stopped being parsed, which reads as health"
    assert {n.config for n in got} == set(ob.CONFIGS)


# ---- what must be flagged -------------------------------------------------

def test_a_bare_stem_cannot_be_checked_by_anyone(tmp_path):
    rel = config(tmp_path, "c.yaml", [
        {"model_name": "q3-8b",
         "litellm_params": {"model": "openai/qwen3-8b-q4_k_m"}}])
    assert ob.unresolvable(tmp_path, (rel,))


def test_the_real_cuda_config_is_the_known_instance():
    """#147, still open. The finding is not that these files are missing: it is
    that the config records WHAT to ask for and never WHERE it comes from, so
    verifying an alias means guessing the repo, and guessing the repo is
    exactly what produced four wrong ones."""
    bad = {n.alias for n, _ in ob.unresolvable(REPO)}
    assert "q3-4b" in bad and "q3-1.7b" in bad


# ---- what must NOT be flagged ---------------------------------------------

def test_a_repo_id_is_left_alone(tmp_path):
    rel = config(tmp_path, "c.yaml", [
        {"model_name": "q3-8b",
         "litellm_params": {"model": "openai/mlx-community/Qwen3-8B-4bit"}}])
    assert ob.unresolvable(tmp_path, (rel,)) == []


def test_a_stem_that_records_its_source_is_left_alone(tmp_path):
    """The fix #147 wants, and the reason this is an inventory rather than a
    gate: it says what would make each row go away."""
    rel = config(tmp_path, "c.yaml", [
        {"model_name": "q3-8b",
         "litellm_params": {"model": "openai/qwen3-8b-q4_k_m",
                            "source_repo": "unsloth/Qwen3-8B-GGUF"}}])
    assert ob.unresolvable(tmp_path, (rel,)) == []


def test_the_mlx_config_is_clean():
    """Its aliases carry full repo ids, so they are checkable by construction.
    If this ever fires, someone has replaced a repo id with a bare name."""
    assert [n.alias for n, _ in ob.unresolvable(REPO, ("gateway/config.yaml",))] == []


# ---- the network half, red-proofed without the network --------------------

def test_an_unanswered_repo_id_is_reported():
    bad = ob.unreachable(REPO, ("gateway/config.yaml",),
                         facts=lambda ref, cache=None: {"size": -1})
    assert bad, "the registry answering nothing must be a finding"


def test_a_repo_that_answers_is_not():
    assert ob.unreachable(REPO, ("gateway/config.yaml",),
                          facts=lambda ref, cache=None: {"size": 4_000_000}) == []


def test_the_network_check_never_asks_about_a_stem():
    """A stem is not a repo id, and asking the registry for one produces a 404
    that reads as "the model is gone" rather than "the config is incomplete"."""
    asked = []

    def facts(ref, cache=None):
        asked.append(ref)
        return {"size": 1}

    ob.unreachable(REPO, ("gateway/config.cuda.yaml",), facts=facts)
    assert asked == []


@pytest.mark.network
def test_every_repo_id_the_gateway_names_still_answers():
    """The real thing, kept out of `make check`: a lint that fails when
    HuggingFace has a bad afternoon teaches people to skip lint."""
    bad = ob.unreachable(REPO)
    assert not bad, "\n".join(f"{n}: {why}" for n, why in bad)
