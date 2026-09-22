"""Two option-variants of one engine are two candidates. Issue #277.

FOUND BY RUNNING A COMPARISON THAT COULD NOT WORK. `--candidates
acestep:...,steps=8,acestep:...,steps=16` produced THREE artifacts where eight
were expected: both specs resolved to `acestep/acestep-v15-turbo`, which is
the receipt key AND the artifact filename, so the second candidate overwrote
the first and the receipt recorded one candidate where two had run.

RULE #262's class -- an axis that changes the exam belongs in the receipt --
and gauntlet #8, the label is not the run.
"""
import pytest

from harness import engines


@pytest.fixture(autouse=True)
def root(monkeypatch, tmp_path):
    monkeypatch.setenv("ACESTEP_ROOT", str(tmp_path))
    monkeypatch.setattr(engines, "acestep_python", lambda r: "python")


def name(spec):
    return engines.resolve(spec).name


def test_two_step_counts_are_two_candidates():
    assert (name("acestep:acestep-v15-turbo,steps=8")
            != name("acestep:acestep-v15-turbo,steps=16"))


def test_mflux_had_it_too():
    """`quantize` was hand-rolled into the name and `steps` was not, so the
    image lane could have been confounded the same way."""
    assert (name("mflux:flux2-klein-4b,steps=2")
            != name("mflux:flux2-klein-4b,steps=20"))


def test_a_bare_spec_keeps_the_name_it_always_had():
    """THE COMPATIBILITY HALF. Every receipt in the store was written under
    the old names, and winners/adopt match candidates BY NAME -- a suffix on
    the plain form would orphan the lot."""
    assert name("mflux:flux2-klein-4b") == "mflux/flux2-klein-4b-q8"
    assert name("acestep:acestep-v15-turbo") == "acestep/acestep-v15-turbo"


def test_where_something_lives_is_not_part_of_what_it_produces():
    """Two machines with the checkout in different places run the same exam.
    Putting `root` in the identity would stop their receipts matching."""
    assert (name("acestep:acestep-v15-turbo,root=/one")
            == name("acestep:acestep-v15-turbo"))


def test_option_order_does_not_change_identity():
    """Sorted, or the same candidate written two ways splits one comparison
    into two piles."""
    assert (name("acestep:acestep-v15-turbo,steps=8,guidance=1.0")
            == name("acestep:acestep-v15-turbo,guidance=1.0,steps=8"))


def test_the_cover_task_is_part_of_the_identity():
    """A cover and a text2music generation from one config are different
    products, and #275 made that a case rather than a hypothetical."""
    assert (name("acestep:acestep-v15-turbo,task=cover")
            != name("acestep:acestep-v15-turbo"))


def test_the_name_still_survives_a_filesystem():
    """It becomes an artifact filename, so it cannot carry a separator or
    anything a path will not hold."""
    got = name("acestep:acestep-v15-turbo,steps=8,task=cover")
    stem = got.replace("/", "_")
    assert "/" not in stem and "\\\\" not in stem
    assert stem == stem.strip() and ".." not in stem
