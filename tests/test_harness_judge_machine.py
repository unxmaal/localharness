"""The rubric has to describe the machine the judge is scoring for.

novelty.yaml told the model "This machine has no CUDA and never will", rewarded
being MLX-native and scored down "Anything that requires CUDA, an NVIDIA card".
Every one of those was true of the Mac it was written on. Run unchanged on a
box bought to run CUDA, it marks down the candidates that box exists for.

The machine-specific claims live in per-runtime fragments now, so a third kind
of machine is a file rather than a fork of the rubric. What stays in
novelty.yaml is the part that is true anywhere.
"""
import pytest

from harness import judge
from harness import machine as mach
from harness.memory import Accelerator

APPLE = mach.Machine(frozenset({"mlx", "cpu"}),
                     Accelerator("unified", 32.0, 25.0, "Mac16,1"))
CARD = mach.Machine(frozenset({"cuda", "cpu"}),
                    Accelerator("discrete", 12.0, 10.5, "RTX 4070"))
LINUX_CARD = mach.Machine(frozenset({"cuda", "cpu"}),
                          Accelerator("discrete", 12.0, 10.5, "RTX 4070"))


def prompt_for(m):
    return judge.load(machine=m).prompt.lower()


def test_a_card_is_not_told_that_cuda_disqualifies_a_candidate():
    """THE POINT OF THIS FILE. The judge is the cheapest tier and it sees every
    proposal, so a rubric that marks CUDA down here removes those candidates
    before anything is inspected, measured or even listed."""
    text = prompt_for(CARD)
    assert "no cuda and never will" not in text
    assert "requires cuda, an nvidia card" not in text


def test_a_card_is_told_what_it_cannot_run():
    """The other half. MLX is the thing this machine cannot run, and saying so
    is what stops it queueing weights that will not load."""
    assert "mlx" in prompt_for(CARD)


def test_a_mac_is_still_told_what_it_could_always_be_told():
    text = prompt_for(APPLE)
    assert "mlx" in text or "apple silicon" in text


def test_running_here_scores_nothing_on_either_machine():
    """Scoring the price of entry as merit put six unrelated candidates at
    10/10 in one sweep. That lesson is about judging, not about hardware, so it
    survives on every machine."""
    for m in (APPLE, CARD):
        assert "counts for nothing by itself" in prompt_for(m)


def test_the_two_machines_are_told_different_things():
    assert prompt_for(APPLE) != prompt_for(CARD)


def test_linux_and_windows_with_the_same_card_are_told_the_same_thing():
    """Nothing here may consult the platform."""
    assert prompt_for(CARD) == prompt_for(LINUX_CARD)


@pytest.mark.parametrize("runtime", sorted(mach._RUNTIME_PROBES))
def test_every_runtime_the_machine_module_knows_has_a_fragment(runtime):
    """A runtime detected but undescribed leaves the judge scoring for a
    machine nobody told it about."""
    assert (judge.RUBRIC_DIR / "machine" / f"{runtime}.yaml").exists(), (
        f"harness/rubrics/machine/{runtime}.yaml is missing; machine.py can "
        f"detect {runtime} and the judge would not be told")


def test_the_rubric_still_loads_without_being_given_a_machine():
    """Callers that predate this keep working, on the local machine."""
    assert judge.load().prompt
