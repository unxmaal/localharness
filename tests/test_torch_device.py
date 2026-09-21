"""Which accelerator the diffusers generators use. Issue #266.

THIS MODULE SAT AT 0% COVERAGE, alongside image_diffusers.py and
video_cuda.py, because torch lives in its own virtualenv
(scripts/diffusers-venv.sh) and the suite cannot import those two at all.
That is the measurable form of the #263 finding: diffusers was registered in
engines.py for months, had never executed on ANY machine, and had collected
three independent defects in series because nothing pushed back.

`accelerator()` takes the torch MODULE as an argument rather than importing
it, so the one piece of that path which is pure logic can be tested here with
a stub. torch.cuda.is_available() reads the machine, and a test that reads the
machine tests the machine (RULE #249).
"""
import pytest

from harness.torch_device import HALF, accelerator


class _Backends:
    def __init__(self, mps):
        if mps is not None:
            self.mps = mps


class _Mps:
    def __init__(self, available):
        self._a = available

    def is_available(self):
        return self._a


class _Cuda:
    def __init__(self, available):
        self._a = available

    def is_available(self):
        return self._a


def torch_like(*, cuda=False, mps=None):
    """A stand-in shaped like the torch attributes this reads."""
    t = type("torch", (), {})()
    t.cuda = _Cuda(cuda)
    t.backends = _Backends(_Mps(mps) if mps is not None else None)
    return t


def test_a_card_is_preferred_where_there_is_one():
    assert accelerator(torch_like(cuda=True, mps=False)) == "cuda"


def test_metal_is_an_accelerator():
    """THE WHOLE DEFECT. Both generators asked torch.cuda.is_available() and
    nothing else, so "no card" and "no Apple GPU" were one answer and the
    image lane had a single engine on Apple Silicon for months."""
    assert accelerator(torch_like(cuda=False, mps=True)) == "mps"


def test_no_accelerator_is_empty_rather_than_cpu():
    """"" is not "cpu". The caller refuses to run without an accelerator
    unless told otherwise, because a diffusion model on a CPU reads as a slow
    model rather than a misconfigured machine."""
    assert accelerator(torch_like(cuda=False, mps=False)) == ""


def test_an_older_torch_without_the_mps_backend_does_not_explode():
    """torch.backends.mps does not exist on every build. Asking for it
    blindly turns "no Apple GPU" into an AttributeError inside a generator."""
    assert accelerator(torch_like(cuda=False, mps=None)) == ""


def test_a_card_still_wins_when_both_are_somehow_present():
    """Not a real machine today, and pinning the order costs nothing: a
    tiebreak nobody stated is a tiebreak that changes silently."""
    assert accelerator(torch_like(cuda=True, mps=True)) == "cuda"


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_both_accelerators_load_half_precision(device):
    """Metal supports float16 and the weights are published in it. fp32 on a
    32 GB unified machine doubles the working set for nothing."""
    assert device in HALF


def test_the_cpu_does_not_load_half_precision():
    """The negative half. fp16 on a CPU is slower than fp32, not faster."""
    assert "cpu" not in HALF
