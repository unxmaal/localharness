"""What this machine can RUN, which is not the same as which OS it is.

Discovery has to answer one question about a candidate: can this machine run
it. That question is about RUNTIMES -- mlx, cuda, rocm -- and the operating
system is only how you find out which ones are present. Writing it the other
way, as a check on sys.platform, is what made every CUDA candidate a rejection
on a machine bought to run them.

The Linux tests here are not hypothetical politeness. A Linux box with an
NVIDIA card is the same machine as a Windows box with one as far as any of this
is concerned, and if that is true then adding it is a row of data rather than a
third branch. These assert that it is true.
"""
import pytest

from harness import machine
from harness.memory import Accelerator

APPLE = machine.Machine(runtimes=frozenset({"mlx", "cpu"}),
                        accelerator=Accelerator("unified", 32.0, 25.0, "Mac16,1"))
WINDOWS_NVIDIA = machine.Machine(
    runtimes=frozenset({"cuda", "cpu"}),
    accelerator=Accelerator("discrete", 12.0, 10.5, "NVIDIA GeForce RTX 4070"))
LINUX_NVIDIA = machine.Machine(
    runtimes=frozenset({"cuda", "cpu"}),
    accelerator=Accelerator("discrete", 12.0, 10.5, "NVIDIA GeForce RTX 4070"))
LINUX_AMD = machine.Machine(
    runtimes=frozenset({"rocm", "cpu"}),
    accelerator=Accelerator("discrete", 16.0, 15.0, "AMD Radeon RX 7800 XT"))


def test_it_detects_something_on_this_machine():
    here = machine.detect()
    assert here.runtimes, "no runtime at all was detected"
    assert "cpu" in here.runtimes, "the cpu is always a runtime"
    assert here.accelerator.total_gb > 0


def test_a_runtime_this_machine_lacks_is_named_in_the_refusal():
    """`needs-cuda` on a Mac and `needs-mlx` on a card are the same rule read
    from two directions. Naming the runtime rather than the platform is what
    makes a third platform a row of data."""
    assert APPLE.refuses("cuda") == "needs-cuda"
    assert WINDOWS_NVIDIA.refuses("mlx") == "needs-mlx"
    assert LINUX_AMD.refuses("cuda") == "needs-cuda"


def test_a_runtime_this_machine_has_is_not_refused():
    assert APPLE.refuses("mlx") is None
    assert WINDOWS_NVIDIA.refuses("cuda") is None
    assert LINUX_AMD.refuses("rocm") is None
    for m in (APPLE, WINDOWS_NVIDIA, LINUX_AMD):
        assert m.refuses("cpu") is None


def test_linux_and_windows_with_the_same_card_are_the_same_machine():
    """THE POINT OF THIS FILE. Nothing downstream may branch on the OS, so a
    Linux box and a Windows box holding the same card have to be
    indistinguishable here. If this ever fails, something has started asking
    which platform it is on rather than what it can run."""
    assert WINDOWS_NVIDIA.runtimes == LINUX_NVIDIA.runtimes
    assert WINDOWS_NVIDIA.accelerator == LINUX_NVIDIA.accelerator
    for probe in ("mlx", "cuda", "rocm", "cpu"):
        assert WINDOWS_NVIDIA.refuses(probe) == LINUX_NVIDIA.refuses(probe)


def test_an_unknown_runtime_is_refused_rather_than_assumed_present():
    """A marker this project has never seen names something this machine
    certainly cannot run. Assuming otherwise proposes a download that fails."""
    assert WINDOWS_NVIDIA.refuses("tenstorrent") == "needs-tenstorrent"


@pytest.mark.parametrize("m", [APPLE, WINDOWS_NVIDIA, LINUX_NVIDIA, LINUX_AMD],
                         ids=["apple", "windows-nvidia", "linux-nvidia", "linux-amd"])
def test_every_machine_can_name_itself_for_a_result_sheet(m):
    """A row has to say which machine produced it, and a runtime set is the
    honest description: two boxes with the same card compare, two with
    different ones do not."""
    assert m.describe()
    assert m.accelerator.name in m.describe()


# ---- where the work executes (issue #170) ---------------------------------

def test_a_machine_outside_a_cluster_is_a_host():
    from harness import machine
    assert machine.where({}) == machine.HOST


def test_a_pod_with_no_card_is_in_pod():
    from harness import machine
    assert machine.where({"KUBERNETES_SERVICE_HOST": "kubernetes.default.svc"},
                         machine=_fake("unified")) == machine.IN_POD


def test_a_pod_the_scheduler_gave_a_card_is_a_gpu_node():
    from harness import machine
    assert machine.where({"KUBERNETES_SERVICE_HOST": "kubernetes.default.svc"},
                         machine=_fake("discrete")) == machine.GPU_NODE


def test_a_declaration_beats_what_the_container_can_see():
    """THE WHOLE POINT. A pod dispatching to a host is still a pod; nothing
    about the container can tell you the Metal work happened on a desk."""
    from harness import machine
    assert machine.where({"KUBERNETES_SERVICE_HOST": "kubernetes.default.svc",
                          "LOCALHARNESS_WHERE": "host"},
                         machine=_fake("discrete")) == machine.HOST


def test_a_place_nothing_recognises_raises_rather_than_reaching_a_receipt():
    """A receipt naming a place nothing recognises is worse than one naming
    none: comparable() would treat it as a real distinction forever."""
    import pytest
    from harness import machine
    with pytest.raises(ValueError):
        machine.where({"LOCALHARNESS_WHERE": "the basement"})


def _fake(kind):
    from harness.machine import Machine
    from harness.memory import Accelerator
    return Machine(runtimes=frozenset({"cpu"}),
                   accelerator=Accelerator(kind, "x", 8.0, 8.0))
