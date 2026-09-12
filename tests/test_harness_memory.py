"""Refusing to load a model that will take the machine down.

WRITTEN AFTER TAKING THE MACHINE DOWN. An eval sweeping seven candidates
reached Qwen3-30B-A3B-4bit -- 16 GB of weights -- while Qwen3-14B was still
resident and Docker held 7.2 GB, on a 32 GB Mac. The log ends mid-line at the
candidate header. There was no panic report and no jetsam entry: the machine
simply stopped.

Nothing in this repo checked a model's size against available memory, on a
machine whose whole design constraint is 32 GB of unified memory. The eval
suite measured peak memory AFTER the fact and reported it in a column.
"""
import os
import sys

import pytest

from harness import memory


def _link(target, link):
    """Symlink where the OS allows one, hardlink where it does not.

    The HuggingFace cache symlinks blobs into snapshots on macOS. An
    unprivileged Windows process cannot create a symlink at all (WinError
    1314) and huggingface_hub falls back to copies or hardlinks there, so
    the dedup that matters on Windows is by file index. Either way the blob
    must be counted once, which is what this exercises."""
    try:
        link.symlink_to(target)
    except OSError:
        os.link(target, link)


def test_a_model_that_fits_is_allowed():
    ok, why = memory.fits(need_gb=4, available_gb=25, ceiling_gb=25)
    assert ok, why


def test_a_model_larger_than_the_ceiling_is_refused():
    ok, why = memory.fits(need_gb=16, available_gb=10, ceiling_gb=25)
    assert not ok
    assert "16" in why and "10" in why


def test_the_headroom_accounts_for_a_model_already_resident():
    """mlx_lm.server hot-swaps per request and the old weights are not
    guaranteed to be freed before the new ones are read. Budgeting as if only
    one is resident is how this went wrong."""
    ok, _ = memory.fits(need_gb=16, available_gb=20, ceiling_gb=25, resident_gb=14)
    assert not ok


def test_headroom_is_reserved_for_the_rest_of_the_system():
    """WindowServer, Docker, the audio server and the MCP server were all up.
    A model that exactly fills free memory does not fit."""
    ok, _ = memory.fits(need_gb=24, available_gb=25, ceiling_gb=25, reserve_gb=6)
    assert not ok


def test_the_ceiling_is_read_from_the_machine_not_hardcoded():
    """This runs on a 32 GB mini today, a 96 GB Studio later, and a 12 GB
    discrete card in the next room.

    The universal invariant is that the budget never exceeds what the
    accelerator can hold. Only UNIFIED memory takes a fraction of it: there
    the GPU is handed a working set out of the same pool as everything else,
    so the ceiling is strictly below the RAM figure. On a discrete card the
    VRAM total IS the wall and the ceiling equals it -- what the desktop is
    already holding comes off via available_gb instead."""
    total = memory.total_gb()
    assert total > 1
    ceiling = memory.ceiling_gb()
    assert ceiling <= total, "the budget cannot exceed what the machine holds"
    if memory.detect().kind == "unified":
        assert ceiling < total, "the GPU working set is not all of RAM"


def test_available_memory_is_measured():
    assert 0 < memory.available_gb() <= memory.total_gb()


def test_a_local_model_reports_its_size_from_disk(tmp_path):
    """The weights on disk are the best available estimate of what loading
    them costs, and it needs no network."""
    d = tmp_path / "models--mlx-community--Fake-30B"
    (d / "snapshots" / "abc").mkdir(parents=True)
    (d / "snapshots" / "abc" / "model.safetensors").write_bytes(b"x" * (3 * 1024 ** 2))
    assert memory.size_gb(str(d)) == pytest.approx(3 / 1024, rel=0.1)


def test_an_unknown_model_returns_none_rather_than_guessing_zero(tmp_path):
    """Zero would read as 'fits easily', which is the dangerous default."""
    assert memory.size_gb(str(tmp_path / "nope")) is None


def test_check_model_is_a_one_call_verdict_for_a_repo_id(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "cache_path", lambda repo: None)
    ok, why = memory.check_model("mlx-community/Unknown-Model")
    # Unknown size must not be silently approved; it warns and allows.
    assert ok and "unknown" in why.lower()


def test_hardlinked_and_symlinked_blobs_are_counted_once(tmp_path):
    """The HF cache keeps real files in blobs/ and symlinks them into
    snapshots/. Counting both reported every model at exactly TWICE its size,
    which made the guard refuse models that fit comfortably -- a guard that
    cries wolf gets switched off."""
    d = tmp_path / "models--mlx-community--Fake"
    blobs = d / "blobs"
    snap = d / "snapshots" / "abc"
    blobs.mkdir(parents=True)
    snap.mkdir(parents=True)
    real = blobs / "deadbeef"
    real.write_bytes(b"x" * (4 * 1024 ** 2))
    _link(real, snap / "model.safetensors")
    assert memory.size_gb(str(d)) == pytest.approx(4 / 1024, rel=0.05)


def test_reclaimable_pages_count_as_available(monkeypatch):
    """Free + inactive alone understates badly on a machine that has just read
    93GB of weights: most of what macOS calls active is file cache it will
    give back. Speculative and purgeable are reclaimable too, and leaving them
    out made the guard refuse a model on a machine `memory_pressure` called
    92% free."""
    fake = '''Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                    65536.
Pages active:                                 100000.
Pages inactive:                                65536.
Pages speculative:                             65536.
Pages purgeable:                               65536.
Pages wired down:                              10000.
'''
    class R:
        stdout = fake
    monkeypatch.setattr(memory.subprocess, "run", lambda *a, **k: R())
    # 4 buckets x 65536 pages x 16KB = 4 GB
    assert memory.available_gb() == pytest.approx(4.0, rel=0.02)


def test_a_quantised_load_costs_less_than_the_repo_on_disk(tmp_path, monkeypatch):
    """CAUGHT BY A FALSE REFUSAL. The guard used on-disk size as the load cost,
    which is right for a pre-quantised MLX repo and WRONG for a bf16 repo
    loaded with --quantize 4. It refused FLUX.1-dev at "needs 31.4 GB" while
    the stage using it was running successfully on the same machine.

    A guard that blocks working work gets switched off, which is worse than
    having no guard."""
    monkeypatch.setattr(memory, "cache_path", lambda repo: "/fake")
    monkeypatch.setattr(memory, "size_gb", lambda path: 32.0)
    ok, why = memory.check_model("x/bf16-model", quantize=4,
                                 reserve_gb=0.0)
    # 32 GB at bf16 is 16 bits per weight; at 4 bits it is a quarter of that.
    assert "8.0" in why, why


def test_no_quantisation_still_costs_the_full_size(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "cache_path", lambda repo: "/fake")
    monkeypatch.setattr(memory, "size_gb", lambda path: 32.0)
    _, why = memory.check_model("x/model", reserve_gb=0.0)
    assert "32.0" in why


# ---- discrete GPUs --------------------------------------------------------
# Added for the CUDA machine. The Mac budgets a FRACTION of system RAM because
# memory is unified. A discrete card is a hard wall, and the system RAM behind
# it is irrelevant to what the GPU can hold: 61.6 GB of RAM behind a 12 GB 4070
# must not read as 46 GB of budget.


def test_a_discrete_gpu_budgets_vram_not_system_ram():
    acc = memory.Accelerator(kind="discrete", total_gb=12.0, available_gb=10.5)
    assert memory.ceiling_for(acc) == pytest.approx(12.0)


def test_a_unified_machine_still_budgets_a_fraction_of_ram():
    """Unchanged Mac behaviour: Metal hands out well under the RAM figure."""
    acc = memory.Accelerator(kind="unified", total_gb=32.0, available_gb=25.0)
    assert memory.ceiling_for(acc) == pytest.approx(32.0 * memory.GPU_FRACTION)
    assert memory.ceiling_for(acc) < 32.0


def test_the_system_reserve_is_smaller_on_a_discrete_card():
    """DEFAULT_RESERVE_GB is 6 GB for a Mac where WindowServer, Docker and a
    browser sit in the SAME pool as the weights. VRAM hosts none of that --
    measured 1.2 GiB in use on this idle 4070 desktop. Reserving 6 of 12 GB
    would refuse models that fit comfortably."""
    discrete = memory.reserve_for(memory.Accelerator("discrete", 12.0, 10.5))
    unified = memory.reserve_for(memory.Accelerator("unified", 32.0, 25.0))
    assert discrete < unified
    assert unified == memory.DEFAULT_RESERVE_GB


def test_the_accelerator_is_detected_on_this_machine():
    """Runs on the Apple Silicon machine today and the CUDA box in the next room."""
    acc = memory.detect()
    assert acc.kind in ("unified", "discrete")
    assert acc.total_gb > 1, "no accelerator memory was detected"
    assert 0 < acc.available_gb <= acc.total_gb


# ---- the machine with no vm_stat and no card ------------------------------


def test_meminfo_is_read_rather_than_every_byte_declared_free(tmp_path):
    """A Linux box with no discrete card fell through to total_gb(), which
    answers "all of it". This module exists because a budget computed from a
    number nobody checked took a machine down."""
    f = tmp_path / "meminfo"
    f.write_text("MemTotal:       65536000 kB\n"
                 "MemFree:         1048576 kB\n"
                 "MemAvailable:   33554432 kB\n"
                 "Buffers:          123456 kB\n", encoding="utf-8")
    assert round(memory._meminfo_available_gb(str(f)), 1) == 32.0


def test_a_missing_meminfo_is_zero_rather_than_a_crash(tmp_path):
    """macOS and Windows have no /proc, and available_gb falls back from here
    rather than failing."""
    assert memory._meminfo_available_gb(str(tmp_path / "nope")) == 0.0
    assert memory._meminfo_available_gb(str(tmp_path)) == 0.0
