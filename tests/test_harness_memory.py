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
import pytest

from harness import memory


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
    """This runs on a 32 GB mini today and a 96 GB Studio later."""
    total = memory.total_gb()
    assert total > 1
    assert memory.ceiling_gb() < total, "the GPU working set is not all of RAM"


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
    (snap / "model.safetensors").symlink_to(real)
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
