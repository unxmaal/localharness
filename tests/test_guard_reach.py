"""Which paths that load weights consult the headroom guard. Issue #284.

RULE #251's class, third occurrence: `memory.check_model` was correct, was
tested against the exact RULE #193 crash, and reached one runner. A guard
nothing invokes is a function.

This is an INVENTORY that is now green for the paths listed. Adding a new
weight-loading path without a headroom check fails it, which is the point.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]

#: file -> why this path loads weights and must check headroom first.
MUST_CHECK = {
    "harness/cli.py": "the screen tier loads discovered weights unattended",
    "evals/runners/chain.py": "two stages hold two models",
}


def reads(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def test_every_weight_loading_path_consults_the_guard():
    missing = [f"{name} ({why})" for name, why in MUST_CHECK.items()
               if "check_model" not in reads(name)]
    assert not missing, (
        "a path that loads weights does not check headroom, which is how the "
        f"2026-09-06 crash happened: {missing}")


def test_the_screen_refuses_without_settling_the_candidate():
    """A candidate declined for what the machine was doing must be asked again.
    `broken` is terminal; a transient memory condition must not use it."""
    src = reads("harness/cli.py")
    block = src[src.index("room, why_not = memory.check_model"):][:900]
    assert '"queued"' in block, block
    assert "until=" in block, "a non-terminal refusal needs its revisit condition"


def test_the_guard_is_asked_before_the_subprocess_starts():
    """Checking after the load has already happened measures nothing."""
    src = reads("harness/cli.py")
    guard = src.index("room, why_not = memory.check_model")
    run = src.index("subprocess.run(screen.argv(r, outdir=outdir)")
    assert guard < run, "the headroom check runs after the weights load"


def test_the_screen_records_the_receipt_it_produced():
    """#281. 27 of 27 terminal verdicts had no run_path, so the evidence behind
    them was a stderr tail."""
    src = reads("harness/cli.py")
    block = src[src.index("ms.decide(store, r[\"name\"], got"):][:400]
    assert "run_path=" in block, block


def test_the_static_ceiling_is_still_only_a_prefilter():
    """MEMORY_CEILING describes one machine and cannot see what is resident.
    It stays as a cheap early cut; it must not be the only check."""
    src = reads("harness/cli.py")
    assert "ins.MEMORY_CEILING" in src
    assert "memory.check_model" in src, (
        "the static ceiling is the only memory check in the screen path")


def test_the_inventory_names_a_reason_for_each_entry():
    """A list nobody prunes becomes a list nobody reads."""
    assert all(why for why in MUST_CHECK.values())
    for name in MUST_CHECK:
        assert (ROOT / name).exists(), f"{name} no longer exists"


def test_check_model_has_a_no_crash_answer_for_an_uncached_repo():
    """THE NEGATIVE CONTROL. A guard that refuses working work gets switched
    off, and check_model's own docstring records it refusing FLUX.1-dev at
    "needs 31.4 GB" while a stage using that model ran fine."""
    from harness import memory
    ok, why = memory.check_model("definitely-not/a-cached-repo-9f3a")
    assert ok, why
    assert "unknown" in why.lower()


def test_no_check_uses_the_unmaintained_swaps_counter():
    """`/usr/bin/time -l` prints `swaps` and macOS never sets it, so a guard
    built on it fires never. #283."""
    offenders = []
    for path in sorted(ROOT.glob("harness/*.py")) + sorted(ROOT.glob("evals/*.py")):
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            if re.search(r'"swaps"|\bswaps\b\s*[><=]', line) and \
                    "UNMAINTAINED" not in line and not line.strip().startswith("#"):
                offenders.append(f"{path.name}: {line.strip()}")
    assert not offenders, offenders
