"""A script with a shebang has to be executable in the index.

The runbook opens a new machine's day with `./scripts/services.sh start`, and
that line failed on the first fresh clone anyone ran it on: five scripts
carrying shebangs were committed 100644. Nothing caught it. `make check` runs
pytest and shellcheck, and neither executes a script as a program, so three
green runners said nothing about the one command every new machine starts with.

The mode is read from the INDEX, not the filesystem. A working tree can lose
permission bits to a filesystem that has no opinion about them; what reaches
the next clone is only ever what git recorded.

The rule is one-directional on purpose. A shebang declares "run me directly",
so it demands the bit. The absence of one does not forbid it: env.sh is sourced
and has been 100755 for its whole life, which is untidy rather than broken.
"""
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _indexed_scripts():
    """(path, mode) for every shell script git is tracking."""
    proc = subprocess.run(
        ["git", "ls-files", "-s", "--", "scripts/*.sh"],
        cwd=REPO, capture_output=True, text=True, check=True,
    )
    for line in proc.stdout.splitlines():
        meta, path = line.split("\t", 1)
        yield path, meta.split()[0]


def test_a_script_with_a_shebang_is_executable_in_the_index():
    offenders = [
        f"{path} is {mode} but starts with a shebang; needs 100755"
        for path, mode in _indexed_scripts()
        if (REPO / path).read_bytes()[:2] == b"#!" and mode != "100755"
    ]
    assert not offenders, "\n".join(
        ["run: git update-index --chmod=+x <path>", *offenders]
    )


def test_the_scripts_the_runbook_names_are_runnable():
    """The specific ones the docs tell a reader to type as ./scripts/..."""
    documented = ("scripts/services.sh", "scripts/smoke.sh")
    modes = dict(_indexed_scripts())
    for path in documented:
        assert path in modes, f"{path} is not tracked"
        assert modes[path] == "100755", f"{path} is {modes[path]}, needs 100755"
