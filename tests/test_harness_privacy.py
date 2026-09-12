"""The checker that keeps one person's desk out of a public repo. Issue #136.

The interesting assertions are the NEGATIVE ones. A checker that flags every
mention of hardware would be turned off within a week, and the rule this
project actually wants is narrower: a measurement keeps its instrument, and
the household around it goes.
"""
from pathlib import Path

import pytest

from harness import privacy


def scan(text):
    return [f.name for f in privacy.scan(text, "t")]


# ---- what must be caught ---------------------------------------------------

@pytest.mark.parametrize("text,name", [
    ("export HF_ROOT=/Volumes/Models/hf", "named-volume"),
    ("weights sat on /Volumes/BigDisk", "named-volume"),
    ("claude mcp add http://kitchen-mac.local:8899/mcp", "private-host"),
    ("the gateway answered on 192.168.1.42:4000", "private-host"),
    ("Linux on a spare NVMe in that same desktop", "spare-drive"),
    ("both halves live in the same tower", "same-box"),
    ("this box plays games and runs this sometimes", "gaming-box"),
    ("a gaming rig that quietly starts a language model", "gaming-box"),
    ("needs-cuda on the mini and needs-mlx on the card", "machine-nickname"),
    ("25.0 GiB of 32 on this mini", "machine-nickname"),
])
def test_the_household_is_caught(text, name):
    assert scan(text) == [name]


def test_a_home_directory_names_an_account():
    assert scan("/Users/somebody/projects/h3.c/h3") == ["home-path"]


# ---- what must NOT be caught, which is the harder half ---------------------

@pytest.mark.parametrize("text", [
    # A measurement keeps its instrument. This is the whole point.
    "Measured on the 4070, a request takes VRAM from 2462 to 3296 MiB",
    "Video takes about 40 minutes a generation on an M2 Pro with 32 GB",
    "the 61.6 GB of RAM behind a 12 GB 4070 is not budget",
    "| lane | Apple Silicon | NVIDIA, on Windows or Linux |",
    # Placeholders are the fix, so they must not read as the defect.
    "export HF_ROOT=/Volumes/FAST/hf",
    "HF_ROOT=/Volumes/NO_SUCH_VOLUME/hf walks up to the root",
    "claude mcp add --transport http lh http://<host>.local:8899/mcp",
    # Apple's own path, not a drive anyone named.
    "$ df -h /System/Volumes/Data",
    # Process co-location, not a floor plan. An early cut flagged this.
    "the client and the server run on the same machine",
    "~/.cache/huggingface is the fallback",
])
def test_provenance_and_placeholders_survive(text):
    assert scan(text) == []


# ---- the escape hatch ------------------------------------------------------

def test_a_marker_on_the_line_exempts_it():
    assert scan("HF_ROOT=/Volumes/Models/hf  # privacy-ok: quoting the old default") == []


def test_a_marker_on_the_line_above_exempts_it():
    """For a line that will not take a trailing comment -- a table row, a
    fenced transcript."""
    assert scan("<!-- privacy-ok -->\n| /Volumes/Models | 959 MB/s |") == []


def test_the_marker_does_not_leak_to_the_line_after_next():
    assert scan("# privacy-ok\nfine\n/Volumes/Models/hf") == ["named-volume"]


# ---- people's names are configured, never shipped --------------------------

def test_no_name_list_means_the_structural_patterns_still_run(tmp_path, monkeypatch):
    monkeypatch.delenv("LH_PRIVATE_NAMES", raising=False)
    assert privacy.name_pattern(tmp_path) is None


def test_names_come_from_the_environment_and_are_case_insensitive(tmp_path, monkeypatch):
    monkeypatch.setenv("LH_PRIVATE_NAMES", "Ada, Grace")
    extra = privacy.name_pattern(tmp_path)
    assert [f.name for f in privacy.scan("grace picked this one by ear", "t", extra)] \
        == ["person-name"]


def test_names_come_from_an_untracked_file(tmp_path, monkeypatch):
    monkeypatch.delenv("LH_PRIVATE_NAMES", raising=False)
    (tmp_path / privacy.NAMES_FILE).write_text("Ada\nGrace\n", encoding="utf-8")
    extra = privacy.name_pattern(tmp_path)
    assert [f.name for f in privacy.scan("Ada's call, and a reasonable one", "t", extra)] \
        == ["person-name"]


def test_a_name_is_not_matched_inside_another_word(tmp_path, monkeypatch):
    """`\\b` matters: 'Ada' inside 'Adaptive' is not a person."""
    monkeypatch.setenv("LH_PRIVATE_NAMES", "Ada")
    extra = privacy.name_pattern(tmp_path)
    assert privacy.scan("an adaptive threshold", "t", extra) == []


def test_the_name_list_is_never_committed():
    """The checker would otherwise publish exactly what it exists to hide."""
    ignored = (Path(__file__).resolve().parent.parent / ".gitignore").read_text()
    assert privacy.NAMES_FILE in ignored


# ---- the repo itself -------------------------------------------------------

def test_this_repo_is_clean():
    """`make lint` runs this too; here it fails with the list rather than a
    bare exit code."""
    found = privacy.scan_paths(Path(__file__).resolve().parent.parent)
    assert not found, "\n" + "\n".join(str(f) for f in found)
