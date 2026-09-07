"""One place for everything this project produces.

There were FOUR, and one of them was relative:

  out/                    the CLI, relative to whatever directory you ran it in
  ~/localharness-out/     the MCP server
  .logs/                  eval runs, mixed in with service logs
  /tmp/                   whatever I was doing at the time

`out/` being relative is the worst of them. `lh` installs onto PATH and runs
from anywhere, so it scattered artifacts into every directory anyone happened
to be standing in.
"""
import os
from pathlib import Path

import pytest

from harness import paths


def test_everything_lives_under_one_root(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALHARNESS_HOME", str(tmp_path))
    for d in (paths.outputs(), paths.runs(), paths.logs()):
        assert tmp_path in d.parents or d == tmp_path, d


def test_the_root_is_absolute_wherever_you_run_from(monkeypatch, tmp_path):
    """The bug that caused this module: a relative `out/` follows the caller's
    cwd, so `lh image` from ~/Desktop wrote to ~/Desktop/out."""
    monkeypatch.delenv("LOCALHARNESS_HOME", raising=False)
    monkeypatch.chdir(tmp_path)
    assert paths.home().is_absolute()
    assert paths.outputs().is_absolute()


def test_the_root_is_overridable(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALHARNESS_HOME", str(tmp_path / "elsewhere"))
    assert paths.home() == tmp_path / "elsewhere"


def test_asking_for_a_directory_creates_it(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALHARNESS_HOME", str(tmp_path))
    assert paths.outputs().is_dir()


def test_outputs_logs_and_runs_are_kept_apart(monkeypatch, tmp_path):
    """A generated artifact and a service's stderr are different things and
    only one of them is worth keeping."""
    monkeypatch.setenv("LOCALHARNESS_HOME", str(tmp_path))
    assert len({paths.outputs(), paths.runs(), paths.logs()}) == 3


def test_a_run_directory_is_named_and_unique(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALHARNESS_HOME", str(tmp_path))
    a = paths.new_run("svg")
    b = paths.new_run("svg")
    assert a != b
    assert "svg" in a.name
    assert a.parent == paths.runs()


def test_an_artifact_name_sorts_by_time_and_does_not_collide(monkeypatch, tmp_path):
    """Two calls a second apart must not overwrite each other; losing a
    generation to a name collision is noticed much later, if at all."""
    monkeypatch.setenv("LOCALHARNESS_HOME", str(tmp_path))
    names = {paths.artifact("image", ".png").name for _ in range(50)}
    assert len(names) == 50


def test_nothing_in_the_tree_still_writes_to_a_relative_out():
    """The regression that matters. A relative default is invisible until
    someone runs the tool from somewhere unexpected."""
    root = Path(__file__).resolve().parents[1]
    for src in list((root / "harness").rglob("*.py")) + list((root / "evals").rglob("*.py")):
        if "__pycache__" in str(src) or src.name == "paths.py":
            continue
        text = src.read_text()
        assert 'Path("out")' not in text, f"{src} writes to a relative out/"
        assert "localharness-out" not in text, f"{src} uses the old MCP path"
