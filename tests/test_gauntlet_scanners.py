"""Repo-wide scanners for the tier-1 gauntlet classes, each proven to fire and to stay quiet (#492 C)."""

from pathlib import Path

from tests.gauntlet import scanners

REPO = Path(__file__).resolve().parent.parent


def test_a_mutable_default_argument_is_found():
    src = "def add(x, acc=[]):\n    return acc\n\ndef put(k, *, into=dict()):\n    return into\n"
    got = scanners.mutable_defaults(src, "m.py")
    assert [f.split(":")[1] for f in got] == ["1", "4"]


def test_a_swallowed_error_is_found():
    src = "try:\n    go()\nexcept Exception:\n    pass\ntry:\n    go()\nexcept:\n    log()\n"
    assert [f.split(":")[1] for f in scanners.swallowed_python(src, "m.py")] == ["3", "7"]
    sh = "#!/usr/bin/env bash\nrm -rf \"$d\" || true\nkill \"$p\" || :\n"
    assert [f.split(":")[1] for f in scanners.swallowed_shell(sh, "s.sh")] == ["2", "3"]


def test_an_unchecked_exit_status_is_found():
    sh = "#!/usr/bin/env bash\nset -u\nproduce | consume\ncd \"$dir\"\nrm -rf ./out\n"
    got = scanners.unchecked_status_shell(sh, "s.sh")
    assert any("pipefail" in f for f in got)
    assert any(":4:" in f and "cd" in f for f in got)


def test_a_fixed_resource_in_a_test_is_found():
    src = ("import socket\nfrom http.server import HTTPServer\n"
           "s = socket.socket()\ns.bind((\"127.0.0.1\", 49221))\n"
           "srv = HTTPServer((\"127.0.0.1\", 8080), H)\n"
           "Path(\"/tmp/lh-test.lock\").write_text(\"x\", encoding=\"utf-8\")\n"
           "open(\"/tmp/out.txt\", \"w\", encoding=\"utf-8\")\n")
    got = scanners.fixed_resources(src, "tests/test_x.py")
    assert [f.split(":")[1] for f in got] == ["4", "5", "6", "7"]


def test_a_live_home_derived_outside_paths_is_found():
    src = ("from pathlib import Path\nimport os\n"
           "A = Path.home() / \"localharness\" / \"omnisvg\"\n"
           "B = Path.home() / \"localharness/runs/x\"\n"
           "C = os.path.join(os.path.expanduser(\"~\"), \"localharness\")\n"
           "D = Path(\"~/localharness/x\").expanduser()\n"
           "OUTDIR = paths.outputs()\n"
           "def f(d=paths.home()):\n    return d\n")
    got = scanners.live_home_python(src, "evals/runners/x.py")
    assert [f.split(":")[1] for f in got] == ["3", "4", "5", "6", "7", "8"]
    sh = "#!/usr/bin/env bash\nDEST=\"$HOME/localharness/x\"\n"
    assert [f.split(":")[1] for f in scanners.live_home_shell(sh, "s.sh")] == ["2"]


def test_an_inventory_above_its_pin_fails_and_names_the_count():
    msg = scanners.ratchet(["a", "b", "c"], 2, "swallowed errors")
    assert msg and "3" in msg and "2" in msg


def _quiet_then_fires(scan, innocent, bad, rel):
    # Quiet on the innocent text, and firing once the bad line joins it: quiet for the right reason.
    assert scan(innocent, rel) == []
    assert scan(innocent + bad, rel) != []


def test_innocent_defaults_are_left_alone():
    innocent = ("def f(x=None, y=(), z=frozenset(), *, w=0, name=\"a\"):\n    acc = []\n    return acc\n"
                "g = lambda k=(): k\n")
    _quiet_then_fires(scanners.mutable_defaults, innocent, "def h(a={}):\n    return a\n", "m.py")


def test_a_handled_error_is_left_alone():
    innocent = ("try:\n    go()\nexcept OSError as e:\n    raise SystemExit(str(e)) from e\n"
                "try:\n    go()\nexcept ValueError:\n    n = 0\n    log(n)\n")
    _quiet_then_fires(scanners.swallowed_python, innocent, "try:\n    go()\nexcept KeyError:\n    pass\n", "m.py")
    sh = ("#!/usr/bin/env bash\n# a comment saying || true is fine\nrm -rf \"$d\" || exit 1\n"
          "[ -f x ] || truncate_log\nfoo || true_name\n")
    _quiet_then_fires(scanners.swallowed_shell, sh, "kill \"$p\" || true\n", "s.sh")


def test_a_checked_exit_status_is_left_alone():
    strict = "#!/usr/bin/env bash\nset -euo pipefail\nproduce | consume\ncd \"$dir\"\n"
    assert scanners.unchecked_status_shell(strict, "s.sh") == []
    loose = ("#!/usr/bin/env bash\nset -uo pipefail\nproduce | consume\ncd \"$dir\" || exit 1\n"
             "if cd \"$x\" && make; then :; fi\n[ a ] || [ b ]\necho cdrom\n")
    _quiet_then_fires(scanners.unchecked_status_shell, loose, "cd \"$dir\"\n", "s.sh")
    probe = ("#!/usr/bin/env bash\nset -u\nif fc-list 2>/dev/null | grep -qi dejavu; then ok; fi\n"
             "need uv \"curl -LsSf https://example.org/install.sh | sh\"\nhint 'a | b'\n")
    _quiet_then_fires(scanners.unchecked_status_shell, probe, "fc-list | sort > fonts.txt\n", "s.sh")
    sourced = "build() {\n  produce | consume\n}\n"
    assert scanners.unchecked_status_shell(sourced, "lib.sh") == []
    assert scanners.unchecked_status_shell("#!/bin/sh\n" + sourced, "x.sh") != []


def test_resources_the_os_hands_out_are_left_alone():
    innocent = ("s.bind((\"127.0.0.1\", 0))\nsrv = HTTPServer((\"127.0.0.1\", 0), H)\n"
                "url = \"http://127.0.0.1:8080/v1\"\nsecurity(host=\"0.0.0.0\", port=8899)\n"
                "(tmp_path / \"x\").write_text(\"y\", encoding=\"utf-8\")\n"
                "argv(Path(\"/tmp/o.wav\"))\nopen(\"/etc/hosts\", encoding=\"utf-8\")\n"
                "Path(\"/tmp/x\").read_text(encoding=\"utf-8\")\n")
    _quiet_then_fires(scanners.fixed_resources, innocent, "s.bind((\"\", 49222))\n", "tests/test_x.py")


def test_a_live_home_asked_of_paths_is_left_alone():
    innocent = ("from harness import paths\nfrom pathlib import Path\n"
                "def out():\n    return paths.home() / \"omnisvg\"\n"
                "TOOLS = Path.home() / \".local/share/uv/tools/mflux/bin\"\n"
                "HF = Path.home() / \".cache\" / \"huggingface\"\n"
                "NOTE = \"see ~/localharness in the README\"\n")
    _quiet_then_fires(scanners.live_home_python, innocent,
                      "X = Path.home() / \"localharness\"\n", "evals/runners/x.py")
    assert scanners.live_home_python("DEFAULT_HOME = Path.home() / \"localharness\"\n", "harness/paths.py") == []
    sh = ("#!/usr/bin/env bash\nDEST=\"${OMNISVG_HOME:-${LOCALHARNESS_HOME:-$HOME/localharness}/omnisvg}\"\n"
          "# default: $HOME/localharness\nCACHE=\"$HOME/.cache/huggingface\"\n")
    _quiet_then_fires(scanners.live_home_shell, sh, "cp x ~/localharness/\n", "s.sh")


def test_a_posix_only_primitive_is_found():
    src = "import os, signal\nimport fcntl\ndef stop(p):\n    os.killpg(p.pid, signal.SIGKILL)\n"
    assert [f.split(":")[1] for f in scanners.posix_primitives(src, "m.py")] == ["2", "4", "4"]


def test_a_posix_primitive_behind_a_platform_guard_is_left_alone():
    innocent = ("import os, signal, sys\n"
                "def take(fd):\n    if sys.platform == \"win32\":\n        import msvcrt\n"
                "        try:\n            return msvcrt.locking(fd, 1, 1)\n        except OSError:\n"
                "            return False\n    import fcntl\n    return fcntl.flock(fd, 1)\n"
                "def stop(p):\n    if hasattr(os, \"killpg\"):\n        os.killpg(p.pid, signal.SIGTERM)\n"
                "    else:\n        p.terminate()\n"
                "try:\n    import pwd\nexcept ImportError:\n    pwd = None\n"
                "SIG = signal.SIGKILL if os.name == \"posix\" else signal.SIGTERM\n")
    _quiet_then_fires(scanners.posix_primitives, innocent, "os.killpg(1, signal.SIGTERM)\n", "m.py")


def test_an_unguarded_import_is_found_and_a_guarded_one_is_not():
    src = ("import yaml\ntry:\n    import torch\nexcept ImportError:\n    torch = None\n"
           "def f():\n    from huggingface_hub import snapshot_download\n    from . import local\n")
    assert scanners.unguarded_imports(src, "m.py") == [(1, "yaml"), (7, "huggingface_hub")]


def test_a_module_level_importorskip_is_read_as_a_skip():
    src = "import pytest\ntorch = pytest.importorskip(\"torch.nn\", reason=\"x\")\ndef t():\n    pytest.importorskip(\"mlx\")\n"
    assert scanners.module_skips(src, "t.py") == {"torch"}


def test_an_inventory_at_its_pin_is_quiet_and_below_it_asks_for_a_lower_pin():
    assert scanners.ratchet(["a", "b"], 2, "x") is None
    msg = scanners.ratchet(["a"], 2, "x")
    assert msg and "lower the pin to 1" in msg


def _shellish(path):
    return scanners.is_shell(path) or path.name == "Makefile"


def _tests_only(rel):
    return rel.startswith("tests/")


def _fail_with(found, why):
    assert not found, why + ":\n  " + "\n  ".join(found)


def test_no_function_has_a_mutable_default():
    _fail_with(scanners.scan_tree(REPO, scanners.mutable_defaults, scanners.is_python),
               "a mutable default is shared by every call; default to None and build it inside")


def test_no_shell_script_ignores_a_failure_it_could_see():
    _fail_with(scanners.scan_tree(REPO, scanners.unchecked_status_shell, scanners.is_shell),
               "set -o pipefail, or end the pipeline in a `grep -q` probe; guard each cd without errexit")


def test_no_test_binds_or_writes_a_fixed_shared_resource():
    _fail_with(scanners.scan_tree(REPO, scanners.fixed_resources, scanners.is_python, _tests_only),
               "two suites at once (parallel worktrees) collide on a fixed resource; ask the OS for one")


def test_the_live_home_is_derived_only_in_harness_paths():
    found = (scanners.scan_tree(REPO, scanners.live_home_python, scanners.is_python)
             + scanners.scan_tree(REPO, scanners.live_home_shell, _shellish))
    _fail_with(found, "conftest redirects LOCALHARNESS_HOME; a path derived any other way is the live one")


# Inventories: reported, never grown. Lower a pin when a fix lands.
SWALLOWED_PIN = 60
CONFIG_LESS_PIN = 154


def test_swallowed_errors_are_an_inventory_that_only_shrinks():
    found = (scanners.scan_tree(REPO, scanners.swallowed_python, scanners.is_python)
             + scanners.scan_tree(REPO, scanners.swallowed_shell, _shellish))
    msg = scanners.ratchet(found, SWALLOWED_PIN, "swallowed errors (empty except, bare except, || true)")
    assert msg is None, msg


def test_numbers_without_their_configuration_are_an_inventory_that_only_shrinks():
    from harness import assertions
    found = [str(c) for c in assertions.collect(REPO) if c.config_less]
    msg = scanners.ratchet(found, CONFIG_LESS_PIN, "numbers in prose with no configuration beside them")
    assert msg is None, msg
