"""Starting and stopping the desktop's servers.

This script was verified by hand, once, with a stub launcher, and the Windows
branch by use. Issue #131. A check that exists only as something a person did
is not a check, which is the standard this project holds everything else to.

No seam was needed. services.sh resolves its launchers relative to its own
location, so each test copies it into a temporary tree beside fake serve-*.sh
scripts and drives the real verbs against them.
"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

import shells

REPO = Path(__file__).resolve().parent.parent
BASH = shells.resolve("bash")

pytestmark = pytest.mark.skipif(
    BASH is None, reason="no usable bash; services.sh is a bash script")


def tree(tmp_path, gateway: str) -> Path:
    """A copy of services.sh whose gateway launcher is `gateway`."""
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy(REPO / "scripts" / "services.sh", scripts / "services.sh")
    for name in ("serve-gateway.sh", "serve-llamacpp.sh", "serve-audio-cuda.sh"):
        body = gateway if name == "serve-gateway.sh" else "exit 0\n"
        path = scripts / name
        path.write_text("#!/usr/bin/env bash\n" + body, encoding="utf-8")
        path.chmod(0o755)
    return scripts / "services.sh"


#: Ports nothing on this machine serves. `status` asks whether anything is
#: LISTENING (issue #173), so without pinning these a test of a fake gateway in
#: a temporary tree reads the real one on :4000 and reports it up. That is the
#: check working; it is also a test whose answer depends on what the developer
#: happens to be running.
TEST_PORTS = {"GATEWAY_PORT": "49221", "LLAMACPP_PORT": "49222",
              "AUDIO_PORT": "49223"}


def run(script: Path, *args, **overrides) -> subprocess.CompletedProcess:
    home = script.parent.parent / "home"
    env = {**os.environ, "LOCALHARNESS_HOME": str(home), **TEST_PORTS,
           **overrides}
    return subprocess.run([BASH, str(script), *args], capture_output=True,
                          text=True, env=env, timeout=120)


#: A launcher that stays up AND SERVES. It used to be `exec sleep 120`, which
#: modelled a live process rather than a live service -- and `status` asking
#: only whether a pid exists is exactly the defect in issue #173. A fake that
#: cannot be reached would let the fixed status be called broken and the broken
#: one correct. `exec` so the pid recorded is the thing to kill.
#: A service that BINDS AND ANSWERS. It was a bare socket, which meant every
#: test here asserted only that a port was held. `status` now asks the service
#: to answer before calling it up (#255), so the fixture has to model one that
#: can: a bound socket with nothing behind it is the WEDGED case below.
ALIVE = (
    'exec python3 -c \''
    'import os\n'
    'from http.server import BaseHTTPRequestHandler, HTTPServer\n'
    'class H(BaseHTTPRequestHandler):\n'
    '    def do_GET(self):\n'
    '        self.send_response(200)\n'
    '        self.send_header("Content-Type", "application/json")\n'
    '        self.end_headers()\n'
    '        self.wfile.write(b"{}")\n'
    '    def log_message(self, *a): pass\n'
    'HTTPServer(("127.0.0.1", int(os.environ["GATEWAY_PORT"])), H).serve_forever()\n'
    '\'\n')
#: A listener that BINDS AND NEVER ANSWERS, which is what mlx_lm.server became
#: after a failed model load: port held, every request hanging, and `status`
#: calling it up for hours while four lanes were unrunnable. #255.
WEDGED = (
    # `exec`, so the pid services.sh records IS the listener. Without it the
    # script kills a shell wrapper and the child keeps the port, which then
    # blocks every later test on the same fixture port.
    'exec python3 -c \'\n'
    'import socket\n'
    's=socket.socket()\n'
    's.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n'
    's.bind(("127.0.0.1", int(__import__("os").environ["GATEWAY_PORT"])))\n'
    's.listen(1)\n'
    'import time; time.sleep(120)\n'
    '\'\n')

def _answers(port: int) -> bool:
    """Can something at `port` serve a GET? The same question status asks."""
    import urllib.error, urllib.request
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=1)
        return True
    except (urllib.error.URLError, OSError):
        return False


#: One that dies on its first line, which is the case this file exists for.
DEAD = 'echo "boom" >&2\nexit 1\n'


def test_a_service_that_starts_is_reported_started_and_is_running(tmp_path):
    script = tree(tmp_path, ALIVE)
    started = run(script, "start", "gateway")
    assert started.returncode == 0, started.stderr
    assert "started" in started.stdout, started.stdout
    try:
        assert "up" in run(script, "status").stdout
    finally:
        run(script, "stop", "gateway")


def test_a_launcher_that_dies_immediately_is_not_reported_as_up(tmp_path):
    """THE PROPERTY THIS FILE EXISTS FOR. Start-Process returns a pid for a
    process that has already exited, so without the second look a dead service
    reads as running and the pid file backs the claim up."""
    script = tree(tmp_path, DEAD)
    got = run(script, "start", "gateway")
    assert got.returncode != 0, got.stdout
    assert "started" not in got.stdout, got.stdout
    assert "down" in run(script, "status").stdout
    # And it leaves no pid file behind to be believed later.
    assert not (tmp_path / "home" / "run" / "gateway.pid").exists()


def test_stop_ends_it_and_status_agrees(tmp_path):
    script = tree(tmp_path, ALIVE)
    run(script, "start", "gateway")
    pidfile = tmp_path / "home" / "run" / "gateway.pid"
    pid = int(pidfile.read_text(encoding="utf-8").strip())
    stopped = run(script, "stop", "gateway")
    assert "stopped" in stopped.stdout, stopped.stdout
    assert not pidfile.exists()
    assert "down" in run(script, "status").stdout
    assert not _alive(pid), "the process outlived its own stop verb"


def test_starting_twice_does_not_start_twice(tmp_path):
    script = tree(tmp_path, ALIVE)
    first = run(script, "start", "gateway")
    try:
        second = run(script, "start", "gateway")
        assert "already up" in second.stdout, second.stdout
        assert _pid(first.stdout) == _pid(second.stdout)
    finally:
        run(script, "stop", "gateway")


def test_stopping_something_already_down_is_not_an_error(tmp_path):
    script = tree(tmp_path, ALIVE)
    got = run(script, "stop", "gateway")
    assert got.returncode == 0
    assert "not running" in got.stdout, got.stdout


def test_an_unknown_service_is_refused(tmp_path):
    script = tree(tmp_path, ALIVE)
    got = run(script, "start", "nosuch")
    assert "unknown service" in got.stderr, got.stderr


def _pid(text: str) -> str:
    return text.split("pid")[-1].strip(" )\n")


def _alive(pid: int) -> bool:
    time.sleep(0.5)
    if sys.platform == "win32":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                             capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


# ---- kokoro weights live with the weights, not with the outputs (#151) -----

def _kokoro_dir(env: dict) -> str:
    """Ask serve-audio-cuda.sh where it would look, without starting it.

    Through shells.resolve, never a bare "bash": on a Windows runner that name
    resolves to WSL, which has no distribution installed and answers "Windows
    Subsystem for Linux has no installed distributions." in UTF-16. That is
    what tests/shells.py exists for, and this file already imported it."""
    script = Path(__file__).resolve().parent.parent / "scripts" / "serve-audio-cuda.sh"
    line = [ln for ln in script.read_text(encoding="utf-8").splitlines()
            if ln.startswith("KOKORO_DIR=")][0]
    r = subprocess.run([BASH, "-c", f'{line}\nprintf "%s" "$KOKORO_DIR"'],
                       capture_output=True, text=True,
                       env={"PATH": os.environ["PATH"], **env})
    return r.stdout


@pytest.mark.skipif(BASH is None, reason="no bash on this machine")
def test_kokoro_weights_default_under_hf_root():
    """They are ~350MB of model. Every other weight in this project lives under
    HF_ROOT, behind env.sh's writability and free-space guard; these were under
    LOCALHARNESS_HOME, which is the ARTIFACT root -- out/, runs/, logs/."""
    got = _kokoro_dir({"HF_ROOT": "/weights/hf", "LOCALHARNESS_HOME": "/artifacts"})
    assert got == "/weights/hf/kokoro", got


@pytest.mark.skipif(BASH is None, reason="no bash on this machine")
def test_kokoro_dir_overrides_everything():
    got = _kokoro_dir({"KOKORO_DIR": "/elsewhere", "HF_ROOT": "/weights/hf"})
    assert got == "/elsewhere", got


@pytest.mark.skipif(BASH is None, reason="no bash on this machine")
def test_without_hf_root_it_falls_back_rather_than_guessing():
    """No HF_ROOT set is the fresh-clone case, and it must still name ONE
    place rather than searching."""
    got = _kokoro_dir({"LOCALHARNESS_HOME": "/artifacts"})
    assert got == "/artifacts/kokoro", got


# ---- status answers "is it up", not "did I start it" (issue #173) --------

def test_a_service_nothing_is_listening_on_is_down(tmp_path):
    got = run(tree(tmp_path, "exit 0\n"), "status")
    assert got.returncode == 0
    assert got.stdout.count("down") == 3, got.stdout


def test_a_service_this_script_did_not_start_is_still_reported_up(tmp_path):
    """THE DEFECT. status read its own pidfile, so on the machine supervised by
    launchd -- the one whose entire job is to serve -- it reported every
    service down while the gateway was answering requests."""
    import socket
    script = tree(tmp_path, "exit 0\n")
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    listener.close()
    # AN ANSWERING SERVER, not a bare socket. The claim under test is about
    # PROVENANCE -- who started it -- and `status` now also asks whether the
    # service can answer (#255), so a socket with nothing behind it would be
    # reported WEDGED and the provenance line would never be reached.
    import subprocess as _sp
    other = _sp.Popen(["bash", "-c", ALIVE],
                      env={**os.environ, "GATEWAY_PORT": str(port)},
                      stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
    try:
        for _ in range(40):
            if _answers(port):
                break
            time.sleep(0.1)
        got = run(script, "status", GATEWAY_PORT=str(port))
    finally:
        other.kill()
        other.wait(timeout=5)
    assert "gateway" in got.stdout
    gateway_line = next(l for l in got.stdout.splitlines()
                        if l.startswith("gateway"))
    assert "up" in gateway_line, gateway_line
    # AND IT SAYS SO. `stop` can only stop what it started, and a reader about
    # to run it should learn that here rather than from it failing.
    assert "not started by this script" in gateway_line


def test_the_port_is_named_so_a_reader_can_check_it_themselves(tmp_path):
    got = run(tree(tmp_path, "exit 0\n"), "status")
    assert ":49221" in got.stdout and ":49222" in got.stdout


# --- a bound port is not a working service (#255) -------------------------

def test_a_wedged_service_is_not_reported_up(tmp_path):
    """THE STATE THAT COST A MORNING. mlx_lm.server held :8081 after a failed
    model load: port bound, /v1/models empty, every request hanging. `status`
    called it `up` for hours while four text lanes were unrunnable, and the
    only visible symptom was models timing out, which reads as the model
    failing rather than the service.
    """
    script = tree(tmp_path, WEDGED)
    run(script, "start", "gateway")
    try:
        line = next(l for l in run(script, "status").stdout.splitlines()
                    if l.startswith("gateway"))
        assert "WEDGED" in line, line
        assert "up" not in line.split(":")[0], line
    finally:
        run(script, "stop", "gateway")


def test_a_service_that_answers_is_still_reported_up(tmp_path):
    """THE NEGATIVE CONTROL. A health probe that fails on a healthy service
    reports everything down and gets ignored, which is worse than the port
    check it replaced."""
    script = tree(tmp_path, ALIVE)
    run(script, "start", "gateway")
    try:
        line = next(l for l in run(script, "status").stdout.splitlines()
                    if l.startswith("gateway"))
        assert "up" in line and "WEDGED" not in line, line
    finally:
        run(script, "stop", "gateway")


def test_every_service_with_a_port_has_a_probe():
    """A probe keyed on a name this script does not use covers nothing. The
    launchd agent for :8081 is `mlx` and this script calls it `llamacpp`, so
    a probe written against the agent's name silently checked nothing."""
    text = (Path(__file__).resolve().parents[1]
            / "scripts" / "services.sh").read_text(encoding="utf-8")
    services = ("gateway", "llamacpp", "audio")
    probe_block = text.split("probe_for()")[1].split("}")[0]
    for name in services:
        assert f"{name})" in probe_block, (
            f"{name} has a port and no health probe, so `status` can only say "
            f"whether something holds it")
