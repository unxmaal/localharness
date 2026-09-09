"""Running the services on a machine whose main job is something else.

scripts/launchd.sh installs units with RunAtLoad and KeepAlive, which is right
for a mini that exists to serve. The Windows box is a gaming rig that also
runs this, so the same behaviour there would hold VRAM and CPU against whatever
is being played, and would do it again after every reboot without being asked.

Everything here is a grep over the scripts, for the reason test_service_pinning
gives: the alternative is registering real scheduled tasks in CI, and the
failure being guarded against is visible in the text.
"""
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SERVICES = REPO / "scripts" / "services.sh"
LLAMACPP = REPO / "scripts" / "serve-llamacpp.sh"

#: Anything that would make Windows start something without being asked. The
#: PowerShell trigger types and the schtasks schedules that fire on their own.
_AUTOSTART = re.compile(
    r"New-ScheduledTaskTrigger|Register-ScheduledTask|"
    r"/sc\s+(ONSTART|ONLOGON|ONIDLE|DAILY|WEEKLY|MINUTE|HOURLY)|"
    r"HKCU.*\\\\Run|shell:startup",
    re.I)


def test_the_windows_services_script_exists():
    assert SERVICES.exists(), (
        "the Windows box needs a way to start and stop the services that is "
        "not launchd")


def test_nothing_registers_itself_to_start_on_its_own():
    """THE POINT OF THIS FILE. A gaming rig that quietly starts a language
    model server at logon has lost 8 GB of VRAM to a machine that was supposed
    to be idle, and the owner finds out from a frame rate rather than a log."""
    text = SERVICES.read_text(encoding="utf-8")
    offenders = [line.strip() for line in text.splitlines()
                 if _AUTOSTART.search(line) and not line.strip().startswith("#")]
    assert not offenders, (
        "services.sh would start something without being asked: "
        + "; ".join(offenders))


def test_stopping_takes_the_whole_process_tree():
    """The launcher is bash and the server is its child. Ending the parent
    leaves llama-server holding the card, which is the state this exists to
    avoid."""
    text = SERVICES.read_text(encoding="utf-8")
    assert re.search(r"taskkill[^\n]*/[Tt]", text), (
        "stop should kill the tree, or the model stays resident in VRAM")


def test_status_reports_what_the_card_is_holding():
    """Whether the services are running is the wrong question before a game.
    The question is whether anything still has VRAM."""
    text = SERVICES.read_text(encoding="utf-8")
    assert "nvidia-smi" in text, (
        "status should say what the card is holding, not only which pids are up")


@pytest.mark.parametrize("verb", ["start", "stop", "status"])
def test_it_offers_the_verbs_a_person_needs(verb):
    text = SERVICES.read_text(encoding="utf-8")
    assert re.search(rf"^\s*{verb}\)", text, re.M), f"no {verb} subcommand"


def test_the_text_server_gives_the_card_back_when_idle():
    """Measured on the 4070: a request takes VRAM from 2462 to 3296 MiB, and
    fifteen seconds later it is 2473 again. Without this a server left running
    holds the model for as long as the process lives, which on this machine
    means until someone notices."""
    text = LLAMACPP.read_text(encoding="utf-8")
    assert "--sleep-idle-seconds" in text
    assert "${LLAMACPP_SLEEP_IDLE:-" in text, (
        "the idle timeout should be overridable on a machine that serves "
        "full time")


def test_windows_tool_flags_are_written_so_msys_does_not_eat_them():
    """A SINGLE-SLASH FLAG IS A PATH TO GIT BASH.

    `taskkill /F /IM x` runs, prints nothing, exits 0 and kills nothing,
    because MSYS rewrites /F into a Windows path before taskkill sees it. The
    double-slash form survives. This cost an hour of believing stop was broken
    when the script was right and the command typed beside it was not, and it
    is invisible in review: both spellings look correct.
    """
    offenders = []
    for path in sorted((REPO / "scripts").glob("*.sh")):
        for number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1):
            if line.lstrip().startswith("#"):
                continue
            for tool in ("taskkill", "tasklist", "schtasks", "sc.exe"):
                if tool not in line:
                    continue
                # Followed by whitespace or end of line, so a real path
                # like /c/Users does not read as a mangled flag.
                rest = line.split(tool, 1)[1]
                if re.search(r"(?<![/\w])/[A-Za-z]{1,4}(?=\s|$)", rest):
                    offenders.append(f"{path.name}:{number}: {line.strip()}")
    assert not offenders, (
        "single-slash flags are rewritten as paths by MSYS; use // -- "
        + "; ".join(offenders))
