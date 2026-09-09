"""Which address the services listen on.

Every service binds 0.0.0.0 by default, on purpose. This is a house LAN, the
models are local, and the point of the machine is that other machines on it can
use the GPU -- which is also the shape the M5 Studio will need, with the Studio
serving and the mini as a client.

There is no authentication and that is a deliberate, stated choice rather than
an oversight. What matters is that the address is a PARAMETER: one variable per
service, so a laptop on a cafe wifi can be locked down with an environment
variable rather than an edit.
"""
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SERVICES = {
    "scripts/serve-gateway.sh": "GATEWAY_HOST",
    "scripts/serve-mlx.sh": "MLX_HOST",
    "scripts/serve-tts.sh": "TTS_HOST",
    # The text lane's server on a machine with an NVIDIA card. Same rule: the
    # address is a parameter, and it listens everywhere by default because the
    # point is that the other machine can use this one's GPU.
    "scripts/serve-llamacpp.sh": "LLAMACPP_HOST",
    # The audio lanes on the same machine. mlx_audio serves these two
    # endpoints on the Mac; harness/audio_server.py serves them here.
    "scripts/serve-audio-cuda.sh": "AUDIO_HOST",
}


@pytest.mark.parametrize("script,var", SERVICES.items())
def test_the_listening_address_is_a_variable(script, var):
    """A hardcoded --host cannot be changed without editing the script."""
    text = (REPO / script).read_text(encoding="utf-8")
    assert f"${{{var}:-" in text, f"{script} should read its host from ${var}"
    assert not re.search(r'--host\s+["\']?127\.0\.0\.1["\']?\s', text), (
        f"{script} still hardcodes 127.0.0.1")


@pytest.mark.parametrize("script,var", SERVICES.items())
def test_the_default_is_every_interface(script, var):
    text = (REPO / script).read_text(encoding="utf-8")
    assert re.search(rf'\$\{{{var}:-0\.0\.0\.0\}}', text), (
        f"{script} should default {var} to 0.0.0.0")


@pytest.mark.parametrize("script", SERVICES)
def test_the_choice_is_argued_for_where_it_is_made(script):
    """A service on every interface with no auth must not look accidental to
    whoever reads this next. It was 0.0.0.0 by accident once already."""
    text = (REPO / script).read_text(encoding="utf-8").lower()
    assert "lan" in text or "no auth" in text or "every interface" in text, (
        f"{script} binds every interface without saying why")


def test_smoke_can_be_pointed_at_another_host():
    """If the Studio serves and the mini is a client, the smoke test has to be
    runnable against the Studio."""
    text = (REPO / "scripts" / "smoke.sh").read_text(encoding="utf-8")
    assert "${SMOKE_HOST:-" in text
    assert not re.search(r'="http://127\.0\.0\.1:', text)
