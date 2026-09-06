"""The CLI. This is the product; everything else exists to serve it.

Tests here never invoke a real generator: they assert the command line built,
the exit status, and what the user is told. Whether mflux draws a good fox is
the eval suite's question.
"""
import struct
from pathlib import Path

import httpx
import pytest
import respx

from harness import cli
from harness.proc import Outcome

GW = "http://127.0.0.1:4000"
AUDIO = "http://127.0.0.1:8890/v1"


@pytest.fixture
def spy(monkeypatch):
    """Capture the argv the CLI would have run, and fake a clean exit."""
    calls = []

    def fake_run(argv, timeout=None, stream=False, cwd=None):
        calls.append(dict(argv=argv, timeout=timeout, stream=stream))
        # Engines are contractually required to leave a file behind.
        out = _output_of(argv)
        if out:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_bytes(png_bytes())
        return Outcome(0, 1.5, 9_000_000)

    monkeypatch.setattr(cli.proc, "run", fake_run)
    return calls


def _output_of(argv):
    for flag in ("--output", "-o"):
        if flag in argv:
            return argv[argv.index(flag) + 1]
    return None


def png_bytes(w=512, h=512):
    """A real, non-uniform PNG so the CLI's own output check passes."""
    import io
    import random
    from PIL import Image
    im = Image.new("RGB", (w, h))
    im.putdata([(random.randint(0, 255),) * 3 for _ in range(w * h)])
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def wav_bytes(samples=16000):
    data = b"\x00\x01" * samples
    return (b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt "
            + struct.pack("<IHHIIHH", 16, 1, 1, 24000, 48000, 2, 16)
            + b"data" + struct.pack("<I", len(data)) + data)


# ---- usage ----------------------------------------------------------------

def test_no_arguments_prints_usage_and_exits_nonzero(capsys):
    assert cli.main([]) != 0
    assert "image" in capsys.readouterr().err


def test_unknown_command_is_a_usage_error():
    with pytest.raises(SystemExit):
        cli.main(["transmogrify", "x"])


# ---- image ----------------------------------------------------------------

def test_image_runs_the_default_engine(spy, tmp_path):
    rc = cli.main(["image", "a red fox", "-o", str(tmp_path / "f.png")])
    assert rc == 0
    argv = spy[0]["argv"]
    # FLUX.2 Klein has its own entry point; mflux-generate rejects it.
    assert argv[0] == "mflux-generate-flux2"
    assert argv[argv.index("--model") + 1] == "flux2-klein-4b"
    assert argv[argv.index("--prompt") + 1] == "a red fox"


def test_image_flags_reach_the_engine(spy, tmp_path):
    cli.main(["image", "fox", "-o", str(tmp_path / "f.png"),
              "--width", "768", "--height", "512", "--steps", "8", "--seed", "42"])
    argv = spy[0]["argv"]
    assert argv[argv.index("--width") + 1] == "768"
    assert argv[argv.index("--steps") + 1] == "8"
    assert argv[argv.index("--seed") + 1] == "42"


def test_image_engine_is_selectable_by_spec(spy, tmp_path):
    cli.main(["image", "fox", "-o", str(tmp_path / "f.png"),
              "-m", "mflux:flux2-klein-4b,quantize=4"])
    argv = spy[0]["argv"]
    assert argv[0] == "mflux-generate-flux2"
    assert argv[argv.index("--model") + 1] == "flux2-klein-4b"
    assert argv[argv.index("--quantize") + 1] == "4"


def test_a_bad_engine_spec_fails_before_anything_runs(spy, capsys):
    assert cli.main(["image", "fox", "-m", "mflux:x,quantise=4"]) != 0
    assert not spy, "the spec must be rejected before the generator is launched"
    assert "quantise" in capsys.readouterr().err


def test_image_reports_time_and_peak_memory(spy, tmp_path, capsys):
    cli.main(["image", "fox", "-o", str(tmp_path / "f.png")])
    out = capsys.readouterr().out
    assert "1.5" in out and "8.6" in out  # 9,000,000 KB is 8.58 GiB


def test_image_prints_the_path_it_wrote(spy, tmp_path, capsys):
    dest = tmp_path / "f.png"
    cli.main(["image", "fox", "-o", str(dest)])
    assert str(dest) in capsys.readouterr().out


def test_a_blank_image_is_a_failure_even_though_the_command_exited_zero(
        monkeypatch, tmp_path, capsys):
    """A broken diffusion pipeline emits a uniform grey square at the right
    resolution with a clean exit status."""
    def blank(argv, timeout=None, stream=False, cwd=None):
        from PIL import Image
        Image.new("RGB", (512, 512), (128, 128, 128)).save(_output_of(argv))
        return Outcome(0, 1.0, 100)
    monkeypatch.setattr(cli.proc, "run", blank)
    assert cli.main(["image", "fox", "-o", str(tmp_path / "f.png")]) != 0
    assert "uniform" in capsys.readouterr().err.lower()


def test_a_nonzero_exit_is_reported_with_the_code(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli.proc, "run",
                        lambda *a, **k: Outcome(3, 1.0, 100, "", "CUDA?"))
    assert cli.main(["image", "fox", "-o", str(tmp_path / "f.png")]) != 0
    err = capsys.readouterr().err
    assert "3" in err and "CUDA?" in err


def test_a_missing_generator_says_how_to_install_it(monkeypatch, tmp_path, capsys):
    def absent(*a, **k):
        raise FileNotFoundError("mflux-generate")
    monkeypatch.setattr(cli.proc, "run", absent)
    assert cli.main(["image", "fox", "-o", str(tmp_path / "f.png")]) != 0
    err = capsys.readouterr().err
    assert "mflux-generate" in err and "uv tool install" in err


def test_the_default_output_path_is_unique_and_under_out(spy, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    cli.main(["image", "a red fox"])
    cli.main(["image", "a red fox"])
    written = sorted((tmp_path / "out").glob("*.png"))
    assert len(written) == 2, "two runs must not overwrite each other"


# ---- video ----------------------------------------------------------------

def test_video_uses_h3_and_streams_its_output(spy, tmp_path):
    cli.main(["video", "a fox running", "-o", str(tmp_path / "v.mp4"),
              "--frames", "22", "--width", "512", "--height", "512"])
    call = spy[0]
    assert call["argv"][0].endswith("h3")
    assert call["argv"][call["argv"].index("--frames") + 1] == "22"
    assert call["stream"] is True, "a 40-minute run with no output is a hang"


def test_video_timeout_is_hours_not_the_image_default(spy, tmp_path):
    cli.main(["video", "fox", "-o", str(tmp_path / "v.mp4")])
    assert spy[0]["timeout"] > 3600


def test_video_rejects_frames_and_seconds_together(spy, capsys):
    assert cli.main(["video", "fox", "--frames", "22", "--seconds", "2"]) != 0
    assert not spy


# ---- svg and web ----------------------------------------------------------

def completion(text):
    return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})


@respx.mock
def test_svg_writes_the_recovered_document(tmp_path):
    respx.post(f"{GW}/v1/chat/completions").mock(return_value=completion(
        "sure!\n```svg\n<svg xmlns='http://www.w3.org/2000/svg' "
        "viewBox='0 0 24 24'><circle cx='12' cy='12' r='8'/></svg>\n```"))
    dest = tmp_path / "g.svg"
    assert cli.main(["svg", "a gear", "-o", str(dest)]) == 0
    body = dest.read_text()
    assert body.startswith("<svg") and "sure!" not in body


@respx.mock
def test_web_writes_html_and_checks_it(tmp_path):
    respx.post(f"{GW}/v1/chat/completions").mock(return_value=completion(
        "<!doctype html><html><head><title>t</title></head>"
        "<body><p>hi</p></body></html>"))
    dest = tmp_path / "p.html"
    assert cli.main(["web", "a landing page", "-o", str(dest)]) == 0
    assert "<html" in dest.read_text()


@respx.mock
def test_svg_that_fails_its_check_is_reported_and_still_written(tmp_path, capsys):
    """Keeping the file is the point: you cannot debug what was deleted."""
    respx.post(f"{GW}/v1/chat/completions").mock(
        return_value=completion("I would rather not."))
    dest = tmp_path / "g.svg"
    assert cli.main(["svg", "a gear", "-o", str(dest)]) != 0
    assert dest.exists()
    assert capsys.readouterr().err.strip()


@respx.mock
def test_gateway_down_is_a_clean_message_not_a_traceback(tmp_path, capsys):
    respx.post(f"{GW}/v1/chat/completions").mock(
        side_effect=httpx.ConnectError("refused"))
    assert cli.main(["svg", "a gear", "-o", str(tmp_path / "g.svg")]) != 0
    assert "serve-gateway.sh" in capsys.readouterr().err


# ---- say ------------------------------------------------------------------

@respx.mock
def test_say_synthesizes_and_plays(monkeypatch, tmp_path):
    respx.post(f"{AUDIO}/audio/speech").mock(
        return_value=httpx.Response(200, content=wav_bytes()))
    played = []
    monkeypatch.setattr(cli.proc, "run",
                        lambda argv, **k: played.append(argv) or Outcome(0, 0.1, 10))
    dest = tmp_path / "s.wav"
    assert cli.main(["say", "bonjour", "-o", str(dest)]) == 0
    assert dest.stat().st_size > 8000
    assert played and "afplay" in played[0][0]


@respx.mock
def test_say_can_skip_playback(monkeypatch, tmp_path):
    respx.post(f"{AUDIO}/audio/speech").mock(
        return_value=httpx.Response(200, content=wav_bytes()))
    played = []
    monkeypatch.setattr(cli.proc, "run",
                        lambda argv, **k: played.append(argv) or Outcome(0, 0.1, 10))
    cli.main(["say", "hi", "-o", str(tmp_path / "s.wav"), "--no-play"])
    assert not played


@respx.mock
def test_say_passes_voice_and_speed(monkeypatch, tmp_path):
    import json
    route = respx.post(f"{AUDIO}/audio/speech").mock(
        return_value=httpx.Response(200, content=wav_bytes()))
    monkeypatch.setattr(cli.proc, "run", lambda *a, **k: Outcome(0, 0.1, 10))
    cli.main(["say", "hi", "-o", str(tmp_path / "s.wav"),
              "--voice", "ff_siwis", "--speed", "1.2"])
    sent = json.loads(route.calls[0].request.read())
    assert sent["voice"] == "ff_siwis" and sent["speed"] == 1.2


@respx.mock
def test_say_reads_stdin_when_the_text_is_a_dash(monkeypatch, tmp_path):
    import io
    respx.post(f"{AUDIO}/audio/speech").mock(
        return_value=httpx.Response(200, content=wav_bytes()))
    monkeypatch.setattr(cli.proc, "run", lambda *a, **k: Outcome(0, 0.1, 10))
    monkeypatch.setattr("sys.stdin", io.StringIO("piped text"))
    assert cli.main(["say", "-", "-o", str(tmp_path / "s.wav")]) == 0


@respx.mock
def test_say_with_the_server_down_is_a_clean_message(tmp_path, capsys):
    respx.post(f"{AUDIO}/audio/speech").mock(side_effect=httpx.ConnectError("x"))
    assert cli.main(["say", "hi", "-o", str(tmp_path / "s.wav")]) != 0
    assert "serve-tts.sh" in capsys.readouterr().err


# ---- hear -----------------------------------------------------------------

@respx.mock
def test_hear_transcribes_an_existing_file(tmp_path, capsys):
    clip = tmp_path / "c.wav"
    clip.write_bytes(wav_bytes())
    respx.post(f"{AUDIO}/audio/transcriptions").mock(
        return_value=httpx.Response(200, json={"text": "the fox"}))
    assert cli.main(["hear", str(clip)]) == 0
    assert "the fox" in capsys.readouterr().out


@respx.mock
def test_hear_records_first_when_given_no_file(monkeypatch, tmp_path, capsys):
    recorded = []

    def fake_run(argv, **k):
        recorded.append(argv)
        Path(argv[argv.index("trim") - 1]).write_bytes(wav_bytes())
        return Outcome(0, 3.0, 100)

    monkeypatch.setattr(cli.proc, "run", fake_run)
    monkeypatch.chdir(tmp_path)
    respx.post(f"{AUDIO}/audio/transcriptions").mock(
        return_value=httpx.Response(200, json={"text": "heard you"}))
    assert cli.main(["hear", "--seconds", "3"]) == 0
    assert recorded and recorded[0][0].endswith("rec")
    assert "heard you" in capsys.readouterr().out


def test_hear_on_a_missing_file_is_a_clean_message(tmp_path, capsys):
    assert cli.main(["hear", str(tmp_path / "nope.wav")]) != 0
    assert "nope.wav" in capsys.readouterr().err
