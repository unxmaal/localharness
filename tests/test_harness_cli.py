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
        calls.append(dict(argv=argv, timeout=timeout, stream=stream, cwd=cwd))
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


def test_the_default_output_path_is_unique(spy, monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALHARNESS_HOME", str(tmp_path))
    cli.main(["image", "a red fox"])
    cli.main(["image", "a red fox"])
    assert len(sorted((tmp_path / "out").glob("*.png"))) == 2, \
        "two runs must not overwrite each other"


def test_output_does_not_follow_the_working_directory(spy, monkeypatch, tmp_path):
    """The bug this test used to ENCODE. `out/` was relative, and `lh` installs
    onto PATH, so running it from ~/Desktop wrote to ~/Desktop/out and running
    it from a repo wrote into that repo. Artifacts went wherever you happened
    to be standing."""
    home = tmp_path / "home"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir(parents=True)
    monkeypatch.setenv("LOCALHARNESS_HOME", str(home))
    monkeypatch.chdir(elsewhere)
    cli.main(["image", "a red fox"])
    assert sorted((home / "out").glob("*.png")), "artifact did not land in the root"
    assert not list(elsewhere.rglob("*.png")), "artifact leaked into the cwd"


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


def test_an_engine_with_a_working_directory_is_run_from_it(spy, tmp_path):
    cli.main(["video", "fox", "-o", str(tmp_path / "v.mp4")])
    assert spy[0]["cwd"], "h3 must not be launched from the caller's cwd"


def test_the_output_path_is_absolute_when_the_cwd_changes(spy, tmp_path, monkeypatch):
    """A relative -o would land inside the engine's own directory, silently,
    where nobody looks for it."""
    monkeypatch.chdir(tmp_path)
    cli.main(["video", "fox", "-o", "clip.mp4"])
    argv = spy[0]["argv"]
    assert argv[argv.index("-o") + 1].startswith("/")


# ---- code -----------------------------------------------------------------
# Both of these lanes existed only in the eval suite. The plumbing to reach
# them -- system prompts, context placement, artifact recovery -- was already
# shared with the CLI; only the verbs were missing.

@respx.mock
def test_code_prints_the_recovered_source(capsys):
    """Stdout by default, because code is something you pipe or read, not an
    artifact you open in a viewer the way an SVG is."""
    respx.post(f"{GW}/v1/chat/completions").mock(return_value=completion(
        "Here you go:\n```python\ndef add(a, b):\n    return a + b\n```"))
    assert cli.main(["code", "add two numbers"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("def add")
    assert "Here you go" not in out and "```" not in out


@respx.mock
def test_code_writes_a_file_when_asked(tmp_path):
    respx.post(f"{GW}/v1/chat/completions").mock(
        return_value=completion("```python\nx = 1\n```"))
    dest = tmp_path / "m.py"
    assert cli.main(["code", "set x", "-o", str(dest)]) == 0
    assert dest.read_text().strip() == "x = 1"


@respx.mock
def test_code_asks_for_code_and_nothing_else(capsys):
    """The steering is shared with the eval. If the CLI asked differently, the
    eval would be ranking a product that does not ship."""
    import json
    route = respx.post(f"{GW}/v1/chat/completions").mock(
        return_value=completion("x = 1"))
    cli.main(["code", "set x"])
    sent = json.loads(route.calls[0].request.read())
    assert sent["messages"][0]["content"] == cli.completion.SYSTEM["code"]


# ---- extract ---------------------------------------------------------------

@respx.mock
def test_extract_answers_from_a_file(tmp_path, capsys):
    log = tmp_path / "build.log"
    log.write_text("ok\nok\n3 tests failed\n")
    respx.post(f"{GW}/v1/chat/completions").mock(return_value=completion("3"))
    assert cli.main(["extract", "how many tests failed?", "-f", str(log)]) == 0
    assert capsys.readouterr().out.strip() == "3"


@respx.mock
def test_extract_puts_the_material_before_the_question(tmp_path):
    """A small model that reads the log and THEN the question does better than
    one that reads the question and has to remember it through forty lines."""
    import json
    log = tmp_path / "build.log"
    log.write_text("3 tests failed")
    route = respx.post(f"{GW}/v1/chat/completions").mock(
        return_value=completion("3"))
    cli.main(["extract", "how many failed?", "-f", str(log)])
    body = json.loads(route.calls[0].request.read())["messages"][1]["content"]
    assert body.index("3 tests failed") < body.index("how many failed?")


@respx.mock
def test_extract_reads_stdin(monkeypatch, capsys):
    import io
    monkeypatch.setattr("sys.stdin", io.StringIO("3 tests failed\n"))
    respx.post(f"{GW}/v1/chat/completions").mock(return_value=completion("3"))
    assert cli.main(["extract", "how many?"]) == 0
    assert capsys.readouterr().out.strip() == "3"


def test_extract_with_no_material_says_so(monkeypatch, capsys):
    """Answering a question about a log nobody supplied would invent one."""
    import io
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    assert cli.main(["extract", "how many?"]) == 1
    assert "material" in capsys.readouterr().err.lower()


@respx.mock
def test_a_missing_extract_file_is_reported_not_a_traceback(tmp_path, capsys):
    assert cli.main(["extract", "q", "-f", str(tmp_path / "gone.log")]) == 1
    assert "gone.log" in capsys.readouterr().err


@respx.mock
def test_not_found_is_passed_through_as_the_answer(tmp_path, capsys):
    """The system prompt asks for NOT FOUND when the material lacks the answer.
    That is a successful extraction, not a failure."""
    log = tmp_path / "a.log"
    log.write_text("nothing relevant")
    respx.post(f"{GW}/v1/chat/completions").mock(
        return_value=completion("NOT FOUND"))
    assert cli.main(["extract", "how many?", "-f", str(log)]) == 0
    assert capsys.readouterr().out.strip() == "NOT FOUND"


def test_the_extract_default_model_is_the_one_that_actually_answers():
    """The lane is for delegating cheap work, but the eval ranked local-small
    at 40% and local-large at 90%. 0.79s is still cheap; being wrong is not."""
    a = cli.build_parser().parse_args(["extract", "q"])
    assert a.model == "local-large"


# ---- cloned voices from the CLI --------------------------------------------

@respx.mock
def test_say_speaks_through_a_cloned_voice_preset(tmp_path):
    import json
    route = respx.post(f"{AUDIO}/audio/speech").mock(
        return_value=httpx.Response(200, content=wav_bytes()))
    dest = tmp_path / "a.wav"
    assert cli.main(["say", "the tests all passed", "--voice", "fr-male",
                     "-o", str(dest), "--no-play"]) == 0
    sent = json.loads(route.calls[0].request.read())
    assert "Chatterbox" in sent["model"]
    assert sent["ref_audio"].endswith("fr-male.wav")
    assert sent["lang_code"] == "en"


@respx.mock
def test_say_still_speaks_through_a_kokoro_voice(tmp_path):
    import json
    route = respx.post(f"{AUDIO}/audio/speech").mock(
        return_value=httpx.Response(200, content=wav_bytes()))
    assert cli.main(["say", "hi", "--voice", "am_adam",
                     "-o", str(tmp_path / "a.wav"), "--no-play"]) == 0
    sent = json.loads(route.calls[0].request.read())
    assert sent["voice"] == "am_adam"
    assert "ref_audio" not in sent


def test_an_unknown_voice_is_reported_before_the_request(tmp_path, capsys):
    assert cli.main(["say", "hi", "--voice", "fr_male",
                     "-o", str(tmp_path / "a.wav"), "--no-play"]) == 1
    err = capsys.readouterr().err
    assert "fr_male" in err and "fr-male" in err


def test_the_voices_command_lists_what_can_be_spoken(capsys):
    """Three coupled settings behind one name is only usable if the names are
    discoverable."""
    assert cli.main(["voices"]) == 0
    out = capsys.readouterr().out
    assert "fr-male" in out and "bm_george" in out


# ---- lh svg --method trace -------------------------------------------------
# The svg lane's LLM path is measured and weak: valid markup that is not the
# picture. Generating a raster and vectorizing it is the method that works, so
# it has to be reachable from the product, not just proven in a scratch file.

@respx.mock
def test_svg_defaults_to_the_language_model_path(tmp_path):
    respx.post(f"{GW}/v1/chat/completions").mock(return_value=completion(
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'>"
        "<circle cx='12' cy='12' r='8'/></svg>"))
    assert cli.main(["svg", "a gear", "-o", str(tmp_path / "g.svg")]) == 0


def test_svg_trace_runs_the_image_engine_then_vectorizes(spy, tmp_path, monkeypatch):
    """One command, two stages. The engine writes a PNG and the tracer turns it
    into paths; neither half is much use to a caller on its own."""
    traced = []
    monkeypatch.setattr(cli.vector, "trace",
                        lambda p, **kw: traced.append(p) or "<svg><path d='M0 0'/></svg>")
    dest = tmp_path / "frog.svg"
    assert cli.main(["svg", "a frog", "--method", "trace", "-o", str(dest)]) == 0
    assert dest.read_text().startswith("<svg")
    assert traced, "the image was never vectorized"
    assert "mflux" in spy[0]["argv"][0]


def test_svg_trace_keeps_the_intermediate_png(spy, tmp_path, monkeypatch):
    """When the SVG is wrong, the question is always whether the raster was
    wrong too. Deleting it throws away the only way to tell."""
    monkeypatch.setattr(cli.vector, "trace", lambda p, **kw: "<svg><path/></svg>")
    dest = tmp_path / "frog.svg"
    cli.main(["svg", "a frog", "--method", "trace", "-o", str(dest)])
    assert dest.with_suffix(".png").exists()


def test_a_trace_failure_is_reported_not_raised(spy, tmp_path, monkeypatch):
    def boom(p, **kw):
        raise cli.vector.VectorError("traced to a blank document")
    monkeypatch.setattr(cli.vector, "trace", boom)
    assert cli.main(["svg", "a frog", "--method", "trace",
                     "-o", str(tmp_path / "f.svg")]) == 1


def test_an_unknown_method_is_rejected_by_the_parser(tmp_path):
    with pytest.raises(SystemExit):
        cli.main(["svg", "a gear", "--method", "magic"])


def test_svg_and_web_have_their_own_defaults():
    """They shared one until the web lane was widened past saturation. At two
    cases local-large and q3-4b both scored 6/6 and speed decided; at five
    cases q3-4b wins 5/5 to 3/5 while svg still goes the other way. One
    'DEFAULT_TEXT_MODEL' could only ever be wrong for one of them."""
    p = cli.build_parser()
    assert p.parse_args(["svg", "x"]).model == cli.DEFAULT_SVG_MODEL
    assert p.parse_args(["web", "x"]).model == cli.DEFAULT_WEB_MODEL
    assert cli.DEFAULT_SVG_MODEL != cli.DEFAULT_WEB_MODEL


def last_json(capsys):
    """The last line printed to stdout, parsed. Uses the capsys FIXTURE: the
    first version read sys.stdout.getvalue(), which is only a thing when
    stdout happens to be a StringIO."""
    import json as _json
    return _json.loads(capsys.readouterr().out.strip().splitlines()[-1])


# ---- --json ---------------------------------------------------------------
# The primary caller is an agent parsing stdout, not a person reading it. Every
# verb currently prints a path or a body and the agent has to guess the shape.

@respx.mock
def test_json_output_carries_the_path_and_the_timing(spy, tmp_path, capsys):
    dest = tmp_path / "a.png"
    assert cli.main(["image", "a fox", "-o", str(dest), "--json"]) == 0
    out = last_json(capsys)
    assert out["ok"] is True
    assert out["path"] == str(dest)
    assert out["seconds"] > 0
    assert out["verb"] == "image"


@respx.mock
def test_json_on_a_text_verb_carries_the_body(tmp_path, capsys):
    respx.post(f"{GW}/v1/chat/completions").mock(
        return_value=completion("<svg xmlns='http://www.w3.org/2000/svg' "
                                "viewBox='0 0 9 9'><circle r='3'/></svg>"))
    assert cli.main(["svg", "a dot", "-o", str(tmp_path / "d.svg"), "--json"]) == 0
    out = last_json(capsys)
    assert out["ok"] is True and out["body"].startswith("<svg")


@respx.mock
def test_json_reports_a_failure_as_data_not_a_stderr_line(tmp_path, capsys):
    """An agent that has to read stderr to find out something went wrong will
    not read stderr."""
    respx.post(f"{GW}/v1/chat/completions").mock(
        return_value=completion("I would rather not."))
    rc = cli.main(["svg", "a dot", "-o", str(tmp_path / "d.svg"), "--json"])
    out = last_json(capsys)
    assert rc == 1
    assert out["ok"] is False
    assert out["error"]


@respx.mock
def test_json_on_extract_carries_the_answer(tmp_path, capsys):
    log = tmp_path / "b.log"
    log.write_text("3 tests failed")
    respx.post(f"{GW}/v1/chat/completions").mock(return_value=completion("3"))
    assert cli.main(["extract", "how many?", "-f", str(log), "--json"]) == 0
    assert last_json(capsys)["body"] == "3"


def test_every_verb_accepts_json():
    """A flag on some verbs is worse than no flag: the caller cannot rely on
    it without knowing which."""
    p = cli.build_parser()
    for verb in ("image", "video", "svg", "web", "code", "extract", "say", "hear"):
        args = p.parse_args([verb, "x"] if verb not in ("hear",) else [verb])
        assert hasattr(args, "json"), verb


# ---- lh discover -----------------------------------------------------------

def test_discover_lists_capabilities_and_marks_the_unmeasured(capsys, monkeypatch):
    from harness import discover as d
    monkeypatch.setattr(d, "capabilities", lambda: [
        d.Capability("model", "ran-before", "text", "gateway", "cmd-a"),
        d.Capability("model", "never-run", "text", "gateway", "cmd-b"),
    ])
    monkeypatch.setattr(d, "measured", lambda: {"ran-before"})
    assert cli.main(["discover"]) == 0
    out = capsys.readouterr().out
    assert "never-run" in out and "ran-before" in out
    assert "cmd-b" in out, "an unmeasured capability must say how to measure it"


def test_discover_can_show_only_the_gap(capsys, monkeypatch):
    from harness import discover as d
    monkeypatch.setattr(d, "capabilities", lambda: [
        d.Capability("model", "ran-before", "text", "gateway", "cmd-a"),
        d.Capability("model", "never-run", "text", "gateway", "cmd-b"),
    ])
    monkeypatch.setattr(d, "measured", lambda: {"ran-before"})
    assert cli.main(["discover", "--gap"]) == 0
    out = capsys.readouterr().out
    assert "never-run" in out and "ran-before" not in out


def test_discover_filters_by_lane(capsys, monkeypatch):
    from harness import discover as d
    monkeypatch.setattr(d, "capabilities", lambda: [
        d.Capability("model", "a-text", "text", "gateway", "x"),
        d.Capability("engine", "an-image", "image", "mflux", "y"),
    ])
    monkeypatch.setattr(d, "measured", lambda: set())
    cli.main(["discover", "--lane", "image"])
    out = capsys.readouterr().out
    assert "an-image" in out and "a-text" not in out


def test_discover_json_is_machine_readable(capsys, monkeypatch):
    import json as _json
    from harness import discover as d
    monkeypatch.setattr(d, "capabilities", lambda: [
        d.Capability("model", "never-run", "text", "gateway", "cmd-b")])
    monkeypatch.setattr(d, "measured", lambda: set())
    assert cli.main(["discover", "--json"]) == 0
    rows = _json.loads(capsys.readouterr().out)["capabilities"]
    assert rows[0]["name"] == "never-run"
    assert rows[0]["measured"] is False
    assert rows[0]["how"] == "cmd-b"


# ---- issue #69 follow-up: the judging path had no test at all ---------------

def test_judging_an_inspected_candidate_uses_only_fields_fit_has(monkeypatch,
                                                                 tmp_path):
    """This shipped reading `Fit.description`, a field Fit did not have, and
    every one of 900+ tests passed because nothing called the function. A code
    path with no test is untested however green the suite is."""
    from harness import cli, inspect as ins, judge

    fit = ins.Fit(repo="org/thing", verdict="fits", why="MLX-native",
                  description="a tool", mlx=True,
                  weights={"org/w": 2 * ins.GIB}, smallest=2 * ins.GIB,
                  largest=2 * ins.GIB)
    shown = {}

    def fake_score(item, rubric=None, **kw):
        shown["item"] = item
        return 8, "because"

    monkeypatch.setattr(judge, "score", fake_score)
    assert cli._judge_fits([fit], store_path=tmp_path / "d.db") == 0
    assert "NAME: org/thing" in shown["item"]
    assert "READ FROM ITS SOURCE: fits: MLX-native" in shown["item"]
    assert "RUNTIME: MLX-native" in shown["item"]
    assert "DESCRIPTION: a tool" in shown["item"]


def test_a_judge_failure_on_one_candidate_does_not_stop_the_rest(monkeypatch,
                                                                 tmp_path):
    from harness import cli, inspect as ins, judge
    fits = [ins.Fit(repo="org/bad", verdict="fits"),
            ins.Fit(repo="org/good", verdict="fits")]
    seen = []

    def fake_score(item, rubric=None, **kw):
        if "org/bad" in item:
            raise RuntimeError("model is down")
        seen.append(item)
        return 5, "ok"

    monkeypatch.setattr(judge, "score", fake_score)
    cli._judge_fits(fits, store_path=tmp_path / "d.db")
    assert len(seen) == 1


def test_sensitivity_lists_its_probes_and_what_it_cannot_cover(capsys):
    assert cli.main(["sensitivity", "--list"]) == 0
    out = capsys.readouterr().out
    assert "min_shared" in out
    assert "not covered" in out


def test_an_unknown_probe_is_named_rather_than_swept(capsys):
    assert cli.main(["sensitivity", "no_such_knob"]) != 0
    assert "no_such_knob" in capsys.readouterr().err
