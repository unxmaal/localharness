"""The MCP surface: the same `lh` a local agent runs, reachable from the LAN.

Rachel gets svg, web, code and image from her own Claude Code on her own
machine. Everything here shells out to `lh` rather than reimplementing it,
because the repo's rule is that every caller runs identical commands -- the CLI
and the eval suite already do, and a third caller that drifted would expose a
product nobody ships.

These tests never generate anything. They assert the command built, the shape
of what comes back, and that the expensive lane goes through the queue.
"""
import pytest

mcp_server = pytest.importorskip(
    "harness.mcp_server",
    reason="needs the `mcp` group: uv run --group mcp pytest")


@pytest.fixture
def spy(monkeypatch, tmp_path):
    """Capture the argv, and fake `lh` writing its artifact."""
    calls = []

    def fake_lh(argv, timeout=None):
        """What `lh` ACTUALLY does, which is not what I first assumed.

        `lh svg` and `lh web` always write a file and print its PATH; only
        `lh code` prints content to stdout. The first version of this fake
        returned markup on stdout, which is what I believed rather than what
        the CLI does, so the tests passed while the tool handed callers a path
        string where an SVG document should have been. Caught live over the
        LAN, not here."""
        calls.append(argv)
        assert "-o" in argv, "every verb should be given an explicit -o"
        out = argv[argv.index("-o") + 1]
        body = (b"\x89PNG\r\n\x1a\n" + b"x" * 200 if argv[1] == "image"
                else b"<svg xmlns='http://www.w3.org/2000/svg'/>")
        open(out, "wb").write(body)
        return out

    monkeypatch.setattr(mcp_server, "run_lh", fake_lh)
    monkeypatch.setattr(mcp_server, "OUTDIR", tmp_path)
    return calls


def test_svg_returns_the_markup_not_the_path_it_was_written_to(spy):
    """`lh svg` prints where it put the file. A caller on another machine
    cannot open that path, and an agent asking for an SVG wants the document."""
    out = mcp_server.svg("two concentric gears")
    assert out.startswith("<svg"), out
    assert spy[0][:2] == ["lh", "svg"]
    assert "two concentric gears" in spy[0]


def test_the_artifact_is_kept_on_the_serving_machine_too(spy, tmp_path):
    """Returned by value AND left on disk: the text is what the caller wanted,
    the file is what makes a bad result inspectable afterwards."""
    mcp_server.svg("a gear")
    written = list(tmp_path.glob("svg-*.svg"))
    assert written and written[0].read_text().startswith("<svg")


def test_web_and_code_are_the_same_shape(spy):
    mcp_server.web("a landing page for a coffee roaster")
    mcp_server.code("a python function that parses an ISO timestamp")
    assert spy[0][1] == "web" and spy[1][1] == "code"


def test_the_model_can_be_chosen_per_call(spy):
    mcp_server.svg("a gear", model="local-large")
    assert "-m" in spy[0] and "local-large" in spy[0]


def test_image_returns_a_job_id_not_an_image(spy):
    """54 seconds cold is past what an MCP client waits for, and 11.4 GiB is
    why it has to be one at a time anyway."""
    out = mcp_server.image("a red fox in snow")
    assert out.job.startswith("image-")
    assert out.state in ("queued", "running")


def test_an_image_job_finishes_and_carries_the_file(spy):
    job = mcp_server.image("a red fox in snow", width=64, height=64)
    done = mcp_server.QUEUE.wait(job.job, timeout=10)
    assert done.state == "done", done.error
    assert done.result.endswith(".png")
    argv = spy[0]
    assert argv[:2] == ["lh", "image"]
    assert "--width" in argv and "64" in argv


def test_job_status_reports_a_wait_it_can_explain(spy):
    job = mcp_server.image("a fox")
    mcp_server.QUEUE.wait(job.job, timeout=10)
    status = mcp_server.job_status(job.job)
    assert status.state == "done"
    assert status.seconds >= 0


def test_an_unknown_job_says_so_rather_than_raising(spy):
    assert mcp_server.job_status("nope-0-abcdef").state == "unknown"


def test_a_failed_generation_is_reported_not_raised(spy, monkeypatch):
    def boom(argv, timeout=None):
        raise RuntimeError("mflux exited 3: out of memory")
    monkeypatch.setattr(mcp_server, "run_lh", boom)
    job = mcp_server.image("a fox")
    mcp_server.QUEUE.wait(job.job, timeout=10)
    status = mcp_server.job_status(job.job)
    assert status.state == "failed"
    assert "out of memory" in status.error


def test_every_tool_is_registered_on_the_server():
    names = {t.name for t in mcp_server.SERVER._tool_manager.list_tools()}
    assert {"svg", "web", "code", "image", "job_status"} <= names


def test_video_and_speech_are_not_exposed():
    """Scoped deliberately: 40 minutes and a large file need job semantics
    nobody has asked for, and speech over the LAN was ruled out."""
    names = {t.name for t in mcp_server.SERVER._tool_manager.list_tools()}
    assert "video" not in names and "say" not in names


# ---- reachable from the LAN, and only from it ------------------------------

def test_the_lan_hostname_is_allowed_or_rachel_gets_a_rejection():
    """MCP 2.x turns DNS-rebinding protection ON by default with an EMPTY
    allowlist, so a request carrying `Host: styx.local:8899` is refused before
    it reaches a tool. Binding 0.0.0.0 is not enough on its own."""
    s = mcp_server.transport_security(host="0.0.0.0", port=8899)
    assert s.enable_dns_rebinding_protection
    joined = " ".join(s.allowed_hosts)
    assert "styx.local:8899" in joined or ".local:8899" in joined
    assert "127.0.0.1:8899" in joined


def test_protection_stays_on_because_the_house_rule_does_not_cover_it():
    """No LAN auth is a decision about who can reach the port. DNS rebinding
    does not need the port to be reachable: it needs someone in the house to
    open a web page. Different threat, so it keeps its guard."""
    assert mcp_server.transport_security("0.0.0.0", 8899).enable_dns_rebinding_protection


def test_an_extra_host_can_be_allowed_for_the_studio():
    s = mcp_server.transport_security("0.0.0.0", 8899, extra=["studio.local"])
    assert any("studio.local:8899" == h for h in s.allowed_hosts)


def test_a_job_comes_back_as_fields_not_json_in_a_string():
    """The consumer is an agent. Without a declared return type the SDK sends
    the dict as a JSON blob inside a text block and structured_content is None,
    so the caller has to parse a string to find out whether the job finished."""
    tools = {t.name: t for t in mcp_server.SERVER._tool_manager.list_tools()}
    for name in ("image", "job_status"):
        schema = tools[name].output_schema
        assert schema, f"{name} declares no output schema"
        assert "job" in schema["properties"] and "state" in schema["properties"]
