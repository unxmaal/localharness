"""localharness over MCP: the same `lh`, reachable from the LAN.

A second agent on another machine (its own Claude Code, its own context) asks
this one to draw something. What crosses the network is a tool call; what runs
here is the same command line a local agent would type.

THAT IS THE DESIGN CONSTRAINT. Every tool shells out to `lh`. The repo's
standing rule is that the CLI and the eval suite run identical commands,
because for a while only the eval knew how to invoke a generator and the
harness could measure something the product did not ship. A third caller obeys
the same rule, and shelling out is how it stays true by construction rather
than by discipline.

SCOPED DELIBERATELY to svg, web, code and image. Video is 40 minutes and a
large file, which needs job semantics past a queue, and speaking over the LAN
was ruled out. Both are reachable locally with `lh`.

THE EXPENSIVE LANE IS QUEUED. `lh image` holds 11.4 GiB for 20-54 seconds and
mlx_lm.server swaps models per request through a single queue. Two callers
without a queue means each inserts a full model load into the other's request
on a machine that swaps if both hold weights at once. So `image` returns a job
id, and the queue reports what a caller is waiting behind.

    ./scripts/serve-mcp.sh          # 0.0.0.0, no auth, house LAN only

Rachel adds one entry pointing at http://styx.local:8899/mcp
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel

from harness import env, jobs, paths


class JobInfo(BaseModel):
    """What a caller gets back about a queued generation.

    A declared return type, not a bare dict: without one the SDK ships the
    result as JSON inside a text block and structured_content comes back None,
    leaving an agent to parse a string to learn whether its image is ready.

    The optional fields are optional in the honest sense. `ahead` means nothing
    once a job is running, and `path` does not exist until it is done.
    """
    job: str
    kind: str
    state: str                       # queued | running | done | failed | unknown
    seconds: float = 0.0
    #: Jobs in front of this one, and what is currently holding the machine, so
    #: a wait is attributable rather than mysterious.
    ahead: int | None = None
    waiting_for: str | None = None
    #: Where the artifact landed. It stays on the serving machine.
    path: str | None = None
    error: str | None = None

# Artifacts stay on this machine and are downloaded when wanted, so the path is
# what a tool result carries rather than the bytes. Under the ONE output root,
# in its own subdirectory: a caller should be able to tell what a remote agent
# asked for from what someone typed here.
OUTDIR = paths.outputs() / "mcp"
LH = "lh"
DEFAULT_TIMEOUT = 300.0

SERVER = MCPServer(
    name="localharness",
    instructions=(
        "Local media generation on Apple Silicon. svg, web and code answer in "
        "a few seconds. image is queued because it holds 11 GiB: it returns a "
        "job id, and job_status carries the file path when it finishes. "
        "Artifacts stay on the serving machine."),
)
QUEUE = jobs.Queue()


def run_lh(argv: list[str], timeout: float = DEFAULT_TIMEOUT) -> str:
    """Run `lh` and return stdout, or raise with what it said on stderr."""
    r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                       env=_child_env())
    if r.returncode != 0:
        raise RuntimeError(
            (r.stderr or r.stdout).strip()[:600] or f"{argv[1]} exited {r.returncode}")
    return r.stdout.strip()


def _child_env() -> dict:
    import os
    child = dict(os.environ)
    # The server may itself be started by launchd with nothing sourced, and an
    # unset HF_HOME sends huggingface_hub to ~/.cache to re-download weights
    # that are already on the volume.
    env.apply(child)
    return child


def _out(kind: str, suffix: str) -> Path:
    return paths.artifact(kind, suffix, where=OUTDIR)


SUFFIX = {"svg": ".svg", "web": ".html", "code": ".txt"}


def _text_tool(verb: str, prompt: str, model: str = "") -> str:
    """Run `lh` and return the DOCUMENT, not the path it was written to.

    `lh svg` and `lh web` always write a file and print where it went; only
    `lh code` prints its content. A caller on another machine cannot open a
    path on this one, so every verb is given an explicit -o and the file is
    read back. It stays on disk afterwards, which is what makes a bad result
    inspectable rather than merely reported.
    """
    out = _out(verb, SUFFIX[verb])
    argv = [LH, verb, prompt, "-o", str(out)]
    if model:
        argv += ["-m", model]
    run_lh(argv)
    if not out.exists():
        raise RuntimeError(f"lh {verb} exited 0 but wrote nothing to {out}")
    return out.read_text()


# ---------------------------------------------------------------------------
# The cheap lane: a few seconds, straight through the gateway, no queue needed.
# ---------------------------------------------------------------------------

@SERVER.tool(description="Generate an SVG document. Returns the markup.")
def svg(prompt: str, model: str = "") -> str:
    return _text_tool("svg", prompt, model)


@SERVER.tool(description="Generate a self-contained HTML page. Returns the "
                         "document, with all CSS and JS inlined.")
def web(prompt: str, model: str = "") -> str:
    return _text_tool("web", prompt, model)


@SERVER.tool(description="Generate code. Returns the source with no prose "
                         "around it.")
def code(prompt: str, model: str = "") -> str:
    return _text_tool("code", prompt, model)


# ---------------------------------------------------------------------------
# The expensive lane: queued, because it holds 11.4 GiB on a 32 GB machine.
# ---------------------------------------------------------------------------

@SERVER.tool(description="Generate an image. Queued: returns a job id "
                         "immediately, then poll job_status for the file path. "
                         "Takes 20-55s once it starts.")
def image(prompt: str, width: int = 512, height: int = 512,
          seed: int = 0, model: str = "") -> JobInfo:
    def work() -> str:
        out = _out("image", ".png")
        argv = [LH, "image", prompt, "-o", str(out),
                "--width", str(width), "--height", str(height)]
        if seed:
            argv += ["--seed", str(seed)]
        if model:
            argv += ["-m", model]
        run_lh(argv)
        return str(out)

    job_id = QUEUE.submit("image", work)
    return _describe(QUEUE.status(job_id))


@SERVER.tool(description="Check a queued job. States: queued, running, done, "
                         "failed. When done, `path` is the artifact on the "
                         "serving machine.")
def job_status(job: str) -> JobInfo:
    status = QUEUE.status(job)
    if status is None:
        return JobInfo(job=job, kind="", state="unknown",
                       error="no such job; jobs do not survive a server restart")
    return _describe(status)


def _describe(job) -> JobInfo:
    info = JobInfo(job=job.id, kind=job.kind, state=job.state,
                   seconds=round(job.seconds, 1))
    if job.state == "queued":
        running = QUEUE.running()
        info.ahead = job.ahead
        # A wait that names what it is behind is a queue; one that does not is
        # a slow tool.
        info.waiting_for = running.kind if running else None
    if job.state == "done":
        info.path = job.result
    if job.state == "failed":
        info.error = job.error
    return info


def transport_security(host: str, port: int, extra=()):
    """Who is allowed to name this server in a Host header.

    MCP 2.x enables DNS-rebinding protection with an EMPTY allowlist, so
    binding 0.0.0.0 is not enough: a request carrying `Host: styx.local:8899`
    is refused before it reaches a tool, and the error says nothing useful.

    The protection STAYS ON. "No LAN auth, this is my house" is a decision
    about who can reach the port, and DNS rebinding does not need the port to
    be reachable from outside -- it needs someone in the house to open a web
    page, and then their browser makes the request. Different threat, so it
    keeps its guard and gets an allowlist instead.
    """
    from mcp.server.transport_security import TransportSecuritySettings

    names = ["localhost", "127.0.0.1", _local_hostname(), *extra]
    if host not in ("0.0.0.0", "::", ""):
        names.append(host)
    allowed = []
    for name in dict.fromkeys(n for n in names if n):
        allowed += [f"{name}:{port}", name]
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=allowed,
        allowed_origins=[f"http://{n}" for n in allowed])


def _local_hostname() -> str:
    import socket
    import subprocess
    try:
        name = subprocess.run(["scutil", "--get", "LocalHostName"],
                              capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        name = ""
    return f"{name}.local" if name else socket.gethostname()


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(prog="lh-mcp")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--allow", action="append", default=[],
                    help="another hostname clients may use, e.g. an IP or the "
                         "Studio's name")
    ap.add_argument("--transport", default="streamable-http",
                    choices=["streamable-http", "stdio", "sse"])
    a = ap.parse_args()
    if a.transport == "stdio":
        SERVER.run(transport="stdio")
        return
    SERVER.run(transport=a.transport, host=a.host, port=a.port,
               transport_security=transport_security(a.host, a.port, a.allow))


if __name__ == "__main__":
    main()
