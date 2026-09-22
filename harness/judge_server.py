"""The page a person votes on. Issue #273.

Serves pairings from a run's receipt, one at a time, and writes each answer
straight to the store beside the programmatic verdicts. Stdlib only: this runs
on a laptop for a few minutes at a time and a web framework would be a
dependency for a form with three buttons.

Artifacts are served from disk by index rather than by path, so a stray
request cannot read an arbitrary file.
"""
from __future__ import annotations

import html
import json
import mimetypes
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from harness import human

PAGE = """<!doctype html><meta charset=utf-8>
<title>{lane} &middot; which is better?</title>
<style>
 body{{font:16px/1.5 system-ui,sans-serif;margin:0;padding:2rem;
   background:#14161a;color:#e8e8ea}}
 h1{{font-size:1.1rem;font-weight:600;margin:0 0 .25rem}}
 .sub{{color:#8b90a0;font-size:.85rem;margin-bottom:1.5rem}}
 .pair{{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem;max-width:960px}}
 .side{{background:#1c1f26;border:1px solid #2a2e38;border-radius:8px;
   padding:1rem;text-align:center}}
 .side h2{{font-size:.8rem;letter-spacing:.08em;color:#8b90a0;margin:0 0 .75rem}}
 img{{max-width:100%;border-radius:4px;display:block}}
 audio{{width:100%}}
 .vote{{margin:1.75rem 0;display:flex;gap:.75rem;max-width:960px}}
 button{{flex:1;padding:.85rem;font:inherit;border-radius:6px;cursor:pointer;
   border:1px solid #2a2e38;background:#232733;color:#e8e8ea}}
 button:hover{{background:#2d3240}}
 .tie{{flex:0 0 11rem;color:#8b90a0}}
 .done{{max-width:960px;color:#8b90a0}}
 code{{color:#c8cad2}}
</style>
<h1>{lane} &middot; {case}</h1>
<div class=sub>{progress}</div>
{body}
"""

DONE = """<div class=done><p>Nothing left to judge in this run.</p>
<p>Answers are in <code>{store}</code>.</p></div>"""


def _media(path: Path) -> str:
    kind = (mimetypes.guess_type(path.name)[0] or "")
    if kind.startswith("audio"):
        return f'<audio controls preload=auto src="{{src}}"></audio>'
    if kind.startswith("image") or path.suffix == ".svg":
        return '<img src="{src}" alt="">'
    return '<a href="{src}">open artifact</a>'


class Judge(BaseHTTPRequestHandler):
    lane = ""
    pairs: list[dict] = []
    files: list[str] = []

    def log_message(self, *a):        # the terminal is for the operator
        pass

    # -- helpers ----------------------------------------------------------
    def _send(self, body: bytes, kind="text/html; charset=utf-8", code=200):
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file_id(self, path: str) -> int:
        if path not in self.files:
            self.files.append(path)
        return self.files.index(path)

    # -- routes -----------------------------------------------------------
    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/file":
            idx = int(parse_qs(url.query).get("i", ["-1"])[0])
            if not 0 <= idx < len(self.files):
                return self._send(b"no", "text/plain", 404)
            p = Path(self.files[idx])
            kind = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
            return self._send(p.read_bytes(), kind)
        if url.path not in ("/", "/index.html"):
            return self._send(b"no", "text/plain", 404)

        todo = human.pending(self.lane, self.pairs)
        if not todo:
            return self._send(
                PAGE.format(lane=self.lane, case="", progress="",
                            body=DONE.format(store=human._file())
                            ).encode("utf-8"))
        p = todo[0]
        left, right = Path(p["left_file"]), Path(p["right_file"])
        sides = "".join(
            f'<div class=side><h2>{name}</h2>'
            + _media(f).format(src=f"/file?i={self._file_id(str(f))}")
            + "</div>"
            for name, f in (("A", left), ("B", right)))
        done = len(self.pairs) - len(todo)
        body = (f'<div class=pair>{sides}</div>'
                f'<form class=vote method=post action="/vote">'
                f'<input type=hidden name=case value="{html.escape(p["case"])}">'
                f'<input type=hidden name=a value="{html.escape(p["a"])}">'
                f'<input type=hidden name=b value="{html.escape(p["b"])}">'
                f'<input type=hidden name=first value="{html.escape(p["left"])}">'
                f'<button name=pick value=left>A is better</button>'
                f'<button class=tie name=pick value=tie>Can&rsquo;t tell</button>'
                f'<button name=pick value=right>B is better</button></form>')
        progress = (f"{done} of {len(self.pairs)} pairings settled &middot; "
                    f"this one needs {p['needs']} more answer(s)")
        self._send(PAGE.format(lane=self.lane, case=html.escape(p["case"]),
                               progress=progress, body=body).encode("utf-8"))

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        form = parse_qs(self.rfile.read(n).decode("utf-8"))
        get = lambda k: (form.get(k) or [""])[0]
        pick, first = get("pick"), get("first")
        a, b = get("a"), get("b")
        # The buttons say left and right; the store wants the candidate. Which
        # side each was on is randomised per showing, so this mapping is the
        # only place that knows.
        answer = ("tie" if pick == "tie"
                  else ("a" if (pick == "left") == (first == a) else "b"))
        human.record(self.lane, get("case"), a, b, answer, shown_first=first)
        self.send_response(303)
        self.send_header("Location", "/")
        self.end_headers()


def serve(lane: str, receipt: dict, port: int = 8765, open_browser=True) -> None:
    Judge.lane = lane
    Judge.pairs = human.pairings(receipt)
    Judge.files = []
    if not Judge.pairs:
        raise ValueError("that receipt has no two candidates sharing a case, "
                         "so there is nothing to compare")
    srv = HTTPServer(("127.0.0.1", port), Judge)
    url = f"http://127.0.0.1:{port}/"
    print(f"judging {len(Judge.pairs)} pairing(s) in the {lane} lane")
    print(f"  {url}\n  ctrl-c when done; answers save as you go")
    if open_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        srv.server_close()
