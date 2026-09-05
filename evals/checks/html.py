"""Is this a usable, self-contained HTML page?

Self-contained matters for a local-first harness: a page that only renders when
a CDN is reachable has failed at the thing it was asked to do.
"""
from __future__ import annotations

import re
from html.parser import HTMLParser

from .base import CheckResult, extract

VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link",
        "meta", "param", "source", "track", "wbr"}
# Elements whose presence in <body> means something was actually rendered.
CONTENT = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "div", "section", "article",
           "ul", "ol", "li", "table", "img", "svg", "canvas", "main", "header",
           "footer", "nav", "form", "button", "a", "span", "pre", "figure"}


class _Parser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.unclosed: list[str] = []
        self.body_content = 0
        self.title = ""
        self._in_title = False
        self._in_body = False

    def handle_starttag(self, tag, attrs):
        if tag == "body":
            self._in_body = True
        if tag == "title":
            self._in_title = True
        if tag not in VOID:
            self.stack.append(tag)
        if self._in_body and tag in CONTENT:
            self.body_content += 1

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag in self.stack:
            while self.stack and self.stack.pop() != tag:
                pass
        else:
            self.unclosed.append(tag)

    def handle_data(self, data):
        if self._in_title:
            self.title += data.strip()


def check(text: str) -> CheckResult:
    src = extract(text, ("html", "!doctype"))
    if not src.strip():
        return CheckResult(False, "no HTML found in output")

    p = _Parser()
    try:
        p.feed(src)
        p.close()
    except Exception as exc:  # noqa: BLE001 - parser can raise anything
        return CheckResult(False, f"HTML parse error: {exc}")

    warnings: list[str] = []
    if not re.match(r"\s*<!doctype", src, re.I):
        warnings.append("missing doctype: triggers quirks mode")
    if p.stack:
        warnings.append(f"unclosed tags: {', '.join(p.stack[:5])}")
    if re.search(r'(?:src|href)\s*=\s*["\']https?://', src, re.I):
        warnings.append("external resource: page is not self-contained")

    if p.body_content == 0:
        return CheckResult(False, "body renders nothing", warnings,
                           has_title=bool(p.title))

    return CheckResult(True, "", warnings, shape_count=p.body_content,
                       has_title=bool(p.title))
