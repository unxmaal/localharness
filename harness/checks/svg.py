"""Is this actually a usable SVG?

Ordered by how badly each failure wastes the reader's time: unparseable is
worthless, an empty canvas is worthless, and a raster in a data: URI is a
bitmap wearing a vector's clothes.
"""
from __future__ import annotations

import re
from xml.etree import ElementTree

from .base import CheckResult, extract

SVG_NS = "{http://www.w3.org/2000/svg}"
# Elements that put marks on the canvas. `defs`, `title`, `metadata` do not.
DRAWING = {"path", "circle", "rect", "ellipse", "line", "polyline", "polygon",
           "text", "use", "image", "g"}


def check(text: str) -> CheckResult:
    src = extract(text, ("svg",))
    if not src.strip():
        return CheckResult(False, "no SVG found in output")

    try:
        root = ElementTree.fromstring(src)
    except ElementTree.ParseError as exc:
        return CheckResult(False, f"XML parse error: {exc}")

    tag = root.tag.replace(SVG_NS, "")
    if tag != "svg":
        return CheckResult(False, f"root element is <{tag}>, not <svg>")

    shapes = [e for e in root.iter()
              if e.tag.replace(SVG_NS, "") in DRAWING and e is not root]
    # A <g> alone draws nothing; require at least one real mark.
    marks = [e for e in shapes if e.tag.replace(SVG_NS, "") != "g"]

    warnings: list[str] = []
    if not root.get("viewBox"):
        warnings.append("no viewBox: will not scale cleanly")
    if not root.get("xmlns") and SVG_NS not in root.tag:
        warnings.append("no xmlns: may not render standalone")
    if re.search(r'data:image/(png|jpe?g|gif|webp)', src, re.I):
        warnings.append("embedded raster image: not really a vector")

    if not marks:
        return CheckResult(False, "SVG draws nothing", warnings, shape_count=0)

    return CheckResult(True, "", warnings, shape_count=len(marks))
