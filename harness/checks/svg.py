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


#: Attributes that, at zero, make a shape invisible whatever else it has.
_EXTENT_ATTRS = {"rect": ("width", "height"), "circle": ("r",),
                 "ellipse": ("rx", "ry"), "line": (), "path": (),
                 "polyline": (), "polygon": (), "text": ()}


def _has_extent(el) -> bool:
    """False for a shape that cannot mark the canvas at any zoom."""
    name = el.tag.replace(SVG_NS, "")
    attrs = _EXTENT_ATTRS.get(name)
    if not attrs:
        # path/line/polygon have no size attribute; an empty `d` is the
        # equivalent and the only one worth catching cheaply.
        if name in ("path",) and not (el.get("d") or "").strip():
            return False
        return True
    for a in attrs:
        raw = (el.get(a) or "").strip()
        if not raw:
            continue
        try:
            if float(raw.rstrip("px%")) == 0:
                return False
        except ValueError:
            continue
    return True


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
    # NOR DOES A SHAPE WITH NO EXTENT. Eight `<rect width="0" height="0"/>`
    # satisfied `min_shapes: 6` and passed the case. The rasterizer catches a
    # wholly blank document, but not six real marks padded out with eight
    # invisible ones, and the count is what the assertion reads.
    marks = [e for e in marks if _has_extent(e)]

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
