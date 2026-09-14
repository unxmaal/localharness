"""Write a prompt for whatever engine this machine actually runs. Issue #138.

A caller asks for "a prompt for an image like X". They do not know what
`lh image` runs and should not have to.

TWO PROPERTIES, and both are about not keeping a second copy of an answer.

ENGINE RESOLUTION IS SHARED. This asks the same resolver `lh image` asks, so a
lane that carries its own idea of the default cannot drift from it. The failure
that prevents is silent: change the image default and a lane with its own copy
keeps writing fluent prompts aimed at the model you stopped using. No error,
just worse pictures.

PER-ENGINE KNOWLEDGE LIVES IN THE TREE, VERSIONED, beside the code that reads
it -- `harness/prompting/<engine>.yaml`, the same shape as the rubric the judge
reads. The H3 grammar in particular was expensive to reverse-engineer and was
sitting in a knowledge base no running code could reach.
"""
from __future__ import annotations

from pathlib import Path

GUIDES = Path(__file__).resolve().parent / "prompting"


class NoGuide(LookupError):
    """No prompting asset for that engine. Never a claim about the engine."""


def resolved_for(lane: str):
    """What this machine would actually run in that lane, from the SAME
    resolver the generating command uses.

    Returns the engine object, so the caller sees the whole identity --
    `mflux/flux2-klein-4b-q8`, quantisation included -- rather than the bare
    constant. A lane keeping its own copy of the default is the silent failure
    this exists to prevent.
    """
    from harness import cli, engines
    defaults = {"image": cli.DEFAULT_IMAGE_ENGINE,
                "video": cli.DEFAULT_VIDEO_ENGINE}
    spec = defaults.get((lane or "").strip().lower())
    if not spec:
        raise NoGuide(f"no generating lane called {lane!r}; "
                      f"one of {', '.join(sorted(defaults))}")
    return engines.resolve(spec)


def engine_for(lane: str) -> str:
    """The engine FAMILY for a lane: `mflux`, `h3`.

    The guide is about the tool, which is what has a prompt grammar; the model
    after the slash is what varies underneath it.
    """
    return resolved_for(lane).name.partition("/")[0]


def load(engine: str, directory: Path | None = None) -> dict:
    """The prompting asset for an engine."""
    import yaml
    path = Path(directory or GUIDES) / f"{engine}.yaml"
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except OSError as exc:
        raise NoGuide(
            f"no prompting guide for {engine!r} at {path.name}. Writing one is "
            f"the work; guessing at it in prose is what this file exists to "
            f"stop") from exc
    data["identity"] = f"{data.get('engine', engine)}@{data.get('version', 0)}"
    return data


def for_lane(lane: str, directory: Path | None = None) -> dict:
    """The guide for whatever this machine runs in that lane."""
    return load(engine_for(lane), directory)


def instructions(guide: dict) -> str:
    """What a model is told before it writes a prompt for this engine.

    Assembled from the asset rather than written here, so the knowledge has one
    home. A field that says nothing is OMITTED rather than rendered as an empty
    heading: "not established" is worth saying once, in the asset, and repeating
    it as a blank section teaches a reader nothing.
    """
    out = [f"You are writing a prompt for {guide.get('engine', 'this engine')}, "
           f"guide {guide['identity']}."]
    for key, heading in (("responds_to", "WHAT IT RESPONDS TO"),
                         ("markers", "MARKERS"),
                         ("ignores", "WHAT IT IGNORES")):
        value = (guide.get(key) or "").strip()
        if value:
            out.append(f"{heading}: {value}")
    sections = guide.get("sections") or {}
    for name, text in sorted(sections.items()):
        if (text or "").strip():
            out.append(f"STRUCTURE ({name}): {text.strip()}")
    for name, heading in (("constraints", "HARD CONSTRAINTS"),
                          ("unknown", "NOT ESTABLISHED, so do not invent it")):
        items = guide.get(name) or []
        if items:
            out.append(heading + ":\n" + "\n".join(f"  - {i}" for i in items))
    return "\n\n".join(out)
