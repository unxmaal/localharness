"""Score a proposal 1-10 with a cheap local model and a versioned rubric. #54.

The cheapest tier. It reads text and spends no GPU generation, so it can triage
the ~47 proposals a sweep produces before anything is run.

A JUDGE IS A METRIC, and a metric without a negative control is noise. This
project retracted a whole set of speaker-similarity numbers for skipping that
step. `control()` scores items whose real outcome is already known; run it and
check the separation before quoting any score.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from harness import completion

RUBRIC_DIR = Path(__file__).resolve().parent / "rubrics"
DEFAULT_RUBRIC = "novelty"


class JudgeError(RuntimeError):
    """The rubric could not be loaded or the model gave no usable score."""


@dataclass
class Rubric:
    name: str
    version: int
    model: str
    low: int
    high: int
    prompt: str
    #: Machine fragments merged into this rubric, sorted. See _machine_sections.
    fragments: tuple[str, ...] = ()

    @property
    def identity(self) -> str:
        """What makes two judged runs comparable. See evals.core.comparable.

        THE MACHINE IS PART OF THE INSTRUMENT. Fragments are merged at load
        time, so `novelty@5` on a Mac and `novelty@5` on a card are two
        different prompts -- different what_scores_high, different
        what_scores_low -- under one name. A candidate scored 8 on the card and
        4 on the Apple Silicon machine is the system working; the same two scores under one
        identity is a contradiction someone will try to reconcile.
        """
        suffix = f"+{'+'.join(self.fragments)}" if self.fragments else ""
        return f"{self.name}@{self.version}{suffix}"

    @property
    def stamp(self) -> str:
        """The identity as RECORDED, which carries the content digest.

        Kept apart from `identity` on the same argument that keeps
        `cases_digest` a field of its own rather than folded into `case_ids`:
        the readable name is what a person writes and reads, and the digest is
        what two runs are actually compared on.
        """
        return f"{self.identity}#{self.digest}"

    @property
    def digest(self) -> str:
        """The prompt itself, hashed, because a version is a name and a name
        survives every edit to the thing it names.

        Editing what_scores_high in place without touching `version` produces a
        second exam under the first one's identity. That is the defect
        evals.core.cases_digest exists to refuse, sitting in the other
        instrument this project ranks with -- and it matters more now that a
        pod scores a queue unattended, where nobody sees the rubric change.
        """
        import hashlib
        return hashlib.sha256(self.prompt.encode("utf-8")).hexdigest()[:8]


def _machine_sections(machine, directory: Path) -> tuple[dict, tuple[str, ...]]:
    """The rubric fragments for the runtimes this machine has.

    A rubric describes what to reward, and half of that depends on what the
    machine can run: telling a judge that CUDA disqualifies a candidate is
    correct on a Mac and removes the whole point of a box with a card. The
    machine-specific claims live in one file per runtime so that a third kind
    of machine is a file rather than a fork of the rubric, and so that nothing
    here has to ask which operating system it is on.
    """
    if machine is None:
        from harness import machine as _machine
        machine = _machine.detect()
    merged: dict = {}
    used: list[str] = []
    for runtime in sorted(machine.runtimes):
        fragment = Path(directory) / "machine" / f"{runtime}.yaml"
        if not fragment.exists():
            continue
        raw = yaml.safe_load(fragment.read_text(encoding="utf-8")) or {}
        for key, items in raw.items():
            merged.setdefault(key, []).extend(items)
        # Named only when it actually contributed. `cpu` is on every machine
        # and has no fragment, so an identity that listed the runtime SET
        # would differ between two machines whose rubrics are identical.
        used.append(runtime)
    return merged, tuple(used)


def load(name: str = DEFAULT_RUBRIC, directory: Path | None = None,
         machine=None) -> Rubric:
    path = Path(directory or RUBRIC_DIR) / f"{name}.yaml"
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except OSError as exc:
        known = sorted(p.stem for p in Path(directory or RUBRIC_DIR).glob("*.yaml"))
        raise JudgeError(f"no rubric {name!r} at {path} "
                         f"(known: {', '.join(known) or 'none'})") from exc
    sections, fragments = _machine_sections(
        machine, Path(directory or RUBRIC_DIR))
    for key, items in sections.items():
        raw[key] = list(raw.get(key) or []) + list(items)
    scale = raw.get("scale") or [1, 10]
    body = [raw.get("question", "").strip(), ""]
    # Every section a rubric can declare. A key not listed here is SILENTLY
    # DROPPED: a "running here is not merit" section was added to the yaml,
    # never rendered, and the judge went on rewarding exactly what it was
    # written to stop.
    for key, header in (("what_scores_high", "SCORES HIGH"),
                        ("what_scores_low", "SCORES LOW"),
                        ("scores_nothing_by_itself",
                         "COUNTS FOR NOTHING BY ITSELF, because it is true of "
                         "EVERY candidate you will be shown")):
        if raw.get(key):
            body.append(f"{header}:")
            body += [f"- {item.strip()}" for item in raw[key]]
            body.append("")
    body.append(raw.get("instructions", "").strip())
    return Rubric(name=raw.get("name", name), version=int(raw.get("version", 0)),
                  model=raw.get("model", "q3-4b"), low=int(scale[0]),
                  high=int(scale[1]), prompt="\n".join(body).strip(),
                  fragments=fragments)


_SCORE = re.compile(r"\b(10|[1-9])\b")


def parse_score(text: str, low: int = 1, high: int = 10) -> tuple[int, str]:
    """First integer in range, plus the rest as the reasoning."""
    for line in text.strip().splitlines():
        m = _SCORE.search(line)
        if m and low <= int(m.group(1)) <= high:
            score = int(m.group(1))
            why = line[m.end():].strip(" .:-")
            rest = text.strip().splitlines()[1:]
            return score, (why or " ".join(rest)).strip()[:300]
    raise JudgeError(f"no score in {low}-{high} found in: {text.strip()[:160]!r}")


def describe(name: str, why: str = "", source: str = "", times_seen: int = 0,
             relevance: int = 0, inspected: str = "", platform: str = "",
             weights: str = "") -> str:
    """What the model is shown.

    Provenance goes in alongside the prose deliberately: a judge given only a
    description is partly scoring copywriting, and the recap entries this reads
    are one-line marketing blurbs.

    THE INSPECT RESULT GOES IN TOO, and that fixed a real disagreement. The
    judge scored apple/coreai-models 3/10 off a generic description while the
    inspect tier had already cloned it and established it was MLX-native with
    weights that fit. The tier holding more evidence lost to the tier holding
    less. These fields are facts read from a source tree, not prose. Issue #69.
    """
    bits = [f"NAME: {name}"]
    if why:
        bits.append(f"DESCRIPTION: {why}")
    if source:
        bits.append(f"SEEN IN: {source}")
    if times_seen:
        bits.append(f"TIMES SEEN: {times_seen}")
    if relevance:
        bits.append(f"APPLE SILICON RELEVANCE: {relevance:+d}")
    if inspected:
        bits.append(f"READ FROM ITS SOURCE: {inspected}")
    if platform:
        bits.append(f"RUNTIME: {platform}")
    if weights:
        bits.append(f"WEIGHTS IT NAMES: {weights}")
    return "\n".join(bits)


def score(item: str, rubric: Rubric | None = None, *, gateway: str = "",
          complete=None) -> tuple[int, str]:
    """Score one described item. Returns (score, reasoning)."""
    rubric = rubric or load()
    complete = complete or completion.complete
    kw = {"model": rubric.model, "modality": "extract"}
    if gateway:
        kw["gateway"] = gateway
    text = complete(f"{rubric.prompt}\n\n---\n{item}\n---", **kw)
    return parse_score(text, rubric.low, rubric.high)


MENTION_PROMPT = """List every software product, model or tool this text names.

One per line, exactly: NAME | what the text claims about it
Use the name as written. If the text names nothing, reply NONE.
Do not list people, usernames, companies, file formats or programming
languages. Do not invent names that are not in the text.

---
{text}
---"""
#: A reply longer than this is the model narrating rather than listing.
MAX_MENTIONS = 12


def mentions(text: str, *, gateway: str = "", complete=None,
             model: str = "q3-4b") -> list[tuple[str, str]]:
    """Names a piece of freeform prose mentions, with what it claims of each.

    The extractor that reads the monthly recap is shaped for `Name - one-line
    description` and finds nothing in comment English. The best line in the
    thread that prompted this ranked four lanes in one sentence and produced
    zero candidates. Issue #79.

    Returns [] rather than raising: a comment nobody can parse is not an error,
    and one bad reply must not empty a sweep.
    """
    complete = complete or completion.complete
    body = (text or "").strip()
    if not body:
        return []
    kw = {"model": model, "modality": "extract"}
    if gateway:
        kw["gateway"] = gateway
    try:
        reply = complete(MENTION_PROMPT.format(text=body[:4000]), **kw)
    except Exception:  # noqa: BLE001
        return []
    out = []
    for line in (reply or "").splitlines():
        line = line.strip().lstrip("-*0123456789. ").strip()
        if not line or line.upper().startswith("NONE"):
            continue
        name, _, claim = line.partition("|")
        name = name.strip().strip("`\"'")
        # A "name" this long is a sentence, and one this short is punctuation.
        if not (2 <= len(name) <= 60) or name.upper() == "NAME":
            continue
        out.append((name, claim.strip()[:160]))
        if len(out) >= MAX_MENTIONS:
            break
    return out


#: Items whose real outcome is already known here, for calibrating a rubric.
#: Descriptions only, as a feed would carry them: the judge must reach the right
#: order without being told the answer.
#:
#: EVERY ITEM CARRIES THE SAME INSPECT VERDICT, and that is the point. When the
#: inspect result was first shown to the judge, six unrelated candidates all
#: scored 10/10 because "MLX-native, weights fit" was true of every one of
#: them; the control could not see it, because control items carried no inspect
#: fields at all. A control has to be shaped like the data the judge will
#: actually meet, and a fact shared by every item cannot be what separates them.
CONTROL_INSPECTED = "fits: MLX-native, weights that fit this machine"
CONTROL = [
    ("trace",
     "Generate a raster image with a diffusion model, then vectorize it to SVG "
     "paths with vtracer. Composes two tools that are already installed.",
     "won"),
    ("repair",
     "Generate, run the existing checker on the output, then ask the same model "
     "again with the checker's complaint attached. No new model or download.",
     "won"),
    ("upscale-seedvr2",
     "SeedVR2 diffusion super-resolution as a second stage after generation.",
     "lost"),
    ("local-small",
     "A 0.5B instruct model, the smallest in its family, for writing SVG markup.",
     "lost"),
    ("q3-14b",
     "A 14B hybrid reasoning model that emits its thinking separately from its "
     "answer.",
     "lost"),
]


def control_repeated(rubric: Rubric | None = None, *, runs: int = 3,
                     gateway: str = "", complete=None,
                     shape: str = "described") -> dict:
    """Run the control several times and report the SPREAD, not one gap.

    The judge samples and nothing pins a seed, so a single control run is one
    draw from a distribution nobody characterised. Measured on the same rubric
    and the same model: gap +4 in one run and +2 in the next, on identical
    inputs. Near the separates/does-not boundary that verdict is a coin flip,
    and it is the sentence that authorises every score this judge produces.
    Issue #85.

    Separates only if EVERY run separates. One pass out of three is not a pass.
    """
    got = [control(rubric, gateway=gateway, complete=complete, shape=shape)
           for _ in range(max(1, runs))]
    gaps = [g["gap"] for g in got]
    return {"runs": len(got), "gaps": gaps, "gap_min": min(gaps),
            "gap_max": max(gaps), "spread": max(gaps) - min(gaps),
            "separates": all(g["separates"] for g in got),
            "separated_in": sum(1 for g in got if g["separates"]),
            "shape": shape,
            "rubric": got[0]["rubric"], "model": got[0]["model"],
            "rows": got[0]["rows"]}


#: THE SAME QUESTION ASKED OF THE DATA THE STORE ACTUALLY HOLDS: a bare
#: `org/name` with no description, which is what a sweep proposes and what the
#: source tier queues. Outcomes are this project's own measurements.
#:
#: The descriptions above are what the rubric was built against, and a control
#: made only of them cannot see a collapse that happens on bare ids -- which is
#: exactly what #175 found, twenty-five candidates at 3/10 behind a control
#: separating at +7. See RULE #211: a control must be shaped like the data the
#: judge will meet.
CONTROL_BARE = [
    # The engine the STT lane measured and chose, on WER and latency (#57).
    ("mlx-community/parakeet-tdt-0.6b-v2", "won"),
    # What the TTS lane serves.
    ("mlx-community/Kokoro-82M-bf16", "won"),
    # The text model that won the svg three-way at 8/9.
    ("mlx-community/Qwen2.5-7B-Instruct-4bit", "won"),
    # Measured against parakeet in the same lane and lost it.
    ("mlx-community/whisper-large-v3-mlx", "lost"),
    # local-small: 0/9, never closed a tag.
    ("mlx-community/Qwen2.5-0.5B-Instruct-4bit", "lost"),
    # Refused on the working set: 16 GB of weights against a 24 GB ceiling (#12).
    ("mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit", "lost"),
]

#: THE SAME SIX MODELS, described the way the store now describes them: from
#: their own registry card, via inspect.card_description(). Frozen here rather
#: than fetched so a gate never depends on somebody else's afternoon; they were
#: read from the registry on 2026-09-14.
#:
#: This is the shape a tier reading the store must gate on, because it is the
#: shape that tier feeds the judge.
CONTROL_CARDED = [
    ("mlx-community/parakeet-tdt-0.6b-v2",
     "task automatic-speech-recognition; served by mlx; tagged mlx, "
     "automatic-speech-recognition, speech, audio, FastConformer, Conformer; "
     "built from nvidia/parakeet-tdt-0.6b-v2; 2.3 GiB of weights", "won"),
    ("mlx-community/Kokoro-82M-bf16",
     "task text-to-speech; served by mlx; tagged mlx, text-to-speech; "
     "built from yl4579/StyleTTS2-LJSpeech; 0.4 GiB of weights", "won"),
    ("mlx-community/Qwen2.5-7B-Instruct-4bit",
     "task text-generation; served by mlx; tagged mlx, qwen2, chat, "
     "text-generation; built from Qwen/Qwen2.5-7B; 4.0 GiB of weights", "won"),
    ("mlx-community/whisper-large-v3-mlx",
     "task automatic-speech-recognition; served by mlx; tagged mlx, whisper, "
     "automatic-speech-recognition; 2.9 GiB of weights", "lost"),
    ("mlx-community/Qwen2.5-0.5B-Instruct-4bit",
     "task text-generation; served by mlx; tagged mlx, qwen2, chat, "
     "text-generation; built from Qwen/Qwen2.5-0.5B; 0.3 GiB of weights",
     "lost"),
    ("mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit",
     "task text-generation; served by mlx; tagged mlx, qwen3_moe, "
     "text-generation, 4-bit; built from Qwen/Qwen3-30B-A3B-Instruct-2507; "
     "16.0 GiB of weights", "lost"),
]

#: Which control shape a caller means. A tier must gate on the shape of ITS OWN
#: input: the store holds bare ids, so a tier reading the store that gates on
#: the described control is answering a question about different data.
SHAPES = {"described": CONTROL, "bare": CONTROL_BARE,
          "carded": CONTROL_CARDED}


def control(rubric: Rubric | None = None, *, gateway: str = "",
            complete=None, shape: str = "described") -> dict:
    """Score known-good against known-bad and report whether they separate.

    Run this BEFORE quoting any figure from this judge. If the classes overlap,
    the rubric does not discriminate on this data and no ranking may be drawn
    from it, however sensible the scores look individually.

    `shape` picks WHICH known set, and it is not a detail: the two differ only
    in how much the judge is shown, which is the axis the instrument actually
    fails on.
    """
    rubric = rubric or load()
    if shape not in SHAPES:
        raise JudgeError(f"unknown control shape {shape!r}; "
                         f"one of {', '.join(sorted(SHAPES))}")
    rows = []
    for item in SHAPES[shape]:
        name, why, outcome = item if len(item) == 3 else (item[0], "", item[1])
        # THE SAME FIELDS A STORE ROW PRODUCES, so the only thing that varies
        # between the two shapes is the description. A control that also
        # changed the provenance fields would be measuring two things at once.
        s, reason = score(describe(name, why, source="recap", times_seen=1,
                                   inspected=CONTROL_INSPECTED,
                                   platform="MLX-native" if why else ""),
                          rubric, gateway=gateway, complete=complete)
        rows.append({"name": name, "outcome": outcome, "score": s,
                     "why": reason})
    won = [r["score"] for r in rows if r["outcome"] == "won"]
    lost = [r["score"] for r in rows if r["outcome"] == "lost"]
    return {"rubric": rubric.identity, "model": rubric.model, "rows": rows,
            "won_min": min(won), "lost_max": max(lost),
            "gap": min(won) - max(lost),
            "separates": min(won) > max(lost)}
