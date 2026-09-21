"""A fake world the whole ladder can turn in, in under a second.

Issue #257. Every defect found on 2026-09-20/21 was at a SEAM between two
tiers that each worked alone, which is why 111 test files stayed green through
all of them:

    inspect ordered the queue by corroboration, fetch read it by rank   #249
    ten candidates tied and the tiebreak was alphabetical               #252
    a terminal verdict did not stop re-screening                       #253
    a candidate that cannot load got no verdict at all                 #254
    the loop's inspect step never passed --from-store
    screen.plan was sliced by --top before filtering for ready

Finding those cost a day of real downloads, a live gateway, a live model
server, and a HuggingFace that was rate-limiting. None of it needed to.

THE CARDS ARE RECORDED, NOT INVENTED, and that distinction is the point. A
hand-written fixture encodes what I expected a registry to say; three of the
defects above exist precisely because the real registry said something else.
`lora` sitting seventh in a tag list is not a thing anyone writes on purpose.
See tests/fixtures/registry/, refreshed by scripts/record_fixtures.py.
"""
from __future__ import annotations

import json
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "registry"

#: Recorded card -> the defect it exists to reproduce. A fixture with no reason
#: to be here gets deleted rather than kept for volume.
WHY = {
    "EschaLabs/Qwen3.6-35B-A3B-Escha-W2":
        "a 2-bit MoE mlx_lm cannot load, and the only queued code candidate "
        "carrying published benchmarks (#254)",
    "Xanthius/Ace-Step-1.5-XL-Concept-Sliders":
        "`lora` is the SEVENTH plain tag, so a six-tag summary drops the one "
        "word that decides whether a runner can load it (#241)",
    "OmniSVG/OmniSVG1.1_8B":
        "pipeline_tag text-generation on a model tagged Text-to-SVG three "
        "times: the supertype swallowing the specific one (#246)",
    "prism-ml/bonsai-image-binary-4B-gemlite-1bit":
        "tagged gemlite and cuda, unrunnable on Apple Silicon (#245)",
    "openbmb/MiniCPM5-1B":
        "an ordinary card with none of the above, so a filter that fires on "
        "it is firing on everything",
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers":
        "the video lane had NO spelling at all, and `h3:{model}` -- the "
        "obvious fix -- silently discards the repo id (#264)",
    "nvidia/parakeet-tdt-0.6b-v2":
        "the stt lane's own incumbent, so a filter that refuses it is "
        "refusing the thing this machine already runs (#264)",
    "hexgrad/Kokoro-82M":
        "the tts lane's incumbent, and the upstream of the repo the lane "
        "actually serves, so the card is a requant's parent (#264)",
}

#: Lanes no recorded card can ever land in, and why that is not a gap.
#: lane_for reads ONE lane off a card, and a text model reads as `code`, so a
#: `web` or `extract` fixture cannot exist. Demanding one would re-create
#: RULE #269 exactly: one engine serving four lanes makes three look empty,
#: and I reported that as a discovery hole once already before measuring it.
#: These lanes are covered through the text-served path instead.
SERVED_NOT_SIGHTED = ("web", "extract")


def card(model_id: str) -> dict:
    """One recorded registry response."""
    path = FIXTURES / (model_id.replace("/", "_") + ".json")
    if not path.is_file():
        raise KeyError(
            f"no recorded card for {model_id}. Record one with "
            f"scripts/record_fixtures.py rather than writing it by hand: an "
            f"invented card tests what you expected a registry to say")
    return json.loads(path.read_text(encoding="utf-8"))


def registry(extra: dict | None = None):
    """A `fetch=` stand-in for inspect's HuggingFace calls.

    Raises on an unrecorded id rather than returning an empty card, because a
    silent empty card is how a test passes while measuring nothing.
    """
    extra = extra or {}

    def fetch(url: str) -> str:
        model_id = url.split("/api/models/", 1)[-1].split("?", 1)[0]
        if model_id in extra:
            return json.dumps(extra[model_id])
        return json.dumps(card(model_id))

    return fetch


class Downloads:
    """A `snapshot=` stand-in. Records what was asked for, downloads nothing.

    The order matters as much as the set: the loop fetched the one candidate
    worth having and then screened six others, and only the ORDER showed it.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.asked: list[str] = []

    def __call__(self, *args, **kwargs):
        repo = kwargs.get("repo_id") or (args[0] if args else "")
        self.asked.append(repo)
        where = self.root / repo.replace("/", "--")
        where.mkdir(parents=True, exist_ok=True)
        (where / "config.json").write_text("{}", encoding="utf-8")
        return str(where)


class Completions:
    """A gateway stand-in: canned output per case id, and a record of asks.

    `scores` maps a candidate to what it returns, so a test can make one
    candidate WIN. That is the whole reason this exists: the adopt tier has
    declined four times for real reasons and its winning path has never run
    outside a unit test, and no real model needs to beat anything for the
    write to be exercised.
    """

    def __init__(self, answers: dict[str, str], default: str = ""):
        self.answers = answers
        self.default = default
        self.asked: list[tuple[str, str]] = []

    def __call__(self, model: str, prompt: str, **kwargs) -> str:
        self.asked.append((model, prompt[:40]))
        return self.answers.get(model, self.default)


def seeded_store(conn, rows):
    """Proposals with an inspect verdict queuing them, as a sweep leaves them.

    `rows` are (name, lane, gib, times). Sizes are written in the format the
    fetch tier reads, because a size sitting in a verdict row the reader did
    not look at settled 16 real candidates permanently (#211).
    """
    from harness import memory_store as ms

    for name, lane, gib, times in rows:
        for i in range(max(1, times)):
            ms.record(conn, ms.Seen(name=name, source=f"s{i}", url="", why="",
                                    relevance=0, kind="candidate",
                                    registry=ms.HUGGINGFACE, lane=lane,
                                    resolved=name,
                                    description=f"{gib:.1f} GiB of weights"))
        ms.decide(conn, name, "queued", tier="inspect",
                  detail=f"bytes={int(gib * 1024 ** 3)} fits")
    return conn
