"""Community aggregation feeds as a discovery source.

A registry says what models exist. It cannot say that everyone moved to a LoRA
that cut generation time 5x. That lives in aggregation posts, and this reads
them. Issue #33.

Proposals only: a subreddit is a popularity signal, never a measurement. Every
byte here is untrusted text written by strangers.
"""
from __future__ import annotations

import html
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path

from harness import paths

ATOM = {"a": "http://www.w3.org/2005/Atom"}

# Reddit serves an HTML block page to anything that looks automated, and returns
# real Atom to a full browser string. Checked by parsing, not by status code.
USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/152.0 Safari/537.36")

#: The field moves weekly and the Reddit feeds are already on that cadence.
#: MUST stay above github.TTL_BY_ENDPOINT_HOURS["/starred"], or a sweep re-reads
#: a cached answer and reports success without looking. Asserted in the tests.
DEFAULT_INTERVAL_DAYS = 7
INTERVAL_ENV = "LOCALHARNESS_DISCOVERY_DAYS"
#: Rapid repeat requests get 429 with an empty body.
RETRIES = 4
RETRY_DELAY = 20.0


class FeedError(RuntimeError):
    """A feed could not be fetched or did not parse as a feed."""


@dataclass
class Source:
    name: str
    url: str
    #: "atom" is a feed of things to try. "releases" is a feed of versions of
    #: something already installed, where the useful signal is drift, not the
    #: repo name -- proposing `ml-explore/mlx` to a project built on MLX is noise.
    kind: str = "atom"
    #: The lane a proposal from here is RECORDED under, and only that. "all"
    #: means the source makes no claim, which `lanes.canonical` turns into no
    #: lane so the candidate's own card decides. Recording a coverage claim as
    #: a candidate's lane put 243 of 323 proposals in a lane that does not
    #: exist (#207), so this field stays a single lane deliberately.
    lane: str = "all"
    #: Lanes this source can SURFACE CANDIDATES FOR, when that is wider than
    #: `lane`. A DIFFERENT QUESTION from the one above, which is why it is a
    #: different field rather than a comma-separated spelling of that one:
    #: mlx-audio is filed `tts` and serves the stt engine too, and putting
    #: "tts,stt" in `lane` would send that string to the recorder. Empty means
    #: "whatever `lane` reaches", which is the answer for every other source.
    reaches: tuple = ()
    enabled: bool = True
    note: str = ""


@dataclass
class Entry:
    title: str
    link: str
    updated: str = ""
    body: str = ""


#: Seeded defaults. Config overrides these; see load_sources().
DEFAULT_SOURCES = [
    Source("reddit-sd-week",
           "https://www.reddit.com/r/StableDiffusion/top/.rss?t=week",
           lane="image",
           note="what the community is actually running this week"),
    Source("reddit-sd-recap",
           "https://www.reddit.com/r/StableDiffusion/search.rss"
           "?q=%22Local+AI+News+You+Missed%22&restrict_sr=1&sort=new&t=year",
           lane="all",
           note="monthly recap series; the whole body rides in <content>"),
    Source("reddit-localllama-week",
           "https://www.reddit.com/r/LocalLLaMA/top/.rss?t=week",
           lane="text",
           note="text lane equivalent"),
    # The reddit feeds carry almost no Apple Silicon signal: measured 0/25 and
    # 1/25 on 2026-09-07 (issue #45). These do, because they are the releases of
    # the stack this project actually runs on.
    Source("mlx-releases", kind="releases", url="https://github.com/ml-explore/mlx/releases.atom",
           lane="all", note="the array framework everything here sits on"),
    Source("mlx-lm-releases", kind="releases",
           url="https://github.com/ml-explore/mlx-lm/releases.atom",
           lane="text", note="the text lane engine"),
    # One server, two lanes. Kokoro speaks through it and parakeet listens
    # through it, so filing it under `tts` alone made `stt` read as a lane
    # with no source while its only source sat right here. #240.
    Source("mlx-audio-releases", kind="releases",
           url="https://github.com/Blaizzy/mlx-audio/releases.atom",
           lane="tts", reaches=("tts", "stt"), note="TTS and STT server"),
    # Not a feed: read through the GitHub API by `lh discover --neighbors`.
    # It lives here so it shares the interval and shows up in --sources, rather
    # than being a source nobody remembers to run. Issue #58.
    Source("github-crowd", kind="crowd",
           url="https://github.com/ (contributors of what this machine runs)",
           lane="all",
           note="what the people who build mlx, mlx-audio and mflux star"),
    Source("mflux-releases", kind="releases",
           url="https://github.com/mflux-community/mflux/releases.atom",
           lane="image",
           note="image lane engine; moved from filipstrand, the old URL 301s"),
    # NO acestep-releases, DELIBERATELY, and the reason is worth keeping.
    # Its feed parses and carries 10 entries, so every check this project has
    # would call it healthy. But a `releases` source reports DRIFT, and
    # ACE-Step tags releases v0.1.8 while the package it installs is version
    # 1.5.0 -- the app and the model generation are numbered separately. So
    # `behind("0.1.8", "1.5.0")` is False forever and the source would propose
    # nothing while looking fine. The music lane is reached through the
    # REGISTRY instead: discover._LANE_QUERIES_ANY["music"]. #240.
    # PROBED BEFORE BEING ADDED, 2026-09-20: 10 entries, whose titles name
    # the coverage -- "New image and video pipelines", "New image and audio
    # pipelines". It is the image AND video engine on the machines with a
    # card, and `video` had no source of any kind before this. Drift is
    # computable here: DIFFUSERS_PIN is in scripts/versions.sh, so
    # installed_version answers 0.40.0 against the feed's 0.40.0. #240.
    Source("diffusers-releases", kind="releases",
           url="https://github.com/huggingface/diffusers/releases.atom",
           lane="video", reaches=("video", "image"),
           note="image and video engine wherever there is a card"),
]


def source_lanes(source) -> tuple:
    """Every lane `source` can surface candidates for.

    ONE FUNCTION, because "what is this filed under" and "what can this find"
    were being answered by reading one field and guessing. Empty means the
    source makes no claim and reaches everything.
    """
    from harness import lanes as L
    if source.reaches:
        return tuple(L.canonical(x) for x in source.reaches)
    lane = L.canonical(source.lane)
    if not lane:
        return ()
    return L.testable_in(lane)


def reach(sources=None) -> dict:
    """lane -> the sources that could surface a candidate for it.

    NOT `harness/coverage.py`, which asks whether a source ACTUALLY surfaced
    the things this machine runs and answered zero of 31 (#99). This asks the
    prior question: could a source reach this lane at all. Two questions, two
    names, because calling both of them coverage is how this project ends up
    filing the same defect five times under different names.

    THROUGH lanes.serves(), which is the whole reason this is a function and
    not a dict comprehension at the call site. One text model serves code,
    web, svg and extract, so a text source covers four lanes; counting by the
    filed lane alone reports web and svg as unreachable and sends somebody
    hunting for feeds about vector graphics that do not exist. That is #207 in
    a new costume.

    A source claiming no lane is left OUT of every row rather than counted for
    all of them: "reddit-sd-recap covers music" is true only in the sense that
    anything might turn up anywhere, and a coverage report that says yes
    everywhere answers nothing.
    """
    from harness import lanes as L
    out = {}
    for lane in L.ALL:
        out[lane] = [s.name for s in (sources or load_sources())
                     if s.enabled and lane in source_lanes(s)]
    return out


def unreachable(sources=None) -> list:
    """Lanes no source can reach. Empty is the state worth holding."""
    return [lane for lane, hits in reach(sources).items() if not hits]

# The vocabulary each runtime is talked about in, and which one is foreign
# depends on the machine asking. A post about a technique names the hardware it
# was measured on, which is what makes this readable at all. Issue #45.
#
# This was one Apple list and one list of "terms that mean it will not run here"
# that included vram -- a word the machine with the card uses about itself. A
# CUDA technique therefore scored negative on the machine it was written for,
# sorted last, and was cut by the limit before anyone saw it.
RUNTIME_TERMS = {
    "mlx": re.compile(
        r"\b(mlx|apple[ -]silicon|metal|macos|mac|coreml|core ?ml|"
        r"unified memory|m[1-9](?:\s*(?:pro|max|ultra))?|neural engine|ane)\b",
        re.I),
    "cuda": re.compile(
        r"\b(cuda|nvidia|rtx|tensorrt|vram|geforce|"
        r"[1-5]0[789]0|a100|h100|xformers)\b", re.I),
    "rocm": re.compile(r"\b(rocm|radeon|hip|instinct|mi[0-9]{3}x?)\b", re.I),
    # A machine with no accelerator has runtimes == {"cpu"}, and without a
    # vocabulary of its own it scored EVERY hardware-naming post negative --
    # the same shape as the vram bug above, one machine further out. The work
    # such a box should actually be shown is exactly this.
    # Deliberately narrow. "quantisation" is not a runtime -- an MLX 4-bit
    # post would score for cpu AND mlx and outrank a technique that names no
    # hardware at all. Only words that mean it runs WITHOUT an accelerator.
    "cpu": re.compile(
        r"\b(gguf|llama[._ -]?cpp|ggml|cpu[- ]only|cpu inference|avx-?[0-9]*)\b",
        re.I),
    # DELIBERATELY OVERLAPS `cpu`, because the two answer different questions.
    # `cpu` asks what an accelerator-less machine should be SHOWN, and the
    # answer really is GGUF work -- without those terms such a box scored
    # every hardware-naming post negative. This asks which RUNTIME a candidate
    # needs, and a GGUF needs llama.cpp wherever it runs: on a card through
    # llama.cpp just as happily, and on a machine without it, not at all
    # whatever the processor. #228.
    "llamacpp": re.compile(r"\b(gguf|llama[._ -]?cpp|ggml)\b", re.I),
    # A SERVING RUNTIME, not hardware, which is why it is its own row rather
    # than part of `cuda`: vllm-mlx exists, so "needs a vLLM server" and
    # "needs a card" are independent questions. Narrow on purpose -- "serving"
    # and "inference" are not runtimes and would score every post about
    # deployment. #245.
    "vllm": re.compile(r"\b(vllm|v-llm|paged[- ]attention|continuous "
                       r"batching)\b", re.I),
}

#: The Apple vocabulary under its old name, for callers that predate the rest.
APPLE_TERMS = RUNTIME_TERMS["mlx"]


def _mentions(pattern: re.Pattern, text: str) -> int:
    """Distinct terms, so one word repeated six times is one mention."""
    return len(set(m.group(0).lower() for m in pattern.finditer(text)))


def relevance(text: str, machine=None) -> int:
    """How much this looks like it runs on THIS machine.

    Positive is a reason to look; negative means it names hardware this machine
    does not have. Zero is the honest default for text that says neither, and a
    technique that names no hardware is not thereby a worse technique.
    """
    if machine is None:
        from harness import machine as _machine
        machine = _machine.detect()
    mine = sum(_mentions(RUNTIME_TERMS[r], text)
               for r in machine.runtimes if r in RUNTIME_TERMS)
    foreign = sum(_mentions(pattern, text)
                  for runtime, pattern in RUNTIME_TERMS.items()
                  if runtime not in machine.runtimes)
    return 2 * mine - foreign


def config_path() -> Path:
    return paths.home() / "discovery-sources.json"


def state_path() -> Path:
    return paths.home() / "discovery-state.json"


def interval_days() -> int:
    raw = os.environ.get(INTERVAL_ENV, "")
    try:
        n = int(raw)
    except ValueError:
        return DEFAULT_INTERVAL_DAYS
    return n if n > 0 else DEFAULT_INTERVAL_DAYS


def load_sources(path: Path | None = None) -> list[Source]:
    """Sources from config, falling back to the seeded defaults."""
    path = path or config_path()
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return list(DEFAULT_SOURCES)
    out = []
    for item in raw.get("sources", []):
        try:
            out.append(Source(**item))
        except TypeError:
            continue
    return out or list(DEFAULT_SOURCES)


def save_sources(sources: list[Source], path: Path | None = None) -> Path:
    path = Path(path or config_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"sources": [asdict(s) for s in sources]},
                               indent=2), encoding="utf-8")
    return path


def _state(path: Path | None = None) -> dict:
    try:
        return json.loads(Path(path or state_path()).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def record_fetch(name: str, when: float | None = None,
                 path: Path | None = None) -> None:
    p = Path(path or state_path())
    s = _state(p)
    s.setdefault("fetched", {})[name] = when if when is not None else time.time()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(s, indent=2), encoding="utf-8")


def last_fetched(name: str, path: Path | None = None) -> float | None:
    return _state(path).get("fetched", {}).get(name)


def staleness(sources: list[Source] | None = None, now: float | None = None,
              days: int | None = None, path: Path | None = None) -> list[dict]:
    """Per source: when it was last read and whether that is too long ago."""
    now = time.time() if now is None else now
    days = interval_days() if days is None else days
    out = []
    for s in sources if sources is not None else load_sources():
        when = last_fetched(s.name, path)
        age = None if when is None else (now - when) / 86400.0
        out.append({"name": s.name, "url": s.url, "enabled": s.enabled,
                    "last_fetched": when, "age_days": age,
                    "interval_days": days,
                    "stale": when is None or age > days})
    return out


def fetch(url: str, timeout: float = 30.0, retries: int = RETRIES,
          delay: float = RETRY_DELAY, sleep=time.sleep) -> str:
    """GET `url` as text, retrying on the rate limit."""
    last = ""
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url,
                                         headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            last = f"HTTP {exc.code}"
            if exc.code not in (429, 500, 502, 503, 504):
                raise FeedError(f"{url}: {last}") from exc
        except (urllib.error.URLError, OSError) as exc:
            last = str(exc)
        except ValueError as exc:
            # Request() rejects a schemeless URL at construction.
            raise FeedError(f"{url}: {exc}") from exc
        if attempt < retries:
            sleep(delay)
    raise FeedError(f"{url}: gave up after {retries + 1} attempts ({last})")


def parse(text: str) -> list[Entry]:
    """Atom entries. A block page parses as HTML, not XML, so it raises here."""
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise FeedError(
            f"not a feed: {exc}. A block page or an error page arrives looking "
            f"like a normal response, so this is the check that catches it") from exc
    # Well-formed is not enough: "<html>...</html>" parses perfectly well.
    tag = root.tag.rsplit("}", 1)[-1].lower()
    if tag not in ("feed", "rss"):
        raise FeedError(f"not a feed: root element is <{tag}>, not <feed>")
    out = []
    for e in root.findall("a:entry", ATOM):
        link = e.find("a:link", ATOM)
        body = e.findtext("a:content", default="", namespaces=ATOM) or ""
        out.append(Entry(
            title=html.unescape(e.findtext("a:title", default="",
                                           namespaces=ATOM) or ""),
            link=(link.get("href", "") if link is not None else ""),
            updated=(e.findtext("a:updated", default="",
                                namespaces=ATOM) or "")[:10],
            body=strip_html(body)))
    return out


def strip_html(raw: str) -> str:
    """Reddit double-escapes the body, so unescape, strip tags, unescape again."""
    text = html.unescape(raw)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


# A recap body is "Name - one sentence." repeated. Names carry digits, dots and
# dashes; the description starts after " - ". Requiring a digit or an internal
# capital keeps ordinary prose out.
_CANDIDATE = re.compile(
    r"([A-Za-z][A-Za-z0-9]*(?:[-_.][A-Za-z0-9]+){1,6})\s+-\s+([A-Z][^.]{5,120}\.)")
_TITLE_NAME = re.compile(r"\b([A-Za-z][A-Za-z0-9]*(?:[-_.][A-Za-z0-9]+){2,6})\b")
# Titles write names with spaces ("MiniMax H3 Lora", "Krea 2"), so a hyphen
# pattern alone found one candidate in twenty-five posts. A version-ish token is
# what makes a product name recognisable in prose.
_TITLE_PRODUCT = re.compile(
    r"\b([A-Z][A-Za-z]{2,}(?:\s+[A-Z][A-Za-z]*)?\s+(?:v?\d+(?:\.\d+)*[A-Za-z]?))\b")
#: A link to a model repo is a VERIFIABLE id, which beats a name lifted from
#: prose. These are the highest-quality proposals a feed produces.
_REPO = re.compile(r"https?://(huggingface\.co|github\.com)/"
                   r"([A-Za-z0-9._-]+/[A-Za-z0-9._-]+)")
#: huggingface.co/blog/... and friends match the owner/name shape but are not
#: repos. "blog/zuanfilm" was offered as a model candidate.
_NOT_AN_OWNER = {"blog", "docs", "papers", "spaces", "datasets", "collections",
                 "models", "tasks", "learn", "join", "pricing", "settings",
                 "orgs", "features", "topics", "trending", "about", "sponsors",
                 "marketplace", "explore", "notifications", "pulls", "issues"}
_STOPWORDS = {"e-mail", "state-of-the-art", "open-source", "text-to-image",
              "image-to-video", "text-to-video", "step-by-step", "so-called"}


@dataclass
class Proposal:
    name: str
    why: str
    source: str
    url: str
    when: str = ""
    kind: str = "candidate"
    #: See relevance(). Higher means more likely to run on this machine.
    relevance: int = 0


#: A comment entry is titled "/u/NAME on <the post's title>". Both halves are
#: traps: the name is a person, and the title is the PARENT post's, repeated on
#: every comment. Reading one thread produced ten usernames and four real
#: candidates before this. Issue #78.
_COMMENT_TITLE = re.compile(r"^/u/([A-Za-z0-9_-]+)\s+on\s+(.*)$", re.S)


def comment_url(permalink: str) -> str:
    """Reddit serves a post's comments as Atom at <permalink>.rss.

    The .json API 403s to everything; this returns 200 with one entry per
    comment, author and body included, and the existing parser reads it
    unchanged.
    """
    return permalink.rstrip("/") + "/.rss"


def comments(permalink: str, fetcher=None) -> list[Entry]:
    """Every comment on one post, as feed entries."""
    return parse((fetcher or fetch)(comment_url(permalink)))


def candidates(entries: list[Entry], source: str = "",
               machine=None) -> list[Proposal]:
    """What a feed is talking about, best evidence first.

    A LINKED repo is a real id and is offered as such. A name lifted from prose
    is a CLAIM about an id and has to be resolved against a registry before
    anyone acts on it -- the same bar `external()` holds a language model to.
    """
    out: dict[str, Proposal] = {}
    # Everyone who wrote a comment. A person is not a candidate, and their
    # handle looks exactly like a product name to the prose extractor.
    authors = {m.group(1).lower() for m in
               (_COMMENT_TITLE.match(e.title) for e in entries) if m}

    def add(name, why, link, when, kind):
        key = name.lower()
        if key in _STOPWORDS or key in out or key in authors:
            return
        why = why.strip()[:160]
        out[key] = Proposal(name, why, source, link, when, kind,
                            relevance(f"{name} {why}", machine))

    # Two passes so a name linked on BOTH huggingface and github is recorded as
    # the model, which is the thing the eval can actually run.
    for want in ("huggingface", "github"):
        for e in entries:
            for host, repo in _REPO.findall(f"{e.body} {e.link}"):
                if want not in host:
                    continue
                if repo.split("/")[0].lower() in _NOT_AN_OWNER:
                    continue
                add(repo, f"linked from: {e.title}", e.link, e.updated,
                    "repo" if want == "huggingface" else "tool")
    for e in entries:
        for name, why in _CANDIDATE.findall(e.body):
            add(name, why, e.link, e.updated, "candidate")
        # A comment's title is the PARENT POST's, repeated on every comment, so
        # mining it once per comment invents the same candidates over and over
        # and attributes them to whoever replied.
        if _COMMENT_TITLE.match(e.title):
            continue
        for name in _TITLE_NAME.findall(e.title):
            add(name, e.title, e.link, e.updated, "candidate")
        for name in _TITLE_PRODUCT.findall(e.title):
            add(" ".join(name.split()), e.title, e.link, e.updated, "candidate")
    return list(out.values())


# Links out of a feed that might themselves be feeds worth following.
_LINK = re.compile(r"https?://[^\s\"'<>)\]]+")
#: Hosts that are media, not places to discover anything.
_SKIP_HOSTS = ("redd.it", "redditstatic", "imgur.com", "youtube.com", "youtu.be",
               "twitter.com", "x.com", "pastebin.com", "i.imgur.com")
#: Already read directly, so pointing at them is not a new source.
_KNOWN_HOSTS = ("huggingface.co", "github.com", "reddit.com")


def candidate_sources(entries: list[Entry],
                      known: list[Source] | None = None) -> list[Proposal]:
    """Places this feed points at that the harness does not read.

    PROPOSED ONLY, never auto-enabled: a source URL lifted from untrusted prose
    is exactly the thing that should need a human nod before the harness starts
    fetching it on a schedule.

    Reported per HOST rather than per link. One civitai link is noise; a host
    that keeps coming up is a hub worth reading.
    """
    have = set()
    for s in known or []:
        have.add(s.url)
        parts = s.url.split("/")
        if len(parts) > 2:
            have.add(parts[2].lower())
    hosts: dict[str, dict] = {}
    for e in entries:
        for url in _LINK.findall(f"{e.body} {e.link}"):
            url = url.rstrip(".,)")
            if url in have:
                continue
            parts = url.split("/")
            if len(parts) < 3:
                continue
            host = parts[2].lower()
            if host in have:
                continue
            if any(h in host for h in _SKIP_HOSTS + _KNOWN_HOSTS):
                continue
            slot = hosts.setdefault(host, {"n": 0, "url": url, "title": e.title,
                                           "when": e.updated})
            slot["n"] += 1
    return [Proposal(host, f"{d['n']} link(s), e.g. from: {d['title'][:70]}",
                     "", d["url"], d["when"], kind="source")
            for host, d in sorted(hosts.items(), key=lambda kv: -kv[1]["n"])]


def probe(url: str, **kw) -> tuple[bool, str]:
    """Does this URL actually serve a parseable feed?"""
    try:
        entries = parse(fetch(url, **kw))
    except FeedError as exc:
        return False, str(exc)[:200]
    if not entries:
        return False, "parses as a feed but has no entries"
    return True, f"{len(entries)} entries"


#: Where a feed lives when the page does not say. Ordered by how often each is
#: the answer, so the first hit is usually the first request.
FEED_PATHS = ("/feed", "/rss", "/feed.xml", "/rss.xml", "/atom.xml",
              "/index.xml", "/feeds/posts/default")

_ALTERNATE = re.compile(
    r'<link\b[^>]*\brel=["\']?alternate["\']?[^>]*>', re.I)
_TYPE_FEED = re.compile(
    r'\btype=["\']?application/(?:rss|atom)\+xml', re.I)
_HREF = re.compile(r'\bhref=["\']([^"\']+)["\']', re.I)


def declared_feeds(html: str, base: str) -> list[str]:
    """Feed URLs a page advertises, the way a browser reads them."""
    out = []
    for tag in _ALTERNATE.findall(html):
        if not _TYPE_FEED.search(tag):
            continue
        href = _HREF.search(tag)
        if href:
            out.append(urllib.parse.urljoin(base, href.group(1)))
    return out


def find_feed(url: str, fetcher=fetch, probe=probe) -> tuple[str, str]:
    """Resolve the feed for a page, returning (feed_url, why).

    A proposed source arrives as ONE ARTICLE LINK lifted from someone's prose,
    which is usually not itself a feed. Ask the page what it advertises, then
    try the conventional paths on its host. Issue #183.
    """
    try:
        text = fetcher(url)
    except FeedError as exc:
        return "", str(exc)[:200]
    # THE URL MAY ALREADY BE THE FEED. Costs nothing -- the body is in hand --
    # and skips guessing at paths for a source that needed no resolving.
    try:
        entries = parse(text)
    except FeedError as exc:
        entries, why_not = [], str(exc)
    else:
        why_not = "parses as a feed but has no entries"
    if entries:
        return url, f"{len(entries)} entries"
    tried = []
    parts = urllib.parse.urlsplit(url)
    root = urllib.parse.urlunsplit((parts.scheme, parts.netloc, "", "", ""))
    for cand in declared_feeds(text, url) + [root + p for p in FEED_PATHS]:
        if cand in tried:
            continue
        tried.append(cand)
        # A SPECULATIVE PATH GETS ONE ATTEMPT. `fetch` defaults to 4 retries
        # 20s apart, which is right for a source we committed to reading and
        # wrong for a guess: seven misses cost nine minutes and timed out the
        # sources report. A 404 on /feed is an answer, not a flaky network.
        ok, why = probe(cand, retries=0)
        if ok:
            return cand, why
    # CARRY WHY THE PAGE ITSELF FAILED. "no feed at 8 paths" says a search
    # happened; it does not say the URL was an HTML article, which is what a
    # reader needs to judge whether the host is worth pursuing by hand.
    return "", f"{why_not}; no feed at {len(tried)} candidate path(s)"


def read(source: Source, cache_dir: Path | None = None,
         ttl_hours: float = 12.0, fetcher=fetch,
         state: Path | None = None) -> list[Entry]:
    """Entries for one source, from a cached copy when it is fresh enough.

    `state` is redirectable for the same reason `cache_dir` is. Without it this
    function wrote half its output to wherever the caller asked and half to the
    real home, so a test that redirected the cache still stamped the user's
    discovery-state.json. Issue #188.
    """
    cache_dir = Path(cache_dir or (paths.home() / "cache" / "feeds"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{re.sub(r'[^A-Za-z0-9_-]', '_', source.name)}.xml"
    if cached.exists() and (time.time() - cached.stat().st_mtime) < ttl_hours * 3600:
        return parse(cached.read_text(encoding="utf-8"))
    text = fetcher(source.url)
    entries = parse(text)          # parse before caching, never cache a block page
    cached.write_text(text, encoding="utf-8")
    record_fetch(source.name, path=state)
    return entries


#: What each releases source tracks, and the package whose installed version it
#: is compared against.
TRACKS = {
    "mlx-releases": "mlx",
    "mlx-lm-releases": "mlx-lm",
    "mlx-audio-releases": "mlx-audio",
    "mflux-releases": "mflux",
    "diffusers-releases": "diffusers",
}

_VERSION = re.compile(r"(\d+\.\d+(?:\.\d+)*)")


def newest_release(entries: list[Entry]) -> str:
    """The highest version in a releases feed, or "" if none parses."""
    best, best_key = "", ()
    for e in entries:
        m = _VERSION.search(e.title)
        if not m:
            continue
        key = tuple(int(x) for x in m.group(1).split("."))
        if key > best_key:
            best, best_key = m.group(1), key
    return best


def installed_version(package: str, pins: Path | None = None) -> str:
    """What this machine actually runs, from the uv tool venv or the pins."""
    from harness import stages
    have = stages.tool_versions()
    if package in have:
        return have[package]
    try:
        import importlib.metadata as md
        return md.version(package)
    except Exception:  # noqa: BLE001
        pass
    # Services install their deps from scripts/versions.sh rather than pyproject.
    path = pins or (Path(__file__).resolve().parent.parent
                    / "scripts" / "versions.sh")
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return ""
    m = re.search(rf'^[A-Z_]+_PIN="{re.escape(package)}(?:\[[^\]]*\])?=='
                  rf'([^"]+)"', text, re.M)
    return m.group(1) if m else ""


def behind(newest: str, have: str) -> bool:
    """Is `have` an older version than `newest`? False if either is unreadable."""
    if not newest or not have:
        return False
    def key(v):
        return tuple(int(x) for x in re.findall(r"\d+", v))
    return key(have) < key(newest)
