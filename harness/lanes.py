"""One definition of what a lane is, because there were four.

Four modules each decided the set of lanes independently and no two agreed:

    discover.LANES          image stt svg text tts video
    screen.LANE_CANDIDATES  image stt code tts
    rank.LANE_PRIORITY      image code web svg video
    evals/cases/*           image stt svg code tts video web extract

`image` was the only lane in all four, so one lane of five could travel the
whole ladder. `code` could be measured and screened and not searched for;
`web` could be measured and nothing else; `svg` could be searched for and not
screened. Issue #207.

These are four genuinely different questions -- what can be searched for, what
can be screened, what is wanted, what can be measured -- so the fix is not one
table. It is one NAMING, here, plus a test that every table covers it.
"""
from __future__ import annotations

#: Every lane, in the priority order this harness was given, then the ones
#: nobody asked for. A directory under evals/cases must exist for each, and
#: tests/test_harness_lanes.py asserts it.
#:
#: `music` is APPENDED rather than inserted. The five before it were ranked
#: explicitly; music was asked for later and has never been ranked against
#: them, so it is wanted and it is wanted last until somebody says otherwise.
#: Issue #237.
WANTED = ("image", "code", "web", "svg", "video", "music")

#: Measured here, and not on the wanted list. `extract` has more cases than any
#: lane but stt and is a text job; stt and tts have the most measurement
#: history in the project and the person this harness is for does not want them
#: ranked ahead of the five above.
UNWANTED = ("extract", "stt", "tts")

ALL = WANTED + UNWANTED

#: Old name -> the name that survives. `text` and `code` were the same lane:
#: discover.py wrote `text` (10 proposals) and inspect.PIPELINE_LANES mapped
#: `text-generation` to `code` (38 proposals). rank.lane_of then discarded
#: `text` as laneless, so half the lane was silently dropped from the queue.
#:
#: `all` is a SOURCE's coverage claim, never a candidate's lane, and recording
#: it as one put 243 of 323 proposals in a lane that does not exist.
ALIASES = {"text": "code", "all": ""}

#: lane -> (why it is not run here, what would change that). A PARKED lane is
#: a DECISION, not a gap, and the difference is the whole point of this table:
#: `unverified` says nobody has got round to it and invites somebody to,
#: whereas parked says somebody looked and chose, and names the condition
#: under which the choice expires.
#:
#: `video` is NOT blocked on memory. RULE #172 measured h3.c generating on
#: this machine at 9.48 GiB peak with zero swaps, superseding the earlier
#: conclusion that local video needed the Studio. It is blocked on throughput:
#: engines.py records 512x512x22 frames at 40.5 minutes, roughly 0.9 seconds
#: of output per 40 minutes, and verify.COST_S puts it at 2700s against 180
#: for music and 60 for code. So the Studio makes it faster, not newly
#: possible. Issue #244.
PARKED = {
    "video": ("40.5 minutes for 22 frames at 512x512 on this machine, an "
              "order of magnitude past every other lane",
              "the Mac Studio arrives"),
}


def parked(lane) -> tuple[str, str]:
    """(why, until) for a parked lane, or ("", "") for one that is not."""
    return PARKED.get(canonical(lane), ("", ""))


#: Lanes a text model serves through mlx_lm.server, which takes the request's
#: `model` as a live repo id and swaps to it. One prompt, one completion, a
#: structural check on the output. The four differ in what they ask for and not
#: in what runs them, which is why they can share both a registry query and a
#: candidate spec.
TEXT_SERVED = ("code", "web", "svg", "extract")

#: Lanes whose winner no program can pick, so a person does. Not a failure to
#: find the right metric: per the 2026 literature there is no per-clip
#: style-similarity metric at all, FAD is distributional and cannot score one
#: clip, and human preference studies are the field's ground truth. This
#: project reached the same wall three times before naming it -- cover
#: similarity (#272), musical quality (KNOWLEDGE #154), and svg aesthetics
#: (KNOWLEDGE #108, "no check can see that OmniSVG's gear is a better-looking
#: gear"). Issue #273.
#:
#: A lane here still runs every programmatic check it has. `music` keeps its
#: sung-lyric WER and its duration adherence; what it gains is a verdict for
#: the part those cannot see.
HUMAN_JUDGED = ("music", "svg")


def human_judged(lane) -> bool:
    """Whether this lane's winner is decided by a person."""
    return canonical(lane) in HUMAN_JUDGED


def canonical(lane) -> str:
    """The surviving name for `lane`. Empty means no lane at all."""
    got = (lane or "").strip().lower()
    return ALIASES.get(got, got)


def known(lane) -> bool:
    """Is this a lane the harness has a name for?"""
    return canonical(lane) in ALL


#: Words in a proposal's own prose that name a lane. Read only when the
#: registry said nothing, because a model card's `pipeline_tag` is the
#: publisher's own answer and this is a guess at it.
#:
#: DELIBERATELY CONSERVATIVE, and the negative control is the point: of 394
#: laneless sightings, 285 carry no lane word at all, and a pattern set that
#: routed those would be worse than the empty lane it replaced. So a term has
#: to name the OUTPUT, not the topic. `model`, `local`, `fast`, `mlx` and
#: `quantized` appear in most of the 285 and are in none of these patterns.
_PROSE = {
    # The model families are here because a card that says "SDXL" or "FLUX"
    # names its output modality as plainly as the word "image" does, and the
    # feeds talk in family names. Kept short and checked against the negative
    # control: adding them moved it by two rows.
    "image": r"\b(images?|photos?|pictures?|txt2img|text[- ]to[- ]image|"
             r"inpaint\w*|upscal\w+|diffusion|sdxl|sd1\.5|flux)\b",
    "video": r"\b(videos?|animat\w+|keyframes?|text[- ]to[- ]video)\b",
    "svg": r"\b(svg|vector graphics?|vectoriz\w+)\b",
    "web": r"\b(web ?pages?|websites?|html|css|front[- ]?end)\b",
    "tts": r"\b(text[- ]to[- ]speech|tts|speech synthesis|voice clon\w+|"
           r"narrat\w+)\b",
    "stt": r"\b(transcri\w+|speech recognition|asr|dictation)\b",
    "code": r"\b(code|coding|programming|language model|llms?|chat|"
            r"instruct\w*|reasoning|agentic|tokens?)\b",
    # Names the OUTPUT, like the rest of this table. `audio` is deliberately
    # absent: it covers tts, stt and music at once, so it would route speech
    # models into the music lane.
    "music": r"\b(music|songs?|instrumentals?|text[- ]to[- ]music|"
             r"music generation|lyrics)\b",
}


def from_prose(text) -> str:
    """The one lane this prose names, or "" when it names none or several.

    AMBIGUITY IS NOT A TIEBREAK. "Juggles a million tokens of text, images and
    video" names three lanes and the honest answer is that a person has to
    read it. Returning the first match would put a video model in the code
    lane, and a wrong lane costs a download and a screen that answers nothing,
    where an empty one costs a line in `lh discover --wanted`.
    """
    import re
    got = [lane for lane, pattern in _PROSE.items()
           if re.search(pattern, str(text or ""), re.I)]
    return got[0] if len(got) == 1 else ""


def testable_in(lane) -> tuple[str, ...]:
    """Every lane a candidate of `lane` can be screened and measured in.

    ONE TEXT MODEL SERVES FOUR LANES. code, web, svg and extract differ in
    what they ask for, not in what runs them: `lh svg` and `lh web` have
    always resolved to the same Qwen the code lane serves, and the svg lane's
    own three-way found a general model writing markup beating both dedicated
    text-to-SVG models (issue #3).

    A proposal carries ONE lane string, so a text candidate was filed under
    `code` and the web and svg lanes read as having zero candidates when they
    had 103. That looked like a reach problem -- no source discusses web page
    design -- and reporting it as one would have sent somebody hunting for
    feeds that do not exist. #207.
    """
    got = canonical(lane)
    if not got:
        return ()
    return TEXT_SERVED if got in TEXT_SERVED else (got,)


def serves(candidate_lane, target_lane) -> bool:
    """Can a candidate filed under `candidate_lane` run `target_lane`'s cases?"""
    return canonical(target_lane) in testable_in(candidate_lane)
