"""The public site: a front page, the lane report and the benchmarks, all from the exports. #561.

Every number on the front page is computed here from the machine exports and
printed beside the machine and date it came from.
"""
from __future__ import annotations

import time
from pathlib import Path

from harness.howitworks import MODEL, REPO_URL, diagram
from harness.howitworks import CSS as FLOW_CSS
PAGES = ("index.html", "reports/index.html", "benchmarks/index.html")

FONTS = ("https://fonts.googleapis.com/css2?family=Silkscreen:wght@400;700"
         "&family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono:wght@400;600"
         "&display=swap")

CSS = """
:root { --coral:#ff6f59; --hot:#ff4f8b; --tang:#ff9f1c; --butter:#ffd84d;
        --turq:#2ec4b6; --pool:#1fb5d6; --deep:#0f7c99;
        --sand:#fff4da; --sand2:#ffe9bd; --ink:#24161f; --dim:#6b5560;
        --line:#24161f; --shadow:#24161f; --on:#24161f; --code:#fbe6bf;
        --sky:linear-gradient(180deg,#ff6f59 0,#ff4f8b 340px,#ff9f1c 620px,#ffd84d 820px,
              #8fe3d4 1150px,#2ec4b6 1500px,#1fb5d6 2100px,#0f7c99 100%);
        --sun1:#fff3a8; --sun2:#ffb627; --sun3:#ff5d73;
        --good:#14593a; --goodbg:#c9f2dc; --warn:#7a4a00; --warnbg:#ffeab0;
        --bad:#8a1f2c; --badbg:#ffd6dc; }
@media (prefers-color-scheme: dark) {
  :root { --coral:#ff6a5c; --hot:#ff3fa4; --tang:#ff8c2b; --butter:#ffd23f;
          --turq:#2de2e6; --pool:#3aa8ff; --deep:#0b3d4a;
          --sand:#2a1d3d; --sand2:#3a2852; --ink:#fbeedd; --dim:#cdb7d6;
          --line:#07040c; --shadow:#07040c; --on:#1a0f22; --code:#1d1430;
          --sky:linear-gradient(180deg,#1b0b3a 0,#4b1366 380px,#9c1f6e 700px,#d9503f 900px,
                #3b1d5c 1150px,#122d4d 1500px,#0b3d4a 100%);
          --sun1:#ffe27a; --sun2:#ff7b39; --sun3:#ff2e88;
          --good:#9bf0bf; --goodbg:#16392a; --warn:#ffd27a; --warnbg:#3d2a0a;
          --bad:#ffa3b2; --badbg:#401624; } }
* { box-sizing:border-box }
html { -webkit-text-size-adjust:100% }
body { margin:0; color:var(--ink); background:var(--sky); background-color:var(--deep);
       font:16px/1.55 "IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
}
body.calm { background:var(--sand2); background-image:radial-gradient(var(--sand) 1.4px,
            transparent 1.5px); background-size:16px 16px }
a { color:inherit }
code, pre, table { font-family:"IBM Plex Mono",ui-monospace,Menlo,Consolas,monospace }
h1, h2, h3, .menubar, .btn, .num, .chip, .ticker { font-family:"Silkscreen",ui-monospace,monospace;
       font-weight:400; letter-spacing:.02em }
.menubar { display:flex; flex-wrap:wrap; align-items:center; gap:.25rem 1.1rem;
           padding:.45rem 16px; background:var(--sand); border-bottom:2px solid var(--line);
           font-size:.85rem; position:relative }
.calm .menubar { box-shadow:0 6px 0 -2px var(--hot), 0 10px 0 -4px var(--tang),
                 0 14px 0 -6px var(--turq) }
.menubar a { text-decoration:none; padding:.1rem .35rem }
.menubar a:hover, .menubar a[aria-current] { background:var(--ink); color:var(--sand) }
.menubar .brand { margin-right:auto }
.ticker { overflow:hidden; border-bottom:2px solid var(--line); background:var(--butter);
          color:var(--on); font-size:.8rem; white-space:nowrap }
.ticker .tape { display:inline-block; padding:.4rem 0; white-space:nowrap }
.ticker span { padding:0 1.4rem }
.ticker span + span::before { content:"*"; margin-right:1.4rem; color:var(--hot) }
main { max-width:64rem; margin:0 auto; padding:2rem 16px 3rem }
.calm main { padding-top:2.4rem }
.win { background:var(--sand); border:2px solid var(--line); box-shadow:6px 6px 0 var(--shadow);
       margin:0 0 1.75rem; min-width:0 }
.bar { display:flex; align-items:center; gap:.6rem; padding:5px 8px;
       border-bottom:2px solid var(--line); background-color:var(--tint, var(--sand));
       background-image:repeating-linear-gradient(var(--line) 0 1px, transparent 1px 4px);
       background-origin:content-box; background-clip:content-box }
.bar .box { flex:none; width:15px; height:15px; border:2px solid var(--line);
            background:var(--sand); box-shadow:0 0 0 3px var(--tint, var(--sand)) }
.bar h2 { margin:0 auto; padding:0 .6rem; background:var(--tint, var(--sand)); color:var(--on);
          font-size:.95rem; line-height:1.5; text-align:center; overflow-wrap:anywhere }
.body { padding:1rem 1.1rem 1.15rem; min-width:0 }
.body > :first-child { margin-top:0 }
.body > :last-child { margin-bottom:0 }
.pink { --tint:var(--hot) } .teal { --tint:var(--turq) } .butter { --tint:var(--butter) }
.sky { --tint:var(--pool) } .coral { --tint:var(--tang) }
.hero .body { position:relative; overflow:hidden; text-align:center; padding:0 0 1.8rem }
.band { background:linear-gradient(180deg,var(--sun3) 0,var(--tang) 55%,var(--butter) 100%);
        padding-bottom:1rem; margin-bottom:1.4rem; border-bottom:2px solid var(--line) }
.scene { position:relative; height:13rem; overflow:hidden }
.rays { position:absolute; left:50%; top:9rem; width:62rem; height:62rem; margin:-31rem 0 0 -31rem;
        background:repeating-conic-gradient(from 0deg, rgba(255,255,255,.22) 0 7deg,
        transparent 7deg 18deg); border-radius:50% }
.sun { position:absolute; left:50%; bottom:-5rem; width:15rem; height:15rem; margin-left:-7.5rem;
       border-radius:50%; border:2px solid var(--line);
       background:linear-gradient(180deg,var(--sun1) 0,var(--sun2) 55%,var(--sun3) 100%);
       -webkit-mask:linear-gradient(#000 0 52%, transparent 52% 56%, #000 56% 64%,
         transparent 64% 69%, #000 69% 76%, transparent 76% 82%, #000 82%);
       mask:linear-gradient(#000 0 52%, transparent 52% 56%, #000 56% 64%,
         transparent 64% 69%, #000 69% 76%, transparent 76% 82%, #000 82%) }
.wave { position:absolute; left:0; bottom:0; width:200%; height:28px; display:block }
.palm { position:absolute; bottom:0; width:72px; height:96px }
.palm.l { left:4% } .palm.r { right:4%; transform:scaleX(-1) }
.hero h1 { font-size:clamp(2.8rem, 13vw, 5.4rem); line-height:1; margin:1.2rem 0 .2rem;
           color:var(--on); text-shadow:4px 4px 0 var(--hot) }
.pun { font-family:"Silkscreen",monospace; font-size:clamp(1.05rem, 4.6vw, 1.6rem);
       margin:0 0 .3rem; color:#fff4da; text-shadow:2px 2px 0 var(--line) }
.expand { font-size:.9rem; margin:0; color:var(--on); font-weight:600 }
.lead { font-size:1.15rem; max-width:38rem; margin:0 auto 1.6rem; padding:0 1rem }
.buttons { display:flex; flex-wrap:wrap; gap:1rem; justify-content:center; padding:0 1rem }
.btn { display:inline-block; padding:.7rem 1.15rem; border:2px solid var(--line);
       box-shadow:4px 4px 0 var(--shadow); background:var(--hot); color:var(--on);
       text-decoration:none; font-size:.95rem }
.btn.alt { background:var(--turq) }
.btn:active { transform:translate(4px,4px); box-shadow:none }
.stats { display:grid; grid-template-columns:repeat(auto-fit, minmax(13rem, 1fr));
         gap:1.4rem; margin:0 0 2rem }
.stats .win { margin:0; transform:rotate(-1.2deg) }
.stats .win:nth-child(2n) { transform:rotate(1.1deg) }
.stats .body { text-align:center }
.num { display:block; font-size:2.5rem; line-height:1.1; margin:.2rem 0 .3rem; color:var(--hot);
       text-shadow:2px 2px 0 var(--line) }
[data-stat=latest] .num { font-size:1.7rem; white-space:nowrap }
.stat-label { display:block; font-weight:600 }
.where { display:block; color:var(--dim); font-size:.8rem; margin-top:.35rem;
         overflow-wrap:anywhere }
.cards { display:grid; grid-template-columns:repeat(auto-fit, minmax(17rem, 1fr));
         gap:1.4rem; margin:0 0 1.75rem }
.cards .win { margin:0 }
.steps { list-style:none; counter-reset:step; padding:0; margin:0;
         display:grid; grid-template-columns:repeat(auto-fit, minmax(14rem, 1fr)); gap:.9rem }
.steps li { counter-increment:step; border:2px solid var(--line); padding:.6rem .75rem;
            background:var(--sand2) }
.steps li::before { content:counter(step) ". "; font-family:"Silkscreen",monospace; color:var(--hot) }
.steps b { font-family:"Silkscreen",monospace; font-weight:400 }
.chips { list-style:none; padding:0; margin:.6rem 0; display:flex; flex-wrap:wrap; gap:.45rem }
.chip { border:2px solid var(--line); padding:.05rem .45rem; background:var(--butter);
        color:var(--on); font-size:.8rem }
.chip.judged { background:var(--turq) }
.chip a { text-decoration:none }
pre { background:var(--code); border:2px solid var(--line); padding:.75rem .9rem;
      overflow-x:auto; font-size:.85rem; line-height:1.45; margin:.4rem 0 1rem }
code { font-size:.92em }
h3 { font-size:.9rem; margin:1.4rem 0 .4rem }
.sub, .note, .dim, .about { color:var(--dim) }
.about { font-size:.82rem; margin-top:.15rem }
.sub { margin:0 0 .5rem }
.note { font-size:.88rem; margin:.5rem 0 0 }
.wide { overflow-x:auto; max-width:100%; margin:.4rem 0 }
table { border-collapse:collapse; width:100%; font-size:.82rem }
th, td { text-align:left; padding:.32rem .55rem; border:1px solid var(--line); vertical-align:top }
th { background:var(--sand2); font-weight:600; white-space:nowrap }
td.num { display:table-cell; font-family:inherit; font-size:inherit; color:inherit;
         text-shadow:none; text-align:right; font-variant-numeric:tabular-nums; margin:0 }
.tag { display:inline-block; padding:0 .35rem; border:1px solid currentColor;
       font-size:.75rem; white-space:nowrap }
.warn { background:var(--warnbg); color:var(--warn) }
.bad { background:var(--badbg); color:var(--bad) }
.good { background:var(--goodbg); color:var(--good) }
tr.changed td, tr.serving td { background:var(--goodbg) }
footer { text-align:center; color:#fff4da; font-size:.85rem; padding:0 16px 2.5rem }
.calm footer { color:var(--dim) }
@media (prefers-reduced-motion: no-preference) {
  .rays { animation:spin 90s linear infinite }
  .sun { animation:haze 5s ease-in-out infinite }
  .wave { animation:roll 9s linear infinite }
  .palm { animation:sway 6s ease-in-out infinite; transform-origin:50% 100% }
  .palm.r { animation-name:sway-r }
  .ticker .tape { animation:tape 40s linear infinite; padding-left:100% }
  .btn:hover, .chip:hover { animation:wiggle .45s ease-in-out }
  @keyframes spin { to { transform:rotate(360deg) } }
  @keyframes haze { 50% { transform:translateY(3px) scaleX(1.015) } }
  @keyframes roll { to { transform:translateX(-50%) } }
  @keyframes sway { 50% { transform:rotate(3deg) } }
  @keyframes sway-r { 0%, 100% { transform:scaleX(-1) } 50% { transform:scaleX(-1) rotate(3deg) } }
  @keyframes tape { to { transform:translateX(-100%) } }
  @keyframes wiggle { 25% { transform:rotate(-3deg) } 75% { transform:rotate(3deg) } }
}
@media (prefers-reduced-motion: reduce) {
  .ticker .tape { padding-left:16px }
}
""" + FLOW_CSS

PALM = ('<svg class="palm {side}" viewBox="0 0 18 24" shape-rendering="crispEdges" '
        'aria-hidden="true"><g fill="var(--line)">'
        '<rect x="8" y="8" width="2" height="16"/><rect x="7" y="14" width="1" height="10"/>'
        '</g><g fill="var(--turq)">'
        '<rect x="2" y="5" width="7" height="2"/><rect x="0" y="7" width="3" height="2"/>'
        '<rect x="9" y="4" width="7" height="2"/><rect x="15" y="6" width="3" height="2"/>'
        '<rect x="5" y="2" width="4" height="2"/><rect x="10" y="1" width="3" height="3"/>'
        '<rect x="3" y="8" width="2" height="3"/><rect x="12" y="7" width="2" height="3"/>'
        '</g><rect x="8" y="5" width="2" height="2" fill="var(--tang)"/></svg>')

WAVE = ('<svg class="wave" viewBox="0 0 200 14" preserveAspectRatio="none" aria-hidden="true">'
        '<path d="M0 7 Q12.5 0 25 7 T50 7 T75 7 T100 7 T125 7 T150 7 T175 7 T200 7 V14 H0Z" '
        'fill="var(--pool)" stroke="var(--line)" stroke-width="1.2"/></svg>')


def _esc(x) -> str:
    from harness.report import _esc as esc
    return esc(x)


def _date(iso) -> str:
    return (iso or "")[:10] or "--"


def window(title: str, body: str, tint: str = "", cls: str = "", tag: str = "div",
           attrs: str = "") -> str:
    """A classic desktop window: striped title bar, close box, sand body."""
    klass = " ".join(c for c in ("win", tint, cls) if c)
    return (f'<{tag}{attrs} class="{klass}"><div class="bar"><span class="box"></span>'
            f'<h2>{title}</h2></div><div class="body">{body}</div></{tag}>')


def page(title: str, body: str, depth: int = 0, here: str = "", now: float | None = None,
         before: str = "") -> str:
    """The shared shell: fonts, theme, menu bar and footer. Table pages get the calm palette."""
    up = "../" * depth
    links = (("", "SoHoT"), ("benchmarks/", "Benchmarks"), ("reports/", "Lane report"))
    menu = "".join(
        f'<a href="{up + href or "./"}"{" class=brand" if not href else ""}'
        f'{" aria-current=page" if href == here else ""}>{label}</a>'
        for href, label in links)
    when = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(time.time() if now is None else now))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(title)}</title>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="{_esc(FONTS)}">
<style>{CSS}</style></head>
<body{' class="calm"' if depth else ''}>
<nav class="menubar">{menu}<a href="{REPO_URL}">Source</a></nav>
{before}<main>
{body}
</main>
<footer>Built {_esc(when)} from each machine's own export.</footer>
</body></html>
"""


# --- the numbers -----------------------------------------------------------

def _where(docs: list[dict], dates=None) -> str:
    """'Mac Studio M5 Ultra 96 GB, 2026-10-06' per machine the number came from."""
    if not docs:
        return "no machine has published"
    return "; ".join(f"{d['machine'].get('label')}, "
                     f"{(dates or {}).get(id(d)) or _date(d.get('generated_at'))}"
                     for d in docs)


def _runs_span(doc: dict) -> str:
    days = sorted(_date(l.get("last_run_at")) for l in doc.get("lanes") or []
                  if l.get("comparison") and l.get("last_run_at"))
    if not days:
        return _date(doc.get("generated_at"))
    return days[0] if days[0] == days[-1] else f"runs {days[0]} to {days[-1]}"


def stats(machines: list[dict]) -> list[dict]:
    """The front page's numbers, each with the machines and dates it was computed from."""
    adopted_docs = [d for d in machines if any(l.get("adopted") for l in d.get("lanes") or [])]
    adopted = {l["lane"] for d in machines for l in d.get("lanes") or [] if l.get("adopted")}
    compared_docs = [d for d in machines if any(l.get("comparison") for l in d.get("lanes") or [])]
    compared = sum(len(l.get("comparison") or []) for d in machines
                   for l in d.get("lanes") or [])
    newest = max(machines, key=lambda d: d.get("generated_at") or "", default=None)
    return [
        {"key": "adopted", "value": str(len(adopted)),
         "label": "lanes with an adopted winner",
         "where": _where(adopted_docs or machines)},
        {"key": "compared", "value": str(compared),
         "label": "candidates compared in each lane's latest run",
         "where": _where(compared_docs or machines,
                         {id(d): _runs_span(d) for d in compared_docs})},
        {"key": "machines", "value": str(len(machines)),
         "label": "machines publishing", "where": _where(machines)},
        {"key": "latest", "value": _date(newest.get("generated_at")) if newest else "--",
         "label": "latest publish",
         "where": _where([newest]) if newest else "no machine has published"},
    ]


def lane_names(machines: list[dict]) -> list[str]:
    """Every lane the harness names, then any an export carries that it does not."""
    from harness import lanes
    out = list(lanes.ALL)
    for d in machines:
        for l in d.get("lanes") or []:
            if l.get("lane") and l["lane"] not in out:
                out.append(l["lane"])
    return out


def ticker_items(machines: list[dict]) -> list[str]:
    """What each adopted lane serves and where, then when each machine last published."""
    items = [f"{l['lane']} serves {l.get('serves_model') or l.get('serves')} on {d['machine'].get('label')}"
             for d in machines for l in d.get("lanes") or [] if l.get("adopted")]
    items += [f"{d['machine'].get('label')} last published {_date(d.get('generated_at'))}"
              for d in machines]
    return items or ["no machine has published yet"]


# --- the front page --------------------------------------------------------

LOOP = (
    ("sweep", "Reads the model registries, the places practitioners talk, and what the "
              "people who build your tools star on GitHub."),
    ("inspect", "Reads each candidate's source or model card with nothing downloaded, and "
                "drops what cannot run on this machine."),
    ("fetch", "Downloads only what is queued, inside its own disk budget, and nothing "
              "more until you say so."),
    ("screen", "One minimal run that answers one question: did it run and emit anything."),
    ("measure", "The lane's full set of cases, paired against the model it would replace."),
    ("adopt", "Replaces the incumbent only on a significant paired win on held-out cases."),
)


def _stat_card(s: dict, tint: str) -> str:
    body = (f'<b class="num">{_esc(s["value"])}</b>'
            f'<span class="stat-label">{_esc(s["label"])}</span>'
            f'<span class="where">{_esc(s["where"])}</span>')
    return window(_esc(s["label"].split()[0]), body, tint, attrs=f' data-stat="{s["key"]}"')


def _ticker(machines: list[dict]) -> str:
    spans = "".join(f"<span>{_esc(t)}</span>" for t in ticker_items(machines))
    return f'<div class="ticker" aria-label="live from the exports"><div class="tape">{spans}</div></div>'


def front(machines: list[dict], now: float | None = None, model: dict | None = None) -> str:
    from harness import gateway, lanes, publish
    tints = ("pink", "teal", "butter", "sky")
    scene = ('<div class="scene" aria-hidden="true"><div class="rays"></div><div class="sun"></div>'
             f'{PALM.format(side="l")}{PALM.format(side="r")}{WAVE}</div>')
    hero = window("SoHoT", (
        f'<div class="band">{scene}<h1>SoHoT</h1>'
        '<p class="pun">So hot right now.</p>'
        '<p class="expand">Self-optimizing Harness of Theseus</p></div>'
        '<p class="lead">It keeps swapping in better local models for each job, '
        'and only on measured evidence.</p>'
        '<div class="buttons"><a class="btn" href="benchmarks/">See the benchmarks</a>'
        '<a class="btn alt" href="reports/">Read the lane report</a></div>'),
        "butter", "hero")
    empty = ('<p class="note">No machine has published yet.</p>' if not machines else "")
    how = window("How it works", (
        "<p>SoHoT runs this loop on its own, on a timer. Each step is cheaper than the next "
        "and only passes on what survived, and every winner becomes the model the next "
        "challenger has to beat. Tap or click a box to read the code behind it.</p>"
        f"{diagram(model)}"), "pink", attrs=' id="how-it-works"')
    cards = "".join(_stat_card(s, tints[i % len(tints)]) for i, s in enumerate(stats(machines)))
    what = window("What is SoHoT?", (
        "<p>Images, video, speech, music, SVG and code, made by models on your own machine. "
        "Like the ship of Theseus, every part of this harness gets replaced over time while "
        "the harness stays itself. The parts are the models serving each lane. SoHoT goes "
        "looking for challengers on its own, measures each one against the model it would "
        "replace on identical cases on your hardware, and swaps it in only on evidence. "
        "Nothing here was adopted because it was popular.</p>"
        "<p>No account, no API key, no per-token bill, and no model retired out from under "
        "you.</p>"), "sky")
    steps = "".join(f"<li><b>{name}</b><br>{_esc(text)}</li>" for name, text in LOOP)
    loop = window("The loop", (
        "<p>Each tier costs more than the last and only sees what survived the one "
        "before, so measurement is spent where it counts.</p>"
        f'<ol class="steps">{steps}</ol>'), "coral")
    chips = "".join(
        f'<li class="chip{" judged" if lanes.human_judged(n) else ""}">{_esc(n)}</li>'
        for n in lane_names(machines))
    judged = ", ".join(lanes.HUMAN_JUDGED)
    aliases = ", ".join(f"<code>{gateway.LANE_ALIAS.format(n)}</code>"
                        for n in gateway.TEXT_LANES)
    features = [
        ("The paired adopt gate", "pink",
         "<p>A challenger and the incumbent run the same held-out cases. The harness works "
         "out how many repeats the paired test needs to see a real gain, and a loss with "
         "too little power is recorded as underpowered, not as no better. Cases are split "
         "by content digest so nothing tunes itself on the cases that decide.</p>"),
        ("Lanes", "butter",
         "<p>A lane is one job with its own cases and checks, and each serves its own "
         f'winner:</p><ul class="chips">{chips}</ul>'
         f"<p class=\"note\">In the teal lanes ({_esc(judged)}) no metric can pick a winner, "
         "so a person votes.</p>"),
        ("One gateway, stable names", "teal",
         "<p>An OpenAI-compatible gateway serves one alias per text lane: "
         f"{aliases}. An adoption re-points the alias to the winner without anyone "
         "editing a config, and without dropping a session in flight. Claude Code can "
         "use it through the Anthropic messages route.</p>"),
        ("A report per machine", "sky",
         "<p>Every machine keeps its own receipts and publishes its own slice, keyed by a "
         "hardware label and never a hostname. Paths are cut, and a privacy scan refuses "
         "any export that still names a person or a place. A machine that has not "
         f"published in {publish.STALE_DAYS:.0f} days is marked stale.</p>"),
        ("The gauntlet", "coral",
         "<p>Every missed defect is traced to the logical failing behind it, which becomes "
         "a generic class with a test shape. Later code is tested against every class "
         "that has bitten, and CI fails new code that trips a detector until it has a "
         "test of its own.</p>"),
        ("It remembers", "pink",
         "<p>Every proposal, sighting and verdict goes into a small database, so a "
         "candidate already answered is not offered again, and one that lost stays "
         "lost until something about it changes.</p>"),
    ]
    feat = "".join(window(_esc(t), b, tint) for t, tint, b in features)
    quick = window("Quick start", (
        "<p>Install the command, <code>soh</code>, into its own environment:</p>"
        f"<pre><code>git clone {REPO_URL}\ncd SoHoT\n"
        "uv tool install --python 3.12 --editable .</code></pre>"
        "<p>Start the engine your machine has. On Apple Silicon:</p>"
        "<pre><code>./scripts/serve-mlx.sh &amp;        # inference engine\n"
        "./scripts/serve-gateway.sh &amp;    # the address clients use</code></pre>"
        "<p>On a machine with an NVIDIA card, one command starts all three:</p>"
        "<pre><code>./scripts/services.sh start     # gateway, text, audio</code></pre>"
        "<p>Then make something, and look for something better:</p>"
        "<pre><code>soh svg \"a settings gear icon\"\n"
        "soh discover --sweep            # read every source family\n"
        "soh discover --queue            # what a screen would teach, best first\n"
        "soh report                      # what each lane serves here, and why</code></pre>"
        "<p>Point Claude Code at the gateway:</p>"
        "<pre><code>ANTHROPIC_BASE_URL=http://&lt;host&gt;:4000 "
        "ANTHROPIC_AUTH_TOKEN=\"$SOHOT_GATEWAY_KEY\" \\\n  claude --model sohot-code</code></pre>"
        f'<p class="note">Everything else is in the <a href="{REPO_URL}#readme">README</a>.</p>'),
        "teal")
    body = (f'{hero}{how}<div class="stats">{cards}</div>{empty}{what}{loop}'
            f'<div class="cards">{feat}</div>{quick}')
    return page("SoHoT", body, 0, "", now, before=_ticker(machines))


# --- the benchmarks --------------------------------------------------------

def _num(x, fmt="{:.2f}") -> str:
    return "--" if x is None else fmt.format(x)


SERVING_ROW = ' class="serving"'


def _bench_table(lane: dict, table: list[dict] | None = None, dated: bool = False) -> str:
    from harness import publish
    rows = []
    for r in lane.get("comparison") or [] if table is None else table:
        tags = []
        serving = not r.get("reference") and r.get("candidate") == lane.get("serves")
        if serving:
            tags.append('<span class="tag good">serving</span>')
        if r.get("reference"):
            tags.append('<span class="tag">reference, not adoptable</span>')
        passed = (f'{r.get("passed")}/{r.get("total")} ' if r.get("total") else "")
        score = (publish.not_run_text(r) if r.get("not_run")
                 else f'{passed}{_num(r.get("pass_rate"))}')
        rows.append(
            f'<tr{SERVING_ROW if serving else ""}>'
            f'<td>{publish.model_html(r)} {" ".join(tags)}</td>'
            f'<td class="num">{_esc(score)}</td>'
            f'<td class="num">{_num(r.get("median_s"))}</td>'
            f'<td class="num">{_num(r.get("first_s"))}</td>'
            f'<td class="num">{_num(r.get("peak_gb"), "{:.1f}")}</td>'
            f'<td>{_esc(publish._metric_text(r.get("metrics")))}</td>'
            + (f'<td>{_date(r.get("run_at"))}</td>' if dated else "") + '</tr>')
    return ('<div class="wide"><table><tr><th>candidate</th><th>pass</th><th>median s</th>'
            '<th>first s</th><th>peak GB</th><th>metrics</th>'
            + ("<th>run</th>" if dated else "") +
            f'</tr>{"".join(rows)}</table></div>')


def _exam_caption(label: str, exam: dict) -> str:
    n = exam.get("runs") or 1
    when = (f'{n} runs, latest {_date(exam.get("run_at"))}' if n > 1
            else f'1 run, {_date(exam.get("run_at"))}')
    reps = exam.get("repeat") or 1
    n_cases = exam.get("cases")
    cases = (f'{n_cases} case{"" if n_cases == 1 else "s"}'
             + (f" x {reps} repeats" if reps > 1 else ""))
    return f'<h3>{_esc(label)} &middot; {cases} &middot; {when}</h3>'


def _lane_tables(label: str, lane: dict) -> str:
    """One table per comparable exam, newest first; an export from before #621 has only its latest run."""
    if "exams" not in lane:
        return (f'<h3>{_esc(label)} &middot; run {_date(lane.get("last_run_at"))}</h3>'
                f'{_bench_table(lane)}')
    return "".join(_exam_caption(label, e) + _bench_table(lane, e.get("rows") or [], True)
                   for e in lane["exams"])


def _has_tables(lane: dict | None) -> bool:
    return bool(lane and (lane.get("exams") or lane.get("comparison")))


def benchmarks(machines: list[dict], now: float | None = None) -> str:
    tints = ("pink", "teal", "butter", "sky", "coral")
    by = [{l["lane"]: l for l in d.get("lanes") or [] if l.get("lane")} for d in machines]
    measured, waiting = [], []
    for n in lane_names(machines):
        (measured if any(_has_tables(b.get(n)) for b in by) else waiting).append(n)
    index = "".join(f'<li class="chip"><a href="#lane-{_esc(n)}">{_esc(n)}</a></li>'
                    for n in measured)
    intro = window("Benchmarks", (
        "<p>Each table is one exam on one machine: every run in it asked the same cases "
        "under the same conditions, and each candidate shows its latest result there. "
        "A changed case set starts a new table, newest first. Reference rows are a "
        "hosted model run as a ceiling and are never adopted. Wall-clock compares only "
        "within one machine; across machines, compare pass rates.</p>"
        + (f'<ul class="chips">{index}</ul>' if index else
           '<p class="note">No machine has published a measure run yet.</p>')), "sky")
    wins = []
    for i, n in enumerate(measured):
        parts = []
        for d, b in zip(machines, by):
            lane = b.get(n)
            if not _has_tables(lane):
                continue
            how = f" (adopted {_esc(lane.get('adopted_how'))})" if lane.get("adopted") else ""
            parts.append(f'{_lane_tables(d["machine"].get("label"), lane)}'
                         f'<p class="note">Serving here: '
                         f'{_esc(lane.get("serves_model") or lane.get("serves")) or "--"}'
                         f'{how}</p>')
        wins.append(window(_esc(n), "".join(parts), tints[i % len(tints)],
                           attrs=f' id="lane-{_esc(n)}"'))
    rest = (window("Not measured yet", "<p>No published measure run for: "
                   + ", ".join(_esc(n) for n in waiting) + ".</p>", "coral")
            if waiting and machines else "")
    return page("SoHoT benchmarks", intro + "".join(wins) + rest, 1, "benchmarks/", now)


# --- the whole site --------------------------------------------------------

def render(machines: list[dict], now: float | None = None) -> dict[str, str]:
    from harness import publish
    now = time.time() if now is None else now
    return {"index.html": front(machines, now),
            "reports/index.html": publish.render_site(machines, now=now),
            "benchmarks/index.html": benchmarks(machines, now)}


def write(machines: list[dict], out) -> list[Path]:
    out = Path(out)
    written = []
    for rel, html in render(machines).items():
        p = out / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(html, encoding="utf-8")
        written.append(p)
    return written
