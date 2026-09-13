# Assertions

`assumptions.md` enumerates what this project **believes** without proof. This
enumerates what it **states as true**, in prose a reader will act on.

The two rot differently. An assumption is honestly labelled and stays put. An
assertion was true when it was written, reads as true forever, and usually
carries no date, no configuration, and no way to tell whether anyone has
checked it since.

Phase 1 was the enumeration. Phase 2, below, checked every row: four were
refuted, three were re-measured, and the scanner learned to tell a measurement
from a number so the next pass costs a command rather than an afternoon.

## Method

Mechanical, so it regenerates rather than being curated from memory:

```sh
uv run python -m harness.assertions                 # 684 live claims
uv run python -m harness.assertions --kind NUM      # 173, the decaying kind
uv run python -m harness.assertions --config-less   # 105, the kind that rots
uv run python -m harness.assertions --archives      # include the dated records
```

It reads what `git ls-files` tracks, since that is what a reader can see, and
looks only at **prose** — comments and markdown. A number in an expression is a
value; a number in a sentence is a claim about the world.

`docs/validation-log.md`, `PLAN.md` and `assumptions.md` are excluded by
default. A validation log **should** be full of numbers from one afternoon;
that is what it is for. What rots is a number a reader takes as current.

| kind | count | why it is worth finding |
|---|---|---|
| `NUM` | 173 | measured once, on one machine, in one configuration, and the sentence says none of that |
| `ABS` | 464 | one counter-example ends it |
| `SAYS-MEASURED` | 90 | the promise that someone checked is usually the last record of the checking |
| `CONFIG-LESS` | 105 | a `NUM` with nothing said about the conditions it was taken in |

`CONFIG-LESS` is the phase 2 addition and the one that does the work. It marks
a number whose neighbourhood says nothing about when, on what, at what size or
with which tool — the difference between these two lines, both of which state
11.4 GiB:

```
README.md:459   it holds 11.4 GiB at the default 512x512      qualified
README.md:33    an image, ~19s                                CONFIG-LESS
```

105 of 173 numbers are bare. 68 of those are outside `tests/`, which is the
list worth working through; a test docstring's number justifies a test rather
than telling a reader what to expect.

## States

**verified** — re-checked, with the date and configuration recorded here ·
**dated** — true when taken, provenance recorded, nobody has re-run it ·
**config-less** — a number with no configuration attached, so it cannot be
compared to anything · **unverified** — asserted, never checked ·
**refuted** — checked and false

---

## The category that bit, and why this file exists

**config-less** is not a tidiness complaint. On 2026-09-12 `lh image` was
measured at 23.9 GiB against a recorded 11.4, and two issues (#141, #142) were
filed claiming the record was stale. The record was right. The runs were at
1024x1024, because the CLI passed no resolution and inherited mflux's default,
while every eval case pins 512.

| resolution | wall | peak |
|---|---|---|
| 256x256 | 47.5s | 10.0 GiB |
| 512x512 | 41.0s | **11.4 GiB** |
| 768x768 | 48.4s | 15.6 GiB |
| 1024x1024 | 73.8s | **23.9 GiB** |

Two correct measurements looked like a regression because neither carried its
configuration. `evals.core.comparable()` exists to refuse exactly that, and
cannot help once a number is loose in prose. That was #157, fixed in phase 2:
`DEFAULT_RESOLUTION = 512` now backs `lh image` and `lh video`, every
generation prints the resolution beside its wall time and peak, and
`tests/test_cli_resolution.py` holds the CLI to what the case files pin.

---

## A. What chose each default

The lane table in `harness/cli.py` is the most load-bearing prose in the repo:
it is the recorded justification for five constants that decide what every user
gets.

| id | assertion | where | state |
|---|---|---|---|
| A1 | svg: `local-large` 7/9 at 4.7s beats `q3-14b` 7/9 at 10.0s | `cli.py` | dated |
| A2 | web: `q3-4b` 5/5 at 21.5s beats `local-large` 3/5 at 19.1s | `cli.py` | dated |
| A3 | code: `q3-4b` 6/9 at 2.9s beats `q3-8b` 6/9 at 130s | `cli.py` | dated |
| A4 | extract: `local-large` 9/10 at 0.79s beats `q3-14b` 9/10 at 13.4s | `cli.py` | dated |
| A5 | `q3-14b` answers "reply with exactly: OK" in 152 completion tokens against 2 | `cli.py` | dated, and the reason a better scorer is not the default |
| A6 | `q3-8b` is 0/9 on svg, every run a timeout | `cli.py` | dated |
| A7 | image: `flux2-klein-4b` beat `z-image-turbo` on 19.4s vs 42.6s and 11.4 vs 13.7 GiB | `README.md` | **fixed 2026-09-12** — the README now says 512x512 on an M2 Pro, which is what those numbers are |

**A1-A6 shared one unstated axis and now state it.** The comment block records
the exam: an M2 Pro with 32 GB, `mlx_lm.server` behind the LiteLLM gateway,
`DEFAULT_TEMPERATURE` 0.2, 2026-09-06. What it cannot fix is that the card has
still ranked nothing (#96), and that temperature was never varied (#90).

## B. What the tool costs

Read by anyone deciding whether to invest an afternoon.

| id | assertion | where | state |
|---|---|---|---|
| B1 | `lh image` produces an image in ~19s | `README.md` | **refuted, removed 2026-09-12** — 19s was a 512 figure for a command that ran 1024. The table now says "tens of seconds" and the CLI prints the real one |
| B2 | `image` holds 11.4 GiB | `README.md` | **fixed** — now reads "11.4 GiB at the default 512x512, against 23.9 at 1024" |
| B3 | video is ~40 minutes a generation | `README.md`, `engines.py` | dated — `512x512x22 frames measured at 40.5 minutes` on the M2 Pro, and still the best-formed claim in the repo: the configuration is attached |
| B4 | `lh say` costs about 2s a line | `README.md` | **refuted and re-measured 2026-09-12** — see F7 |
| B5 | models run 4 GB to 31 GB each | `README.md` | **verified and narrowed 2026-09-12** — see below |
| B6 | a request takes VRAM from 2462 to 3296 MiB, returning to 2473 after 15s idle | `README.md` | dated, and the best-formed claim in the file: instrument, direction and recovery all stated |

**B5, measured against the hub cache on 2026-09-12.** The range was roughly
right and the floor was wrong:

| what | on disk |
|---|---|
| Z-Image-Turbo | 30.6 GiB |
| FLUX.2-klein-4B | 14.9 GiB |
| PickScore v1 | 3.7 GiB |
| parakeet-tdt-0.6b-v2 | 2.3 GiB |
| HPSv2 | 1.8 GiB |

4-bit text models are fetched on demand and run from under a gigabyte to about
17, so "4 GB to 31 GB each" understated both ends. The README now says what the
two image engines cost, because that is the number that decides whether the
lane is worth starting.

## C. Gates, ceilings and thresholds

These decide what runs at all, so a wrong one is silent.

| id | assertion | where | state |
|---|---|---|---|
| C1 | 22 GiB is the largest weight this machine can load | `inspect.py` | **fixed** — it is a WEIGHT ceiling, not a job ceiling, and the comment now says so along with both job figures. While the CLI ran 1024 the tool's own default exceeded its own gate |
| C2 | 20 GB free is enough for a normal model | `env.py` | tuned, stated as such |
| C3 | H3 needs its own 160 GB, OmniSVG its own 40 | `env.py` | dated |
| C4 | healthy speech is 0.25-0.56 s/word; a runaway was 3.4-5.3 | `audio.py` | dated, one defect — assumption A7 says the thresholds are ~20x loose (#91) |
| C5 | `lh extract` measured at 0.96s, so it must not queue behind a video | `exclusive.py` | **fixed** — labelled as one run of one prompt on a named machine and date, with the argument restated so it does not rest on the number |

**C5 was this session's own contribution to the problem** and is the useful
kind of fix: the number stays, its provenance is attached, and the sentence now
says what would actually change the decision. Two orders of magnitude is the
claim; a re-measurement would have to find seconds rather than milliseconds.

## D. Claims about other people's software

The most decay-prone class: upstream moves and nothing here notices.

| id | assertion | where | state |
|---|---|---|---|
| D1 | Chromium writes no screenshot with its own `--user-data-dir`; Chrome at the same version is fine | `render.py`, RULE #227 | verified 2026-09-11 by a 2x2 matrix on one runner carrying both |
| D2 | Ubuntu 23.10+ denies the userns Chrome's sandbox needs | `render.py` | verified — read from `/proc` rather than assumed, which is the right shape |
| D3 | tesseract reads this project's generated images 6/18 against Vision and RapidOCR at 18/18 | RULE #224, `ocr.py` | verified 2026-09-11 on 18 real generated images |
| D4 | `mlx_lm.server` hot-swaps models per request through one queue | KNOWLEDGE #90 | dated 2026-09-05, mlx-lm 0.31.3 |
| D5 | `config.cuda.yaml` names four GGUFs that cannot be fetched as written | #147 | verified 2026-09-12 against the HF API |
| D6 | `ubuntu-latest` is 24.04 | `ci.yml` | **now checked on every run** — `check-linux` reads `/etc/os-release` and fails loudly against `EXPECT_UBUNTU` |
| D7 | `/usr/bin/time` is not installed by default on Ubuntu | `proc.py`, KNOWLEDGE #110 | verified on a bare machine, 2026-09-12 |

**D6 was the model case for this whole file** and is now the model fix. CI
parity with 24.04 is the entire reason the distro question was settled that
way, so the morning GitHub moves the label, the job says which claim died and
what the two ways out are, instead of going quietly green on a different
operating system.

## E. Claims about this machine

| id | assertion | where | state |
|---|---|---|---|
| E1 | this machine has 32 GB and a ceiling of 22 GiB | throughout | dated; #96 is the whole point |
| E2 | `iogpu.wired_limit_mb` reads 0, meaning ~70-75% of RAM | KNOWLEDGE #90 | dated 2026-09-05, formula undocumented and version-dependent |
| E3 | the weights volume reads 959 MB/s cold | `env.sh`, validation log | dated 2026-09-05, method recorded |

## F. Refuted, 2026-09-12

Kept because a refuted assertion is the most useful row in the table.

| id | was asserted | by | what is true |
|---|---|---|---|
| F1 | `lh image` runs at 512x512 | me, in #141 and #142 | it inherited mflux's 1024; #157 gave it a declared default of 512 |
| F2 | several pods writing one SQLite file is how you corrupt it | me, in #148 | SQLite serialises writers; the failure is `database is locked`, and this code sets neither WAL nor a busy timeout |
| F3 | Ubuntu 24.04 is required by GitHub runner limitations | another session, reported | nothing gates on a version; the decision was CI parity, and the real LTS argument is NVIDIA packaging |
| F4 | "There is no Linux path" | `README.md`, until #135 | false since #130 |
| F5 | kokoro's path is derived from an assumed clone location | #151 | it is `LOCALHARNESS_HOME`; the defect is which root it hangs off |
| F6 | the image lane regressed from 19.4s to 87s | me, in #142 | different resolutions; ~2x remains at matching resolution, plausibly swap, and still open |
| F7 | `lh say` costs about 2s a line | `README.md` | 2s is the fixed cost per call, not the whole cost. Four words took 4.4s for 1.6s of speech and twenty-eight took 11.3s for 6.2s, so it is ~2s plus about 1.5x the length of the audio. A line is the wrong unit |
| F8 | Kokoro is sub-second | `README.md` | 3.5s for a four-word line through the CLI. Kokoro's synthesis may well be sub-second; what a user waits for is not |

F7 and F8 were measured on an M2 Pro already swapping (19.9 GB of 21.5 GB in
use), so both are upper bounds. The ratio is the durable part and the fixed
cost is what "2s a line" originally was.

## What phase 2 changed

Every row above that says *fixed*, plus:

- **`harness/assertions.py` can now tell a measurement from a number.** That is
  the durable artifact: the next pass over this file is one command, not a
  reading of the repo. The qualifier is deliberately generous — a scanner that
  flags everything gets switched off within a week.
- **`lh image` and `lh video` declare `DEFAULT_RESOLUTION = 512`**, and print
  it beside every timing and peak they report.
- **`tests/test_cli_resolution.py`** holds the CLI to the case files, the fifth
  instance of the parity-test shape in this repo.
- **`check-linux` asserts its own runner.**

## What is left

1. The 68 config-less numbers outside `tests/`, which the scanner will now
   list on demand. Most are in comments explaining a decision, where the fix is
   usually four words.
2. #96, which no amount of prose fixes: every number in section A was taken on
   one machine, and the card has ranked nothing under either operating system.
3. F6, the ~2x image gap at matching resolution, which needs a Mac that is not
   swapping.
4. Whether `--config-less` should ever become a gate like `harness/privacy.py`.
   It should not yet: 105 findings is a backlog, and a gate that starts red
   teaches people to pass `--no-verify`.
