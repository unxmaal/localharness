# Assertions

`assumptions.md` enumerates what this project **believes** without proof. This
enumerates what it **states as true**, in prose a reader will act on.

The two rot differently. An assumption is honestly labelled and stays put. An
assertion was true when it was written, reads as true forever, and usually
carries no date, no configuration, and no way to tell whether anyone has
checked it since.

Phase 1 is this enumeration; phase 2 validates each entry, the same shape
`assumptions.md` used.

## Method

Mechanical, so it regenerates rather than being curated from memory:

```sh
uv run python -m harness.assertions              # 672 live claims
uv run python -m harness.assertions --kind NUM   # the decaying kind
uv run python -m harness.assertions --archives   # include the dated records
```

It reads what `git ls-files` tracks, since that is what a reader can see, and
looks only at **prose** — comments and markdown. A number in an expression is a
value; a number in a sentence is a claim about the world.

`docs/validation-log.md`, `PLAN.md` and `assumptions.md` are excluded by
default. A validation log **should** be full of numbers from one afternoon;
that is what it is for. What rots is a number a reader takes as current.

| kind | count | why it is worth finding |
|---|---|---|
| `NUM` | 165 | measured once, on one machine, in one configuration, and the sentence says none of that |
| `ABS` | 458 | one counter-example ends it |
| `SAYS-MEASURED` | 89 | the promise that someone checked is usually the last record of the checking |

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
1024x1024, because the CLI passes no resolution and inherits mflux's default,
while every eval case pins 512.

| resolution | wall | peak |
|---|---|---|
| 256x256 | 47.5s | 10.0 GiB |
| 512x512 | 41.0s | **11.4 GiB** |
| 768x768 | 48.4s | 15.6 GiB |
| 1024x1024 | 73.8s | **23.9 GiB** |

Two correct measurements looked like a regression because neither carried its
configuration. `evals.core.comparable()` exists to refuse exactly that, and
cannot help once a number is loose in prose. That is #157, and it is the
strongest argument for this file.

---

## A. What chose each default

The lane table in `harness/cli.py:45-73` is the most load-bearing prose in the
repo: it is the recorded justification for five constants that decide what
every user gets.

| id | assertion | where | state |
|---|---|---|---|
| A1 | svg: `local-large` 7/9 at 4.7s beats `q3-14b` 7/9 at 10.0s | `cli.py:47` | dated, 2026-09-06 |
| A2 | web: `q3-4b` 5/5 at 21.5s beats `local-large` 3/5 at 19.1s | `cli.py:48` | dated |
| A3 | code: `q3-4b` 6/9 at 2.9s beats `q3-8b` 6/9 at 130s | `cli.py:62` | dated |
| A4 | extract: `local-large` 9/10 at 0.79s beats `q3-14b` 9/10 at 13.4s | `cli.py:63` | dated |
| A5 | `q3-14b` answers "reply with exactly: OK" in 152 completion tokens against 2 | `cli.py:57` | dated, and the reason a better scorer is not the default |
| A6 | `q3-8b` is 0/9 on svg, every run a timeout | `cli.py:60` | dated |
| A7 | image: `flux2-klein-4b` beat `z-image-turbo` on 19.4s vs 42.6s and 11.4 vs 13.7 GiB | `README.md:70` | **config-less** — verified 2026-09-12 as 512 numbers, but the README does not say so, and `lh image` runs 1024 |

**A1-A6 share one unstated axis.** None records the resolution question, because
text lanes have none — but none records the *gateway*, the *temperature*
(`DEFAULT_TEMPERATURE = 0.2`, assumption A3), or the *machine*, and all were
taken on the M2 Pro. That is #96.

## B. What the tool costs

Read by anyone deciding whether to invest an afternoon.

| id | assertion | where | state |
|---|---|---|---|
| B1 | `lh image` produces an image in ~19s | `README.md:33` | **refuted as written** — 19s is the 512 figure; the CLI defaults to 1024 and takes 73.8s. #157 |
| B2 | `image` holds 11.4 GiB | `README.md:459` | **config-less** — true at 512, 23.9 GiB at the CLI's own default |
| B3 | video is ~40 minutes a generation | `README.md:119`, `engines.py:244` | dated — `512x512x22 frames measured at 40.5 minutes` on the M2 Pro, and unusually good practice: the configuration IS attached |
| B4 | `lh say` costs about 2s a line | `README.md:238` | dated; measured 23.2s for one line on 2026-09-12 with the default cloned voice, so this is a Kokoro number quoted for a Chatterbox default |
| B5 | models run 4 GB to 31 GB each | `README.md:118` | unverified |
| B6 | a request takes VRAM from 2462 to 3296 MiB, returning to 2473 after 15s idle | `README.md:219` | dated, and the best-formed claim in the file: instrument, direction, and recovery all stated |

## C. Gates, ceilings and thresholds

These decide what runs at all, so a wrong one is silent.

| id | assertion | where | state |
|---|---|---|---|
| C1 | 22 GiB is the largest weight this machine can load | `inspect.py:32` | **config-less** — the default image job uses 11.4 at 512 and 23.9 at 1024, so the gate sits between the tool's two configurations |
| C2 | 20 GB free is enough for a normal model | `env.py:32` | tuned, stated as such |
| C3 | H3 needs its own 160 GB, OmniSVG its own 40 | `env.py:36` | dated |
| C4 | healthy speech is 0.25-0.56 s/word; a runaway was 3.4-5.3 | `audio.py:127` | dated, one defect — assumption A7 says the thresholds are ~20x loose (#91) |
| C5 | `lh extract` measured at 0.96s, so it must not queue behind a video | `exclusive.py:39` | **config-less, mine, 2026-09-12** — one run, one prompt, no repeat, and it is now load-bearing for what `EXCLUSIVE` contains |

**C5 is this session's own contribution to the problem.** The reasoning is
sound and the number is a single unlabelled sample quoted in a comment that
justifies a design decision.

## D. Claims about other people's software

The most decay-prone class: upstream moves and nothing here notices.

| id | assertion | where | state |
|---|---|---|---|
| D1 | Chromium writes no screenshot with its own `--user-data-dir`; Chrome at the same version is fine | `render.py`, RULE #227 | verified 2026-09-11 by a 2x2 matrix on one runner carrying both |
| D2 | Ubuntu 23.10+ denies the userns Chrome's sandbox needs | `render.py:147` | verified — read from `/proc` rather than assumed, which is the right shape |
| D3 | tesseract reads this project's generated images 6/18 against Vision and RapidOCR at 18/18 | RULE #224, `ocr.py` | verified 2026-09-11 on 18 real generated images |
| D4 | `mlx_lm.server` hot-swaps models per request through one queue | KNOWLEDGE #90 | dated 2026-09-05, mlx-lm 0.31.3 |
| D5 | `config.cuda.yaml` names four GGUFs that cannot be fetched as written | #147 | verified 2026-09-12 against the HF API |
| D6 | `ubuntu-latest` is 24.04 | `ci.yml:66` | verified 2026-09-12 from a live run; **will silently become false** and nothing checks it |
| D7 | `/usr/bin/time` is not installed by default on Ubuntu | `proc.py`, KNOWLEDGE #110 | verified on a bare machine, 2026-09-12 |

**D6 is the model case for this whole file.** A true, dated, load-bearing claim
about someone else's infrastructure, with no mechanism to notice when it stops
being true.

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
| F1 | `lh image` runs at 512x512 | me, in #141 and #142 | it inherits mflux's 1024; the CLI declares no default |
| F2 | several pods writing one SQLite file is how you corrupt it | me, in #148 | SQLite serialises writers; the failure is `database is locked`, and this code sets neither WAL nor a busy timeout |
| F3 | Ubuntu 24.04 is required by GitHub runner limitations | another session, reported | nothing gates on a version; the decision was CI parity, and the real LTS argument is NVIDIA packaging |
| F4 | "There is no Linux path" | `README.md`, until #135 | false since #130 |
| F5 | kokoro's path is derived from an assumed clone location | #151 | it is `LOCALHARNESS_HOME`; the defect is which root it hangs off |
| F6 | the image lane regressed from 19.4s to 87s | me, in #142 | different resolutions; ~2x remains, plausibly swap |

## What to do with this

1. **A number in prose gets its configuration, or it gets deleted.** B1, B2,
   C1 and C5 are all the same defect and all are one edit each.
2. **D6 wants a check, not a correction.** `runs-on: ubuntu-latest` resolving
   to something else is a one-line assertion in CI.
3. **Every `dated` row needs a date in the text**, not in git blame. A1-A6
   carry none.
4. `harness/assertions.py` should probably grow a rule that flags a `NUM` claim
   in prose with no nearby date or configuration, the way `harness/privacy.py`
   flags a home path. That is the mechanical version of this whole document.
