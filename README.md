# localharness

[![check](https://github.com/unxmaal/localharness/actions/workflows/ci.yml/badge.svg)](https://github.com/unxmaal/localharness/actions/workflows/ci.yml)

**Make pictures, video, speech and code on your own Mac. Nothing leaves the machine.**

One command, `lh`, generates an image, a short video, an SVG icon, a web page,
some code, or speech in a voice you chose — and transcribes what you say back.
No account, no API key, no per-token bill, no rate limit, and no model quietly
retired out from under you.

The part that makes it more than a pile of scripts: **it measures.** Every model
and every method here earned its place by winning a run against the others, on
this hardware, on real cases. When something new appears, `lh discover` proposes
it and the eval suite settles it. Nothing gets adopted because it was trending.

## What it does

| | |
|---|---|
| `lh image "a red fox in falling snow"` | an image, ~19s |
| `lh video "a fox running" --seconds 2` | a video with sound (slow — see below) |
| `lh svg "a settings gear icon"` | a real vector icon |
| `lh web "a landing page for a coffee roaster"` | a self-contained HTML page |
| `lh code "parse an ISO timestamp"` | code, to stdout |
| `lh extract --file build.log "which tests failed?"` | ask a question about a file |
| `lh say "the tests all passed"` | speak it aloud |
| `lh hear --seconds 5` | record and transcribe |
| `lh discover` | what this machine can do that nobody has measured |

Everything lands in `~/localharness/out/`.

## Why bother

**It is yours.** The prompts, the logs you pipe into it, the voice clips — none
of it is uploaded anywhere. For anything touching work, health or family that is
the whole argument.

**It costs nothing to run.** After the download, generating a thousand images
costs electricity.

**It does not rot.** A hosted model changes under you or is deprecated. Weights
on your disk keep behaving the same way in a year.

**It tells you what is actually best.** This is the unusual part. The suite runs
candidates against the same cases and reports pass rates and quality metrics, and
it refuses to rank two runs that were not comparable. Some of what that has
found:

- for SVG, **tracing beats every language model** — draw a raster, then
  vectorize it: 4/4 against 2/6 on the same cases
- **checking your own output and trying again works**: code 20/27 → 24/27, SVG
  6/9 → 9/9
- the fast image model was also the better one, at 19.4s against 42.6s, with
  nothing traded away
- a speaker-similarity metric that looked broken was fine, and the *experiment*
  was wrong

## Try it

```bash
uv tool install --python 3.12 --editable .
./scripts/serve-mlx.sh &        # inference engine
./scripts/serve-gateway.sh &    # the address clients use
lh svg "a settings gear icon"
```

That prints a path. Open it. For speech, also start `./scripts/serve-tts.sh` and
run `lh say "hello"`.

## What it is not

Worth knowing before you invest an afternoon:

- **Apple Silicon only.** It is built on MLX and Metal. There is no Linux or
  Intel path and there is not going to be one.
- **It needs disk.** Weights are tens of gigabytes.
- **Video is slow here.** About 40 minutes a generation on an M2 Pro with 32 GB.
  It works, and it is not something you will use casually. A faster machine is
  the fix.
- **It is a workshop, not a product.** There is no GUI, and some lanes are better
  than others.

---

# Using it

## Voices

`lh voices` lists them. The default is **`fr-male`**, a French-accented English
voice, cloned from a reference clip rather than picked from a table.

That works because Chatterbox clones *across* languages: the reference clip
speaks French, the output speaks English, and the accent comes along with the
voice. No accented-English corpus was needed. It costs about 2s a line.

```bash
lh say "the tests all passed"                     # cloned, ~2s
lh say "the tests all passed" --voice bm_george   # Kokoro, sub-second
```

Use `bm_george` when a line needs to come back immediately.

## Piping

`code` and `extract` print to stdout, because you pipe or read them rather than
open them in a viewer. `extract` reads stdin when given no `--file`:

```bash
make test 2>&1 | lh extract "which test failed, and why?"
```

That is the lane for handing a cheap question to a small model instead of
spending a large one's context on a log.

## Finding what to try next

```bash
lh discover                  # what is here, and what has never been measured
lh discover --gap            # only the gaps
lh discover --external --lane image   # ask the model registries
lh discover --feeds          # read the community feeds
lh discover --sources        # which feeds, last read when
```

`--feeds` reads aggregation posts, because a registry can tell you a model
exists but not that everyone has moved to a LoRA that made generation five times
faster. Feeds are a **popularity signal, never a measurement**: they propose
candidates with a URL and a date, and the eval decides. Names lifted from prose
are checked against the registry before they are offered, because a name a
stranger typed is no more verified than one a language model invented.

Sources live in `~/localharness/discovery-sources.json`. They go stale after 30
days (`$LOCALHARNESS_DISCOVERY_DAYS`), and a stale source is reported by plain
`lh discover` rather than waiting to be asked about. New sources the feeds point
at are proposed and never auto-enabled.

## Serving it to another machine

```bash
./scripts/serve-mcp.sh      # MCP on 0.0.0.0:8899
claude mcp add --transport http localharness http://styx.local:8899/mcp
```

Exposes `svg`, `web`, `code` and `image`, so another machine's agent can borrow
this one's GPU. Every tool shells out to `lh`, so the CLI, the eval suite and the
MCP server run identical commands — which is how the thing being measured stays
the thing that ships.

`image` is queued: it holds 11.4 GiB and the inference server swaps models
through a single queue, so it returns a job id and `job_status` carries the queue
position and the artifact path. Artifacts stay here, in `~/localharness/out/mcp/`.

Video and speech are deliberately not exposed. There is **no authentication** —
this is a house LAN by choice. Set `TTS_HOST=127.0.0.1` on an untrusted network.
DNS-rebinding protection stays on with an allowlist (`MCP_ALLOW`), because that
is a different threat: it needs someone here to open a web page, not the port to
be reachable from outside.

---

# Setup and upkeep

## Install

```bash
uv tool install --python 3.12 --editable .
```

`lh` lands on PATH via `~/.local/bin`, in its own venv. It never touches the
system Python.

**`--python 3.12` is not optional.** `uv tool install` picks an interpreter
silently, and mflux once installed against 3.9 where every entry point died on
`int | None`. `--editable` keeps the checkout as the source of truth.

`lh` finds the weights on its own, by the same free-space rule the service
scripts use. It has to: on PATH it runs with nothing sourced, and an unset
`HF_HOME` sends huggingface_hub off to re-download what is already on the volume.

The install carries no torch. `mlx-whisper` needs it unconditionally, so the
multilingual ear lives in the `whisper` dependency group — 370MB installed
rather than 1.1GB.

## Keeping the services up

Installed as launchd agents, so the machine comes back serving after a reboot:

```bash
./scripts/launchd.sh probe       # can an agent read the weights volume?
./scripts/launchd.sh install
./scripts/launchd.sh status
```

**Run `probe` first, always.** macOS TCC denies `/Volumes` to launchd jobs, and
the failure is horrible unprepared: the volume stats fine, reports free space and
appears in `/Volumes`, so nothing looks wrong until mlx_lm hangs forever inside
`os.listdir` — accepting connections, answering none, logging nothing, at 0% CPU.

The fix is granting Full Disk Access to **`/bin/bash`** (System Settings →
Privacy & Security → Full Disk Access, then Cmd+Shift+G to reach `/bin`). TCC
propagates the grant to child processes. It is broad, and that is the trade: a
dedicated binary keys its grant to a code hash and needs re-granting on every
rebuild, and `uv` is a symlink into `Cellar/uv/<version>/` so a grant to it dies
on the next upgrade. `/bin/bash` is SIP-protected and never moves.

Restart one after editing it:

```bash
launchctl kickstart -k gui/$UID/com.unxmaal.localharness.mcp
```

Or run them by hand, from a terminal that already has the access:

```bash
./scripts/serve-mlx.sh       # inference engine on :8081
./scripts/serve-gateway.sh   # gateway on :4000, the only address clients use
./scripts/serve-tts.sh       # Kokoro TTS + Parakeet STT on :8890
./scripts/smoke.sh           # assert the seam still holds
```

## Where things land

One root, `~/localharness`, overridable with `$LOCALHARNESS_HOME`:

```
out/                        artifacts: images, audio, svg, pages
out/mcp/                    artifacts asked for over MCP
runs/<stamp>-<modality>/    one eval run: artifacts and results.json
logs/                       service stdout and stderr
```

There were four places before this, and one was relative. `lh` runs from
anywhere, so a relative `out/` scattered artifacts into whatever directory the
caller was standing in, and a generation you cannot find is a generation you did
not make.

## Measuring candidates

```bash
uv run python -m evals.run --modality image --out .logs/img \
  --candidates mflux:flux2-klein-4b,mflux:z-image-turbo
```

A candidate is a gateway alias for text, an engine spec for anything that runs as
a process, or `tts:<model>,voice=<name>` for speech. A candidate can also be a
**workflow** rather than a model:

```bash
--candidates trace:mflux:flux2-klein-4b        # raster, then vectorize
--candidates trace-icon:mflux:flux2-klein-4b   # same, tuned small: 3.2x fewer bytes
--candidates repair:q3-4b                      # generate, check, repair
```

`repair` closes a loop that was always available and never used: every checker
here is an automated verifier — it executes generated code, rasterizes SVG,
renders HTML in a browser — and none of them was fed back into generation. Same
model, asked again with the checker's own complaint attached.

Compare two finished runs with `--compare a/results.json b/results.json`. It
refuses runs that are not comparable and names the axis that differs, because
ranking a run from before a sampling change against one from after is comparing
two different exams.

## Developing

```bash
make check     # shellcheck + bash -n + unit tests, no services needed
make test      # unit tests only
make smoke     # end-to-end, REQUIRES the services running
```

CI runs `make check` on an Apple Silicon runner for every push and pull request,
and reports which tests it skipped and why — a silently skipped test reports
green for something it never checked.

Tests are red-proofed by mutation: every guard has been broken deliberately and
the corresponding test confirmed to fail.

Work is tracked in [issues](https://github.com/unxmaal/localharness/issues);
[#16](https://github.com/unxmaal/localharness/issues/16) is the roadmap.
`PLAN.md` holds the reasoning, the issues hold the state, and the issues win when
they disagree. `docs/validation-log.md` holds the evidence behind every claim,
including conclusions that turned out wrong and how they were caught.
`docs/testing.md` records which mutations were run and which two survived.

## Two traps this repo exists to remember

`mlx_lm.server` has no `/v1/responses`, and LiteLLM routes `/v1/messages` there
by default. The opt-out is in both `gateway/config.yaml` and
`scripts/serve-gateway.sh`.

LiteLLM's `/health/readiness` returns 200 before the proxy can serve. Never
conclude anything from a request made in that window, and when restarting, wait
for port 4000 to actually free before probing or you will test the dying process.
