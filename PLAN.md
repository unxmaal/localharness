# localharness: local LLM serving, routing, and evaluation on Apple Silicon

Status: plan, revision 2 (2026-09-05). Claims marked VALIDATED were executed on this
machine; evidence is in `docs/validation-log.md`.

Revision 2 incorporates the stated use case, which is **intelligent routing between
cloud and local models, plus media generation (images and video)**. That changes the
shape of the plan: routing moves from an implementation detail to the product, and a
second service lane appears for media, because media generation does not fit a chat
API.

## 1. Problem statement

The recurring failure is not a bad architecture. It is that the architecture has no
seam, so every new Apple Silicon inference method found on GitHub reads as a reason to
rebuild rather than a candidate to measure. Two things fix that permanently:

1. A stable interface boundary, so swapping an engine costs a config line.
2. A benchmark on real work, so a new method gets measured in twenty minutes instead
   of adopted on impression.

## 2. Hardware and storage

### Machine (VALIDATED 2026-09-05)

| Fact | Value |
|---|---|
| Model | Mac14,12 (M2 Pro Mac mini), 32 GiB unified |
| macOS | 26.5.1, build 25F80 |
| GPU wired limit | `iogpu.wired_limit_mb: 0`, meaning system default |
| MLX Metal backend | available, `Device(gpu, 0)` |

### Storage, resolved

`HF_HOME` is chosen at launch by `scripts/env.sh`, preferring the faster volume and
failing closed if none is usable.

Link speeds, from `ioreg` `"Device Speed"` (3 = SuperSpeed 5 Gbps,
4 = SuperSpeed+ 10 Gbps). Neither drive is Thunderbolt; `SPThunderboltDataType`
reports no device connected on any port.

| Volume | Device | Link | Topology | Free |
|---|---|---|---|---|
| **2TB** | DockCase SSD Enclosure C1P | **Speed=4, 10 Gbps** | direct to host controller | 780 GiB |
| T7 | Samsung PSSD T7 | Speed=3, 5 Gbps | behind a VIA Labs USB3.0 hub | 470 GiB |
| internal | APPLE SSD AP1024Z | NVMe | n/a | 76 GiB at 92% |

**The 2TB is the target.** It has double the link budget and a direct path.

Measured throughput, 1 GiB `dd`:

- T7 cold sequential read: **460 MB/s**, consistent across two different
  pre-existing files. That is 5 Gbps saturation, so the T7 is bus-limited, not
  media-limited. Note the T7 is itself a 10 Gbps device: the VIA hub is halving it,
  and moving it to a direct port should roughly double it.
- Internal write: 1.15 GB/s.
- 2TB: **not fairly measurable today.** Reads of existing files returned 229 MB/s and
  119 MB/s, but every file on it is Time Machine backup data stored with APFS clones
  and heavy fragmentation, so those numbers measure the backup structure rather than
  the drive. A clean write/read test after reclaiming the volume is the real
  measurement. Expect roughly 900 MB/s to 1 GB/s for a decent NVMe at 10 Gbps.

**Blocker on the 2TB, being resolved by the user.** It is an active Time Machine
destination and its root is `root:wheel` with this user not in `wheel`, so writes fail
with `Permission denied`. `scripts/env.sh` therefore probes writability, not just
mount state, and falls back to the T7 until that changes. No code change is needed
when it does: the preference order picks up the 2TB automatically.

Why this matters more than raw capacity: model load time scales directly with it, and
the hot-swap design (section 4) pays that cost on every model switch. A 40 GB model is
roughly 90 s at 460 MB/s against roughly 40 s at 1 GB/s.

**Mount guard.** `scripts/env.sh` refuses to start unless it finds a mounted,
writable model volume, rather than letting `huggingface_hub` silently recreate the
cache and re-download tens of GB onto a volume with no room. An explicit `HF_ROOT` is
validated the same way; it pins a location but does not bypass the check. VALIDATED
across four cases: auto-pick succeeds, explicit-and-writable succeeds, explicit-but-
unwritable fails with exit 1, explicit-but-unmounted fails with exit 1.

**Ollama reclaim, DONE.** All eight stale models (44 GB, 9 to 20 months old) deleted
2026-09-05 with the user's approval; `~/.ollama` is now 1.0 MB. Note `df` did not
move, because local Time Machine snapshots retain the blocks. The space is purgeable
and macOS reclaims it under pressure; `diskutil` already reports 81.4 GB container
free space against `df`'s 76 GiB. To force it: `tmutil deletelocalsnapshots /`.

**Not migrated.** `~/.cache/huggingface` still holds 9.3 GiB (FLUX.1-dev,
FLUX.1-schnell, flux_text_encoders, CLIP, Chatterbox), directly relevant to the media
lane. Moving it waits on a decision to set `HF_HOME` globally rather than per-project,
since it affects `tsatd_video` and anything else assuming the default path.

### Target hardware

M5 Ultra Mac Studio, 96 GB unified, expected around November 2026.

- Weight bytes is approximately `params x bits / 8`. A 70B at 4-bit is about 40 GB
  loaded, before KV cache.
- The macOS default GPU wired limit is roughly 70 to 75% of unified memory. Apple
  does not document the formula and it has changed across releases, so measure rather
  than compute. On 96 GB the default is roughly 67 GB, raisable via
  `sudo sysctl -w iogpu.wired_limit_mb=<mb>`.
- The setting does not survive a reboot, and setting it too aggressively wedges the
  machine, which has already happened here. Leave the OS 8 GB minimum. Persist via a
  LaunchDaemon; `/etc/sysctl.conf` is not reliably honored on modern macOS.

## 3. Architecture

Two lanes, because chat and media have incompatible shapes. Chat is a request/response
of seconds. Media is a job of minutes producing a binary artifact. Forcing both
through one API costs more than it saves.

```
TEXT LANE (built, validated)

  Claude Code        opencode        OpenExecutive        scripts
  (Anthropic)        (OpenAI)        (OpenAICompatible)
        |                |                 |                |
        +----------------+--------+--------+----------------+
                                  |
                   LiteLLM gateway  127.0.0.1:4000
                   - serves /v1/chat/completions AND /v1/messages
                   - ROUTING: aliases, fallback chains, cost and
                     latency strategies, per-model rate limits
                   - logs latency, tokens, cost per request
                                  |
        +-------------------------+-------------------------+
        |                         |                         |
  mlx_lm.server :8081     llama-server :8082         cloud providers
  PRIMARY LOCAL           GGUF-only fallback         Anthropic / OpenAI /
  (hot-swaps models)      (not installed)            OpenRouter
        |                         |
                    $HF_HOME = /Volumes/T7/hf


MEDIA LANE (not built)

  scripts / CLI  ->  job dispatcher  ->  ComfyUI :8188  ->  outputs on T7
                     (async, minutes)     own graph API
                                          not OpenAI-shaped
```

The rule that keeps the text lane swappable: **no client ever learns an engine name
or port.** Clients address aliases on :4000.

### Why the routing use case validates the gateway choice

LiteLLM's router already provides the mechanical half: named fallback chains, retry
policy, `routing_strategy` (latency-based, least-busy, cost-based), and per-deployment
rate limits. Declaring a local deployment and a cloud deployment under one alias, with
the cloud one as fallback, is configuration rather than code.

What it does **not** provide is the interesting half: routing by *task*. Deciding that
a request is cheap classification (local 1.5B), bulk summarization (local mid), or
hard multi-step reasoning (cloud frontier) is a judgment about the prompt, not about
latency or cost. LiteLLM has no hook for that. Realistic options, in ascending order
of effort:

1. **Explicit aliases.** The caller names `fast`, `mid`, or `frontier`. Zero
   machinery, and honest. Start here.
2. **A cheap local classifier in front.** A 0.5B to 1.5B model labels the request and
   picks the tier. This is itself a local-model job, which makes it the second natural
   customer for the eval harness. The measurement that matters is whether the router
   plus a small model beats always-frontier on cost without losing quality, and that
   is exactly what the harness is for.
3. **Learned routing from logged outcomes.** Only worth considering once the gateway
   has logged enough real traffic to train on. Not a starting point.

The plan commits to option 1 now and treats option 2 as the first real experiment
once the harness exists.

### Why media generation is a separate lane

ComfyUI speaks its own API (`POST /prompt`, `GET /history/{id}`, a websocket for
progress) built around graphs. LiteLLM's `/v1/images/generations` maps to hosted
DALL-E-shaped services and cannot express a ComfyUI graph, which is the entire reason
to run ComfyUI. Putting them behind one endpoint would discard the graph and gain
nothing.

The shared piece worth building is not an API shim but a **job dispatcher**, since
image and especially video generation are long async jobs needing queueing, progress,
retry, and artifact management. That is a different service from a chat proxy.

Existing knowledge that applies directly, from the `tsatd_video` work:

- FLUX weights are already in the HF cache and image generation is viable on this
  32 GB machine today.
- MiniMax H3 video needs roughly 40 GB resident and is **not** viable on 32 GB
  (KNOWLEDGE #67). It becomes plausible on the 96 GB Studio. Video is therefore a
  post-Studio phase, and renting stays the correct answer until then.
- `ComfyUI_MiniMax_H3_Extender` is genuinely local, not an API wrapper
  (KNOWLEDGE #70), and its Motion Context feature carries a previous clip's sampled
  latent forward. That is a real continuity capability the hosted API does not expose,
  and it is the reason to revisit local video on the Studio.

### Engine choices, text lane

MLX (`mlx_lm.server`) primary: Apple's own framework, best memory efficiency and
prompt throughput on M-series, toolchain already proven here. llama.cpp second lane
for GGUF-only models and architectures MLX has not ported; not installed yet. Ollama
is not the serving path: it wraps llama.cpp, lags upstream, and hides tuning flags.
It remains installed, now with no models.

## 4. Validated behaviour

All executed 2026-09-05. Versions: uv 0.12.5, mlx 0.32.2, mlx-lm 0.31.3,
litellm 1.99.0.

**MLX generation.** `mlx-community/Qwen2.5-0.5B-Instruct-4bit`: 25.1 tok/s prompt,
784 tok/s generation, 0.336 GB peak, correct output.

**`mlx_lm.server` is OpenAI-compatible.** `/v1/models` and `/v1/chat/completions`
correct. SSE emits proper `chat.completion.chunk` deltas terminated by `data: [DONE]`.

**`mlx_lm.server` hot-swaps models per request.** Significant and mostly
undocumented. The `model` field is treated as a live Hugging Face `repo_id`. Serving
with `--model A` then requesting B loads B, and `/v1/models` afterwards lists both.
Consequences:

- One server process serves the whole local library. No port-per-model.
- A typo triggers a **network fetch**, not a clean error: `"model":"x"` produced a
  404 from `huggingface.co/api/models/x`. The gateway must be the only supplier of
  model ids.
- Switching costs a disk load, roughly 90 s for 40 GB over the T7's USB link.
  Sequence evals by model, never interleaved.

**The gateway serves both API shapes over MLX**, with aliases resolving to different
upstream repo ids.

**Tool calling round-trips on both shapes**, verified on a 1.5B model: OpenAI
`tool_calls[]`, and Anthropic `{"type":"tool_use"}` with `stop_reason: "tool_use"`.

**Weights land on T7.** After wiring `HF_HOME`, a cold start populated
`/Volumes/T7/hf/hub` and all six smoke checks passed.

**Trap 1.** LiteLLM routes `/v1/messages` for any `custom_llm_provider: openai`
deployment through its Responses API adapter to `POST /v1/responses`, which
`mlx_lm.server` does not implement. The opt-out
`use_chat_completions_url_for_anthropic_messages` works **both** as a
`litellm_settings:` YAML key and as `LITELLM_USE_CHAT_COMPLETIONS_URL_FOR_ANTHROPIC_MESSAGES=1`;
controlled A/B confirmed each works alone. The launcher sets both.

**Trap 2, which nearly poisoned this document.** LiteLLM's `GET /health/readiness`
returns 200 while the proxy still cannot serve. Revision 1 asserted that the YAML key
was ignored and only the env var worked, with a plausible source-level mechanism.
That was wrong; it was the readiness race. The red-proof caught it. `scripts/smoke.sh`
now polls a real completion on the route under test, and when restarting, waits for
port 4000 to actually free, or the probe succeeds against the dying process.

### Not yet validated

- Claude Code driven end to end via `ANTHROPIC_BASE_URL`. The endpoint answers
  correctly in isolation; the full agent loop has not been run.
- Any llama.cpp lane, or any cloud deployment in the gateway.
- Throughput above 1.5B on this hardware.
- Concurrent requests against a hot-swapping backend. Suspected to thrash badly when
  two clients want different models. Needs a test before any multi-client use, and it
  is a real risk for the routing use case specifically.
- Prompt caching behaviour across the gateway.
- Anything in the media lane.

## 5. Phases

**Phase 0, reclaim disk. DONE.** 44 GB of stale Ollama models deleted.

**Phase 1, pin the seam. DONE.** Gateway config, launchers, `HF_HOME` on T7 with a
mount guard, and `scripts/smoke.sh` asserting six behaviours. Red-proofed: removing
the opt-out fails exactly the two Anthropic checks and passes the other four.

**Phase 2, the eval harness. The actual deliverable. Not built.**

Without it Phase 1 is only plumbing. Scope:

- `evals/cases/*.yaml`: 20 to 50 prompts from real work already in these repos.
  Available today: Swift diff review from `xdripswift`, Python from `mcm-engine`,
  the rules-adjudication classification job from KNOWLEDGE #10 which is explicitly
  designed for a cheap local model and is the best first customer, and shell/config
  generation.
- `evals/run.py`: runs each case against one or more aliases, recording tokens/sec,
  time to first token, peak resident memory, wall time, and pass or fail against
  per-case assertions.
- Results stored to the mcm KB via `add_knowledge`, so comparisons survive sessions.
- Sequenced by model because of the hot-swap load cost.

Acceptance: pointing it at a newly discovered engine or model produces a comparable
score sheet in under twenty minutes with no code changes.

**Phase 3, routing.** Add a cloud tier to the gateway alongside the local one, with
explicit `fast` / `mid` / `frontier` aliases and fallback chains. Then run the first
real routing experiment: does a small local classifier in front beat always-frontier
on cost without losing quality? That question is only answerable because Phase 2
exists, which is the argument for that ordering. Test concurrency here too, since
routing implies more than one caller.

**Phase 4, wire the clients.** Claude Code via `ANTHROPIC_BASE_URL`, validating the
full agent loop rather than a single completion. opencode as an OpenAI-compatible
provider. OpenExecutive already has `OpenAICompatibleProvider` at its
`providers/provider.py` seam (KNOWLEDGE #79); note that entry's caveat that prompt
caching is load-bearing there and local models will not reproduce it.

**Phase 5, media lane, images.** ComfyUI on :8188 with outputs on T7. FLUX weights
are already cached and this is viable on 32 GB today. Build the job dispatcher here,
where jobs are minutes rather than the tens of minutes video costs.

**Phase 6, Studio migration.** Should be close to a no-op if the seam held. Move
`$HF_HOME`, re-run the eval suite, and treat the score sheet as the migration test.
Only then revisit `iogpu.wired_limit_mb`.

**Phase 7, media lane, video.** Post-Studio by necessity: H3 needs roughly 40 GB
resident against 32 GB available today. Revisit `ComfyUI_MiniMax_H3_Extender` for
Motion Context. Until then, renting remains correct.

## 6. Resolved and open

Resolved 2026-09-05: stale Ollama models deleted; use case is cloud/local routing plus
media generation; privacy is not a constraint, so cloud deployments in the gateway are
fine; `HF_HOME` is `/Volumes/T7/hf`.

Still open:

1. Set `HF_HOME` globally in the shell profile and migrate the existing 9.3 GiB
   cache, or keep it per-project? Global is better for the media lane, which wants
   the FLUX weights, but it affects `tsatd_video` and anything else that assumes the
   default path.
2. Which cloud providers should the gateway carry, and under which alias names?
3. Once the 2TB is writable, benchmark it clean and record the real number. If it
   lands near 1 GB/s, storage stops being the bottleneck until the Studio.
4. Separately, the T7 sits behind a VIA Labs USB3.0 hub that halves it to 5 Gbps.
   Moving it to a direct port is free and roughly doubles it, worth doing regardless
   of which volume ends up holding weights.
