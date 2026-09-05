# Validation log

Every claim in PLAN.md marked VALIDATED traces to a command here. Run on
2026-09-05, macOS 26.5.1 (25F80), Mac14,12 M2 Pro Mac mini, 32 GiB.

Versions at time of validation: uv 0.12.5, mlx 0.32.2, mlx-lm 0.31.3,
litellm 1.99.0, ollama 0.32.1, Homebrew 6.0.18.

## Hardware and OS

    $ sysctl hw.model hw.memsize
    hw.model: Mac14,12
    hw.memsize: 34359738368

    $ sysctl iogpu.wired_limit_mb
    iogpu.wired_limit_mb: 0          # 0 means "system default", not "no GPU memory"

    $ sw_vers
    ProductVersion: 26.5.1   BuildVersion: 25F80

sysctl is present at /usr/sbin/sysctl. Reading needs no privileges; writing needs
sudo and does not survive a reboot.

## Storage

    $ df -h /System/Volumes/Data
    /dev/disk3s5  926Gi  804Gi  77Gi  92%

    $ du -sh ~/.ollama
     44G

    $ du -sh ~/.cache/huggingface
    9.3G                              # FLUX, CLIP, Chatterbox. Not LLM weights.

/Volumes/Raid and /Volumes/External1 both resolve to /dev/disk3s5, the internal Data
volume. They are stale mountpoints, not attached drives. Confirmed by `df` reporting
the same filesystem and by `mount | grep -E 'Raid|External1'` returning nothing.

## External drives (measured after the user reconnected them)

    $ df -h /Volumes/T7 /Volumes/2TB
    /dev/disk53s1   931Gi  461Gi  470Gi  50%  /Volumes/T7
    /dev/disk51s1   1.8Ti  1.1Ti  780Gi  59%  /Volumes/2TB

Link speeds, ioreg "Device Speed" (3 = SuperSpeed 5Gbps, 4 = SuperSpeed+ 10Gbps):

    +-o AppleT8112USBXHCI@00000000
    | +-o DockCase SSD Enclosure C1P@00200000     <- /Volumes/2TB
    |       "Device Speed" = 4                     direct to host controller
    +-o AppleEmbeddedUSBXHCIASMedia3142@08000000
      +-o USB3.0 Hub@08100000  (VIA Labs)
      |     "Device Speed" = 3
      +-o PSSD T7@08200000                         <- /Volumes/T7
            "Device Speed" = 3                     behind the hub

`SPThunderboltDataType` reports "No device connected" on every port, so neither drive
is Thunderbolt.

Throughput, 1 GiB dd:

    T7,  cold read of pre-existing itunes_backup.tar.gz : 459,927,758 B/s (460 MB/s)
    T7,  same file at offset 400M (page cache)          : 1,117,559,080 B/s
    T7,  write                                          : 430,744,510 B/s
    internal, write                                     : 1,152,784,821 B/s
    2TB, read of a 12GB TM backup .img                  : 229,010,986 B/s
    2TB, same file at offset 2G                         : 118,718,271 B/s

The T7 numbers are trustworthy: 460 MB/s cold, reproduced across two different
pre-existing files, which is 5 Gbps saturation. The T7 is a 10 Gbps device, so the
VIA hub is halving it.

### Superseded: the 2TB reclaimed and re-measured

The user repartitioned the 2 TB NVMe into `Backups` and `Models`, two APFS volumes in
one container with 1 TB quotas each. `Models` is not a Time Machine target, so it
mounts `eric:staff` and writable.

Re-measured with 4 GiB dd and a cold read forced by `diskutil unmount` +
`diskutil mount`, so the page cache cannot inflate it:

    Models write     : 4294967296 bytes in 4.238520 secs (1,013,317,690 B/s)
    Models COLD read : 4294967296 bytes in 4.478884 secs (  958,936,935 B/s)
    Models warm read : 1073741824 bytes in 0.390211 secs (2,751,695,426 B/s)  <- cache, not the drive

    T7 write         : 4294967296 bytes in 10.168272 secs (422,389,104 B/s)
    T7 COLD read     : 4294967296 bytes in  9.932652 secs (432,408,917 B/s)

Models is 2.2x the T7 and effectively matches the internal SSD. This vindicates
discarding the earlier 229 MB/s reading rather than recording it: the clean volume is
over four times faster, confirming that number measured Time Machine's fragmented
backup structure and not the drive.

HF_HOME moved to /Volumes/Models/hf; the 1.1 GiB of existing weights were rsynced
from the T7 (82 files, 1,169,990,654 bytes) and all six smoke checks pass on the new
volume.

Guard re-verified by exit code after the switch: auto-pick 0, explicit Models 0,
explicit T7 0, explicit nonexistent 1.

### Original (superseded) reading

The 2TB numbers below are NOT trustworthy as a drive measurement. Every file on that volume
is Time Machine backup data stored with APFS clones and heavy fragmentation, and a
write test is impossible:

    $ dd if=/dev/zero of=/Volumes/2TB/.bench_tmp bs=1m count=1024
    dd: /Volumes/2TB/.bench_tmp: Permission denied

    $ ls -ld /Volumes/2TB
    drwxrwxr-x@ 8 root wheel /Volumes/2TB
    $ id -Gn eric | grep -x wheel   ->  eric NOT in wheel

Time Machine owns the volume root. Re-measure with a clean write/read test once the
user reclaims it.

## MLX backend

    $ uv run python -c "import mlx.core as mx; print(mx.default_device(), mx.metal.is_available())"
    Device(gpu, 0) True

    $ uv run mlx_lm.generate --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
        --prompt "Reply with exactly: OK" --max-tokens 20
    OK
    Prompt: 34 tokens, 25.120 tokens-per-sec
    Generation: 2 tokens, 784.455 tokens-per-sec
    Peak memory: 0.336 GB

## mlx_lm.server OpenAI surface

    $ curl -s localhost:8081/v1/models
    {"object":"list","data":[{"id":"mlx-community/Qwen2.5-0.5B-Instruct-4bit",...}]}

    $ curl -s localhost:8081/v1/chat/completions -d '{...,"max_tokens":10}'
    ...{"message":{"role":"assistant","content":"PONG!"}}...

Streaming, counting prompt, 30 max tokens:

    total lines: 26   data chunks: 11   keepalives: 2
    first: data: {..."object":"chat.completion.chunk"...,"delta":{"content":"1"}}
    last:  data: [DONE]

## Model hot-swap (undocumented, important)

Server launched with `--model mlx-community/Qwen2.5-0.5B-Instruct-4bit`. Request sent
with `"model":"mlx-community/Qwen2.5-1.5B-Instruct-4bit"` returned HTTP 200 answered
by the 1.5B, and afterwards:

    $ curl -s localhost:8081/v1/models
    {"data":[{"id":"...Qwen2.5-1.5B-Instruct-4bit"},{"id":"...Qwen2.5-0.5B-Instruct-4bit"}]}

The `model` field is treated as a live HF repo_id. Sending `"model":"x"` produced:

    {"error": "404 Client Error ... Repository Not Found for url:
     https://huggingface.co/api/models/x/revision/main"}

A bad model id is a network fetch, not a validation error.

## Gateway, both API shapes

    $ curl -s localhost:4000/v1/chat/completions -d '{"model":"local-small",...}'
    ..."content":"GATEWAY-OK"...   usage.prompt_tokens_details.cached_tokens: 28

    $ curl -s localhost:4000/v1/chat/completions -d '{"model":"local-mid",...}'
    model: local-mid | content: Hello! How can I help you today?

Anthropic shape, after the env-var fix (see below):

    $ curl -s localhost:4000/v1/messages -H 'anthropic-version: 2023-06-01' -d '{...}'
    {"type":"message","role":"assistant","content":[{"type":"text","text":"ANTHROPIC-OK"}],
     "stop_reason":"end_turn"}   HTTP=200

Upstream path confirmed in mlx-server.log as `"POST /v1/chat/completions"`, not
/v1/responses.

## Tool calling, both shapes, 1.5B model

OpenAI:

    tool_calls: [{"function":{"name":"get_weather","arguments":"{\"city\": \"Paris\"}"},
                  "id":"fa50aa26-...","type":"function"}]

Anthropic:

    content: [{"type":"tool_use","id":"c3a0f025-...","name":"get_weather",
               "input":{"city":"Paris"}}]
    stop_reason: tool_use

## The LiteLLM /v1/messages 404

Initial failure:

    litellm.NotFoundError: OpenAIException - Not Found. Received Model Group=local-small
    MaskedHTTPStatusError: Client error '404 Not Found' for url
      'http://127.0.0.1:8081/v1/responses'

Source read at
`litellm/llms/anthropic/experimental_pass_through/messages/handler.py`:

    _RESPONSES_API_PROVIDERS: Final = frozenset({"openai"})

    def _should_route_to_responses_api(...):
        if litellm.use_chat_completions_url_for_anthropic_messages:
            return False
        if custom_llm_provider in _RESPONSES_API_PROVIDERS:
            return True

and at `litellm/__init__.py:239`:

    use_chat_completions_url_for_anthropic_messages: bool = bool(
        os.getenv("LITELLM_USE_CHAT_COMPLETIONS_URL_FOR_ANTHROPIC_MESSAGES", False)
    )

### First conclusion, and why it was wrong

Attempt 1 put `use_chat_completions_url_for_anthropic_messages: true` under
`litellm_settings:` in config.yaml. Tested immediately after
`/health/readiness` returned healthy: appeared to FAIL with the identical 404.

Attempt 2 set `LITELLM_USE_CHAT_COMPLETIONS_URL_FOR_ANTHROPIC_MESSAGES=1` in the
launcher environment. Tested after a longer delay: SUCCEEDED, HTTP 200.

Conclusion drawn: "the YAML key is ignored, only the env var works." **This was
wrong.** It survived initial scrutiny because the source read supported it, which is
exactly what made it dangerous.

### The red-proof that falsified it

Restarting the gateway *without* the env var was expected to reproduce the 404. The
first smoke run after restart failed five of six checks including checks that had
nothing to do with Anthropic routing. Re-running the same script seconds later passed
all six. That inconsistency, not the original test, is what exposed the race.

Controlled A/B, each waiting on a real completion rather than `/health/readiness`:

CASE A, no YAML key and no env var:

    $ sed -n '/litellm_settings/,$p' /tmp/cfg_nokey.yaml
    litellm_settings:
      drop_params: true

    HTTP=404
    {"error":{"message":"litellm.NotFoundError: ... Received Model Group=local-small"}}
    caseA.log: for url 'http://127.0.0.1:8081/v1/responses'
    mlx-server.log: "POST /v1/responses HTTP/1.1" 404

CASE B, YAML key present, env var explicitly removed with
`env -u LITELLM_USE_CHAT_COMPLETIONS_URL_FOR_ANTHROPIC_MESSAGES`:

    HTTP=200
    {"type":"message","role":"assistant",
     "content":[{"type":"text","text":"OK, thank you for asking! ..."}],
     "stop_reason":"end_turn"}
    $ grep -c "v1/responses" .logs/caseB.log
    0

**Both mechanisms work.** The YAML key is honored. The env var is honored.

### Standing correction

Never gate a config A/B on `/health/readiness`. Poll a real request on the route
under test and require a successful body before concluding anything. Then run the
negative case and confirm the failure returns. See KB rule "LiteLLM
/health/readiness lies".

### The env var's bool() quirk (unchanged, still true)

`litellm/__init__.py:239` reads the flag as `bool(os.getenv(...))`, so any non-empty
value is true, including the string `false`. Set it to 1 or leave it unset.


## HF_HOME made global, cache migrated

`~/.zshrc` is a symlink to `projects/github/unxmaal/dotfiles/zshrc`, so the export
went in the repo, not the home directory:

    export HF_HOME=/Volumes/Models/hf

Verified in a fresh login shell and through the library itself:

    $ zsh -lic 'echo $HF_HOME'
    /Volumes/Models/hf
    $ python -c "from huggingface_hub import constants; print(constants.HF_HUB_CACHE)"
    /Volumes/Models/hf/hub

Deliberately not guarded by a mount check. `/Volumes` is `drwxr-xr-x root:wheel`, and
a user `mkdir` there fails with `Permission denied` (tested), so an absent drive makes
`huggingface_hub` fail loudly instead of silently re-downloading to a full disk.

Migration verified before deleting the source:

    $ rsync -a --checksum --dry-run --itemize-changes ~/.cache/huggingface/ /Volumes/Models/hf/
    differences: 0
    src md5 (6.2GB fastchat-t5 blob): 583dfa4de314b9226939172dd8a9b914
    dst md5                         : 583dfa4de314b9226939172dd8a9b914
    src sum of regular file bytes: 10.3955 GB
    dst sum of regular file bytes: 10.3957 GB

### Correction: the FLUX weights were never there

A `du` discrepancy looked at first like 1.1 GB had gone missing. It had not: the Qwen
models were present in BOTH trees, because the MLX server had run against the default
`HF_HOME` before it was repointed. Source and destination byte sums match.

Investigating it did surface a real error in an earlier draft of PLAN.md, which
claimed FLUX weights were already cached and image generation was ready to go:

    $ find ~/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev
    .../models--black-forest-labs--FLUX.1-dev
    .../models--black-forest-labs--FLUX.1-dev/refs
    .../models--black-forest-labs--FLUX.1-dev/refs/main

    $ find .../FLUX.1-dev -type f -size +1M | wc -l
    0

All three FLUX directories hold only a `refs/main` pointer. No blobs. The 10 GB cache
is fastchat-t5 (6.2 GB) plus Chatterbox (3.0 GB) plus the Qwen test models. FLUX has
to be downloaded. The plan was corrected.

Lesson worth keeping: a populated-looking `models--*` directory in an HF cache proves
a repo was referenced, not that its weights were fetched. Check for blobs.

## h3.c built and probed on this machine (2026-09-05)

Clone: `antirez/h3.c` at 8974cc0 "Clarify SSD streaming memory and speed tradeoff".
Toolchain: Apple clang 21.0.0, macOS SDK 26.5, ffmpeg/ffprobe present via Homebrew.

    $ make -j8
    build exit=0    errors: 0    warnings: 0

Zero warnings under the project's own `-Wall -Wextra -Wpedantic -Wshadow
-Wconversion`. Produces `h3` (546K) and `libh3.a` (745K).

    $ make test
    exit=0
    ok: 1768 checks
    ok: native AudioVAE Metal primitives match host references
    ok: concurrent FFmpeg video/PCM pipes created /tmp/h3-av-mux-test.mp4 (51750 bytes)
    fail/error/abort/assert lines: 0

    audio primitive Conv1d           max abs 5.960464e-08
    audio primitive ConvTranspose1d  max abs 2.980232e-08
    audio primitive SnakeBeta        max abs 2.384186e-07

All skips are weight-dependent fixtures ("released ... weights/fixture are not
installed"). So the Metal compute path is verified working on an M2 Pro, which the
project documentation neither claims nor denies.

### Device probe, no weights required

`h3_metal_probe()` is standalone and exported from `libh3.a`, so it runs before any
checkpoint download. `tools/h3probe.c` calls it:

    Device: Apple M2 Pro (applegpu_g14s)
      physical memory       32.0 GiB
      recommended GPU set   25.0 GiB
      max Metal buffer      18.7 GiB
      Apple GPU family      8
      Metal 4               yes
      unified memory        yes

Three things this settles that were previously estimates:

1. **The GPU working set is 25.0 GiB**, not the "roughly 70 to 75% of unified RAM"
   the plan had been assuming. That is 78%, and it is a measured value from
   `device.recommendedMaxWorkingSetSize` rather than a guess.
2. **`max Metal buffer` is 18.7 GiB**, a hard per-allocation cap that had not been
   considered at all. No single tensor may exceed it regardless of total memory.
3. 25.0 GiB does not fit the 36.5 GiB full-residency DiT, so `--ssd-streaming` is
   mandatory here, not optional. It comfortably fits the 2.0 GiB streamed DiT.

### Correction: Metal 4 does not mean int8

The probe initially concluded "int8 MLP engine available (Metal 4 TensorOps)" from
the `metal4` flag reading yes. That was wrong. h3 gates TensorOps on a literal
substring match of the device name, in `h3_gpu.m`:

    BOOL m5 = [gpu.device.name rangeOfString:@"M5"].location != NSNotFound;
    BOOL wantsTensorOps = m5 && (!nax || !*nax || strcmp(nax, "0") != 0);

It is not a capability query. `[device supportsFamily:MTLGPUFamilyMetal4]` returns
yes on this M2 Pro because macOS 26 extends the Metal 4 API family well beyond the
hardware that has tensor units, and h3 never consults that field for gating. The
device name is "Apple M2 Pro", which contains no "M5", so TensorOps is off.

`H3_NAX` can disable TensorOps on an M5. Nothing enables it on a non-M5.

Practical result: this machine gets the BF16 path only and forfeits the int8 gain
(36.30 s BF16 versus 25.80 s int8 on an M5 Max). The `--use-int8-row-fc2` flag is
inert here.

## h3.c generation on 32 GiB, SETTLED (2026-09-05)

The open question was whether MiniMax H3 can generate on a 32 GiB M2 Pro.
It can. KNOWLEDGE #67's "not viable on 32GB, video is post-Studio" is superseded
by measurement.

Weights: FL2VA only, 134.2 GiB, downloaded in 24.1 min (~95 MB/s sustained to the
Models volume), 81 files, zero partials. `Ref2VA DiT  0 files  0 tensors  0.000 GiB`
in `--info` confirms the optional component degrades cleanly.

Command:

    ./h3 -d /Volumes/Models/MiniMax-H3 \
      -p "A red fox walks through falling snow in a quiet forest." \
      -o outputs/test1.mp4 --ssd-streaming --profile \
      --width 320 --height 320 --frames 8 --steps 6 --layers 40

Result: `h3: wrote outputs/test1.mp4`, a real h264 + aac file, 320x320, 22 frames,
0.925 s, 32 kHz stereo audio, 131 KB.

### The numbers that matter

    750 s wall (12.5 min)
    maximum resident set size  9.48 GiB      <- against a 19 GiB practical ceiling
    swaps                      0
    page faults                1849

Per phase, from `--profile`:

    Qwen text encoder  total  wall=131.750s  peak=2.727GiB  alloc=46.862GiB
    H3 DiT             load   wall= 81.117s  peak=1.607GiB  alloc=27.466GiB
    H3 DiT      Euler denoise  wall=501.436s peak=1.607GiB  alloc= 0.000GiB  wait=68.250s
    H3 DiT             total  wall=582.624s  peak=1.607GiB  alloc=27.466GiB
    audio VAE decoder  total  wall=  3.137s  peak=0.282GiB  alloc= 0.543GiB
    video VAE decoder  total  wall= 31.481s  peak=0.775GiB  alloc= 9.554GiB

`peak` versus `alloc` is the whole story. The text encoder allocates **46.9 GiB**
cumulatively but peaks at **2.7 GiB**, because it streams 50 layers with a
prefetch depth of 2 (`h3_gpu_is_m5(gpu) ? 3 : 2`) at roughly 0.98 GiB per layer.
The DiT allocates 27.5 GiB and peaks at 1.6 GiB under `--ssd-streaming`. Nothing
is ever fully resident, so a 62 GiB text encoder and a 61.7 GiB DiT both run in
under 3 GiB each.

Highest per-phase peak was the text encoder at 2.727 GiB. Process max RSS of
9.48 GiB is well above the sum of phase peaks, which is expected: RSS includes
page cache from streaming ~74 GiB off disk, not just live tensors.

### Predictions checked

- "The text encoder phase is the peak-memory candidate": **right**, it is the
  highest per-phase peak at 2.727 GiB, but the magnitude was wrong by an order of
  magnitude in the safe direction.
- "62 GiB text encoder does not fit in 19 GiB": **wrong as stated.** It never
  needs to fit; it is layer-streamed by design via `text_layer_prefetch`.
- "Plan against ~22 GiB, not 25.0": moot. Actual demand was under 10 GiB.
- `TEXT_LAYERS = 50` against the checkpoint's `num_hidden_layers = 64` is
  deliberate, not an incompatibility. The progress display confirms it: the
  text encoder counts to 50/50.

### Where the time goes

Denoise is 501 s of the 750 s, and **68.25 s of that is `wait`**, the GPU idle on
streaming I/O. That is the `--ssd-streaming` tax, about 14% of denoise wall here.
It scales with disk throughput, which is the concrete payoff of putting weights on
the Models volume at 959 MB/s rather than the T7 at 432.

### Flag note

`--reuse` and `--core-reuse` cannot be combined; h3 rejects it in 0.04 s with
"core reuse and denoiser reuse cannot be combined". It also warns when `--reuse`
is paired with very few steps.

## h3.c quality run at 512x512 (2026-09-05)

    ./h3 -d /Volumes/Models/MiniMax-H3 \
      -p "A red fox walks through falling snow in a quiet forest, late afternoon
          light, shallow depth of field." \
      -o outputs/quality1.mp4 --ssd-streaming --profile \
      --width 512 --height 512 --frames 8 --steps 20 --layers 50 --seed 42

Output is photographic and coherent: correct fox anatomy, fur detail, believable
depth of field, the prompted late-afternoon light and falling snow both present.

    2431.40 real (40.5 min)     <- predicted ~90 min, over by 2.2x
    maximum resident set size  9.60 GiB
    swaps                      0

### The peak phase moved

                        peak @320x320   peak @512x512
    Qwen text encoder      2.727 GiB       2.728 GiB    unchanged
    H3 DiT                 1.607 GiB       2.002 GiB    ~flat
    video VAE decoder      0.775 GiB       9.454 GiB    12.2x

The video VAE decoder is the binding constraint, not the DiT and not the text
encoder. It is the only major component that does not stream. 2.56x the pixels
produced 12.2x the memory, which is superlinear and consistent with attention
over spatial tokens: its `attention` count went 36 -> 144 over the same change.

Two points are a trend and not a law, but if that curve holds, h3's default
864x480 needs far more than the 25.0 GiB working set and would fail. **512x512
is at or near the ceiling on this hardware, and VAE decode is what breaks
first.** Worth measuring 640x640 before assuming anything.

This also revises the earlier conclusion. At 320x320 the text encoder looked like
the peak-memory component; that was an artifact of a resolution small enough to
make the VAE trivial. The text encoder is fixed-cost and streams; the VAE is
resolution-driven and does not.

### The streaming tax roughly doubled

                     wait / denoise wall     share
    320x320, 6 steps, 40 layers    68.25s / 501.44s     13.6%
    512x512, 20 steps, 50 layers  644.29s / 2177.53s    29.6%

Block loads went from 6x40=240 to 20x50=1000. The `--ssd-streaming` cost scales
with steps x layers, so quality settings pay for it twice: more compute and more
I/O stalls. At 29.6% idle, storage throughput is now a first-order term in wall
time, which is the clearest argument yet for the Models volume at 959 MB/s over
the T7 at 432.

### Prediction accuracy

- Wall time: predicted ~90 min from linear scaling of resolution x steps x layers,
  actual 40.5 min. Linear scaling overestimated by 2.2x.
- Peak memory: predicted "should stay near 9.5 GiB". Held (9.48 -> 9.60 GiB RSS),
  but for the wrong reason: per-phase peaks moved substantially and RSS is
  dominated by streaming page cache rather than live tensors.

## 640x640 probe: the VAE plateaus, it does not scale superlinearly (2026-09-05)

Prediction from the 320->512 pair was that the video VAE scaled with a ~2.66
exponent, putting 640x640 near 30 GiB and expecting failure. Run with a swap
watchdog to fail fast rather than thrash. It succeeded comfortably.

    --width 640 --height 640 --frames 8 --steps 4 --layers 40

    596.61 real (9.9 min)
    maximum resident set size  10.84 GiB
    swap                       flat at 1027M for the entire run, never moved

    video VAE peak:  320x320  0.775 GiB
                     512x512  9.454 GiB
                     640x640  9.365 GiB   <- FLAT, slightly lower

The VAE peak plateaus around 9.4 GiB. The 12x jump between 320 and 512 was a
threshold effect, almost certainly the decoder switching to a tiled path, not a
smooth curve. Fitting an exponent to two points on opposite sides of a threshold
produced a confident and completely wrong extrapolation.

Practical consequence: **the resolution ceiling is well above 640x640.** h3's
default 864x480 is 414,720 px against 640x640's 409,600, essentially identical,
so the default resolution should run on this machine.

VAE `attention` counts across the three runs were 36, 144, 324, which does grow
quadratically in linear resolution. Memory did not follow it, which is what
tiling is for.

### Wait-per-block-load grows with resolution

    320x320,  6 steps x 40 layers =  240 loads,  68.25s wait -> 0.28 s/load
    512x512, 20 steps x 50 layers = 1000 loads, 644.29s wait -> 0.64 s/load
    640x640,  4 steps x 40 layers =  160 loads, 173.53s wait -> 1.08 s/load

So the `--ssd-streaming` stall is not purely a function of block count. It grows
with resolution too, which points at activation-related stalls rather than pure
weight I/O. At 640x640 the wait was 51.9% of denoise wall.

### Watchdog bug worth remembering

The first watchdog checked `pgrep -qx h3` immediately after launching
`nohup bash ... &`, before `bash -> /usr/bin/time -> h3` had spawned, and
declared the run dead at t=0. Same shape as the LiteLLM readiness race earlier in
this log: **wait for a process to exist before watching for its absence.**

## Image generation via mflux, measured (2026-09-05)

Engine: `mflux` (2310 stars, MLX-native), model Z-Image Turbo 6B, 8 steps,
512x512, seed 42, 8-bit quantized. Weights 6.1 GB to /Volumes/Models/hf.

    candidate                     pass   rate   median    total      peak
    mflux/z-image-turbo          3/3     100%  119.16s   624.4s    6.2GiB

    fox-snow      387.2s   <- includes the cold model load
    product-shot  119.2s
    text-render   118.0s

Peak RSS 6.2 GiB against the 25.0 GiB working set measured by tools/h3probe, so
image generation is comfortable on this machine and could run concurrently with
most things.

The 387s first case versus ~119s after is exactly why the summary reports a
median rather than a mean: a cold load would otherwise triple the apparent cost
of every candidate.

`text-render` is the notable pass. Legible, correctly spelled "OPEN" in carved
serif, which is historically where local image models fail.

### Two install traps

`uv tool install mflux` picked **Python 3.9** from a pyenv shim, and mflux uses
`int | None` annotations, so every entry point died at import with

    TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'

`uv tool install --python 3.12 mflux` fixes it. Worth checking the interpreter
whenever a `uv tool install` produces a binary that cannot import itself.

The first install also produced only 2 executables against 37 on the second,
which is a quieter symptom of the same failure.

## Image head to head: Z-Image Turbo 6B vs FLUX.2 klein 4B (2026-09-05)

512x512, 8 steps, seed 42, 8-bit quantized, same three prompts.

    candidate                     pass   rate   median    total      peak
    mflux/flux2-klein-4b         3/3     100%   64.65s   351.8s    6.6GiB
    mflux/z-image-turbo          3/3     100%  120.89s   378.6s    3.3GiB

    per case (s)        fox-snow  product-shot  text-render
    flux2-klein-4b        222.53*        64.65        64.61
    z-image-turbo         137.80*       120.89       119.96
                          * cold model load

**FLUX.2 klein is 1.9x faster warm.** The memory column above is RETRACTED: it
came from a before/after delta of `resource.getrusage(RUSAGE_CHILDREN).ru_maxrss`,
which is a monotone high-water mark across all waited children, so the delta
reads 0 for every child after the largest, and plain RSS falls under memory
pressure besides. The same Z-Image config measured 6.2 GiB on a quiet machine
and 3.3 GiB on a loaded one. The two models are probably comparable on memory
and the "2x" claim was never supported.

Peak is now measured with `/usr/bin/time -l` "peak memory footprint", the same
phys_footprint metric the h3 runs used, so image and video numbers are finally
on one scale. The comparison needs re-running to get a real number.

Cold load is worth 158s on klein and only 18s on Z-Image Turbo, so klein's
advantage only exists once it is warm. For one-shot generation from cold they
are roughly level.

Both rendered legible, correctly spelled text, which is where local image models
usually fail. klein's output is the more photographic of the two and Z-Image
Turbo's the more studio-clean; that difference is taste and the harness
deliberately does not score it.

### Measurement caveat

Discord started partway through and swap climbed from 1.0GB to 8.8GB. The wall
times above were taken under that pressure and are therefore pessimistic, though
both candidates ran in the same session so the comparison between them stands.
Re-measure on a quiet machine before treating the absolute numbers as a baseline.

### The eval missed the thing that decided it

Both candidates scored 3/3. The suite therefore ranked them on speed and memory
and presented the result as a tradeoff: klein faster, Z-Image Turbo lighter.

Eric looked at the six images and said: "flux2 looks real. z-image-turbo not so
much."

That inverts the conclusion. If klein is both faster AND visibly better, 3.3 GiB
of extra peak against a 25.0 GiB working set buys nothing, and Z-Image Turbo has
no remaining case. **Decision: mflux/flux2-klein-4b is the image engine.**

The objective checks did their job and it was a smaller job than it looked.
"Does it decode, is it the right size, is it non-uniform" is a COMPETENCE GATE:
it separates a working pipeline from a broken one. It cannot rank two working
pipelines, and reporting a median latency next to a 100% pass rate implies a
ranking it has not earned.

Two honest options, and the suite should say which one it is doing:

1. Keep the checks as a gate, and state plainly that ranking beyond the gate is
   a human judgement. Cheap, truthful, and it still catches real failures.
2. Add a quality dimension. For images that means a local VLM scoring prompt
   adherence and artefacts against the generated file. That is a materially
   bigger build, needs its own eval to know whether the judge is any good, and
   introduces a model whose taste is not the user's.

Recorded before choosing, because the failure mode to avoid is a suite that
looks rigorous and quietly answers a different question than the one asked.
