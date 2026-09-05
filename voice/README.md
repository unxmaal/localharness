# voice

Speaking and hearing for Claude Code in wezterm. Local models only.

    you speak  -> rec (sox, silence-gated) -> parakeet-mlx -> wezterm cli send-text
    claude ends turn -> Stop hook -> strip markdown -> local summary via :4000
                     -> mlx-audio kokoro :8083 -> afplay

## Pieces

| file | role |
|---|---|
| `config.sh` | shared settings, endpoints, mute flag |
| `speak.sh` | Claude Code `Stop` hook, speaks a 2-sentence summary |
| `listen.sh` | record, transcribe, type into the pane |
| `strip.py` | markdown -> speakable prose, splits off the closing question |
| `wezterm-snippet.lua` | reference copy of the CTRL+SHIFT+V keybinding |

## Measured

    TTS   0.5s to generate 6.45s of audio      12-13x realtime
    STT   0.26s to transcribe 6.5s of speech   resident, no per-call load
    hook  0.024s to return                     playback is detached

## Why the closing question never reaches the summarizer

`strip.py` splits a trailing question out and `speak.sh` reattaches it verbatim
after summarization. This is not tidiness. Qwen2.5-1.5B, asked to condense
"Want me to wire up the voice layer next?", produced "No, I don't need to wire up
the voice layer next" and, on another run, "Next, I'll wire up the voice layer."
The first answers a question that was never asked; the second converts it into a
commitment. Both spoken in Claude's own voice.

It happened non-deterministically, which is worse than consistently: 4 runs of
the same input gave different handling. So the highest-stakes sentence is taken
out of the model's hands entirely. Verified 4/4 verbatim afterwards.

## Why a 7B and not the 1.5B

The 1.5B also mangles technical claims. Condensing "does NOT scale
superlinearly, it plateaus" it produced "It scales superlinearly up to ~9.4 GiB",
the opposite of the source. Qwen2.5-7B-Instruct-4bit (`local-summarize` in the
gateway) gave 4/4 identical, correct output on the same input. Anything speaking
in Claude's voice needs a model that keeps facts straight.

## Why a summary rather than the whole response

Responses contain code blocks, tables, and file paths. Read aloud verbatim they
are unusable, and long. `speak.sh` strips markdown mechanically, then asks a
small local model through the gateway for at most two spoken sentences, keeping
any closing question last so the conversational turn survives. If the gateway is
down it falls back to the first two sentences rather than going silent.

## Why the transcript is typed, not sent

`listen.sh` uses `--no-paste`, so the text lands in the prompt and you press
Enter. A speech recognizer that both puts words in your mouth and submits them
is one mis-hearing away from an unwanted action.

## What did it say?

    tail -f ${TMPDIR:-/tmp}/claude-voice/spoken.log

Every spoken line is logged with a timestamp. A summarizer that said something
wrong and one that never ran are otherwise indistinguishable.

## Mute

    touch ~/.claude/voice-mute     # quiet
    rm ~/.claude/voice-mute        # speaking again

No settings.json edit, no restart.

## Setup, in order

1. Start TTS: `./scripts/serve-tts.sh &` (Kokoro-82M on the GPU via mlx-audio,
   port 8083). First run pulls ~330MB to `$HF_HOME`.
2. Start the gateway: `./scripts/serve-mlx.sh &` and `./scripts/serve-gateway.sh &`
3. Register the Stop hook in `~/.claude/settings.json` (see `settings-snippet.json`).
   DONE on this machine.
4. Add the keybinding to `~/.wezterm.lua`. DONE on this machine, in
   `dotfiles/.wezterm.lua`; `wezterm-snippet.lua` is kept as a portable copy.
5. Weights land in `$HF_HOME` on first use: Kokoro-82M ~330MB, Parakeet ~600MB,
   Qwen2.5-7B-Instruct-4bit ~4.3GB for the summarizer.

Voice ships MUTED. `rm ~/.claude/voice-mute` to turn it on.

## Latency

Measured end to end for a 4.1s utterance, total 6.76s:

    +0.13s  invoked        script startup
    +6.13s  captured       4.1s speech + 0.9s hang + ~1.0s waiting for onset
    +6.49s  transcript     0.36s transcription
    +6.76s  done           0.27s typing into the pane

Fixed overhead is ~1.66s. The models are not the bottleneck: Parakeet transcribes
a 6.5s clip in 0.24s, held resident.

The variable cost is the gap between pressing the key and starting to talk, which
was 2-4s in early use. sox discards leading silence, so starting immediately
cannot clip you. `VOICE_HANG` is the only real knob; 0.6s feels snappier and will
cut you off if you pause mid-sentence.

`$VOICE_RUN/listen.log` timestamps every stage with elapsed time, so a slow run
can be attributed rather than guessed at.

## Recording gate

`VOICE_THRESHOLD` (default 1%) is a percentage of full scale. sox starts
capturing above it and stops after `VOICE_HANG` seconds below it. The original
2% never triggered on a quiet source: a Yeti picking up speaker output measured
RMS 179, or 0.55%. A quiet room with nobody talking measures ~0.18%, so 1% sits
comfortably between. Raise it if the gate self-triggers.

`VOICE_ONSET_TIMEOUT` (default 12s) is a hard stop on waiting for speech. Without
it, a gate that never triggers leaves sox holding the microphone open forever,
which is exactly what happened the first time this was tested: sox wrote a
0-byte file and never returned.

Note that sox cannot always honour device-level format flags. A Yeti is 48 kHz
stereo and `rec -c 1 -r 16000` only warns, then records 48k stereo anyway.
`channels 1 rate 16000` as EFFECTS are applied in software and always hold.

## Known gaps

- Parakeet is English-only. `mlx-whisper` is the multilingual swap.
- The gate has only been exercised against speaker playback, not a person
  talking directly into the mic. Direct speech is far louder, so 1% should be
  conservative, but the threshold may want tuning in practice.
- Kokoro-82M-bf16 is the default. `-8bit` and `-4bit` variants exist if load
  time or memory ever matter, though at 82M parameters neither is likely to.
- Nothing interrupts Claude mid-response by voice; you can only cut off playback
  by starting a new recording.
