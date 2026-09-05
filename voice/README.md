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
| `wezterm-snippet.lua` | CTRL+SHIFT+V keybinding |

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

## Mute

    touch ~/.claude/voice-mute     # quiet
    rm ~/.claude/voice-mute        # speaking again

No settings.json edit, no restart.

## Setup, in order

1. Start TTS: `./scripts/serve-tts.sh &` (Kokoro-82M on the GPU via mlx-audio,
   port 8083). First run pulls ~330MB to `$HF_HOME`.
2. Start the gateway: `./scripts/serve-mlx.sh &` and `./scripts/serve-gateway.sh &`
3. Register the Stop hook in `~/.claude/settings.json` (see `settings-snippet.json`).
4. Add the keybinding from `wezterm-snippet.lua` to `~/.wezterm.lua`.
5. First run of `listen.sh` downloads the Parakeet weights (~600MB) to `$HF_HOME`.

## Known gaps

- Parakeet is English-only. `mlx-whisper` is the multilingual swap.
- Kokoro-82M-bf16 is the default. `-8bit` and `-4bit` variants exist if load
  time or memory ever matter, though at 82M parameters neither is likely to.
- Nothing interrupts Claude mid-response by voice; you can only cut off playback
  by starting a new recording.
