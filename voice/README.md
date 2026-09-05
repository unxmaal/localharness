# voice

Speaking and hearing for Claude Code in wezterm. Local models only.

    you speak  -> rec (sox, silence-gated) -> parakeet-mlx -> wezterm cli send-text
    claude ends turn -> Stop hook -> strip markdown -> local summary via :4000
                     -> kokoro /v1/audio/speech -> afplay

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

1. Start Kokoro: `cd ~/projects/github/unxmaal/local_chat && docker compose up -d kokoro-tts`
   Raise `cpus: 0.3` in that compose file first; it is throttled for batch VO,
   not for conversation.
2. Start the gateway: `./scripts/serve-mlx.sh &` and `./scripts/serve-gateway.sh &`
3. Register the Stop hook in `~/.claude/settings.json` (see `settings-snippet.json`).
4. Add the keybinding from `wezterm-snippet.lua` to `~/.wezterm.lua`.
5. First run of `listen.sh` downloads the Parakeet weights (~600MB) to `$HF_HOME`.

## Known gaps

- Parakeet is English-only. `mlx-whisper` is the multilingual swap.
- Kokoro runs in a CPU container. `mlx-audio` would move it onto the GPU and is
  the obvious upgrade if latency disappoints.
- Nothing interrupts Claude mid-response by voice; you can only cut off playback
  by starting a new recording.
