# Voice cloning: what works, what is staged, what blocks

Status 2026-09-06. The goal is a male French voice, which Kokoro cannot do —
its only French voice, `ff_siwis`, is female — so it needs a cloning model.

## Working

Chatterbox clones from a reference clip and speaks French. Measured:

    curl -s http://127.0.0.1:8890/v1/audio/speech \
      -H 'Content-Type: application/json' \
      -d '{"model":"litmudoc/Chatterbox-Multilingual-MLX-v2-Q8",
           "input":"Bonjour, la passerelle est en marche.",
           "ref_audio":"/Volumes/Models/corpora/voice-refs/fleurs-fr-male-1.wav",
           "lang_code":"fr",
           "response_format":"wav"}' -o out.wav

4.2s of audio from a 9.8s reference, generated in 5.2s.

Three things had to be right at once, and each failed differently:

- **The checkpoint must be multilingual.** `mlx-community/Chatterbox-TTS-8bit`
  refuses French with "The English Chatterbox checkpoint does not support 'a'.
  Use a multilingual v2 or v3 checkpoint." There is no multilingual Chatterbox
  under `mlx-community`; `litmudoc/Chatterbox-Multilingual-MLX-v2-Q8` is one
  that works.
- **It must ship `conds.safetensors`.** Without it Chatterbox raises "No
  conditionals available" even with a reference clip. The mlx-community repo
  does not have this file; the litmudoc one does.
- **The field is `lang_code`, not `language`.** mlx_audio's `SpeechRequest`
  defaults `lang_code` to `"a"` (Kokoro's American English), so a request that
  sets `language` is silently ignored and fails as "Unsupported language code
  'a'" — naming a code the caller never sent.

Chatterbox also needs a second repo, `mlx-community/S3TokenizerV2`, which
nothing mentions until it fails.

## Staged reference material

`/Volumes/Models/corpora/voice-refs/` — three French male clips, 9-11 seconds,
with `manifest.json` giving the transcript for each.

Source: **google/fleurs**, `fr_fr` dev split, CC-BY-4.0. It ships a TSV with
the transcript AND a gender label per utterance, which is why it was chosen
over Common Voice (gated) or Multilingual LibriSpeech (60GB for French). 227
male French clips are available; three were copied out.

`/Volumes/Models/corpora/fleurs-fr/` holds the whole dev split (289 clips),
`/Volumes/Models/corpora/LibriSpeech/` the English test-clean used by the stt
lane.

## The blocker

**Nothing can score French speech on this machine yet.** The tts lane measures
intelligibility by transcribing what was spoken, and the resident STT is
Parakeet, which is English-only: it hears the French sample as "Monjour passed
in March and Test in Russia". That is not a measurement of anything.

Both multilingual candidates fail through mlx_audio's server:

| model | failure |
|---|---|
| `mlx-community/whisper-large-v3-mlx` | `Processor not found. Make sure the model was loaded with a HuggingFace processor.` The repo ships `weights.npz` and `config.json` only — no `preprocessor_config.json`. |
| `mlx-community/whisper-large-v3-turbo` | same |
| `mlx-community/Voxtral-Mini-3B-2507-bf16` | `Model.generate() got an unexpected keyword argument 'word_timestamps'` — the server passes it unconditionally and this model family does not accept it. |

So the reference clips are ready, cloning works, and the ruler is missing.
