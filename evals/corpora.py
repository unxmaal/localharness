"""Generating stt cases from a speech corpus.

    uv run python -m evals.corpora --limit 300 --seed 1

LibriSpeech test-clean is 2620 utterances with human transcripts, which is what
makes an stt measurement the model's own rather than joint with a TTS.

USE AT LEAST A FEW HUNDRED. At 40 clips this project got the winner right and
both effect sizes wrong in the same run: it called a 1.43x gap "twice as bad"
and called a significant loss a tie. 40 is enough to decide "do not switch";
it is not enough to quote a ratio. `evals.compare` puts an interval on it. Writing
case files for a sample by hand would be silly, and shipping them in the repo
would be wrong: the audio paths are machine-specific. So the generator is
committed and the cases are not, and the sample is seeded so a comparison run a
week later is the same comparison.

    curl -LO https://www.openslr.org/resources/12/test-clean.tar.gz
    tar xzf test-clean.tar.gz -C "$(dirname "${HF_ROOT:-./hf_root}")/corpora"
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from harness import env

DEFAULT_ROOT = env.beside() / "corpora" / "LibriSpeech"
DEFAULT_OUT = Path(__file__).resolve().parent / "cases" / "stt"
DOWNLOAD = "https://www.openslr.org/resources/12/test-clean.tar.gz"


@dataclass(frozen=True)
class Utterance:
    uid: str
    audio: Path
    transcript: str


def librispeech(root: str | Path = DEFAULT_ROOT, limit: int | None = None,
                seed: int = 1) -> list[Utterance]:
    """Every utterance under `root`, or a seeded sample of `limit` of them."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"no corpus at {root}. Fetch it from openslr:\n"
            f"    curl -LO {DOWNLOAD}\n"
            f"    tar xzf test-clean.tar.gz -C {root.parent}")

    found: list[Utterance] = []
    for trans in sorted(root.rglob("*.trans.txt")):
        for line in trans.read_text(encoding="utf-8").splitlines():
            uid, _, text = line.partition(" ")
            if not uid or not text:
                continue
            audio = trans.parent / f"{uid}.flac"
            if audio.exists():
                found.append(Utterance(uid, audio, text.strip()))

    if limit is None or limit >= len(found):
        return found
    # Seeded and sorted first, so the sample does not depend on filesystem
    # ordering. A comparison run next week has to be the same comparison.
    return random.Random(seed).sample(found, limit)


def readable(transcript: str) -> str:
    """LibriSpeech transcripts are bare uppercase with no punctuation.

    Scoring does not care -- the WER normalizer lowercases and strips anyway --
    but a case file a person has to read should not shout at them.
    """
    text = transcript.strip().lower()
    return re.sub(r"^(\w)", lambda m: m.group(1).upper(), text)


def write_cases(utterances: list[Utterance], out: str | Path = DEFAULT_OUT,
                max_wer: float | None = None) -> list[Path]:
    """Write one case per utterance, replacing whatever was there before."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("*.yaml"):
        stale.unlink()

    written = []
    for u in utterances:
        body: dict = {
            "id": u.uid,
            "modality": "stt",
            "prompt": readable(u.transcript),
            # Absolute: the corpus lives on a volume, not beside the case.
            "audio_file": str(u.audio.resolve()),
        }
        if max_wer is not None:
            body["assert"] = {"max_wer": max_wer}
        path = out / f"{u.uid}.yaml"
        path.write_text(yaml.safe_dump(body, sort_keys=False, width=200), encoding="utf-8")
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="evals.corpora", description=__doc__)
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--limit", type=int, default=40,
                    help="utterances to sample; 0 for all 2620")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-wer", type=float, default=None,
                    help="fail a case above this; omit to measure without "
                         "judging, which is what ranking wants")
    args = ap.parse_args(argv)

    try:
        found = librispeech(args.root, limit=args.limit or None, seed=args.seed)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    written = write_cases(found, args.out, max_wer=args.max_wer)
    print(f"wrote {len(written)} stt cases to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
