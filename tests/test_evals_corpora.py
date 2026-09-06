"""Generating stt cases from a speech corpus.

LibriSpeech is 2620 utterances on a volume. Hand-writing case files for a
sample of them is silly and shipping them in the repo is wrong -- the audio
paths are machine-specific. So the generator is committed and the cases are
not, and the sample is seeded so a comparison is repeatable.
"""
from pathlib import Path

import pytest
import yaml

from evals import corpora


@pytest.fixture
def fake_corpus(tmp_path):
    """The LibriSpeech layout: speaker/chapter/*.flac plus a .trans.txt."""
    for speaker, chapter, count in (("101", "1", 3), ("202", "2", 2)):
        d = tmp_path / "test-clean" / speaker / chapter
        d.mkdir(parents=True)
        lines = []
        for i in range(count):
            uid = f"{speaker}-{chapter}-{i:04d}"
            (d / f"{uid}.flac").write_bytes(b"fLaC" + b"\0" * 900)
            lines.append(f"{uid} HELLO THERE UTTERANCE {i}")
        (d / f"{speaker}-{chapter}.trans.txt").write_text("\n".join(lines) + "\n")
    return tmp_path


def test_finds_every_utterance(fake_corpus):
    found = corpora.librispeech(fake_corpus)
    assert len(found) == 5
    assert all(u.audio.exists() for u in found)


def test_the_transcript_comes_from_the_corpus_not_a_model(fake_corpus):
    u = corpora.librispeech(fake_corpus)[0]
    assert "HELLO THERE" in u.transcript.upper()


def test_a_sample_is_reproducible(fake_corpus):
    a = corpora.librispeech(fake_corpus, limit=3, seed=7)
    b = corpora.librispeech(fake_corpus, limit=3, seed=7)
    assert [u.uid for u in a] == [u.uid for u in b]


def test_a_different_seed_gives_a_different_sample(fake_corpus):
    a = corpora.librispeech(fake_corpus, limit=3, seed=1)
    b = corpora.librispeech(fake_corpus, limit=3, seed=2)
    assert [u.uid for u in a] != [u.uid for u in b] or len(a) < 2


def test_the_sample_spans_speakers(fake_corpus):
    """A sample from one speaker measures one voice, not a model."""
    got = corpora.librispeech(fake_corpus, limit=4, seed=3)
    assert len({u.uid.split("-")[0] for u in got}) > 1


def test_a_missing_corpus_says_where_to_get_it(tmp_path):
    with pytest.raises(FileNotFoundError) as e:
        corpora.librispeech(tmp_path / "nope")
    assert "openslr" in str(e.value).lower()


def test_writes_loadable_cases(fake_corpus, tmp_path):
    out = tmp_path / "stt"
    written = corpora.write_cases(corpora.librispeech(fake_corpus, limit=2, seed=1),
                                  out, max_wer=0.2)
    assert len(written) == 2
    from evals.core import load_cases
    cases = load_cases(out)
    assert len(cases) == 2
    assert all(c.modality == "stt" for c in cases)
    assert all(c.audio.exists() for c in cases)
    assert all(c.assertions["max_wer"] == 0.2 for c in cases)


def test_generated_cases_use_absolute_audio_paths(fake_corpus, tmp_path):
    out = tmp_path / "stt"
    corpora.write_cases(corpora.librispeech(fake_corpus, limit=1, seed=1), out)
    body = yaml.safe_load(next(out.glob("*.yaml")).read_text())
    assert Path(body["audio_file"]).is_absolute()


def test_regenerating_replaces_rather_than_accumulates(fake_corpus, tmp_path):
    out = tmp_path / "stt"
    corpora.write_cases(corpora.librispeech(fake_corpus, limit=4, seed=1), out)
    corpora.write_cases(corpora.librispeech(fake_corpus, limit=2, seed=1), out)
    assert len(list(out.glob("*.yaml"))) == 2


def test_the_transcript_is_normalized_to_readable_text(fake_corpus, tmp_path):
    """LibriSpeech transcripts are bare uppercase with no punctuation. Storing
    them verbatim is fine for scoring -- the WER normalizer lowercases anyway --
    but a case file a person has to read should not shout."""
    out = tmp_path / "stt"
    corpora.write_cases(corpora.librispeech(fake_corpus, limit=1, seed=1), out)
    body = yaml.safe_load(next(out.glob("*.yaml")).read_text())
    assert body["prompt"] != body["prompt"].upper()
