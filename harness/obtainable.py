"""Can the weights this repo names actually be obtained?

CLASS 11 OF THE GAUNTLET: external state with no assertion on it. Every alias
in a gateway config points at somebody else's published artifact. Each was
obtainable when it was written, nothing here notices when one stops being, and
the failure arrives on a machine rather than in CI.

#147 is the proof and it was found by hand, on a bare Linux box, by trying:
config.cuda.yaml names four GGUFs that cannot be fetched as written. One is
split into shards, one is published only at Q8_0, one lives behind a repo that
401s, and two differ from their upstream filenames ONLY IN CASE. The router
echoes the stem verbatim, so the two aliases that worked were exactly the two
whose upstream names happened to match.

TWO CHECKS, because the two configs name two different kinds of thing:

  resolvable  OFFLINE and structural. A repo id says where it comes from. A
              bare GGUF stem does not, so nothing -- no person and no script --
              can tell whether it is obtainable without guessing the repo. That
              missing provenance IS #147's defect rather than a symptom of it.

  reachable   NETWORK. The repo id answers. Marked slow, never in `make check`:
              a lint that fails when HuggingFace has a bad afternoon teaches
              people to skip lint.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

#: LiteLLM prefixes a provider onto the upstream name. `openai/` here means
#: "speak the OpenAI protocol", not an OpenAI model, and it is not part of the
#: identifier being resolved.
PROVIDERS = ("openai/", "huggingface/", "ollama/")

#: A HuggingFace repo id: exactly one slash, and both halves present.
REPO_ID = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")

#: Where a GGUF stem may record where it came from, since the stem alone
#: cannot. Either key answers "which published file is this".
PROVENANCE_KEYS = ("source_repo", "source_file")

CONFIGS = ("gateway/config.yaml", "gateway/config.cuda.yaml")


@dataclass(frozen=True)
class Named:
    config: str
    alias: str
    ref: str

    @property
    def is_repo_id(self) -> bool:
        return bool(REPO_ID.match(self.ref))

    def __str__(self) -> str:
        kind = "repo" if self.is_repo_id else "stem"
        return f"{self.config}\t{self.alias}\t{kind}\t{self.ref}"


def strip_provider(model: str) -> str:
    for p in PROVIDERS:
        if model.startswith(p):
            return model[len(p):]
    return model


def named(root: Path, configs=CONFIGS) -> list[Named]:
    """Every external artifact the gateway configs name."""
    out = []
    for rel in configs:
        path = root / rel
        if not path.exists():
            continue
        body = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for entry in body.get("model_list", []):
            params = entry.get("litellm_params") or {}
            model = params.get("model")
            if not model:
                continue
            out.append(Named(rel, entry.get("model_name", "?"),
                             strip_provider(model)))
    return out


def unresolvable(root: Path, configs=CONFIGS) -> list[tuple[Named, str]]:
    """Named artifacts nothing could look up, offline.

    A bare stem with no recorded source is the whole of #147: the config says
    WHAT to ask for and not WHERE it comes from, so verifying it means guessing
    the repo, which is exactly the guess that produced four wrong aliases.
    """
    out = []
    for n in named(root, configs):
        if n.is_repo_id:
            continue
        body = yaml.safe_load((root / n.config).read_text(encoding="utf-8"))
        entry = next((e for e in body["model_list"]
                      if e.get("model_name") == n.alias), {})
        params = entry.get("litellm_params") or {}
        if any(params.get(k) or entry.get(k) for k in PROVENANCE_KEYS):
            continue
        out.append((n, "a filename stem with no source_repo: nothing can check "
                       "it, and the router echoes it verbatim, so case matters"))
    return out


def unreachable(root: Path, configs=CONFIGS, facts=None) -> list[tuple[Named, str]]:
    """Repo ids the registry does not answer for. NEEDS THE NETWORK."""
    if facts is None:
        from harness.inspect import hf_facts
        facts = hf_facts
    cache: dict = {}
    out = []
    for n in named(root, configs):
        if not n.is_repo_id:
            continue
        if facts(n.ref, cache=cache).get("size", -1) <= 0:
            out.append((n, "the registry returns nothing for this repo id"))
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m harness.obtainable",
                                 description=__doc__.splitlines()[0])
    ap.add_argument("--network", action="store_true",
                    help="also ask the registry whether each repo id answers")
    a = ap.parse_args(argv)

    root = Path(__file__).resolve().parent.parent
    findings = unresolvable(root)
    if a.network:
        findings += unreachable(root)
    for n, why in findings:
        print(f"{n}\n    -> {why}")
    print(f"{len(findings)} finding(s) of {len(named(root))} named")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
