"""Issue #61: read a candidate's source before downloading its weights."""
import pytest

from harness import inspect as ins


def tree(tmp_path, files: dict):
    for rel, text in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return tmp_path


# ---- what the source says --------------------------------------------------

def test_a_declared_cuda_dependency_is_found(tmp_path):
    t = tree(tmp_path, {"requirements.txt": "torch\nflash_attn==2.5\n"})
    assert "flash_attn" in ins.scan(t)["cuda"]


def test_cuda_in_code_is_recorded_but_not_as_a_dependency(tmp_path):
    """apple/coreai-models mentions torch.cuda in one export recipe and is an
    Apple on-device repo. Calling that "needs CUDA" threw away the most
    relevant candidate in the sweep."""
    t = tree(tmp_path, {"export.py": "if torch.cuda.is_available():\n    pass\n"})
    found = ins.scan(t)
    assert found["cuda"] == []
    assert "torch.cuda" in found["cuda_mentioned"]


def test_mlx_native_is_distinguished_from_a_torch_mps_fallback(tmp_path):
    native = ins.scan(tree(tmp_path / "a", {"m.py": "import mlx.core as mx\n"}))
    fallback = ins.scan(tree(tmp_path / "b", {"m.py": 'device = "mps"\n'}))
    assert native["mlx"] and not native["mps"]
    assert fallback["mps"] and not fallback["mlx"]


def test_model_ids_are_read_out_of_code(tmp_path):
    t = tree(tmp_path, {"g.py": 'load("Lightricks/LTX-2")\nx = "google/umt5-xxl"\n'})
    assert set(ins.scan(t)["hf_ids"]) == {"Lightricks/LTX-2", "google/umt5-xxl"}


def test_infrastructure_ids_are_not_weights(tmp_path):
    t = tree(tmp_path, {"w.yaml": 'uses: "actions/checkout"\n'})
    assert ins.scan(t)["hf_ids"] == []


def test_dot_directories_are_not_the_project(tmp_path):
    """An agent skills folder full of helper scripts was being read as the
    project's own entry points."""
    t = tree(tmp_path, {".claude/skills/helper.py": 'if __name__ == "__main__": pass'})
    assert ins.scan(t)["entry_points"] == []


def test_an_entry_point_is_found(tmp_path):
    t = tree(tmp_path, {"pyproject.toml": "[project.scripts]\nx = 'a:b'\n"})
    assert ins.scan(t)["entry_points"] == ["pyproject scripts"]


# ---- the verdict -----------------------------------------------------------

def fit(**kw):
    return ins.decide(ins.Fit(repo="a/b", entry_points=["main.py"], **kw))


def test_a_declared_cuda_dependency_outranks_everything(tmp_path):
    """Absolute on this machine: a repo that cannot run here at any size is
    not a size question."""
    assert fit(cuda=["flash_attn"], smallest=1).verdict == "needs-cuda"


def test_the_smallest_weight_decides_not_the_largest():
    """Measured the hard way. Blaizzy/nativ names a 1774 GiB model and was
    reported as too big for this machine; it is a Mac app with a CATALOGUE of
    models it can serve. Source cannot tell a requirement from an option, so
    the honest question is whether ANYTHING it names could run here."""
    got = fit(smallest=2 * ins.GIB, largest=1774 * ins.GIB)
    assert got.verdict == "fits"


def test_a_model_that_cannot_fit_at_all_is_refused():
    assert fit(smallest=90 * ins.GIB, largest=90 * ins.GIB).verdict == "too-big"


def test_weights_that_could_not_be_sized_are_unknown_not_fitting():
    """An unknown size reported as zero reads as "small enough", which is the
    opposite of what is known. HuggingFace 429s this endpoint under a sweep."""
    got = fit(unsized=["org/a", "org/b"])
    assert got.verdict == "unknown" and "not known" in got.why


def test_naming_no_weights_at_all_is_not_the_same_as_unknown():
    got = fit()
    assert got.verdict == "fits" and "no weights named" in got.why


def test_a_repo_nobody_has_touched_in_years_is_dead():
    assert fit(last_commit="2019-01-01T00:00:00+00:00").verdict == "dead"


def test_nothing_to_call_is_not_a_candidate():
    assert ins.decide(ins.Fit(repo="a/b")).verdict == "no-entry-point"


def test_an_unparseable_commit_date_does_not_decide_anything():
    got = fit(last_commit="whenever")
    assert got.verdict == "fits"


# ---- guards ----------------------------------------------------------------

def test_a_repo_carrying_weights_in_git_is_refused_before_cloning(tmp_path):
    """Cloning it IS the download this tier exists to avoid."""
    got = ins.inspect("a/b", tmp_path, meta={"size": 900_000},
                      run=lambda *a, **k: pytest.fail("must not clone"))
    assert got.verdict == "too-big" and "weights in git" in got.why


def test_the_clone_is_shallow_and_single_branch(tmp_path):
    seen = []

    def run(argv, cwd=None, timeout=180.0):
        seen.append(argv)
        (tmp_path / "a__b").mkdir(exist_ok=True)
        return "2026-01-01T00:00:00+00:00"

    ins.inspect("a/b", tmp_path, meta={"size": 10}, run=run,
                sizer=lambda m: -1)
    assert "--depth" in seen[0] and "1" in seen[0]
    assert "--single-branch" in seen[0] and "--no-tags" in seen[0]


def test_an_unreachable_registry_sizes_nothing_rather_than_zero():
    assert ins.hf_size("org/x", fetch=lambda url: (_ for _ in ()).throw(
        RuntimeError("429"))) == -1


def test_a_registry_reply_with_no_files_is_unknown_not_zero():
    assert ins.hf_size("org/x", fetch=lambda url: '{"siblings": []}') == -1


def test_sizes_are_summed_across_the_repo():
    reply = '{"siblings": [{"size": 100}, {"size": 250}]}'
    assert ins.hf_size("org/x", fetch=lambda url: reply) == 350


# ---- the tier has to stay cheap --------------------------------------------

def test_only_a_bounded_number_of_weights_is_sized(tmp_path):
    """A repo listing a hundred models is showing a CATALOGUE. Sizing all of
    them turned a seconds-long tier into hours: the registry 429s under a
    sweep, and retrying at the feed cadence costs 80s per id."""
    asked = []

    def run(argv, cwd=None, timeout=180.0):
        d = tmp_path / "a__b"
        d.mkdir(exist_ok=True)
        (d / "m.py").write_text("\n".join(
            f'load("org/model-{i:03d}")' for i in range(60)))
        return "2026-01-01T00:00:00+00:00"

    def sizer(model_id, cache=None):
        asked.append(model_id)
        return -1

    got = ins.inspect("a/b", tmp_path, meta={"size": 10}, run=run, sizer=sizer)
    assert 0 < len(asked) <= ins.SIZE_LIMIT
    assert len(got.unsized) == 60


def test_the_sizing_retry_is_short_not_the_feed_cadence():
    """20s between four tries is right for a feed nobody is waiting on, and
    wrong for a tier justified by costing seconds."""
    assert ins.SIZE_RETRIES <= 1 and ins.SIZE_DELAY <= 5


def test_a_known_size_is_answered_from_cache_without_asking():
    calls = []

    def fetch(url):
        calls.append(url)
        return '{"siblings": [{"size": 5}]}'

    cache = {"org/x": 4242}
    assert ins.hf_size("org/x", fetch=fetch, cache=cache) == 4242
    assert calls == []


def test_a_rate_limit_is_not_cached_as_a_permanent_unknown():
    """Caching -1 would freeze one bad afternoon into a fact."""
    cache = {}
    ins.hf_size("org/x", fetch=lambda url: (_ for _ in ()).throw(RuntimeError()),
                cache=cache)
    assert cache == {}


def test_a_real_size_is_kept_so_the_next_sweep_is_free():
    cache = {}
    ins.hf_size("org/x", fetch=lambda url: '{"siblings": [{"size": 7}]}',
                cache=cache)
    assert cache == {"org/x": 7}


def test_an_mlx_project_is_not_disqualified_by_an_optional_cuda_build(tmp_path):
    """ml-explore/mlx itself came back needs-cuda. Its setup.py adds nvidia-*
    inside `if toolkit == 12:`, a branch this machine never takes, and an
    `import mlx` is positive proof the project runs on Apple Silicon."""
    got = ins.decide(ins.Fit(repo="ml-explore/mlx", entry_points=["setup.py"],
                             mlx=True, cuda=["nvidia-cublas"]))
    assert got.verdict == "fits"
    assert "nvidia-cublas" in got.cuda_mentioned


def test_a_non_mlx_project_is_still_disqualified_by_a_cuda_dependency():
    got = ins.decide(ins.Fit(repo="a/b", entry_points=["setup.py"], mlx=False,
                             cuda=["nvidia-resiliency-ext"]))
    assert got.verdict == "needs-cuda"


def test_awq_is_a_quantisation_format_not_a_cuda_requirement(tmp_path):
    """mlx-lm was flagged on an entry point named `mlx_lm.awq`, and mlx-lm
    implements AWQ natively."""
    t = tree(tmp_path, {"setup.py": '"mlx_lm.awq = mlx_lm.quant.awq:main"'})
    assert ins.scan(t)["cuda"] == []


def test_too_big_is_not_claimed_from_a_truncated_size_scan():
    """The smallest of twelve sized ids out of three hundred named is an upper
    bound on the floor, not the floor. mlx-video was refused on exactly this."""
    got = ins.decide(ins.Fit(repo="a/b", entry_points=["m.py"],
                             weights={"org/x": 90 * ins.GIB},
                             smallest=90 * ins.GIB, largest=90 * ins.GIB,
                             unsized=["org/%d" % i for i in range(50)]))
    assert got.verdict == "unknown" and "never sized" in got.why


def test_too_big_still_holds_when_the_scan_was_complete():
    got = ins.decide(ins.Fit(repo="a/b", entry_points=["m.py"],
                             weights={"org/x": 90 * ins.GIB},
                             smallest=90 * ins.GIB, largest=90 * ins.GIB))
    assert got.verdict == "too-big"


# ---- issue #68: which weight is the repo actually FOR ----------------------

def test_a_model_named_in_the_readme_outranks_one_buried_in_code():
    """The first queue built from "smallest named weight" filled with
    tokenizers and a 0.6B somebody used in a test."""
    got = ins.headline("org/thing", ["org/buried", "org/introduced"],
                       {"org/buried": 1, "org/introduced": 1},
                       ["org/introduced"])
    assert got[0] == "org/introduced"


def test_a_model_named_repeatedly_outranks_one_named_once():
    """The model a repo is ABOUT gets named again and again; a helper appears
    once."""
    got = ins.headline("org/thing", ["org/once", "org/everywhere"],
                       {"org/once": 1, "org/everywhere": 9}, [])
    assert got[0] == "org/everywhere"


def test_a_model_whose_name_echoes_the_repo_outranks_a_stranger():
    """mlx-video naming Lightricks/LTX-2 beats it naming google/umt5-xxl."""
    got = ins.headline("Blaizzy/mlx-video", ["google/umt5-xxl", "org/mlx-video-base"],
                       {}, [])
    assert got[0] == "org/mlx-video-base"


def test_the_ranking_keeps_everything(tmp_path):
    """A ranking, not a filter: a weak signal is not evidence of irrelevance."""
    ids = ["org/a", "org/b", "org/c"]
    assert sorted(ins.headline("x/y", ids, {}, [])) == sorted(ids)


def test_both_the_smallest_and_the_headline_get_sized(tmp_path):
    """They are rarely the same models, and they answer different questions:
    can anything here run, and what is worth downloading."""
    asked = []

    def run(argv, cwd=None, timeout=180.0):
        d = tmp_path / "a__thing"
        d.mkdir(exist_ok=True)
        (d / "README.md").write_text('see "org/the-headline-model-with-long-name"')
        (d / "m.py").write_text("\n".join(
            f'load("org/helper-{i:02d}")' for i in range(30)))
        return "2026-01-01T00:00:00+00:00"

    def sizer(model_id, cache=None):
        asked.append(model_id)
        return 2 * ins.GIB

    ins.inspect("a/thing", tmp_path, meta={"size": 10}, run=run, sizer=sizer)
    assert "org/the-headline-model-with-long-name" in asked
    assert any(i.startswith("org/helper-") for i in asked)


def test_the_fit_carries_the_repo_description(tmp_path):
    """_judge_fits reads it to show the judge prose AND source facts together.
    It shipped reading a field Fit did not have, and the whole suite passed
    because nothing exercised that path."""
    def run(argv, cwd=None, timeout=180.0):
        (tmp_path / "a__b").mkdir(exist_ok=True)
        return "2026-01-01T00:00:00+00:00"

    got = ins.inspect("a/b", tmp_path, meta={"size": 10, "description": "a tool"},
                      run=run, sizer=lambda m, cache=None: -1)
    assert got.description == "a tool"
