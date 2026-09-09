"""Issue #61: read a candidate's source before downloading its weights."""
import pytest

from harness import inspect as ins


def tree(tmp_path, files: dict):
    for rel, text in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
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
            f'load("org/model-{i:03d}")' for i in range(60)), encoding="utf-8")
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

    cache = {"org/x": {"size": 4242, "lane": "stt"}}
    assert ins.hf_size("org/x", fetch=fetch, cache=cache) == 4242
    assert calls == []


def test_a_rate_limit_is_not_cached_as_a_permanent_unknown():
    """Caching -1 would freeze one bad afternoon into a fact."""
    cache = {}
    ins.hf_size("org/x", fetch=lambda url: (_ for _ in ()).throw(RuntimeError()),
                cache=cache)
    assert cache == {}


def test_a_real_size_is_kept_so_the_next_sweep_is_free():
    """Size and lane come out of ONE registry call and are cached together:
    the registry rate-limits, so asking twice for one model is a request spent
    on nothing."""
    cache = {}
    ins.hf_size("org/x", fetch=lambda url: '{"siblings": [{"size": 7}]}',
                cache=cache)
    assert cache == {"org/x": {"size": 7, "lane": ""}}


def test_an_older_size_only_cache_entry_is_upgraded_not_trusted():
    """Trusting it reported every already-sized model as unmeasurable: the
    first real run showed 0 queued and 14 orphans, several plainly STT and
    image models. A size-only entry predates lanes, so it is refetched."""
    got = ins.hf_facts("org/x", cache={"org/x": 99},
                       fetch=lambda url: '{"siblings": [{"size": 99}],'
                                         ' "pipeline_tag": "text-to-speech"}')
    assert got == {"size": 99, "lane": "tts"}


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
        (d / "README.md").write_text('see "org/the-headline-model-with-long-name"', encoding="utf-8")
        (d / "m.py").write_text("\n".join(
            f'load("org/helper-{i:02d}")' for i in range(30)), encoding="utf-8")
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


# ---- issue #81: a weight no lane can test ---------------------------------

def test_the_registrys_own_task_label_names_the_lane():
    assert ins.lane_for({"pipeline_tag": "automatic-speech-recognition"}) == "stt"
    assert ins.lane_for({"pipeline_tag": "text-to-audio"}) == "tts"


def test_free_text_tags_are_read_when_the_task_label_is_missing():
    """pipeline_tag was absent on three of six real models checked."""
    assert ins.lane_for({"pipeline_tag": None, "tags": ["mlx", "asr"]}) == "stt"


def test_a_model_nothing_here_can_measure_has_no_lane():
    """silero-vad and MossFormer2 are both good models and neither can be
    scored by anything in this repo. Empty is a fact about the harness, not a
    rejection of the model."""
    assert ins.lane_for({"pipeline_tag": "audio-classification",
                         "tags": ["mlx", "vad"]}) == ""
    assert ins.lane_for({}) == ""


def test_the_lane_travels_with_the_weight(tmp_path):
    def run(argv, cwd=None, timeout=180.0):
        d = tmp_path / "a__b"
        d.mkdir(exist_ok=True)
        (d / "m.py").write_text('load("org/ears")', encoding="utf-8")
        return "2026-01-01T00:00:00+00:00"

    got = ins.inspect("a/b", tmp_path, meta={"size": 10}, run=run,
                      facts=lambda m, cache=None: {"size": 2 * ins.GIB,
                                                   "lane": "stt"})
    assert got.lanes["org/ears"] == "stt"


# ---- a training extra is not a runtime requirement -------------------------

def test_an_optional_extra_is_not_a_runtime_dependency(tmp_path):
    """Reading a pyproject wholesale called starvector CUDA-dependent partly on
    `deepspeed`, which sits in `[project.optional-dependencies] train`. The
    verdict was right for another reason, which is worse than being wrong: it
    hid the defect."""
    t = tree(tmp_path, {"pyproject.toml": (
        '[project]\ndependencies = ["torch", "numpy"]\n\n'
        '[project.optional-dependencies]\ntrain = ["deepspeed", "ninja"]\n')})
    found = ins.scan(t)
    assert found["cuda"] == []
    assert "deepspeed" in found["cuda_mentioned"]


def test_a_runtime_dependency_before_the_extras_still_disqualifies(tmp_path):
    """starvector really does pin flash_attn==2.7.3 inside `dependencies`, so
    the verdict stands -- for the right reason this time."""
    t = tree(tmp_path, {"pyproject.toml": (
        '[project]\ndependencies = ["torch", "flash_attn==2.7.3"]\n\n'
        '[project.optional-dependencies]\ntrain = ["deepspeed"]\n')})
    assert "flash_attn" in ins.scan(t)["cuda"]


def test_poetry_dev_groups_are_optional_too(tmp_path):
    t = tree(tmp_path, {"pyproject.toml": (
        '[project]\ndependencies = ["torch"]\n\n'
        '[tool.poetry.group.dev.dependencies]\ntriton = "*"\n')})
    assert ins.scan(t)["cuda"] == []


def test_a_dev_requirements_file_is_not_a_runtime_requirement():
    """It is read to decide whether a thing can RUN here."""
    assert "requirements-dev.txt" not in ins.DEPENDENCY_FILES


def _clone_pushed(tmp_path, when: str):
    """A fake checkout with one entry point, last touched at `when`."""
    def run(argv, cwd=None, timeout=180.0):
        tree = tmp_path / "a__b"
        tree.mkdir(exist_ok=True)
        (tree / "main.py").write_text("if __name__ == '__main__':\n    pass\n", encoding="utf-8")
        return when
    return run


def test_inspect_passes_dead_days_through_to_decide(tmp_path):
    """It accepted a ceiling override and not this one, so the abandonment
    threshold was reachable only as decide()'s captured default and no caller
    could vary it. Issue #103."""
    run = _clone_pushed(tmp_path, "2024-01-01T00:00:00+00:00")
    lenient = ins.inspect("a/b", tmp_path, meta={"size": 10}, run=run,
                          sizer=lambda m: -1, dead_days=100_000)
    assert lenient.verdict != "dead"

    strict = ins.inspect("a/b", tmp_path, meta={"size": 10}, run=run,
                         sizer=lambda m: -1, dead_days=1)
    assert strict.verdict == "dead"


def test_inspect_still_defaults_to_the_module_threshold(tmp_path):
    run = _clone_pushed(tmp_path, "2019-01-01T00:00:00+00:00")
    got = ins.inspect("a/b", tmp_path, meta={"size": 10}, run=run,
                      sizer=lambda m: -1)
    assert got.verdict == "dead"

# ---- capacity gating ------------------------------------------------------
# The ceiling was a constant describing ONE machine. Two Windows boxes differ
# from each other as much as either differs from the mini, so what a candidate
# is measured against has to be read off the machine running the sweep.


def test_the_ceiling_is_read_off_the_accelerator_not_a_constant():
    from harness import memory
    discrete = memory.Accelerator("discrete", 12.0, 10.5)
    big_card = memory.Accelerator("discrete", 24.0, 22.0)
    assert ins.ceiling_bytes(discrete) < ins.ceiling_bytes(big_card)
    assert ins.ceiling_bytes(discrete) <= 12 * ins.GIB


def test_a_weight_over_the_cards_vram_is_too_big_for_that_card():
    """Qwen3-30B-A3B at 4-bit is ~17 GB: it fits a 24 GB card and does not fit
    a 12 GB one. The same candidate, two verdicts, two machines."""
    from harness import memory
    small = ins.decide(ins.Fit(repo="x/y", weights=["w"], smallest=17 * ins.GIB,
                                 entry_points=["run.py"]),
                        ceiling=ins.ceiling_bytes(
                            memory.Accelerator("discrete", 12.0, 10.5)))
    roomy = ins.decide(ins.Fit(repo="x/y", weights=["w"], smallest=17 * ins.GIB,
                                 entry_points=["run.py"]),
                        ceiling=ins.ceiling_bytes(
                            memory.Accelerator("discrete", 24.0, 22.0)))
    assert small.verdict == "too-big"
    assert roomy.verdict != "too-big"
