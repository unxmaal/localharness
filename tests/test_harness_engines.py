"""Engine specs: turning a candidate string into a command line.

An engine is data, so adding one is a string rather than a code change. That is
the same bet the gateway makes for text, and it is what lets the CLI and the
eval suite invoke identical commands instead of two drifting copies.
"""
from pathlib import Path

import pytest

from harness.engines import Engine, resolve, spec_error

OUT = Path("/tmp/o.png")


def argv(spec, **params):
    eng = resolve(spec)
    return eng, eng.argv("a red fox", OUT, params)


# ---- mflux ----------------------------------------------------------------

def test_the_model_reaches_the_command_line_whichever_binary_runs_it():
    for model in ("z-image-turbo", "flux2-klein-4b", "schnell"):
        _, a = argv(f"mflux:{model}")
        assert a[0].startswith("mflux-generate")
        assert a[a.index("--model") + 1] == model


def test_prompt_and_output_are_single_arguments():
    _, a = argv("mflux:z-image-turbo")
    assert a[a.index("--prompt") + 1] == "a red fox"
    assert a[a.index("--output") + 1] == str(OUT)


def test_quantization_is_in_the_name_so_two_quants_do_not_collide():
    """mflux/z-image-turbo at q4 and at q8 were the same row."""
    assert resolve("mflux:z-image-turbo").name == "mflux/z-image-turbo-q8"
    assert resolve("mflux:z-image-turbo,quantize=4").name == "mflux/z-image-turbo-q4"
    assert resolve("mflux:z-image-turbo,quantize=none").name == "mflux/z-image-turbo-bf16"


def test_quantize_none_passes_no_quantize_flag():
    _, a = argv("mflux:z-image-turbo,quantize=none")
    assert "--quantize" not in a


def test_quantize_is_passed_as_the_long_flag():
    _, a = argv("mflux:z-image-turbo,quantize=4")
    assert a[a.index("--quantize") + 1] == "4"


def test_metadata_is_written_so_a_result_can_be_reproduced():
    _, a = argv("mflux:z-image-turbo")
    assert "--metadata" in a


def test_generation_params_are_forwarded():
    _, a = argv("mflux:z-image-turbo", width=512, height=768, steps=8, seed=42)
    assert a[a.index("--width") + 1] == "512"
    assert a[a.index("--height") + 1] == "768"
    assert a[a.index("--steps") + 1] == "8"
    assert a[a.index("--seed") + 1] == "42"


def test_spec_defaults_are_overridden_by_call_params():
    """The candidate pins a default; the case or the CLI flag wins."""
    _, a = argv("mflux:z-image-turbo,steps=4", steps=20)
    assert a[a.index("--steps") + 1] == "20"
    assert a.count("--steps") == 1


def test_unset_params_are_omitted_rather_than_passed_as_none():
    _, a = argv("mflux:z-image-turbo")
    assert "--width" not in a and "None" not in a


def test_mflux_produces_a_png():
    assert resolve("mflux:z-image-turbo").output_suffix == ".png"
    assert resolve("mflux:z-image-turbo").modality == "image"


# ---- h3 -------------------------------------------------------------------

def test_h3_is_a_video_engine_producing_mp4():
    eng = resolve("h3")
    assert eng.modality == "video"
    assert eng.output_suffix == ".mp4"
    assert eng.name == "h3/minimax-h3"


def test_h3_passes_the_model_directory_and_prompt(monkeypatch, tmp_path):
    monkeypatch.setenv("H3_MODEL_DIR", str(tmp_path))
    eng, a = argv("h3", frames=22, width=512, height=512, steps=20, seed=7)
    assert a[a.index("-d") + 1] == str(tmp_path)
    assert a[a.index("-p") + 1] == "a red fox"
    assert a[a.index("--frames") + 1] == "22"
    assert a[a.index("--seed") + 1] == "7"


def test_h3_streams_from_ssd_by_default():
    """134GiB of weights on a 32GB machine. Without this it cannot start."""
    _, a = argv("h3")
    assert "--ssd-streaming" in a


def test_h3_ssd_streaming_can_be_turned_off_for_the_studio():
    _, a = argv("h3", ssd_streaming=False)
    assert "--ssd-streaming" not in a


def test_h3_seconds_and_frames_are_mutually_exclusive():
    """Passing both makes h3 silently pick one; the caller should be told."""
    with pytest.raises(ValueError):
        argv("h3", frames=22, seconds=2)


# ---- diffusers, the image lane on a machine with an NVIDIA card -----------
#
# A different tool for the same lane. mflux is MLX and does not run here, so
# the engine is named for what it runs rather than for the lane it serves,
# the same way mflux and h3 are.


def test_diffusers_is_an_image_engine_producing_png():
    eng = resolve("diffusers:stabilityai/sdxl-turbo")
    assert eng.modality == "image"
    assert eng.output_suffix == ".png"
    assert "sdxl-turbo" in eng.name


def test_diffusers_passes_the_model_prompt_and_output():
    _, a = argv("diffusers:stabilityai/sdxl-turbo")
    assert a[a.index("--model") + 1] == "stabilityai/sdxl-turbo"
    assert a[a.index("--prompt") + 1] == "a red fox"
    assert a[a.index("--output") + 1] == str(OUT)


def test_diffusers_forwards_the_generation_params():
    _, a = argv("diffusers:stabilityai/sdxl-turbo",
                steps=4, width=768, height=768, seed=7)
    for flag, value in (("--steps", "4"), ("--width", "768"),
                        ("--height", "768"), ("--seed", "7")):
        assert a[a.index(flag) + 1] == value


def test_diffusers_omits_a_param_that_was_never_set():
    """--seed None is a seed of the string None, which reproduces nothing."""
    _, a = argv("diffusers:stabilityai/sdxl-turbo")
    assert "--seed" not in a


def test_diffusers_spec_defaults_are_overridden_by_call_params():
    _, a = argv("diffusers:stabilityai/sdxl-turbo,steps=1", steps=8)
    assert a[a.index("--steps") + 1] == "8"


def test_the_diffusers_generator_is_a_path_not_a_name(monkeypatch, tmp_path):
    """Same reason H3_BIN is overridable: this one is a script in the checkout
    rather than something installed onto PATH, and a test or another machine
    has to be able to point it elsewhere."""
    fake = tmp_path / "image-cuda.sh"
    monkeypatch.setenv("IMAGE_CUDA_BIN", str(fake))
    _, a = argv("diffusers:stabilityai/sdxl-turbo")
    assert a[0] == str(fake)


def test_diffusers_rejects_an_unknown_option_before_the_model_loads():
    with pytest.raises(ValueError) as e:
        resolve("diffusers:stabilityai/sdxl-turbo,quantise=4")
    assert "quantise" in str(e.value)


def test_diffusers_needs_a_model():
    with pytest.raises(ValueError):
        resolve("diffusers")


# ---- diffusers-video (the video lane on a machine with an NVIDIA card) ----
#
# h3 is Metal shaders over a 134 GiB checkpoint and has no build that runs
# here, so this lane is a different model as well as a different tool. WHICH
# model is left to the eval: the engine takes any diffusers video repo id, the
# same way the image one takes any text-to-image id.


def test_diffusers_video_is_a_video_engine_producing_mp4():
    eng = resolve("diffusers-video:Lightricks/LTX-Video")
    assert eng.modality == "video"
    assert eng.output_suffix == ".mp4"
    assert "LTX-Video" in eng.name


def test_diffusers_video_passes_the_model_prompt_and_output():
    _, a = argv("diffusers-video:Lightricks/LTX-Video")
    assert a[a.index("--model") + 1] == "Lightricks/LTX-Video"
    assert a[a.index("--prompt") + 1] == "a red fox"
    assert a[a.index("--output") + 1] == str(OUT)


def test_diffusers_video_forwards_the_generation_params():
    _, a = argv("diffusers-video:Lightricks/LTX-Video",
                frames=25, steps=30, width=512, height=512, seed=7)
    for flag, value in (("--frames", "25"), ("--steps", "30"),
                        ("--width", "512"), ("--height", "512"),
                        ("--seed", "7")):
        assert a[a.index(flag) + 1] == value


def test_diffusers_video_offloads_to_host_memory_by_default():
    """A video model does not fit 12 GB resident. Offloading is what makes the
    lane run at all here, so it is the default rather than a flag to remember."""
    _, a = argv("diffusers-video:Lightricks/LTX-Video")
    assert "--no-offload" not in a


def test_diffusers_video_offload_can_be_turned_off_on_a_larger_card():
    _, a = argv("diffusers-video:Lightricks/LTX-Video", offload=False)
    assert "--no-offload" in a


def test_the_video_generator_is_a_path_not_a_name(monkeypatch, tmp_path):
    fake = tmp_path / "video-cuda.sh"
    monkeypatch.setenv("VIDEO_CUDA_BIN", str(fake))
    _, a = argv("diffusers-video:Lightricks/LTX-Video")
    assert a[0] == str(fake)


def test_diffusers_video_gets_the_long_timeout_a_video_needs():
    """The image engine's 15 minutes is not enough; h3 allows six hours."""
    assert resolve("diffusers-video:Lightricks/LTX-Video").timeout >= 3600


def test_diffusers_video_needs_a_model():
    with pytest.raises(ValueError):
        resolve("diffusers-video")


# ---- spec parsing ---------------------------------------------------------

def test_unknown_engine_names_the_known_ones():
    with pytest.raises(ValueError) as e:
        resolve("stablediffusion:xl")
    assert "mflux" in str(e.value) and "h3" in str(e.value)


def test_unknown_option_is_rejected_at_parse_time_not_after_a_40_minute_run():
    with pytest.raises(ValueError) as e:
        resolve("mflux:z-image-turbo,quantise=4")
    assert "quantise" in str(e.value)


def test_a_bare_engine_name_is_an_error_when_a_model_is_required():
    with pytest.raises(ValueError):
        resolve("mflux")


def test_spec_error_lists_the_grammar():
    assert "engine:model" in spec_error("x")


def test_engine_is_immutable_so_a_run_cannot_mutate_the_candidate():
    with pytest.raises(Exception):
        resolve("mflux:z-image-turbo").name = "other"


def test_engines_are_plain_data_and_carry_their_spec_for_the_record():
    assert resolve("mflux:z-image-turbo,steps=8").spec == "mflux:z-image-turbo,steps=8"
    assert isinstance(resolve("h3"), Engine)


# ---- mflux entry points ----------------------------------------------------
#
# `mflux-generate --help` lists every built-in model under --model, because all
# the binaries share one parser. It is not a list of what mflux-generate can
# run: it rejects flux2-klein-4b at RUNTIME with "not supported by
# mflux-generate. Use mflux-generate-flux2 instead." Reading the help and
# believing it cost a real run.

import shutil  # noqa: E402

from harness.engines import ENTRY_POINTS, mflux_binary  # noqa: E402


def test_flux2_models_use_the_flux2_entry_point():
    assert mflux_binary("flux2-klein-4b") == "mflux-generate-flux2"
    assert mflux_binary("flux2-klein-9b") == "mflux-generate-flux2"
    _, a = argv("mflux:flux2-klein-4b")
    assert a[0] == "mflux-generate-flux2"


def test_z_image_turbo_and_z_image_are_different_entry_points():
    assert mflux_binary("z-image-turbo") == "mflux-generate-z-image-turbo"
    assert mflux_binary("z-image") == "mflux-generate-z-image"


def test_the_flux1_family_uses_the_bare_entry_point():
    for model in ("dev", "schnell", "krea-dev"):
        assert mflux_binary(model) == "mflux-generate"


def test_the_model_is_still_passed_so_the_binary_does_not_use_its_own_default():
    """Every family binary takes --model with a family default, so omitting it
    silently generates with a different model than the candidate names."""
    _, a = argv("mflux:flux2-klein-9b")
    assert a[a.index("--model") + 1] == "flux2-klein-9b"


def test_an_unknown_model_is_rejected_at_spec_time_with_the_known_list():
    with pytest.raises(ValueError) as e:
        resolve("mflux:stable-diffusion-xl")
    assert "z-image-turbo" in str(e.value)


@pytest.mark.skipif(shutil.which("mflux-generate") is None,
                    reason="mflux is not installed")
def test_every_entry_point_in_the_table_actually_exists():
    """The table is hand-maintained against mflux's entry_points.txt. This is
    what catches it going stale on an upgrade."""
    missing = [b for b in set(ENTRY_POINTS.values()) if shutil.which(b) is None]
    assert not missing, f"engines.py names binaries mflux does not ship: {missing}"


# ---- working directory -----------------------------------------------------

def test_h3_runs_from_its_own_directory():
    """h3 compiles h3_shaders.metal at startup and looks for it RELATIVE TO
    CWD, so invoking it from anywhere else dies with

        h3: cannot compile h3_shaders.metal: ... no such file

    after loading the tokenizer and half the text encoder, which reads like a
    model problem rather than a path one.
    """
    eng = resolve("h3")
    assert eng.cwd, "h3 must declare a working directory"
    assert Path(eng.cwd).name == "h3.c"


def test_h3_cwd_follows_the_binary_when_it_is_overridden(monkeypatch, tmp_path):
    binary = tmp_path / "elsewhere" / "h3"
    binary.parent.mkdir()
    monkeypatch.setenv("H3_BIN", str(binary))
    assert resolve("h3").cwd == str(binary.parent)


def test_mflux_needs_no_working_directory():
    """It resolves everything through HF_HOME, so pinning a cwd would only
    invent a way to break it."""
    assert resolve("mflux:z-image-turbo").cwd is None


# ---- h3 frame floor --------------------------------------------------------

def test_h3_rejects_a_frame_count_below_its_floor():
    """h3 CLAMPS to 22 rather than refusing: ask for 8 and you get 22, twelve
    minutes later, and the eval case that asked for 8 then fails its own
    assertion. Saying so up front is the same principle as rejecting a bad
    engine spec before the run.

    The floor is h3.c:861, `if (h3_align_frame_count(params->frames) < 22)`.
    """
    with pytest.raises(ValueError) as e:
        argv("h3", frames=8)
    assert "22" in str(e.value) and "8" in str(e.value)


def test_h3_rejects_a_frame_count_below_its_hard_minimum():
    """h3.c:514 refuses fewer than 5 outright."""
    with pytest.raises(ValueError):
        argv("h3", frames=2)


def test_h3_accepts_its_floor():
    _, a = argv("h3", frames=22)
    assert a[a.index("--frames") + 1] == "22"


def test_h3_seconds_below_a_second_is_rejected_for_the_same_reason():
    """--seconds 0 is 0 frames at 24fps, well under the floor."""
    with pytest.raises(ValueError):
        argv("h3", seconds=0)


def test_h3_one_second_is_above_the_floor():
    _, a = argv("h3", seconds=1)
    assert a[a.index("--seconds") + 1] == "1"
