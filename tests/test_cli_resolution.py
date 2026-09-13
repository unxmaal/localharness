"""The CLI and the suite must sit the same exam.

#157. `lh image` declared --width with no default, so it inherited mflux's
1024 while every evals/cases/image/*.yaml pinned 512. Both numbers were
correct and they were not comparable, which is how two good measurements came
to look like a regression (#141, #142).

Same shape as tests/test_preflight_sh.py holding the shell script to the
player list in harness/audio.py: two files that must agree, made to prove it.
"""
from pathlib import Path

import yaml

from harness import cli

CASES = Path(__file__).resolve().parent.parent / "evals" / "cases"


def parse(*argv):
    return cli.build_parser().parse_args(argv)


def test_the_image_lane_runs_what_the_suite_measured():
    for case in sorted((CASES / "image").glob("*.yaml")):
        params = yaml.safe_load(case.read_text(encoding="utf-8"))["params"]
        assert params["width"] == cli.DEFAULT_RESOLUTION, case.name
        assert params["height"] == cli.DEFAULT_RESOLUTION, case.name


def test_image_and_video_carry_a_resolution_without_being_asked():
    for command in ("image", "video"):
        a = parse(command, "a red fox")
        assert a.width == cli.DEFAULT_RESOLUTION
        assert a.height == cli.DEFAULT_RESOLUTION


def test_an_explicit_size_still_wins():
    a = parse("image", "a red fox", "--width", "768", "--height", "768")
    assert (a.width, a.height) == (768, 768)


def test_the_svg_tracer_agrees_with_the_image_lane():
    """It generates a raster first, so a different default would mean the icon
    preset was tuned at a size the image lane never runs."""
    a = parse("svg", "a gear", "--method", "trace")
    assert (a.width, a.height) == (cli.DEFAULT_RESOLUTION,
                                   cli.DEFAULT_RESOLUTION)


def test_the_video_case_is_smaller_on_purpose():
    """Not a parity failure. 512x512x22 frames is 40.5 minutes on the M2 Pro,
    so the case is sized to let the suite finish; the CLI default is the size
    that timing was taken at."""
    params = yaml.safe_load(
        (CASES / "video" / "fox-run.yaml").read_text(encoding="utf-8"))["params"]
    assert params["width"] < cli.DEFAULT_RESOLUTION


def test_a_reported_number_carries_the_size_that_produced_it(capsys):
    cli.say(path="/tmp/x.png", seconds=41.0, peak_kb=11.4 * 1024 * 1024,
            size="512x512",
            human="/tmp/x.png  (41.0s, peak 11.4 GiB, 512x512)")
    assert "512x512" in capsys.readouterr().out
