"""Is this a usable generated video?

Same argument as the image checker, one dimension further. "The command exited
0" is not evidence, and neither is "ffprobe demuxed it": a broken sampler emits
a technically valid MP4 of the right length in which nothing moves, and a
stalled one emits N copies of the first frame. Both play. Both are failures.

So: does it demux, is it the size and length asked for, does each frame contain
an image rather than one colour, and does anything actually change between
frames.
"""
import shutil
import subprocess

import pytest

from harness.checks import video

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="needs ffmpeg and ffprobe")


def make(path, source, frames=8, size="128x128", fps=8):
    """Render a short clip with ffmpeg's own synthetic sources.

    The separator matters: a source that already carries an option takes ':'
    for the next one, so "color=c=gray" + "=size=..." is a parse error.
    """
    sep = ":" if "=" in source else "="
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", f"{source}{sep}size={size}:rate={fps}",
         "-frames:v", str(frames), "-pix_fmt", "yuv420p", str(path)],
        check=True, capture_output=True)
    return path


def moving(path, **kw):
    return make(path, "testsrc", **kw)


def static(path, **kw):
    return make(path, "color=c=gray", **kw)


# ---- probing ---------------------------------------------------------------

def test_probe_reports_dimensions_and_frame_count(tmp_path):
    info = video.probe(moving(tmp_path / "m.mp4", frames=8))
    assert (info.width, info.height) == (128, 128)
    assert info.frames == 8
    assert info.duration > 0


def test_probe_on_a_missing_file_raises(tmp_path):
    with pytest.raises(video.VideoError):
        video.probe(tmp_path / "nope.mp4")


def test_probe_on_something_that_is_not_a_video_raises(tmp_path):
    p = tmp_path / "n.mp4"
    p.write_bytes(b"this is not an mp4" * 100)
    with pytest.raises(video.VideoError):
        video.probe(p)


# ---- the check -------------------------------------------------------------

def test_a_moving_clip_passes(tmp_path):
    r = video.check(moving(tmp_path / "m.mp4"), expect=(128, 128), frames=8)
    assert r.ok, r.reason


def test_a_clip_where_nothing_moves_fails(tmp_path):
    """The failure worth catching: valid container, right length, plays fine,
    and the sampler never did anything."""
    r = video.check(static(tmp_path / "s.mp4"), expect=(128, 128), frames=8)
    assert not r.ok
    assert "static" in r.reason.lower() or "uniform" in r.reason.lower()


def test_wrong_dimensions_fail_with_both_numbers(tmp_path):
    r = video.check(moving(tmp_path / "m.mp4"), expect=(512, 512))
    assert not r.ok
    assert "512" in r.reason and "128" in r.reason


def test_a_short_clip_fails_when_more_frames_were_asked_for(tmp_path):
    r = video.check(moving(tmp_path / "m.mp4", frames=4), frames=22)
    assert not r.ok
    assert "22" in r.reason and "4" in r.reason


def test_the_frame_count_tolerates_a_codec_rounding_by_one(tmp_path):
    """Encoders drop or pad a frame at the boundary. Failing a 40-minute
    generation over one frame would be measuring the muxer."""
    r = video.check(moving(tmp_path / "m.mp4", frames=9), frames=8)
    assert r.ok, r.reason


def test_a_missing_file_is_a_failure_not_an_exception(tmp_path):
    r = video.check(tmp_path / "nope.mp4")
    assert not r.ok
    assert "no video" in r.reason.lower()


def test_an_empty_file_fails(tmp_path):
    p = tmp_path / "e.mp4"
    p.write_bytes(b"")
    r = video.check(p)
    assert not r.ok


def test_motion_is_published_as_a_metric(tmp_path):
    r = video.check(moving(tmp_path / "m.mp4"), expect=(128, 128), frames=8)
    assert r.metrics["motion"] > 0


def test_a_static_clip_scores_zero_motion(tmp_path):
    r = video.check(static(tmp_path / "s.mp4"), expect=(128, 128), frames=8)
    assert r.metrics["motion"] == 0.0


def test_motion_is_graded_not_binary(tmp_path):
    """The point of a metric rather than a verdict: both of these pass the
    gate, and one is far more of a video than the other. mandelbrot redraws the
    whole frame every step; testsrc moves a bar and a counter."""
    whole_frame = video.check(
        make(tmp_path / "b.mp4", "mandelbrot")).metrics["motion"]
    part_frame = video.check(moving(tmp_path / "m.mp4")).metrics["motion"]
    assert whole_frame > part_frame > 0
